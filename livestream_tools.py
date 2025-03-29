#!/usr/bin/env python3
"""
Live Stream Tools - YouTube Live Stream Discovery, Connection, and Transcription

This script provides functionality to:
- List available YouTube live streams based on search criteria
- Connect to YouTube live streams
- Transcribe YouTube live streams in real-time

All downloads and transcripts are saved to a 'downloads' folder within the current directory.
"""

import os
import sys
import time
import json
import argparse
import logging
import threading
import tempfile
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Generator

import yt_dlp
import speech_recognition as sr
from pydub import AudioSegment
import ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: The path to the directory to ensure exists.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def list_live_streams(query: Optional[str] = None, 
                     max_results: int = 10,
                     order: str = "relevance") -> List[Dict[str, Any]]:
    """List currently available YouTube live streams.
    
    This function uses yt-dlp to search for live streams based on the provided query.
    
    Args:
        query: Search query to filter live streams. If None, returns most popular live streams.
        max_results: Maximum number of results to return (default: 10)
        order: Order of results ('relevance', 'date', 'viewCount')
    
    Returns:
        List of dictionaries containing live stream information (title, channel, URL, etc.)
    """
    # Ensure the downloads directory exists
    downloads_dir = os.path.join(os.getcwd(), 'downloads')
    ensure_directory_exists(downloads_dir)
    
    # Build the yt-dlp search query
    search_query = "ytsearch{}:live".format(max_results)
    if query:
        search_query += " " + query
    
    logger.info(f"Searching for live streams with query: {search_query}")
    
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'skip_download': True,
        'no_warnings': True,
        'simulate': True,
        'ignoreerrors': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            
            if info and 'entries' in info:
                live_streams = []
                for entry in info['entries']:
                    # Get detailed info for each entry to check if it's a live stream
                    try:
                        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                        detailed_info = ydl.extract_info(video_url, download=False)
                        
                        # Only include entries that are actually live
                        if detailed_info.get('is_live', False):
                            stream_info = {
                                'id': entry.get('id', 'Unknown ID'),
                                'title': entry.get('title', 'Unknown Title'),
                                'channel': entry.get('channel', 'Unknown Channel'),
                                'url': video_url,
                                'uploader': entry.get('uploader', 'Unknown Uploader'),
                                'viewer_count': detailed_info.get('concurrent_views', 'Unknown'),
                                'description': detailed_info.get('description', 'No description'),
                            }
                            live_streams.append(stream_info)
                            
                            # Stop once we've found max_results live streams
                            if len(live_streams) >= max_results:
                                break
                    except Exception as e:
                        logger.warning(f"Could not get detailed info for video {entry.get('id', 'Unknown')}: {e}")
                        continue
                
                logger.info(f"Found {len(live_streams)} live streams")
                return live_streams
            else:
                logger.warning("No live streams found matching the query")
                return []
    except Exception as e:
        logger.error(f"Error searching for live streams: {e}")
        return []


def get_live_stream_url(video_url: str, format: str = "best") -> Optional[str]:
    """Get the direct streaming URL for a YouTube live stream.
    
    Args:
        video_url: YouTube video URL
        format: Format specification for yt-dlp (default: 'best')
    
    Returns:
        Direct streaming URL, or None if extraction failed
    """
    logger.info(f"Extracting live stream URL for: {video_url}")
    
    ydl_opts = {
        'quiet': True,
        'format': format,
        'skip_download': True,
        'no_warnings': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if not info.get('is_live', False):
                logger.error(f"The video at {video_url} is not a live stream")
                return None
            
            # Get the URL for the requested format
            stream_url = info.get('url')
            if stream_url:
                logger.info(f"Successfully extracted live stream URL")
                return stream_url
            else:
                logger.error(f"Could not extract stream URL from the live video")
                return None
    except Exception as e:
        logger.error(f"Error extracting live stream URL: {e}")
        return None


class LiveStreamTranscriber:
    """Class for transcribing YouTube live streams in real-time."""
    
    def __init__(self, 
                 stream_url: str, 
                 output_path: Optional[str] = None,
                 buffer_size: int = 10,  # seconds
                 output_interval: int = 30):  # seconds
        """Initialize the live stream transcriber.
        
        Args:
            stream_url: Direct URL to the live stream
            output_path: Path to save the transcript (default: None, generates a filename)
            buffer_size: Size of audio buffer in seconds (default: 10)
            output_interval: How often to write transcript to file in seconds (default: 30)
        """
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.output_interval = output_interval
        
        # Create downloads directory if it doesn't exist
        downloads_dir = os.path.join(os.getcwd(), 'downloads')
        ensure_directory_exists(downloads_dir)
        
        # Set output path
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(
                downloads_dir, f"live_transcript_{timestamp}.txt"
            )
        else:
            self.output_path = output_path
        
        # Initialize the speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # For tracking state
        self.is_running = False
        self.threads = []
        self.transcription = []
        self.last_write_time = time.time()
        
    def _stream_audio(self):
        """Stream audio from the live stream URL and queue chunks for processing."""
        temp_dir = tempfile.mkdtemp()
        chunk_file = os.path.join(temp_dir, "chunk.wav")
        
        try:
            logger.info(f"Starting audio streaming from live stream")
            
            # Use ffmpeg to capture audio from the stream in chunks
            while self.is_running:
                # Capture a chunk of audio using ffmpeg
                try:
                    # Use ffmpeg to extract audio for buffer_size seconds
                    process = (
                        ffmpeg
                        .input(self.stream_url, t=self.buffer_size)
                        .output(chunk_file, acodec='pcm_s16le', ar=16000, ac=1, loglevel='error')
                        .overwrite_output()
                        .run_async(pipe_stdout=True, pipe_stderr=True)
                    )
                    
                    # Wait for the process to complete
                    stdout, stderr = process.communicate()
                    
                    # If the file was created successfully, add it to the queue
                    if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                        # Load the audio file
                        audio = AudioSegment.from_wav(chunk_file)
                        self.audio_queue.put(audio)
                    else:
                        logger.warning("Failed to capture audio chunk, retrying...")
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error capturing audio chunk: {e}")
                    if not self.is_running:
                        break
                    time.sleep(1)  # Wait before retrying
        
        except Exception as e:
            logger.error(f"Error in audio streaming thread: {e}")
        finally:
            # Clean up temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temporary files: {e}")
    
    def _process_audio(self):
        """Process audio chunks from the queue and perform transcription."""
        try:
            logger.info("Starting audio processing for transcription")
            
            while self.is_running:
                try:
                    # Get an audio chunk from the queue, with a timeout
                    audio = self.audio_queue.get(timeout=5)
                    
                    # Export to a temporary file for Speech Recognition
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        
                    audio.export(temp_path, format="wav")
                    
                    # Transcribe the audio
                    with sr.AudioFile(temp_path) as source:
                        audio_data = self.recognizer.record(source)
                        try:
                            # Using Google Speech Recognition
                            text = self.recognizer.recognize_google(audio_data)
                            if text:
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                transcription_entry = f"[{timestamp}] {text}"
                                logger.info(f"Transcribed: {transcription_entry}")
                                
                                # Add to transcription list with timestamp
                                self.transcription.append(transcription_entry)
                                
                                # Write to file periodically
                                current_time = time.time()
                                if current_time - self.last_write_time >= self.output_interval:
                                    self._write_transcription()
                                    self.last_write_time = current_time
                                    
                        except sr.UnknownValueError:
                            logger.debug("Speech Recognition could not understand audio")
                        except sr.RequestError as e:
                            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {temp_path}: {e}")
                        
                    # Mark the task as done
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # No audio in the queue, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    
        except Exception as e:
            logger.error(f"Error in audio processing thread: {e}")
            
    def _write_transcription(self):
        """Write current transcription to the output file."""
        if not self.transcription:
            return
            
        try:
            ensure_directory_exists(os.path.dirname(self.output_path))
            
            # Append to the file
            with open(self.output_path, 'a', encoding='utf-8') as f:
                for entry in self.transcription:
                    f.write(entry + "\n")
            
            logger.info(f"Wrote {len(self.transcription)} new transcription entries to {self.output_path}")
        except Exception as e:
            logger.error(f"Error writing transcription to file: {e}")
    
    def start(self) -> bool:
        """Start the live stream transcription process.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Transcription is already running")
            return False
            
        logger.info(f"Starting live stream transcription")
        logger.info(f"Transcript will be saved to: {self.output_path}")
        
        self.is_running = True
        self.transcription = []
        
        # Initialize empty output file
        try:
            ensure_directory_exists(os.path.dirname(self.output_path))
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Live Stream Transcription\n")
                f.write(f"# Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            logger.error(f"Error initializing output file: {e}")
            self.is_running = False
            return False
        
        # Start audio streaming thread
        stream_thread = threading.Thread(target=self._stream_audio)
        stream_thread.daemon = True
        stream_thread.start()
        self.threads.append(stream_thread)
        
        # Start audio processing thread
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.daemon = True
        process_thread.start()
        self.threads.append(process_thread)
        
        logger.info("Transcription process started")
        return True
    
    def stop(self) -> None:
        """Stop the live stream transcription process."""
        if not self.is_running:
            logger.warning("Transcription is not running")
            return
            
        logger.info("Stopping transcription process...")
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Final write to file
        self._write_transcription()
        
        # Add end marker to the file
        try:
            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write(f"\n# Transcription ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            logger.error(f"Error finalizing output file: {e}")
        
        logger.info(f"Transcription process stopped")
        logger.info(f"Full transcript saved to: {self.output_path}")


def transcribe_live_stream(
    video_url: str, 
    output_path: Optional[str] = None,
    format: str = "best",
    buffer_size: int = 10,
    output_interval: int = 30,
    duration: Optional[int] = None
) -> Optional[str]:
    """Connect to a YouTube live stream and transcribe it in real-time.
    
    Args:
        video_url: YouTube video URL for the live stream
        output_path: Path to save the transcript (default: None, generates a filename)
        format: Format specification for yt-dlp (default: 'best')
        buffer_size: Size of audio buffer in seconds (default: 10)
        output_interval: How often to write transcript to file in seconds (default: 30)
        duration: How long to transcribe for in seconds (default: None, runs until stopped)
    
    Returns:
        Path to the saved transcript, or None if failed
    """
    # Extract the direct streaming URL
    stream_url = get_live_stream_url(video_url, format)
    if not stream_url:
        logger.error("Failed to get live stream URL")
        return None
    
    # Create and start the transcriber
    transcriber = LiveStreamTranscriber(
        stream_url=stream_url,
        output_path=output_path,
        buffer_size=buffer_size,
        output_interval=output_interval
    )
    
    if not transcriber.start():
        logger.error("Failed to start transcription process")
        return None
    
    try:
        logger.info("Transcription in progress...")
        logger.info("Press Ctrl+C to stop transcription")
        
        start_time = time.time()
        
        # Run for the specified duration or until interrupted
        while True:
            if duration and time.time() - start_time >= duration:
                logger.info(f"Reached specified duration of {duration} seconds")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
    finally:
        # Stop transcription
        transcriber.stop()
    
    return transcriber.output_path


def main() -> int:
    """Main entry point for the command line interface.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(description='YouTube Live Stream Tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List live streams command
    list_parser = subparsers.add_parser('list', help='List available YouTube live streams')
    list_parser.add_argument('--query', '-q', help='Search query to filter live streams')
    list_parser.add_argument(
        '--max-results', '-m', type=int, default=10,
        help='Maximum number of results to return (default: 10)'
    )
    list_parser.add_argument(
        '--order', '-o', default='relevance', 
        choices=['relevance', 'date', 'viewCount'],
        help='Order of results (default: relevance)'
    )
    
    # Transcribe live stream command
    transcribe_parser = subparsers.add_parser(
        'transcribe', 
        help='Connect to a YouTube live stream and transcribe it in real-time'
    )
    transcribe_parser.add_argument('url', help='URL of the YouTube live stream')
    transcribe_parser.add_argument(
        '--output', '-o', 
        help='Output path for the transcript (default: auto-generated)'
    )
    transcribe_parser.add_argument(
        '--format', '-f', default='best',
        help='Format specification for yt-dlp (default: best)'
    )
    transcribe_parser.add_argument(
        '--buffer-size', '-b', type=int, default=10,
        help='Size of audio buffer in seconds (default: 10)'
    )
    transcribe_parser.add_argument(
        '--output-interval', '-i', type=int, default=30,
        help='How often to write transcript to file in seconds (default: 30)'
    )
    transcribe_parser.add_argument(
        '--duration', '-d', type=int,
        help='How long to transcribe for in seconds (default: runs until stopped)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'list':
        live_streams = list_live_streams(args.query, args.max_results, args.order)
        
        if live_streams:
            print(f"\nFound {len(live_streams)} live streams:")
            print("=" * 80)
            
            for i, stream in enumerate(live_streams, 1):
                print(f"{i}. {stream['title']}")
                print(f"   Channel: {stream['channel']}")
                print(f"   URL: {stream['url']}")
                print(f"   Viewers: {stream['viewer_count']}")
                print("-" * 80)
            
            return 0
        else:
            logger.error("No live streams found matching your criteria")
            return 1
    
    elif args.command == 'transcribe':
        transcript_path = transcribe_live_stream(
            video_url=args.url,
            output_path=args.output,
            format=args.format,
            buffer_size=args.buffer_size,
            output_interval=args.output_interval,
            duration=args.duration
        )
        
        if transcript_path:
            logger.info(f"Transcription complete! Transcript saved to: {transcript_path}")
            return 0
        else:
            logger.error("Failed to transcribe live stream")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
