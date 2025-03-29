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
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Generator

import yt_dlp
import speech_recognition as sr
from pydub import AudioSegment

# Configure main logger for operational logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Default to WARNING level to reduce noise

# Create a special logger just for transcriptions
transcript_logger = logging.getLogger("transcript")
transcript_logger.setLevel(logging.INFO)

# Add handlers with formatters
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))  # Simple format for transcript output
transcript_logger.addHandler(console_handler)

debug_handler = logging.StreamHandler()
debug_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", 
                                            "%Y-%m-%d %H:%M:%S"))
logger.addHandler(debug_handler)

# Global verbosity flag
VERBOSE_OUTPUT = False

def configure_logging(verbose=False):
    """Configure logging based on verbosity level.
    
    Args:
        verbose: If True, enable detailed logging
    """
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = verbose
    
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


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
        
        # Add stream position tracking
        self.stream_position = 0  # Track position in the stream in seconds
        self.last_successful_position = 0  # Last position that produced good transcription
        
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
        self.last_transcription_time = time.time()
        
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
                    # For live streams, we need a special approach that works with HTTP streaming
                    # and doesn't rely on seeking within the stream
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-reconnect", "1",  # Reconnect if the connection is lost
                        "-reconnect_streamed", "1",  # Reconnect if the stream fails
                        "-reconnect_delay_max", "5",  # Max delay between reconnection attempts
                        "-i", self.stream_url,
                        "-t", str(self.buffer_size),  # Duration to capture
                        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                        "-loglevel", "warning",
                        chunk_file
                    ]
                    
                    logger.info(f"Capturing audio chunk from stream at position approximately {self.stream_position}s")
                    
                    # Run ffmpeg process with a timeout
                    process = subprocess.run(
                        ffmpeg_cmd, 
                        capture_output=True, 
                        text=True,
                        check=False,
                        timeout=self.buffer_size * 2  # Timeout after 2x buffer_size seconds
                    )
                    
                    # If the file was created successfully, add it to the queue
                    if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                        # Load the audio file
                        audio = AudioSegment.from_wav(chunk_file)
                        
                        # Add metadata about which segment of the stream this represents
                        metadata = {
                            'start_position': self.stream_position,
                            'duration': self.buffer_size,
                            'timestamp': time.time()
                        }
                        
                        # Put both audio and metadata in the queue
                        self.audio_queue.put((audio, metadata))
                        
                        # For live streams, we need to ensure we don't miss content
                        # Use more aggressive overlapping to ensure continuous coverage
                        # Use a smaller increment to create more overlap between chunks
                        position_increment = max(1, self.buffer_size // 4)  # Use 1/4 of buffer or at least 1 second
                        self.stream_position += position_increment
                        logger.info(f"Stream position advanced to {self.stream_position} seconds")
                        
                        # Reset the consecutive empty counter if we got a good chunk
                        consecutive_empty = 0
                    else:
                        logger.warning(f"Failed to capture audio chunk: file empty or not created")
                        
                        # If we failed to get a chunk, log more information
                        if process.stderr:
                            logger.info(f"FFmpeg stderr: {process.stderr}")
                        if process.stdout:
                            logger.info(f"FFmpeg stdout: {process.stdout}")
                            
                        # Use a smaller increment when there's an error
                        self.stream_position += 0.5
                        logger.info(f"Advanced stream position by 0.5 seconds due to error")
                        
                        # Add a delay before retrying to avoid hammering the server
                        time.sleep(1)
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"FFmpeg process timed out after {self.buffer_size * 2} seconds")
                    self.stream_position += 1  # Move forward a little bit
                    time.sleep(1)  # Wait a bit before retrying
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
            last_transcript = ""
            last_transcript_position = 0
            consecutive_repeats = 0
            consecutive_empty = 0
            no_transcription_counter = 0
            
            while self.is_running:
                try:
                    # Get an audio chunk from the queue, with a timeout
                    audio, metadata = self.audio_queue.get(timeout=5)
                    logger.debug(f"Processing chunk from position {metadata['start_position']} (queue size: {self.audio_queue.qsize()})")
                    
                    # Export to a temporary file for Speech Recognition
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        
                    audio.export(temp_path, format="wav")
                    
                    # Log information about the segment being processed
                    logger.debug(f"Processing audio segment from position {metadata['start_position']} to {metadata['start_position'] + metadata['duration']} seconds")
                    
                    # Transcribe the audio
                    with sr.AudioFile(temp_path) as source:
                        audio_data = self.recognizer.record(source)
                        try:
                            # Using Google Speech Recognition
                            text = self.recognizer.recognize_google(audio_data)
                            if text:
                                # Check for repetition issues
                                if text == last_transcript and metadata['start_position'] > last_transcript_position:
                                    consecutive_repeats += 1
                                    logger.debug(f"Detected repeated transcription ({consecutive_repeats} times): {text}")
                                    
                                    # If we've seen the same text too many times, trigger resync
                                    if consecutive_repeats >= 2:
                                        logger.warning(f"Detected repeated transcription {consecutive_repeats} times, resynchronizing stream")
                                        self._resync_stream(last_successful_position=last_transcript_position)
                                        consecutive_repeats = 0
                                        # Skip adding this repeated text to the transcript
                                        self.audio_queue.task_done()
                                        continue
                                else:
                                    # New text, reset the counters
                                    consecutive_repeats = 0
                                    consecutive_empty = 0
                                    last_transcript = text
                                    last_transcript_position = metadata['start_position']
                                    self.last_successful_position = metadata['start_position']
                                
                                # Update transcription time
                                self.last_transcription_time = time.time()
                                
                                # Create timestamped entry for file output
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                transcription_entry = f"[{timestamp}] {text}"
                                
                                # For console display, use the appropriate logger based on verbosity
                                if not VERBOSE_OUTPUT:
                                    # Just show the plain text without timestamp or prefix
                                    transcript_logger.info(text)
                                else:
                                    # Show full debug info
                                    logger.info(f"Transcribed: {transcription_entry}")
                                
                                # Add to transcription list with timestamp for file output
                                self.transcription.append(transcription_entry)
                                
                                # Write to file periodically
                                current_time = time.time()
                                if current_time - self.last_write_time >= self.output_interval:
                                    self._write_transcription()
                                    self.last_write_time = current_time
                            else:
                                consecutive_empty += 1
                                logger.debug(f"No text detected in segment (empty count: {consecutive_empty})")
                                
                                # If too many consecutive empty segments, we might be in a silent part
                                if consecutive_empty >= 5:
                                    logger.info(f"Multiple empty segments detected, may be in silent section. Advancing position.")
                                    consecutive_empty = 0
                                    # Advance a bit more to get to new content
                                    self.stream_position += 5
                                    
                        except sr.UnknownValueError:
                            logger.debug("Speech Recognition could not understand audio")
                            consecutive_empty += 1
                            
                            # If too many consecutive failures to understand, we might need to resync
                            if consecutive_empty >= 8:
                                logger.warning(f"Multiple recognition failures, resynchronizing stream")
                                self._resync_stream()
                                consecutive_empty = 0
                                
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
                    # Check if we haven't had a transcription in a while
                    if time.time() - self.last_transcription_time > 30:  # 30 seconds without transcription
                        logger.warning("No transcription for 30 seconds, attempting to resync")
                        self._resync_stream()
                        self.last_transcription_time = time.time()  # Reset timer
                    
                    # No audio in the queue, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    
        except Exception as e:
            logger.error(f"Error in audio processing thread: {e}")
            
    def _resync_stream(self, last_successful_position=None):
        """Attempt to resynchronize the stream if we detect issues.
        
        Args:
            last_successful_position: Last stream position that produced good transcription
        """
        logger.warning("Attempting to resynchronize stream")
        
        previous_position = self.stream_position
        
        # Determine the best resyncing strategy
        if last_successful_position is not None and last_successful_position > 0:
            # We have a good reference point - try multiple strategies
            
            # Strategy 1: Try continuing from the last known good position with a small jump
            small_jump = 2  # 2 seconds is a small jump
            new_position = last_successful_position + small_jump
            logger.info(f"Resync strategy 1: Using last successful position {last_successful_position} + {small_jump}s")
        else:
            # No good reference - make a more aggressive jump
            # This helps if we're stuck in a problematic section of the stream
            jump_amount = 15  # More aggressive 15 second jump
            new_position = self.stream_position + jump_amount
            logger.info(f"Resync strategy: Aggressive jump forward by {jump_amount}s (no good reference point)")
        
        # Update stream position
        self.stream_position = new_position
        
        # Clear the audio queue to discard any potentially problematic chunks
        cleared_chunks = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
                cleared_chunks += 1
            except queue.Empty:
                break
                
        logger.info(f"Stream resynced from {previous_position}s to {self.stream_position}s (cleared {cleared_chunks} chunks)")
        
        # Force an immediate write of current transcriptions to avoid losing data
        self._write_transcription()
    
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
            
            # Clear the list of transcribed entries that have been written
            self.transcription = []
            
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
        self.stream_position = 0  # Reset position when starting
        self.last_successful_position = 0
        self.last_transcription_time = time.time()
        
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
    
    # Enable more verbose debugging for troubleshooting
    logger.info(f"Setting up transcription with buffer size: {buffer_size}s and output interval: {output_interval}s")
    
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
    transcribe_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output with debugging information (default: False)'
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
        # Configure logging based on verbose flag
        configure_logging(verbose=getattr(args, 'verbose', False))
        
        # If not in verbose mode, show a simple starting message
        if not VERBOSE_OUTPUT:
            print("Starting transcription... Press Ctrl+C to stop")
            print("---------------------------------------------")
        
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
