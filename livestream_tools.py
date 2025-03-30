#!/usr/bin/env python3
"""
Live Stream Tools - YouTube Live Stream Discovery, Connection, Transcription, and Clip Extraction

This script provides functionality to:
- List available YouTube live streams based on search criteria
- Connect to YouTube live streams
- Transcribe YouTube live streams in real-time
- Extract periodic short clips from YouTube live streams

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
import shutil

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
                 buffer_size: int = 2,  # seconds - reduced from 4
                 min_buffer_size: int = 1,  # seconds - reduced from 2
                 max_buffer_size: int = 4,  # seconds - reduced from 8
                 overlap_ratio: float = 0.15,  # overlap between consecutive chunks - reduced from 0.25 
                 output_interval: int = 5,  # seconds - reduced from 10
                 max_queue_size: int = 20):  # maximum audio chunks in queue
        """Initialize the live stream transcriber.
        
        Args:
            stream_url: Direct URL to the live stream
            output_path: Path to save the transcript (default: None, generates a filename)
            buffer_size: Initial size of audio buffer in seconds (default: 4)
            min_buffer_size: Minimum buffer size in seconds for adaptive sizing (default: 2)
            max_buffer_size: Maximum buffer size in seconds for adaptive sizing (default: 8)
            overlap_ratio: Ratio of overlap between consecutive chunks (default: 0.25)
            output_interval: How often to write transcript to file in seconds (default: 10)
        """
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        self.overlap_ratio = overlap_ratio
        self.output_interval = output_interval
        
        # Add enhanced stream position tracking
        self.stream_position = 0  # Track position in the stream in seconds
        self.last_successful_position = 0  # Last position that produced good transcription
        self.position_history = []  # Keep track of recent positions
        
        # Queue management
        self.max_queue_size = max_queue_size
        self.backpressure_applied = False
        
        # Performance tracking
        self.transcription_success_count = 0
        self.transcription_attempt_count = 0
        self.consecutive_empty = 0
        self.consecutive_repeats = 0
        self.consecutive_failures = 0
        self.performance_metrics = {
            'processing_times': [],
            'queue_sizes': [],
            'resync_events': 0,
            'duplicate_skips': 0,
            'backpressure_events': 0
        }
        self.adaptive_stats = {
            'buffer_adjustments': 0,
            'successful_resyncs': 0,
            'failed_resyncs': 0
        }
        
        # Text deduplication - store hashes of transcribed text
        self.seen_text_hashes = set()
        self.recent_transcripts = []  # Store recent transcriptions to avoid duplicates
        self.max_recent_transcripts = 20  # Maximum number of recent transcripts to remember
        
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
        
    def _is_duplicate_text(self, text: str) -> bool:
        """Check if text is a duplicate of recently transcribed text using improved fuzzy matching.
        
        Args:
            text: The transcribed text to check
            
        Returns:
            True if the text is a duplicate, False otherwise
        """
        # Skip very short texts as they're often false positives for duplicates
        normalized_text = text.lower().strip()
        if len(normalized_text) < 5:  # Increased from potentially 0 to minimum 5 chars
            return False
            
        # Create a hash of the text for exact match comparison
        text_hash = hash(normalized_text)
        
        # Check if we've seen this exact text before (exact match)
        if text_hash in self.seen_text_hashes:
            self.performance_metrics['duplicate_skips'] += 1
            return True
            
        # Calculate word-level fingerprint for fuzzy matching
        text_words = normalized_text.split()
        word_count = len(text_words)
        
        # Skip further checking if text is too short (less than 3 words)
        if word_count < 3:
            self.seen_text_hashes.add(text_hash)
            self.recent_transcripts.append(normalized_text)
            return False
            
        # Check for substrings and fuzzy matches in recent transcripts
        for recent in self.recent_transcripts:
            recent_normalized = recent.lower().strip()
            
            # Full substring check (if this text is completely contained in a recent transcript)
            if normalized_text in recent_normalized:
                self.performance_metrics['duplicate_skips'] += 1
                return True
                
            # Sliding window check for partial matches
            # This catches cases where the new text overlaps with the end of a previous text
            recent_words = recent_normalized.split()
            
            # Compare using word n-grams for more robust matching
            if len(text_words) >= 3 and len(recent_words) >= 3:
                # Create sets of 3-word phrases (trigrams) from each text
                text_trigrams = set()
                for i in range(len(text_words) - 2):
                    text_trigrams.add(" ".join(text_words[i:i+3]))
                    
                recent_trigrams = set()
                for i in range(len(recent_words) - 2):
                    recent_trigrams.add(" ".join(recent_words[i:i+3]))
                
                # If there's significant trigram overlap, it's likely a duplicate
                if text_trigrams and recent_trigrams:
                    common_trigrams = text_trigrams.intersection(recent_trigrams)
                    # More strict threshold - needs 40% of trigrams to match
                    if len(common_trigrams) >= 0.4 * len(text_trigrams):
                        self.performance_metrics['duplicate_skips'] += 1
                        return True
                
            # Check for significant word overlap (more than 75% of words - reduced from 80%)
            recent_word_set = set(recent_words)
            text_word_set = set(text_words)
            
            if text_word_set and recent_word_set:
                overlap = len(text_word_set.intersection(recent_word_set))
                shorter_len = min(len(text_word_set), len(recent_word_set))
                
                # If 75% of words in the shorter text are in the longer text, likely duplicate
                if overlap / shorter_len > 0.75:
                    self.performance_metrics['duplicate_skips'] += 1
                    return True
        
        # Not a duplicate, add to our seen text
        self.seen_text_hashes.add(text_hash)
        self.recent_transcripts.append(normalized_text)
        
        # Keep the recent transcripts list at a reasonable size
        if len(self.recent_transcripts) > self.max_recent_transcripts:
            old_text = self.recent_transcripts.pop(0)
            # Remove the hash for the old text
            old_hash = hash(old_text.lower().strip())
            self.seen_text_hashes.discard(old_hash)
            
        return False
    
    def _adjust_buffer_size(self):
        """Dynamically adjust buffer size based on transcription success rate."""
        if self.transcription_attempt_count < 3:  # Reduced from 5 for faster adaptation
            # Not enough data to make a good decision yet
            return
            
        success_rate = self.transcription_success_count / max(1, self.transcription_attempt_count)
        
        # Adjust buffer size based on success rate
        old_buffer_size = self.buffer_size
        
        if success_rate > 0.8:  # More aggressive reduction (was 0.9)
            # High success rate - we can try reducing buffer size for better responsiveness
            self.buffer_size = max(self.min_buffer_size, self.buffer_size - 1)
        elif success_rate < 0.7:  # More aggressive increase (was 0.6)
            # Low success rate - increase buffer size for better recognition
            self.buffer_size = min(self.max_buffer_size, self.buffer_size + 1)
            
        # If buffer size changed, log at debug level
        if old_buffer_size != self.buffer_size:
            self.adaptive_stats['buffer_adjustments'] += 1
            logger.debug(f"Adjusted buffer size from {old_buffer_size}s to {self.buffer_size}s (success rate: {success_rate:.2f})")
            
        # Reset counters more frequently to adapt to changing conditions 
        if self.transcription_attempt_count > 10:  # Was 20
            self.transcription_success_count = int(self.transcription_success_count * 0.7)  # Less aggressive decay (was 0.5)
            self.transcription_attempt_count = int(self.transcription_attempt_count * 0.7)  # Less aggressive decay (was 0.5)
    
    def _stream_audio(self):
        """Stream audio from the live stream URL and queue chunks for processing."""
        temp_dir = tempfile.mkdtemp()
        chunk_file = os.path.join(temp_dir, "chunk.wav")
        
        try:
            logger.debug(f"Starting audio streaming from live stream")
            
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
                    
                    logger.debug(f"Capturing audio chunk from stream at position approximately {self.stream_position}s")
                    
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
                        
                        # Calculate the increment to advance the stream position
                        # Use the overlap ratio to determine how much to advance
                        position_increment = max(1, self.buffer_size * (1 - self.overlap_ratio))
                        self.stream_position += position_increment
                        logger.debug(f"Stream position advanced to {self.stream_position} seconds")
                        
                        # If we're getting good chunks, dynamically adjust the buffer
                        self._adjust_buffer_size()
                        
                        # Reset the consecutive empty counter if we got a good chunk
                        self.consecutive_empty = 0
                    else:
                        logger.debug(f"Failed to capture audio chunk: file empty or not created")
                        
                        # If we failed to get a chunk, log more information at debug level
                        if process.stderr:
                            logger.debug(f"FFmpeg stderr: {process.stderr}")
                        if process.stdout:
                            logger.debug(f"FFmpeg stdout: {process.stdout}")
                            
                        # Use a smaller increment when there's an error
                        self.stream_position += 0.5
                        logger.debug(f"Advanced stream position by 0.5 seconds due to error")
                        
                        # Add a delay before retrying to avoid hammering the server
                        time.sleep(1)
                        
                except subprocess.TimeoutExpired:
                    logger.debug(f"FFmpeg process timed out after {self.buffer_size * 2} seconds")
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
            logger.debug("Starting audio processing for transcription")
            last_transcript = ""
            last_transcript_position = 0
            
            while self.is_running:
                try:
                    # Get an audio chunk from the queue, with a timeout
                    audio, metadata = self.audio_queue.get(timeout=3)  # Reduced timeout for responsiveness
                    logger.debug(f"Processing chunk from position {metadata['start_position']} (queue size: {self.audio_queue.qsize()})")
                    
                    # Export to a temporary file for Speech Recognition
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        
                    audio.export(temp_path, format="wav")
                    
                    # Increment attempt counter for adaptive sizing
                    self.transcription_attempt_count += 1
                    
                    # Log information about the segment being processed
                    logger.debug(f"Processing audio segment from position {metadata['start_position']} to {metadata['start_position'] + metadata['duration']} seconds")
                    
                    # Store current position for tracking
                    self.position_history.append(metadata['start_position'])
                    # Keep position history manageable
                    if len(self.position_history) > 10:
                        self.position_history.pop(0)
                    
                    # Submit the audio file for parallel transcription
                    self.batch_queue.put((temp_path, metadata))
                    
                    # Try to transcribe using the main process while waiting for workers
                    with sr.AudioFile(temp_path) as source:
                        audio_data = self.recognizer.record(source)
                        try:
                            # Using Google Speech Recognition
                            text = self.recognizer.recognize_google(audio_data)
                            
                            # Process the transcription result
                            if text and len(text.strip()) > 0:
                                # Check if this is duplicate text using our new method
                                if self._is_duplicate_text(text):
                                    self.consecutive_repeats += 1
                                    logger.debug(f"Detected duplicate text (skipping): {text}")
                                    
                                    # If we've seen too many duplicates, trigger resync
                                    if self.consecutive_repeats >= 2:
                                        logger.debug(f"Too many duplicates ({self.consecutive_repeats}), resynchronizing stream")
                                        self._resync_stream(last_successful_position=last_transcript_position)
                                        self.consecutive_repeats = 0
                                        
                                    # Skip adding this duplicate text 
                                    self.audio_queue.task_done()
                                    
                                    # Clean up worker queue entry for this chunk if it exists
                                    try:
                                        while not self.results_queue.empty():
                                            self.results_queue.get_nowait()
                                            self.results_queue.task_done()
                                    except queue.Empty:
                                        pass
                                    continue
                                else:
                                    # New unique text, reset counters and update success metrics
                                    self.consecutive_repeats = 0
                                    self.consecutive_empty = 0
                                    self.consecutive_failures = 0
                                    last_transcript = text
                                    last_transcript_position = metadata['start_position']
                                    self.last_successful_position = metadata['start_position']
                                    self.transcription_success_count += 1
                                
                                # Update transcription time
                                self.last_transcription_time = time.time()
                                
                                # Create timestamped entry for file output
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                transcription_entry = f"[{timestamp}] {text}"
                                
                                # Only output unique content
                                if not VERBOSE_OUTPUT:
                                    # Just show the plain text without timestamp or prefix
                                    transcript_logger.info(text)
                                else:
                                    # Show full debug info
                                    logger.info(f"Transcribed: {transcription_entry}")
                                
                                # Add to transcription list with timestamp for file output
                                self.transcription.append(transcription_entry)
                                
                                # Write to file more frequently with smaller buffer
                                current_time = time.time()
                                if current_time - self.last_write_time >= self.output_interval:
                                    self._write_transcription()
                                    self.last_write_time = current_time
                            else:
                                self.consecutive_empty += 1
                                logger.debug(f"No text detected in segment (empty count: {self.consecutive_empty})")
                                
                                # If too many consecutive empty segments, we might be in a silent part
                                # Reduced from 5 to 3 for faster detection
                                if self.consecutive_empty >= 3:
                                    logger.debug(f"Multiple empty segments detected, likely silent section. Advancing position.")
                                    self.consecutive_empty = 0
                                    # Advance a bit more to get to new content
                                    self.stream_position += self.buffer_size
                                    
                        except sr.UnknownValueError:
                            logger.debug("Speech Recognition could not understand audio")
                            self.consecutive_empty += 1
                            self.consecutive_failures += 1
                            
                            # Reduced from 8 to 5 for faster detection of issues
                            if self.consecutive_failures >= 5:
                                logger.debug(f"Multiple recognition failures, resynchronizing stream")
                                self._resync_stream()
                                self.consecutive_empty = 0
                                
                        except sr.RequestError as e:
                            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                            # Add a more aggressive recovery from API errors
                            logger.debug("Attempting recovery from API error")
                            self._resync_stream()
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.debug(f"Could not delete temporary file {temp_path}: {e}")
                        
                    # Mark the task as done
                    self.audio_queue.task_done()
                    
                    # Actively check for results from worker threads even if we already got our own results
                    self._check_worker_results()
                    
                except queue.Empty:
                    # Still check for worker results even during quiet periods
                    self._check_worker_results()
                    # Check if we haven't had a transcription in a while (reduced from 30 to 15 seconds)
                    if time.time() - self.last_transcription_time > 15:
                        logger.debug("No transcription for 15 seconds, attempting to resync")
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
        # Only log at INFO level if in verbose mode
        if VERBOSE_OUTPUT:
            logger.info("Attempting to resynchronize stream")
        else:
            logger.debug("Attempting to resynchronize stream")
        
        previous_position = self.stream_position
        
        # Determine the best resyncing strategy based on failure patterns
        if last_successful_position is not None and last_successful_position > 0:
            # We have a good reference point - use adaptive strategy
            
            # Scale jump size based on how many resyncs we've done recently
            # This creates a backoff mechanism for persistent issues
            if self.consecutive_failures < 3:
                # Small jump first attempt
                jump_ratio = 0.2  # 20% of buffer size
            elif self.consecutive_failures < 5:
                # Medium jump for repeated failures
                jump_ratio = 0.5  # 50% of buffer size
            else:
                # Large jump after many failures
                jump_ratio = 1.5  # 150% of buffer size
                
            jump_amount = max(1, self.buffer_size * jump_ratio)
            new_position = last_successful_position + jump_amount
            
            logger.debug(f"Resync using last successful position {last_successful_position} + {jump_amount:.1f}s jump")
        else:
            # No good reference - use more aggressive strategy
            # This helps if we're stuck in a problematic section of the stream
            jump_amount = self.buffer_size * 1.5  # 150% of buffer size
            new_position = self.stream_position + jump_amount
            logger.debug(f"Aggressive resync with {jump_amount:.1f}s forward jump (no good reference)")
        
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
                
        logger.debug(f"Stream resynced from {previous_position:.1f}s to {self.stream_position:.1f}s (cleared {cleared_chunks} chunks)")
        
        # Adjust buffer size during resync based on failure pattern
        if self.consecutive_failures > 3:
            old_buffer = self.buffer_size
            # Try increasing buffer size to capture more context
            self.buffer_size = min(self.max_buffer_size, self.buffer_size + 2)
            if old_buffer != self.buffer_size:
                logger.debug(f"Increased buffer size from {old_buffer}s to {self.buffer_size}s due to repeated failures")
        
        # Track resync statistics
        self.adaptive_stats['successful_resyncs'] += 1
        
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
    
    def _manage_backpressure(self):
        """Manage backpressure to prevent queue overflow."""
        while self.is_running:
            try:
                current_size = self.audio_queue.qsize()
                self.performance_metrics['queue_sizes'].append(current_size)
                
                # Trim the queue sizes list to avoid memory issues
                if len(self.performance_metrics['queue_sizes']) > 100:
                    self.performance_metrics['queue_sizes'] = self.performance_metrics['queue_sizes'][-100:]
                
                # If queue is getting too full, apply backpressure by increasing stream position
                if current_size >= self.max_queue_size:
                    if not self.backpressure_applied:
                        logger.debug(f"Applying backpressure: queue size {current_size} >= max {self.max_queue_size}")
                        # Jump ahead to reduce pressure on the processing pipeline
                        self.stream_position += self.buffer_size * 2
                        logger.debug(f"Backpressure applied: jumped to position {self.stream_position}s")
                        self.backpressure_applied = True
                        self.performance_metrics['backpressure_events'] += 1
                elif current_size < self.max_queue_size // 2:
                    # Reset backpressure flag when queue is less than half full
                    if self.backpressure_applied:
                        logger.debug("Backpressure released: queue size reduced")
                        self.backpressure_applied = False
                
                # Track performance analytics
                if len(self.performance_metrics['queue_sizes']) > 10:
                    avg_queue_size = sum(self.performance_metrics['queue_sizes'][-10:]) / 10
                    if avg_queue_size > self.max_queue_size * 0.8 and self.overlap_ratio > 0.05:
                        # If we're consistently close to max queue size, reduce overlap ratio
                        old_overlap = self.overlap_ratio
                        self.overlap_ratio = max(0.05, self.overlap_ratio - 0.05)
                        logger.debug(f"High queue load: reduced overlap ratio from {old_overlap:.2f} to {self.overlap_ratio:.2f}")
                
                # Sleep for a short time before checking again
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in backpressure management: {e}")
                time.sleep(1)
                
    def _check_worker_results(self):
        """Check for and process results from worker threads."""
        try:
            # Process as many results as are available without blocking
            while True:
                try:
                    # Try to get a result with a very short timeout
                    text, metadata, audio_path = self.results_queue.get_nowait()
                    
                    # Process valid transcription
                    if text and len(text.strip()) > 0:
                        # Check if this is duplicate text
                        if not self._is_duplicate_text(text):
                            # New unique text
                            # Update transcription time
                            self.last_transcription_time = time.time()
                            self.last_successful_position = metadata['start_position']
                            
                            # Create timestamped entry for file output
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            transcription_entry = f"[{timestamp}] {text}"
                            
                            # Only output unique content
                            if not VERBOSE_OUTPUT:
                                # Just show the plain text without timestamp or prefix
                                transcript_logger.info(f"[WORKER] {text}")
                            else:
                                # Show full debug info
                                logger.info(f"Worker transcribed: {transcription_entry}")
                            
                            # Add to transcription list with timestamp for file output
                            self.transcription.append(transcription_entry)
                            self.transcription_success_count += 1
                    
                    # Mark the task as done
                    self.results_queue.task_done()
                    
                except queue.Empty:
                    # No more results available, break out of the loop
                    break
        except Exception as e:
            logger.error(f"Error checking worker results: {e}")
    
    def _parallel_transcribe(self, batch_queue, results_queue):
        """Transcribe audio in parallel worker thread."""
        worker_recognizer = sr.Recognizer()
        
        while self.is_running:
            try:
                # Get the next item to process
                item = batch_queue.get(timeout=3)
                if item is None:  # Sentinel value to indicate shutdown
                    batch_queue.task_done()
                    break
                    
                audio_path, metadata = item
                
                try:
                    with sr.AudioFile(audio_path) as source:
                        audio_data = worker_recognizer.record(source)
                        text = worker_recognizer.recognize_google(audio_data)
                        
                    # Put the result in the results queue
                    results_queue.put((text, metadata, audio_path))
                except sr.UnknownValueError:
                    # No text detected
                    results_queue.put((None, metadata, audio_path))
                except sr.RequestError as e:
                    logger.error(f"API error in worker thread: {e}")
                    results_queue.put((None, metadata, audio_path))
                except Exception as e:
                    logger.error(f"Error in worker thread: {e}")
                    results_queue.put((None, metadata, audio_path))
                    
                batch_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Unexpected error in worker thread: {e}")
                time.sleep(0.5)
    
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
        
        # Setup thread communication queues
        self.batch_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
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
        
        # Start backpressure management thread
        backpressure_thread = threading.Thread(target=self._manage_backpressure)
        backpressure_thread.daemon = True
        backpressure_thread.start()
        self.threads.append(backpressure_thread)
        
        # Start parallel transcription worker threads
        num_workers = min(4, os.cpu_count() or 2)  # Use up to 4 workers or CPU count
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._parallel_transcribe, 
                args=(self.batch_queue, self.results_queue)
            )
            worker.daemon = True
            worker.start()
            self.threads.append(worker)
            
        logger.info(f"Transcription process started with {num_workers} parallel workers")
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
    buffer_size: int = 2,  # Reduced from 4s for faster sampling
    min_buffer_size: int = 1,  # Reduced from 2s for faster adaptation
    max_buffer_size: int = 4,  # Reduced from 8s
    overlap_ratio: float = 0.15,  # Reduced from 0.25 for less duplication
    output_interval: int = 5,  # Reduced from 10s for more frequent updates
    duration: Optional[int] = None,
    max_queue_size: int = 20  # Control queue backpressure
) -> Optional[str]:
    """Connect to a YouTube live stream and transcribe it in real-time.
    
    Args:
        video_url: YouTube video URL for the live stream
        output_path: Path to save the transcript (default: None, generates a filename)
        format: Format specification for yt-dlp (default: 'best')
        buffer_size: Initial size of audio buffer in seconds (default: 10)
        min_buffer_size: Minimum buffer size in seconds for adaptive sizing (default: 5)
        max_buffer_size: Maximum buffer size in seconds for adaptive sizing (default: 20)
        overlap_ratio: Ratio of overlap between consecutive chunks (default: 0.5)
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
    
    # Log at debug level unless in verbose mode
    if VERBOSE_OUTPUT:
        logger.info(f"Setting up transcription with buffer size: {buffer_size}s and output interval: {output_interval}s")
    else:
        logger.debug(f"Setting up transcription with buffer size: {buffer_size}s (min: {min_buffer_size}s, max: {max_buffer_size}s), overlap: {overlap_ratio}")
    
    # Create and start the transcriber with all the new parameters
    transcriber = LiveStreamTranscriber(
        stream_url=stream_url,
        output_path=output_path,
        buffer_size=buffer_size,
        min_buffer_size=min_buffer_size,
        max_buffer_size=max_buffer_size,
        overlap_ratio=overlap_ratio,
        output_interval=output_interval,
        max_queue_size=max_queue_size
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


class ClipExtractor:
    """Class for extracting periodic clips from YouTube live streams."""
    
    def __init__(self, 
                 stream_url: str,
                 clip_duration: int,
                 interval: int,
                 format: str = "best"):
        """Initialize the clip extractor.
        
        Args:
            stream_url: Direct URL to the live stream
            clip_duration: Duration of each clip in seconds
            interval: Interval between clips in seconds
            format: Format specification for video quality
        """
        self.stream_url = stream_url
        self.clip_duration = clip_duration
        self.interval = interval
        self.format = format
        
        # Create clips directory if it doesn't exist
        downloads_dir = os.path.join(os.getcwd(), 'downloads')
        ensure_directory_exists(downloads_dir)
        self.clips_dir = os.path.join(downloads_dir, 'clips')
        ensure_directory_exists(self.clips_dir)
        
        # For tracking state
        self.is_running = False
        self.threads = []
        
        # For tracking clip count
        self.clip_count = 0
        
        # Print setup information
        print(f"Clip Extractor initialized:")
        print(f"• Clip duration: {self.clip_duration} seconds")
        print(f"• Interval between clips: {self.interval} seconds")
        print(f"• Output directory: {self.clips_dir}")
    
    def _extract_clips(self):
        """Extract clips from the live stream at specified intervals."""
        try:
            print(f"Starting clip extraction from YouTube live stream")
            logger.debug(f"Using stream URL: {self.stream_url}")
            
            while self.is_running:
                try:
                    # Generate filename with epoch timestamps
                    start_time = int(time.time())
                    end_time = start_time + self.clip_duration
                    filename = f"{start_time}-{end_time}.mp4"
                    output_path = os.path.join(self.clips_dir, filename)
                    
                    print(f"[{time.strftime('%H:%M:%S')}] Extracting clip: {filename}")
                    
                    # Use ffmpeg to capture a clip of the specified duration
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-reconnect", "1",  # Reconnect if the connection is lost
                        "-reconnect_streamed", "1",  # Reconnect if the stream fails
                        "-reconnect_delay_max", "5",  # Max delay between reconnection attempts
                        "-i", self.stream_url,
                        "-t", str(self.clip_duration),  # Duration to capture
                        "-c:v", "copy",  # Copy video codec to maintain quality
                        "-c:a", "copy",  # Copy audio codec to maintain quality
                        "-loglevel", "warning",
                        output_path
                    ]
                    
                    # Run ffmpeg process with a timeout
                    process = subprocess.run(
                        ffmpeg_cmd, 
                        capture_output=True, 
                        text=True,
                        check=False,
                        timeout=self.clip_duration * 2  # Timeout after 2x clip_duration seconds
                    )
                    
                    # Verify the clip was created successfully
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        self.clip_count += 1
                        print(f"[{time.strftime('%H:%M:%S')}] ✓ Clip saved: {filename}")
                        print(f"  Total clips: {self.clip_count} | Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] ✗ Failed to extract clip: {filename}")
                        if process.stderr and len(process.stderr) > 0:
                            print(f"  Error: {process.stderr.strip()}")
                    
                    # Wait for the next interval
                    # We subtract the time it took to capture the clip to maintain regular intervals
                    elapsed = time.time() - start_time
                    wait_time = max(0, self.interval - elapsed)
                    if wait_time > 0:
                        print(f"Waiting {wait_time:.1f} seconds until next clip...")
                    time.sleep(wait_time)
                    
                except subprocess.TimeoutExpired:
                    logger.error(f"FFmpeg process timed out after {self.clip_duration * 2} seconds")
                    time.sleep(1)  # Brief pause before retrying
                except Exception as e:
                    logger.error(f"Error extracting clip: {e}")
                    if not self.is_running:
                        break
                    time.sleep(1)  # Brief pause before retrying
                    
        except Exception as e:
            logger.error(f"Error in clip extraction thread: {e}")
        finally:
            logger.info("Clip extraction thread stopped")
    
    def start(self) -> bool:
        """Start the clip extraction process.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Clip extraction is already running")
            return False
        
        self.is_running = True
        
        # Start clip extraction thread
        extract_thread = threading.Thread(target=self._extract_clips)
        extract_thread.daemon = True
        extract_thread.start()
        self.threads.append(extract_thread)
        
        logger.info("Clip extraction process started")
        return True
    
    def stop(self) -> None:
        """Stop the clip extraction process."""
        if not self.is_running:
            logger.warning("Clip extraction is not running")
            return
        
        print("Stopping clip extraction process...")
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Final summary
        print(f"\nClip extraction complete!")
        print(f"• Total clips extracted: {self.clip_count}")
        print(f"• Clips saved to: {self.clips_dir}")


def extract_clips_from_stream(
    video_url: str,
    clip_duration: int,
    interval: int,
    format: str = "best",
    max_clips: Optional[int] = None
) -> bool:
    """Extract periodic clips from a YouTube live stream.
    
    Args:
        video_url: YouTube video URL for the live stream
        clip_duration: Duration of each clip in seconds
        interval: Interval between clips in seconds
        format: Format specification for yt-dlp (default: 'best')
        max_clips: Maximum number of clips to extract (default: None, runs until stopped)
    
    Returns:
        True if successful, False otherwise
    """
    # Extract the direct streaming URL
    print(f"\nConnecting to YouTube live stream: {video_url}")
    stream_url = get_live_stream_url(video_url, format)
    if not stream_url:
        print(f"❌ Failed to get live stream URL. Make sure this is a valid live stream.")
        return False
    
    print(f"✓ Successfully connected to live stream")
    
    # Create and start the extractor
    extractor = ClipExtractor(
        stream_url=stream_url,
        clip_duration=clip_duration,
        interval=interval,
        format=format
    )
    
    if not extractor.start():
        print("❌ Failed to start clip extraction process")
        return False
    
    try:
        if max_clips:
            print(f"Will automatically stop after extracting {max_clips} clips")
        
        start_time = time.time()
        last_report_time = start_time
        
        # Run until interrupted or max_clips reached
        while True:
            if max_clips and extractor.clip_count >= max_clips:
                print(f"\nReached specified maximum number of clips: {max_clips}")
                break
            
            # Sleep a bit to avoid hammering the CPU
            time.sleep(1)
            
            # Periodically report status if no clips have been extracted in a while
            current_time = time.time()
            if current_time - last_report_time > 30 and extractor.clip_count > 0:
                print(f"Status: {extractor.clip_count} clips extracted so far. Still running...")
                last_report_time = current_time
            
    except KeyboardInterrupt:
        logger.info("Clip extraction interrupted by user")
    except Exception as e:
        logger.error(f"Error during clip extraction: {e}")
    finally:
        # Stop extraction
        extractor.stop()
    
    return True


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
    
    # Get clips command
    get_clips_parser = subparsers.add_parser(
        'get-clips',
        help='Extract periodic clips from a YouTube live stream'
    )
    get_clips_parser.add_argument('clip_duration', type=int, help='Duration of each clip in seconds')
    get_clips_parser.add_argument('interval', type=int, help='Interval between clips in seconds')
    get_clips_parser.add_argument('url', help='URL of the YouTube live stream')
    get_clips_parser.add_argument(
        '--format', '-f', default='best',
        help='Format specification for yt-dlp (default: best)'
    )
    get_clips_parser.add_argument(
        '--max-clips', '-m', type=int,
        help='Maximum number of clips to extract before stopping (default: runs until stopped)'
    )
    get_clips_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output with debugging information (default: False)'
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
    
    # Add buffer and adaptive sizing parameters
    transcribe_parser.add_argument(
        '--buffer-size', '-b', type=int, default=2,
        help='Initial size of audio buffer in seconds (default: 2)'
    )
    transcribe_parser.add_argument(
        '--min-buffer-size', type=int, default=1,
        help='Minimum buffer size for adaptive sizing in seconds (default: 1)'
    )
    transcribe_parser.add_argument(
        '--max-buffer-size', type=int, default=4,
        help='Maximum buffer size for adaptive sizing in seconds (default: 4)'
    )
    transcribe_parser.add_argument(
        '--overlap-ratio', type=float, default=0.15,
        help='Ratio of overlap between consecutive chunks (default: 0.15)'
    )
    transcribe_parser.add_argument(
        '--output-interval', '-i', type=int, default=5,
        help='How often to write transcript to file in seconds (default: 5)'
    )
    transcribe_parser.add_argument(
        '--max-queue-size', type=int, default=20,
        help='Maximum size of the audio processing queue (default: 20)'
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
            min_buffer_size=args.min_buffer_size,
            max_buffer_size=args.max_buffer_size,
            overlap_ratio=args.overlap_ratio,
            output_interval=args.output_interval,
            duration=args.duration,
            max_queue_size=args.max_queue_size
        )
        
        if transcript_path:
            logger.info(f"Transcription complete! Transcript saved to: {transcript_path}")
            return 0
        else:
            logger.error("Failed to transcribe live stream")
            return 1
            
    elif args.command == 'get-clips':
        # Configure logging based on verbose flag
        configure_logging(verbose=getattr(args, 'verbose', False))
        
        # If not in verbose mode, show a simple starting message
        if not VERBOSE_OUTPUT:
            print(f"Starting clip extraction (duration: {args.clip_duration}s, interval: {args.interval}s)...")
            print("Press Ctrl+C to stop")
            print("---------------------------------------------")
        
        success = extract_clips_from_stream(
            video_url=args.url,
            clip_duration=args.clip_duration,
            interval=args.interval,
            format=args.format,
            max_clips=args.max_clips
        )
        
        if success:
            clips_dir = os.path.join(os.getcwd(), 'downloads', 'clips')
            logger.info(f"Clip extraction complete! Clips saved to: {clips_dir}")
            return 0
        else:
            logger.error("Failed to extract clips from live stream")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
