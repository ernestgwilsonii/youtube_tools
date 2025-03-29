#!/usr/bin/env python3
"""
Video to Text Transcription Script

This script downloads a video, extracts the audio, and converts the speech to text.
It builds on the functionality of the download_demo.py script.
"""

import os
import sys
import argparse
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple

from yt_dlp import YoutubeDL
import speech_recognition as sr
from pydub import AudioSegment
import math

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
        directory: Path to the directory to ensure exists.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def download_audio(url: str, output_path: Optional[str] = None) -> Optional[str]:
    """Download a video's audio using yt-dlp.
    
    Args:
        url: URL of the video to download
        output_path: Path to save the audio file (default: None, uses temp file)
        
    Returns:
        Path to the downloaded audio file, or None if download failed
    """
    # Create downloads directory if it doesn't exist
    downloads_dir = os.path.join(os.getcwd(), 'downloads')
    ensure_directory_exists(downloads_dir)
    
    # Use temp file if no output path specified
    if output_path is None:
        # Create a temp file with .mp3 extension
        temp_dir = os.path.join(downloads_dir, 'temp')
        ensure_directory_exists(temp_dir)
        output_path = os.path.join(temp_dir, f"{next(tempfile._get_candidate_names())}.mp3")
    
    # Configure yt-dlp options for audio download
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'noplaylist': True,
    }
    
    # Execute download
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            logger.info(f"Downloaded audio from: {info.get('title', 'Video')}")
            
            # Get the output filename from ydl
            if 'entries' in info:
                # For playlists
                filename = ydl.prepare_filename(info['entries'][0])
            else:
                # For single videos
                filename = ydl.prepare_filename(info)
            
            # YoutubeDL adds the mp3 extension to the output path after extraction
            # The file path will be output_path + '.mp3'
            if output_path.endswith('.mp3'):
                return output_path + '.mp3'
            else:
                return filename.rsplit('.', 1)[0] + '.mp3'
    except Exception as e:
        logger.error(f"Error downloading video audio: {e}")
        return None


def transcribe_audio(
    audio_path: str, 
    output_text_path: Optional[str] = None,
    chunk_size_minutes: int = 5
) -> Optional[str]:
    """Transcribe an audio file to text using Google Speech Recognition.
    
    Args:
        audio_path: Path to the audio file
        output_text_path: Path to save the transcript (default: None, uses audio filename)
        chunk_size_minutes: Size of audio chunks in minutes for processing (default: 5)
        
    Returns:
        Transcription text, or None if transcription failed
    """
    logger.info(f"Transcribing audio file: {audio_path}")
    recognizer = sr.Recognizer()
    
    # If output_text_path is None, create one based on the audio file
    if output_text_path is None:
        output_text_path = os.path.splitext(audio_path)[0] + '_transcript.txt'
    
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Get audio duration in milliseconds
        duration_ms = len(audio)
        chunk_size_ms = chunk_size_minutes * 60 * 1000
        
        # Calculate number of chunks
        num_chunks = math.ceil(duration_ms / chunk_size_ms)
        
        full_transcript = ""
        
        logger.info(f"Processing audio in {num_chunks} chunks...")
        
        # Process audio in chunks to avoid memory issues
        for i in range(num_chunks):
            logger.info(f"Processing chunk {i+1}/{num_chunks}...")
            start_ms = i * chunk_size_ms
            end_ms = min((i + 1) * chunk_size_ms, duration_ms)
            
            # Extract chunk
            chunk = audio[start_ms:end_ms]
            
            # Export chunk to a temporary WAV file
            temp_path = os.path.join(tempfile.gettempdir(), f"chunk_{i}.wav")
            chunk.export(temp_path, format="wav")
            
            # Recognize speech in the chunk
            with sr.AudioFile(temp_path) as source:
                audio_data = recognizer.record(source)
                try:
                    # Using Google Speech Recognition
                    logger.info(f"Sending chunk {i+1} to Google Speech Recognition...")
                    chunk_transcript = recognizer.recognize_google(audio_data)
                    logger.info(f"Successfully transcribed chunk {i+1}")
                    full_transcript += chunk_transcript + " "
                except sr.UnknownValueError:
                    logger.warning(f"Speech Recognition could not understand audio in chunk {i+1}")
                    full_transcript += "[inaudible segment] "
                except sr.RequestError as e:
                    logger.error(
                        f"Could not request results from Google Speech Recognition "
                        f"service in chunk {i+1}; {e}"
                    )
                    full_transcript += "[recognition error] "
                    
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_path}: {e}")
            
        # Save transcript to file
        cleaned_transcript = full_transcript.strip()
        ensure_directory_exists(os.path.dirname(output_text_path))
        
        with open(output_text_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_transcript)
            
        logger.info(f"Transcription complete. Saved to: {output_text_path}")
        
        # Return the transcript even if it's mostly [inaudible segment] markers
        if cleaned_transcript:
            return cleaned_transcript
        else:
            logger.warning("The transcript is empty or contains only inaudible markers")
            return "[The audio could not be transcribed due to recognition limitations]"
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None


def video_to_text(
    url: str, 
    output_audio_path: Optional[str] = None, 
    output_text_path: Optional[str] = None, 
    chunk_size_minutes: int = 5
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Download a video, extract audio, and transcribe to text.
    
    Args:
        url: URL of the video to download
        output_audio_path: Path to save the audio (default: None)
        output_text_path: Path to save the transcript (default: None)
        chunk_size_minutes: Size of audio chunks in minutes for processing (default: 5)
        
    Returns:
        Tuple of (audio_path, transcript_path, transcript_text), or (None, None, None) if failed
    """
    # Download the video and extract audio
    audio_path = download_audio(url, output_audio_path)
    if not audio_path:
        logger.error("Failed to download audio. Exiting.")
        return None, None, None
    
    # Transcribe the audio to text
    transcript = transcribe_audio(audio_path, output_text_path, chunk_size_minutes)
    
    if output_text_path is None:
        output_text_path = os.path.splitext(audio_path)[0] + '_transcript.txt'
    
    return audio_path, output_text_path, transcript


def main() -> int:
    """Main entry point for the command line interface.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(description='Download a video and transcribe its audio to text')
    parser.add_argument('url', help='URL of the video to download')
    parser.add_argument('--audio-output', '-a', help='Path to save the audio file')
    parser.add_argument('--text-output', '-t', help='Path to save the transcript file')
    parser.add_argument(
        '--chunk-size', '-c', type=int, default=5, 
        help='Size of audio chunks in minutes for processing (default: 5)'
    )
    
    args = parser.parse_args()
    
    audio_path, transcript_path, transcript = video_to_text(
        args.url, 
        args.audio_output, 
        args.text_output,
        args.chunk_size
    )
    
    if audio_path and transcript_path:
        logger.info("\nTranscription summary:")
        logger.info(f"- Audio saved to: {audio_path}")
        logger.info(f"- Transcript saved to: {transcript_path}")
        if transcript and not transcript.startswith("[The audio could not be transcribed"):
            logger.info(f"- Transcript preview (first 150 chars): {transcript[:150]}...")
        else:
            logger.warning("- Note: Transcription had limited success recognizing speech in the audio")
        return 0
    else:
        logger.error("Download or transcription process failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
