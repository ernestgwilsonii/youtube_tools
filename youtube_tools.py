#!/usr/bin/env python3
"""
YouTube Tools - Video Download and Transcription

This script provides functionality to download videos from YouTube and other
platforms supported by yt-dlp, and transcribe their audio content to text.

All downloads are saved to a 'downloads' folder within the current directory.
"""

import os
import sys
import argparse
import tempfile
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

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
        directory: The path to the directory to ensure exists.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def download_video(
    url: str, 
    output_path: Optional[str] = None, 
    format: Optional[str] = None, 
    extract_audio: bool = False, 
    audio_format: str = "mp3", 
    audio_quality: str = "192"
) -> Optional[str]:
    """Download a video using yt-dlp.
    
    Args:
        url: URL of the video to download.
        output_path: Path to save the downloaded video. If None, saves to
            downloads directory.
        format: Format specification for yt-dlp.
        extract_audio: Whether to extract audio from the video.
        audio_format: Format for extracted audio (if extract_audio is True).
        audio_quality: Quality for extracted audio (if extract_audio is True).
        
    Returns:
        Path to the downloaded file, or None if download failed.
    """
    # Create downloads directory if it doesn't exist
    downloads_dir = os.path.join(os.getcwd(), 'downloads')
    ensure_directory_exists(downloads_dir)
    
    # Configure yt-dlp options
    if output_path is None:
        output_template = os.path.join(downloads_dir, '%(title)s.%(ext)s')
    else:
        output_template = output_path
    
    ydl_opts: Dict[str, Any] = {
        'outtmpl': output_template,
        'verbose': False,
        'quiet': False,
        'no_warnings': False,
        'progress': True,
        'noplaylist': True,
    }
    
    if format:
        ydl_opts['format'] = format
    
    # Add audio extraction if requested
    if extract_audio:
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': audio_quality,
        }]
        
    # Execute download
    try:
        logger.info(f"Downloading video from: {url}")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            logger.info(f"Downloaded: {info.get('title', 'Video')}")
            
            # Get the output filename from ydl
            if 'entries' in info:  # For playlists
                filename = ydl.prepare_filename(info['entries'][0])
            else:  # For single videos
                filename = ydl.prepare_filename(info)
                
            return filename
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return None


def download_audio(
    url: str, 
    output_path: Optional[str] = None
) -> Optional[str]:
    """Download a video's audio using yt-dlp.
    
    Args:
        url: URL of the video to download audio from.
        output_path: Path to save the downloaded audio. If None, saves to
            downloads directory.
            
    Returns:
        Path to the downloaded audio file, or None if download failed.
    """
    # Create downloads directory if it doesn't exist
    downloads_dir = os.path.join(os.getcwd(), 'downloads')
    ensure_directory_exists(downloads_dir)
    
    # Use default output path if none specified
    if output_path is None:
        output_path = os.path.join(downloads_dir, '%(title)s.%(ext)s')
    
    # Configure yt-dlp options for audio download
    ydl_opts: Dict[str, Any] = {
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
        logger.info(f"Downloading audio from: {url}")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            logger.info(f"Downloaded audio from: {info.get('title', 'Video')}")
            
            # Get the output filename from ydl
            if 'entries' in info:  # For playlists
                filename = ydl.prepare_filename(info['entries'][0])
            else:  # For single videos
                filename = ydl.prepare_filename(info)
            
            # yt-dlp adds the .mp3 extension due to the FFmpegExtractAudio post-processor
            mp3_filename = os.path.splitext(filename)[0] + '.mp3'
            if os.path.exists(mp3_filename):
                return mp3_filename
            else:
                # Some versions might create filename.ext.mp3 instead
                alt_filename = filename + '.mp3'
                if os.path.exists(alt_filename):
                    return alt_filename
                else:
                    logger.error(
                        f"Could not find downloaded audio file. Expected at: "
                        f"{mp3_filename} or {alt_filename}"
                    )
                    return None
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None


def convert_format(
    input_path: str, 
    output_path: Optional[str] = None, 
    to_format: str = "mp4", 
    audio_only: bool = False, 
    quality: str = "192"
) -> Optional[str]:
    """Convert a video/audio file to a different format using FFmpeg.
    
    Args:
        input_path: Path to the file to convert.
        output_path: Path for the converted file. If None, saves to downloads
            directory.
        to_format: Target format to convert to.
        audio_only: Whether to extract audio only.
        quality: Quality setting for audio conversion.
        
    Returns:
        Path to the converted file, or None if conversion failed.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return None
    
    # Create downloads directory if it doesn't exist
    downloads_dir = os.path.join(os.getcwd(), 'downloads')
    ensure_directory_exists(downloads_dir)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(
            downloads_dir, f"{base_name}_converted.{to_format}"
        )
    
    # Construct FFmpeg command
    cmd: List[str] = ['ffmpeg', '-y', '-i', input_path]
    
    if audio_only:
        # Audio conversion
        if to_format in ['mp3', 'aac', 'wav', 'ogg', 'flac', 'm4a']:
            if to_format == 'mp3':
                cmd.extend(['-q:a', quality])
            elif to_format == 'aac':
                cmd.extend(['-c:a', 'aac', '-b:a', f'{quality}k'])
            elif to_format in ['ogg', 'flac']:
                cmd.extend(['-c:a', to_format])
            elif to_format == 'm4a':
                cmd.extend(['-c:a', 'aac', '-b:a', f'{quality}k'])
            # For WAV, no special options needed
        else:
            logger.warning(
                f"Unsupported audio format: {to_format}, defaulting to mp3"
            )
            to_format = 'mp3'
            cmd.extend(['-q:a', quality])
            output_path = os.path.splitext(output_path)[0] + '.mp3'
    else:
        # Video conversion
        if to_format in ['mp4', 'mkv', 'webm', 'avi', 'mov']:
            # Different codec settings based on format
            if to_format == 'mp4':
                cmd.extend([
                    '-c:v', 'libx264', '-crf', '23', 
                    '-c:a', 'aac', '-b:a', f'{quality}k'
                ])
            elif to_format == 'webm':
                cmd.extend([
                    '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0', 
                    '-c:a', 'libopus', '-b:a', f'{quality}k'
                ])
            elif to_format == 'mkv':
                cmd.extend([
                    '-c:v', 'libx264', '-crf', '23', 
                    '-c:a', 'aac', '-b:a', f'{quality}k'
                ])
            elif to_format == 'avi':
                cmd.extend([
                    '-c:v', 'mpeg4', '-q:v', '6', 
                    '-c:a', 'libmp3lame', '-q:a', '4'
                ])
            elif to_format == 'mov':
                cmd.extend([
                    '-c:v', 'libx264', '-crf', '23', 
                    '-c:a', 'aac', '-b:a', f'{quality}k'
                ])
        else:
            logger.warning(
                f"Unsupported video format: {to_format}, defaulting to mp4"
            )
            to_format = 'mp4'
            cmd.extend([
                '-c:v', 'libx264', '-crf', '23', 
                '-c:a', 'aac', '-b:a', f'{quality}k'
            ])
            output_path = os.path.splitext(output_path)[0] + '.mp4'
    
    # Add output path to command
    cmd.append(output_path)
    
    try:
        logger.info(f"Converting {'audio' if audio_only else 'video'} to {to_format} format...")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(output_path):
            logger.info(f"Conversion successful. Output saved to: {output_path}")
            return output_path
        else:
            logger.error("Conversion failed. Output file not created.")
            logger.error(f"FFmpeg stdout: {result.stdout}")
            logger.error(f"FFmpeg stderr: {result.stderr}")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during conversion: {e}")
        logger.error(f"FFmpeg stdout: {e.stdout}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        return None


def download_youtube_transcript(
    url: str, 
    output_text_path: Optional[str] = None, 
    languages: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Download a YouTube video's transcript/subtitle directly from YouTube.
    
    Args:
        url: URL of the YouTube video.
        output_text_path: Path to save the transcript. If None, saves to
            downloads directory.
        languages: List of language codes to try downloading transcripts for.
            Defaults to ['en'].
            
    Returns:
        Tuple of (transcript text, path to saved transcript), or (None, None)
        if download failed.
    """
    # Create downloads directory if it doesn't exist
    downloads_dir = os.path.join(os.getcwd(), 'downloads')
    ensure_directory_exists(downloads_dir)
    
    # Configure yt-dlp options for subtitle/transcript download
    ydl_opts: Dict[str, Any] = {
        'skip_download': True,  # Skip downloading the video
        'writesubtitles': True,  # Download subtitles if available
        'writeautomaticsub': True,  # Download auto-generated subtitles if no regular subs
        'subtitleslangs': languages if languages else ['en'],  # Language preference
        'subtitlesformat': 'json3',  # Download as json3 format for easier processing
        'outtmpl': os.path.join(downloads_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }
    
    try:
        logger.info(f"Downloading transcript for: {url}")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            # Try to find the subtitle file that was downloaded
            video_title = info.get('title', 'video')
            base_filename = ydl.prepare_filename(info)
            base_path = os.path.splitext(base_filename)[0]
            
            # Check common subtitle file patterns
            subtitle_files: List[Tuple[str, bool]] = []
            for lang in languages if languages else ['en']:
                # Regular subtitles
                filename = f"{base_path}.{lang}.json3"
                if os.path.exists(filename):
                    subtitle_files.append((filename, False))  # (path, is_auto)
                
                # Auto-generated subtitles
                auto_filename = f"{base_path}.{lang}.auto.json3"
                if os.path.exists(auto_filename):
                    subtitle_files.append((auto_filename, True))  # (path, is_auto)
            
            if not subtitle_files:
                logger.warning(f"No transcript files found for video: {video_title}")
                return None, None
            
            # Sort to prefer regular subtitles over auto-generated ones
            subtitle_files.sort(key=lambda x: x[1])  # Sort by is_auto (False comes first)
            subtitle_path, is_auto = subtitle_files[0]
            
            # Parse the JSON3 file
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                subtitle_data = json.load(f)
            
            # Process transcript data
            full_transcript = ""
            
            # Handle different JSON formats
            if 'events' in subtitle_data:
                # YouTube JSON3 format
                for event in subtitle_data['events']:
                    if 'segs' in event:
                        for seg in event['segs']:
                            if 'utf8' in seg:
                                full_transcript += seg['utf8'] + " "
            
            # Clean up transcript
            full_transcript = full_transcript.strip()
            
            # Set output path if not provided
            if output_text_path is None:
                output_text_path = os.path.join(
                    downloads_dir, 
                    f"{os.path.basename(base_path)}_youtube_transcript.txt"
                )
            
            # Save transcript to file
            logger.info(f"Saving YouTube transcript to: {output_text_path}")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(full_transcript)
            
            # Remove the original subtitle file
            try:
                os.remove(subtitle_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary subtitle file {subtitle_path}: {e}")
            
            source_type = "auto-generated" if is_auto else "official"
            logger.info(f"Successfully downloaded {source_type} YouTube transcript")
            
            return full_transcript, output_text_path
            
    except Exception as e:
        logger.error(f"Error downloading YouTube transcript: {e}")
        return None, None


def transcribe_audio(
    audio_path: str, 
    output_text_path: Optional[str] = None, 
    chunk_size_minutes: int = 2
) -> Optional[str]:
    """Transcribe an audio file to text using Google Speech Recognition.
    
    Args:
        audio_path: Path to the audio file to transcribe.
        output_text_path: Path to save the transcript. If None, saves alongside
            the audio file.
        chunk_size_minutes: Size of audio chunks in minutes for processing.
            Larger chunks may be more accurate but use more memory.
            
    Returns:
        Transcription text, or None if transcription failed.
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None
    
    logger.info(f"Transcribing audio file: {audio_path}")
    recognizer = sr.Recognizer()
    
    # If output_text_path is None, create one based on the audio file
    if output_text_path is None:
        output_text_path = os.path.splitext(audio_path)[0] + '_transcript.txt'
    
    try:
        # Load audio file
        logger.info(f"Loading audio file into memory")
        audio = AudioSegment.from_file(audio_path)
        
        # Get audio duration in milliseconds
        duration_ms = len(audio)
        logger.info(f"Audio duration: {duration_ms/1000/60:.2f} minutes")
        
        chunk_size_ms = chunk_size_minutes * 60 * 1000
        
        # Calculate number of chunks
        num_chunks = math.ceil(duration_ms / chunk_size_ms)
        
        full_transcript = ""
        logger.info(f"Processing audio in {num_chunks} chunks of {chunk_size_minutes} minutes each")
        
        # Process audio in chunks to avoid memory issues
        for i in range(num_chunks):
            logger.info(f"Processing chunk {i+1}/{num_chunks}...")
            start_ms = i * chunk_size_ms
            end_ms = min((i + 1) * chunk_size_ms, duration_ms)
            
            # Extract chunk
            chunk = audio[start_ms:end_ms]
            
            # Export chunk to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
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
                    logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                    full_transcript += "[recognition error] "
                    
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_path}: {e}")
        
        # Save transcript to file
        cleaned_transcript = full_transcript.strip()
        logger.info(f"Saving transcript to: {output_text_path}")
        
        # Create directory for transcript if it doesn't exist
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


def download_and_transcribe(
    url: str, 
    audio_output: Optional[str] = None, 
    text_output: Optional[str] = None, 
    chunk_size_minutes: int = 2, 
    use_youtube_transcript: bool = True, 
    languages: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Download a video, extract audio, and transcribe to text.
    
    Args:
        url: URL of the video to download.
        audio_output: Path to save the audio file. If None, saves to downloads
            directory.
        text_output: Path to save the transcript. If None, saves alongside the
            audio file.
        chunk_size_minutes: Size of audio chunks in minutes for processing.
        use_youtube_transcript: Whether to try downloading YouTube's transcript
            first (if available).
        languages: List of language codes to try downloading transcripts for.
            Defaults to ['en'].
            
    Returns:
        Tuple of (audio path, transcript path, transcript text, source),
        or (None, None, None, None) if download failed.
    """
    # Try to download YouTube's transcript first if requested
    if use_youtube_transcript and 'youtube.com' in url:
        logger.info("Attempting to download YouTube's transcript first...")
        youtube_transcript, transcript_path = download_youtube_transcript(url, text_output, languages)
        
        if youtube_transcript:
            logger.info("Successfully downloaded YouTube's transcript")
            
            # Still download audio for completeness if audio_output was specified
            audio_path = None
            if audio_output:
                audio_path = download_audio(url, audio_output)
                
            return audio_path, transcript_path, youtube_transcript, "youtube"
    
    # If YouTube transcript wasn't requested, unavailable, or failed, do our own transcription
    logger.info("Performing custom transcription using downloaded audio...")
    
    # Download the video's audio
    audio_path = download_audio(url, audio_output)
    if not audio_path:
        logger.error("Failed to download audio. Exiting.")
        return None, None, None, None
    
    # Transcribe the audio to text
    transcript = transcribe_audio(audio_path, text_output, chunk_size_minutes)
    
    if text_output is None:
        text_output = os.path.splitext(audio_path)[0] + '_transcript.txt'
    
    return audio_path, text_output, transcript, "custom"


def compare_transcripts(
    transcript1_path: str, 
    transcript2_path: str, 
    output_path: Optional[str] = None
) -> Optional[str]:
    """Compare two transcript files and output the comparison results.
    
    Args:
        transcript1_path: Path to the first transcript file.
        transcript2_path: Path to the second transcript file.
        output_path: Path to save the comparison results. If None, saves to
            downloads directory.
            
    Returns:
        Path to the comparison file, or None if comparison failed.
    """
    if not os.path.exists(transcript1_path) or not os.path.exists(transcript2_path):
        logger.error(f"One or both transcript files don't exist.")
        return None
    
    try:
        # Read transcripts
        with open(transcript1_path, 'r', encoding='utf-8') as f1:
            transcript1 = f1.read().strip()
        
        with open(transcript2_path, 'r', encoding='utf-8') as f2:
            transcript2 = f2.read().strip()
        
        # Generate output path if not provided
        if output_path is None:
            downloads_dir = os.path.join(os.getcwd(), 'downloads')
            ensure_directory_exists(downloads_dir)
            
            base_name1 = os.path.splitext(os.path.basename(transcript1_path))[0]
            base_name2 = os.path.splitext(os.path.basename(transcript2_path))[0]
            output_path = os.path.join(
                downloads_dir, f"{base_name1}_vs_{base_name2}_comparison.txt"
            )
        
        # Perform simple comparison (word count, character count)
        words1 = transcript1.split()
        words2 = transcript2.split()
        
        comparison_text = f"""Transcript Comparison Results
=========================

Transcript 1: {os.path.basename(transcript1_path)}
- Word count: {len(words1)}
- Character count: {len(transcript1)}

Transcript 2: {os.path.basename(transcript2_path)}
- Word count: {len(words2)}
- Character count: {len(transcript2)}

Word count difference: {abs(len(words1) - len(words2))} ({abs(len(words1) - len(words2))/max(1, len(words1)):.2%} difference)
Character count difference: {abs(len(transcript1) - len(transcript2))} ({abs(len(transcript1) - len(transcript2))/max(1, len(transcript1)):.2%} difference)

Transcript Samples:
------------------
Transcript 1 (first 150 chars): {transcript1[:150]}...

Transcript 2 (first 150 chars): {transcript2[:150]}...

Complete Transcripts:
------------------
Transcript 1:
{transcript1}

Transcript 2:
{transcript2}
"""
        
        # Save comparison to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(comparison_text)
        
        logger.info(f"Transcript comparison saved to: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error comparing transcripts: {e}")
        return None


def main() -> int:
    """Main entry point for the command line interface.
    
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(description='YouTube video download and transcription tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a video')
    download_parser.add_argument('url', help='URL of the video to download')
    download_parser.add_argument(
        '--output', '-o', 
        help='Output path (default: downloads/%(title)s.%(ext)s)'
    )
    download_parser.add_argument(
        '--format', '-f', 
        help='Format to download (e.g., "bestvideo[height<=720]+bestaudio/best[height<=720]")'
    )
    
    # Download-audio command
    download_audio_parser = subparsers.add_parser(
        'download-audio', 
        help='Download a video\'s audio'
    )
    download_audio_parser.add_argument('url', help='URL of the video to download')
    download_audio_parser.add_argument(
        '--output', '-o', 
        help='Output path (default: downloads/%(title)s.mp3)'
    )
    
    # Convert format command
    convert_parser = subparsers.add_parser(
        'convert', 
        help='Convert a video/audio file to a different format'
    )
    convert_parser.add_argument('input_path', help='Path to the file to convert')
    convert_parser.add_argument(
        '--output', '-o', 
        help='Output path (default: [input_path]_converted.ext)'
    )
    convert_parser.add_argument(
        '--to-format', '-t', required=True, 
        help='Target format (e.g., mp4, mkv, webm, mp3, aac, wav, etc.)'
    )
    convert_parser.add_argument(
        '--audio-only', '-a', action='store_true',
        help='Extract audio only'
    )
    convert_parser.add_argument(
        '--quality', '-q', default='192',
        help='Audio quality for audio conversion (default: 192)'
    )
    
    # Download-transcript command
    download_transcript_parser = subparsers.add_parser(
        'download-transcript', 
        help='Download a YouTube video\'s transcript (subtitles)'
    )
    download_transcript_parser.add_argument('url', help='URL of the YouTube video')
    download_transcript_parser.add_argument(
        '--output', '-o', 
        help='Output path for the transcript'
    )
    download_transcript_parser.add_argument(
        '--languages', '-l', nargs='+', default=['en'], 
        help='Language preference codes (e.g., en en-US es). Default: en'
    )
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe an audio file')
    transcribe_parser.add_argument('audio_path', help='Path to the audio file')
    transcribe_parser.add_argument(
        '--output', '-o', 
        help='Output path for the transcript (default: [audio_path]_transcript.txt)'
    )
    transcribe_parser.add_argument(
        '--chunk-size', '-c', type=int, default=2, 
        help='Size of audio chunks in minutes for processing (default: 2)'
    )
    
    # Compare transcripts command
    compare_parser = subparsers.add_parser(
        'compare-transcripts', 
        help='Compare two transcript files'
    )
    compare_parser.add_argument('transcript1', help='Path to the first transcript file')
    compare_parser.add_argument('transcript2', help='Path to the second transcript file')
    compare_parser.add_argument(
        '--output', '-o', 
        help='Output path for the comparison results'
    )
    
    # Download and transcribe command
    download_transcribe_parser = subparsers.add_parser(
        'download-and-transcribe', 
        help='Download a video and transcribe its audio'
    )
    download_transcribe_parser.add_argument('url', help='URL of the video to download')
    download_transcribe_parser.add_argument(
        '--audio-output', '-a', 
        help='Output path for the audio'
    )
    download_transcribe_parser.add_argument(
        '--text-output', '-t', 
        help='Output path for the transcript'
    )
    download_transcribe_parser.add_argument(
        '--chunk-size', '-c', type=int, default=2, 
        help='Size of audio chunks in minutes for processing (default: 2)'
    )
    download_transcribe_parser.add_argument(
        '--force-custom', '-f', action='store_true',
        help='Force using custom transcription even for YouTube videos'
    )
    download_transcribe_parser.add_argument(
        '--languages', '-l', nargs='+', default=['en'], 
        help='Language preference codes for YouTube transcript (e.g., en en-US es). Default: en'
    )
    
    args = parser.parse_args()
    
    if args.command == 'download':
        result = download_video(args.url, args.output, args.format)
        if result:
            logger.info(f"Video downloaded successfully to: {result}")
            
    elif args.command == 'download-audio':
        result = download_audio(args.url, args.output)
        if result:
            logger.info(f"Audio downloaded successfully to: {result}")
    
    elif args.command == 'convert':
        result = convert_format(args.input_path, args.output, args.to_format, args.audio_only, args.quality)
        if result:
            logger.info(f"File converted successfully to: {result}")
            
    elif args.command == 'download-transcript':
        transcript, output_path = download_youtube_transcript(args.url, args.output, args.languages)
        if transcript:
            logger.info(f"YouTube transcript downloaded successfully to: {output_path}")
            logger.info(f"Transcript preview (first 150 chars): {transcript[:150]}...")
        else:
            logger.error("Failed to download YouTube transcript. The video may not have subtitles available.")
    
    elif args.command == 'compare-transcripts':
        output_path = compare_transcripts(args.transcript1, args.transcript2, args.output)
        if output_path:
            logger.info(f"Transcript comparison saved to: {output_path}")
        else:
            logger.error("Failed to compare transcripts.")
            
    elif args.command == 'transcribe':
        if not os.path.exists(args.audio_path):
            logger.error(f"Audio file not found: {args.audio_path}")
            return 1
            
        transcript = transcribe_audio(args.audio_path, args.output, args.chunk_size)
        if transcript:
            output_path = args.output or os.path.splitext(args.audio_path)[0] + '_transcript.txt'
            logger.info(f"Transcription complete. Saved to: {output_path}")
            logger.info(f"Transcript preview (first 150 chars): {transcript[:150]}...")
            
    elif args.command == 'download-and-transcribe':
        use_youtube_transcript = not args.force_custom
        
        audio_path, transcript_path, transcript, source = download_and_transcribe(
            args.url, 
            args.audio_output, 
            args.text_output,
            args.chunk_size,
            use_youtube_transcript,
            args.languages
        )
        
        if transcript_path and transcript:
            logger.info("Download and transcription complete!")
            if audio_path:
                logger.info(f"- Audio saved to: {audio_path}")
            logger.info(f"- Transcript saved to: {transcript_path}")
            logger.info(f"- Transcript source: {source.capitalize()}")
            if not transcript.startswith("[The audio could not be transcribed"):
                logger.info(f"- Transcript preview (first 150 chars): {transcript[:150]}...")
            else:
                logger.warning("- Note: Transcription had limited success recognizing speech in the audio")
        else:
            logger.error("Transcription process failed.")
    
    else:
        parser.print_help()
        return 1
        
    return 0


if __name__ == '__main__':
    sys.exit(main())
