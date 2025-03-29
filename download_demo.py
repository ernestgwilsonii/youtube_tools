#!/usr/bin/env python3
"""
yt-dlp Demo Script

This script demonstrates how to use yt-dlp programmatically in Python.
Run this file with 'python download_demo.py' after activating the virtual environment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from yt_dlp import YoutubeDL

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


def download_video(
    url: str, 
    output_format: str = 'mp4', 
    quality: str = 'best', 
    audio_only: bool = False
) -> bool:
    """Download a video using yt-dlp.
    
    Args:
        url: URL of the video to download
        output_format: Format of the output file (default: mp4)
        quality: Quality of the video (default: best)
        audio_only: Whether to download only audio (default: False)
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    # Create downloads directory if it doesn't exist
    downloads_dir = os.path.join(os.getcwd(), 'downloads')
    ensure_directory_exists(downloads_dir)
    
    # Configure yt-dlp options
    if audio_only:
        ydl_opts: Dict[str, Any] = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(downloads_dir, '%(title)s.%(ext)s'),
            'verbose': False
        }
    else:
        ydl_opts: Dict[str, Any] = {
            'format': f'{quality}[ext={output_format}]/best[ext={output_format}]/best',
            'outtmpl': os.path.join(downloads_dir, '%(title)s.%(ext)s'),
            'verbose': False
        }
    
    # Execute download
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            logger.info(f"Downloaded: {info.get('title', 'Video')}")
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return False
    
    return True


def list_formats(url: str) -> None:
    """List all available formats for a video.
    
    Args:
        url: URL of the video to list formats for
    """
    ydl_opts: Dict[str, Any] = {
        'listformats': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=False)


def main() -> int:
    """Main entry point for the command line interface.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(description='Download videos using yt-dlp')
    parser.add_argument('url', help='URL of the video to download')
    parser.add_argument('--audio-only', '-a', action='store_true', help='Download audio only')
    parser.add_argument('--format', '-f', default='mp4', help='Format of the output file (default: mp4)')
    parser.add_argument('--quality', '-q', default='best', help='Quality of the video (default: best)')
    parser.add_argument('--list', '-l', action='store_true', help='List available formats')
    
    args = parser.parse_args()
    
    if args.list:
        list_formats(args.url)
        return 0
    else:
        success = download_video(args.url, args.format, args.quality, args.audio_only)
        if success:
            logger.info("Download completed successfully!")
            return 0
        else:
            logger.error("Download failed. Check the error messages above.")
            return 1


if __name__ == '__main__':
    sys.exit(main())
