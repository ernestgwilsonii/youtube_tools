"""
yt-dlp Demo Script

This script demonstrates how to use yt-dlp programmatically in Python.
Run this file with 'python download_demo.py' after activating the virtual environment.
"""

import os
import sys
import argparse
from yt_dlp import YoutubeDL

def download_video(url, output_format='mp4', quality='best', audio_only=False):
    """
    Download a video using yt-dlp.
    
    Args:
        url: URL of the video to download
        output_format: Format of the output file (default: mp4)
        quality: Quality of the video (default: best)
        audio_only: Whether to download only audio (default: False)
    """
    # Create downloads directory if it doesn't exist
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    
    # Configure yt-dlp options
    if audio_only:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'downloads/%(title)s.%(ext)s',
            'verbose': True
        }
    else:
        ydl_opts = {
            'format': f'{quality}[ext={output_format}]/best[ext={output_format}]/best',
            'outtmpl': 'downloads/%(title)s.%(ext)s',
            'verbose': True
        }
    
    # Execute download
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print(f"Downloaded: {info.get('title', 'Video')}")
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False
    
    return True

def list_formats(url):
    """List all available formats for a video."""
    ydl_opts = {
        'listformats': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=False)

def main():
    parser = argparse.ArgumentParser(description='Download videos using yt-dlp')
    parser.add_argument('url', help='URL of the video to download')
    parser.add_argument('--audio-only', '-a', action='store_true', help='Download audio only')
    parser.add_argument('--format', '-f', default='mp4', help='Format of the output file (default: mp4)')
    parser.add_argument('--quality', '-q', default='best', help='Quality of the video (default: best)')
    parser.add_argument('--list', '-l', action='store_true', help='List available formats')
    
    args = parser.parse_args()
    
    if args.list:
        list_formats(args.url)
    else:
        success = download_video(args.url, args.format, args.quality, args.audio_only)
        if success:
            print("Download completed successfully!")
        else:
            print("Download failed. Check the error messages above.")

if __name__ == '__main__':
    main()
