# YouTube Tools - Video Download, Format Conversion, and Transcription

This project provides a comprehensive toolset for interacting with YouTube content:

- **Download YouTube Videos**: Get videos in various formats and quality levels
- **Format Conversion**: Convert videos and audio to multiple different formats
- **Transcription Generation**: Extract speech-to-text from any video using built-in transcription capabilities
- **Transcript Download**: Option to download existing YouTube transcriptions when available
- **Transcript Comparison**: Compare transcriptions from different sources to evaluate quality
- **Live Stream Discovery**: Find currently active YouTube live streams based on search criteria
- **Live Stream Transcription**: Real-time transcription of YouTube live streams as they happen

Built with [yt-dlp](https://github.com/yt-dlp/yt-dlp), this versatile toolkit works across Windows, macOS, and Linux using pure Python, handling both pre-recorded videos and (in future updates) live streams.

## Project Goals

The primary goals of this project are:

1. **Complete Video Flexibility**: Download and save YouTube videos in any desired format
2. **Comprehensive Transcription Support**:
   - Create high-quality transcripts from any video using custom transcription tools
   - Efficiently download existing YouTube transcripts when available
   - Compare transcriptions from different sources to evaluate quality
3. **Format Conversion**:
   - Convert between various video formats (MP4, MKV, WEBM, AVI, MOV)
   - Extract audio in multiple formats (MP3, WAV, AAC, OGG, FLAC, M4A)
4. **Live Stream Support**:
   - Connect to YouTube live streams
   - Process and transcribe live streams in real-time
   - Extract valuable information while streams are happening

All downloads and generated files are saved to a `downloads` folder within the project directory.

## Prerequisites

1. **Python 3.8 or later**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify with: `python --version` or `python3 --version`

2. **FFmpeg**
   - Required for audio extraction and format conversion
   - Verify with: `ffmpeg -version`
   - Installation varies by platform (see platform-specific setup sections below)

3. **Speech Recognition Requirements**
   - Internet connection (for Google Speech Recognition API)
   - Audio capabilities (for processing audio files)

## Platform-Specific Setup

### Windows Setup

1. **Python Installation**
   - When installing Python, ensure "Add Python to PATH" is checked
   - You can use Command Prompt, PowerShell, or Git Bash

2. **FFmpeg Installation**
   - Install via chocolatey: `choco install ffmpeg`
   - Or download directly from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Add to PATH if needed

### macOS Setup

1. **Python Installation**
   - macOS may come with Python, but it's recommended to install the latest version
   - You can use [Homebrew](https://brew.sh): `brew install python`

2. **FFmpeg Installation**
   - Install with Homebrew: `brew install ffmpeg`

### Linux Setup

1. **Python Installation**
   - Many distributions come with Python pre-installed
   - For Ubuntu/Debian: `sudo apt install python3 python3-pip`
   - For Fedora: `sudo dnf install python3 python3-pip`

2. **FFmpeg Installation**
   - For Ubuntu/Debian: `sudo apt install ffmpeg`
   - For Fedora: `sudo dnf install ffmpeg`

## Installation Steps

1. **Clone or download this repository**

2. **Create the downloads directory** (if it doesn't exist):

   ```bash
   mkdir -p downloads
   ```

3. **Virtual Environment Setup**

   ```bash
   # Create a virtual environment
   python -m venv venv
   # On some systems, you may need to use python3 instead
   # python3 -m venv venv
   
   # Activate the virtual environment
   # On Linux/macOS:
   source venv/bin/activate
   
   # On Windows:
   # With Command Prompt:
   venv\Scripts\activate
   # With PowerShell:
   .\venv\Scripts\Activate.ps1
   # With Git Bash:
   source venv/Scripts/activate
   ```

4. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   # On some systems, you may need to use pip3 instead
   # pip3 install -r requirements.txt
   ```

   This will install:
   - yt-dlp: For video downloading
   - SpeechRecognition: For speech-to-text conversion
   - pydub: For audio processing

5. **Verify Installation**

   ```bash
   # Check that all dependencies are installed correctly
   python -c "import yt_dlp, speech_recognition, pydub; print('All dependencies installed successfully!')"
   
   # Check yt-dlp version
   python -c "import yt_dlp; print(f'yt-dlp version: {yt_dlp.version.__version__}')"
   ```

## Usage Examples

The `youtube_tools.py` script provides a comprehensive command-line interface for all
functionality:

### 1. Download a Video

```bash
# Activate the virtual environment if not already active
python youtube_tools.py download "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Specify output path
python youtube_tools.py download "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output "downloads/my_video.mp4"

# Specify format
python youtube_tools.py download "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --format "bestvideo[height<=720]+bestaudio/best[height<=720]"
```

### 2. Download Audio Only

```bash
# Download audio in MP3 format
python youtube_tools.py download-audio "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Specify output path
python youtube_tools.py download-audio "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output "downloads/my_audio.mp3"
```

### 3. Convert Video/Audio Formats

```bash
# Convert a video file to a different format
python youtube_tools.py convert "downloads/my_video.mp4" --to-format mkv

# Extract audio from a video file
python youtube_tools.py convert "downloads/my_video.mp4" --to-format mp3 --audio-only

# Convert audio to a different format with quality setting
python youtube_tools.py convert "downloads/my_audio.mp3" --to-format aac --audio-only --quality 256
```

### 4. Transcribe an Audio File

```bash
# Transcribe an audio file
python youtube_tools.py transcribe "downloads/my_audio.mp3"

# Specify output path for transcript
python youtube_tools.py transcribe "downloads/my_audio.mp3" --output "downloads/my_transcript.txt"

# Adjust chunk size for better processing (in minutes)
python youtube_tools.py transcribe "downloads/my_audio.mp3" --chunk-size 1
```

### 5. Download YouTube Transcript

```bash
# Download the transcript for a YouTube video
python youtube_tools.py download-transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Specify output path and language
python youtube_tools.py download-transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --output "downloads/custom_transcript.txt" --languages en es
```

### 6. Download and Transcribe in One Step

```bash
# Download a video and transcribe its audio
python youtube_tools.py download-and-transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Specify output paths
python youtube_tools.py download-and-transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --audio-output "downloads/my_audio.mp3" \
    --text-output "downloads/my_transcript.txt"

# Adjust chunk size for better processing
python youtube_tools.py download-and-transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --chunk-size 1

# Force using custom transcription even if YouTube transcript is available
python youtube_tools.py download-and-transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --force-custom
```

### 7. Compare Transcripts

```bash
# Compare two transcript files
python youtube_tools.py compare-transcripts "downloads/youtube_transcript.txt" "downloads/custom_transcript.txt"

# Specify output path for comparison results
python youtube_tools.py compare-transcripts "downloads/youtube_transcript.txt" "downloads/custom_transcript.txt" \
    --output "downloads/transcript_comparison.txt"
```

### 8. List Live Streams

```bash
# List popular live streams (default: top 10)
python livestream_tools.py list

# Search for live streams with a query
python livestream_tools.py list --query "music concert"

# Adjust number of results
python livestream_tools.py list --query "gaming" --max-results 5

# Change results order
python livestream_tools.py list --order viewCount
```

### 9. Transcribe Live Streams

```bash
# Connect to a live stream and transcribe in real-time
python livestream_tools.py transcribe "https://www.youtube.com/watch?v=live_video_id"

# Connect to a live stream and transcribe in real-time verbose logging
python livestream_tools.py transcribe -v "https://www.youtube.com/watch?v=live_video_id"

# Specify output path for transcript
python livestream_tools.py transcribe "https://www.youtube.com/watch?v=live_video_id" \
    --output "downloads/live_transcript.txt"

# Adjust buffer size for audio processing
python livestream_tools.py transcribe "https://www.youtube.com/watch?v=live_video_id" \
    --buffer-size 5

# Set how often to write to transcript file (in seconds)
python livestream_tools.py transcribe "https://www.youtube.com/watch?v=live_video_id" \
    --output-interval 60

# Set a duration to stop transcription after (in seconds)
python livestream_tools.py transcribe "https://www.youtube.com/watch?v=live_video_id" \
    --duration 600
```

### 10. Extract Video Clips from Live Streams

```bash
# Extract 6-second clips every 60 seconds from a live stream
python livestream_tools.py get-clips 6 60 "https://www.youtube.com/watch?v=live_video_id"

# Extract 10-second clips every 30 seconds with verbose logging
python livestream_tools.py get-clips 10 30 "https://www.youtube.com/watch?v=live_video_id" -v

# Specify video format
python livestream_tools.py get-clips 6 60 "https://www.youtube.com/watch?v=live_video_id" \
    --format "best[height<=720]"

# Set maximum number of clips to extract before stopping
python livestream_tools.py get-clips 6 60 "https://www.youtube.com/watch?v=live_video_id" \
    --max-clips 10
```

The `get-clips` command extracts short video clips from a YouTube live stream at regular intervals:
- The first parameter is the duration (in seconds) of each clip
- The second parameter is the interval (in seconds) between clips
- Clips are saved to the `downloads/clips` directory with filenames using EPOCH timestamps (e.g., `1616935321-1616935327.mp4`)
- The highest resolution video format is used by default
- The command provides real-time feedback on extraction progress, including clip count and file sizes

### 11. Video Analysis with Twelve Labs API

```bash
# First, set your Twelve Labs API key as an environment variable
# On Linux/macOS:
export TWELVE_LABS_API_KEY=your_api_key

# On Windows:
set TWELVE_LABS_API_KEY=your_api_key

# Analyze a video file
python twelve_labs_tools.py get-video-info downloads/clips/1743357070-1743357076.mp4

# Analyze with verbose logging
python twelve_labs_tools.py get-video-info --verbose downloads/clips/1743357070-1743357076.mp4

# Provide API key directly (useful for one-time operations without setting environment variable)
python twelve_labs_tools.py get-video-info --api-key your_api_key downloads/clips/1743357070-1743357076.mp4
```

The `twelve_labs_tools.py` script provides AI-powered video analysis using the Twelve Labs API:

- **get-video-info**: Analyzes a video file and extracts detailed information
  - Creates an index in Twelve Labs for organizing videos
  - Uploads and processes the specified video file
  - Identifies key moments in the video
  - Generates a summary of the video content
  - Displays analysis results including duration, key moments and summary

Options:
- `--verbose, -v`: Enable detailed logging for debugging
- `--api-key, -k`: Directly specify your Twelve Labs API key (overrides environment variable)

Note: You must have a valid Twelve Labs API key to use this feature. The environment variable approach is recommended for regular use, while the --api-key option is convenient for one-time operations.

### 12. Getting Help

```bash
# Show all available commands for youtube_tools
python youtube_tools.py --help

# Show help for a specific command
python youtube_tools.py download-and-transcribe --help

# Show help for twelve_labs_tools
python twelve_labs_tools.py --help

# Show help for specific twelve_labs_tools command
python twelve_labs_tools.py get-video-info --help
```

## Advanced Usage

### Batch Processing

Create a script to process multiple videos:

```python
import os
from subprocess import run

urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=another_video_id",
    # Add more URLs as needed
]

for url in urls:
    run(["python", "youtube_tools.py", "download-and-transcribe", url])
```

### Format Conversion Pipeline

Create a script to download a video and convert it to multiple formats:

```python
import os
import subprocess

video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
formats = ["mp4", "mkv", "webm", "mp3", "wav"]

# Download the video
result = subprocess.run(["python", "youtube_tools.py", "download", video_url], capture_output=True, text=True)
video_path = result.stdout.strip().split("to: ")[1]

# Convert to various formats
for fmt in formats:
    audio_only = fmt in ["mp3", "wav", "aac", "ogg", "flac"]
    cmd = ["python", "youtube_tools.py", "convert", video_path, "--to-format", fmt]
    if audio_only:
        cmd.append("--audio-only")
    subprocess.run(cmd)
```

### Speech Recognition Tips

- For better speech recognition results:
  - Use shorter chunk sizes (0.5-1 minute) for clearer audio
  - Use longer chunk sizes (2-5 minutes) for less clear audio to provide more context
  - Music in the background can significantly impact recognition quality

## Troubleshooting

### Common Issues

#### Downloads Issues

- **Downloads Not Appearing**:
  - Check that you have a `downloads` folder in your project directory
  - Make sure you have write permissions to this folder
  - Verify that the URL is valid and the content is available

#### Speech Recognition Issues

- If you see "[inaudible segment]" markers:
  - The audio may have background music or noise interfering with recognition
  - Try a video with clearer speech and less background noise
  - You might need to use professional transcription services for better quality
- If you get "Could not request results from Google Speech Recognition service":
  - Check your internet connection
  - Google Speech Recognition has daily/hourly limits for free usage
- For processing very large files:
  - Adjust the chunk size parameter to find the optimal balance
  - Use `--chunk-size 1` for better accuracy on shorter clips
  - Use `--chunk-size 5` for longer videos to reduce API calls

#### Format Conversion Issues

- If converting to certain formats fails:
  - Ensure FFmpeg is correctly installed with all necessary codecs
  - Some formats may require additional codecs or libraries
  - Check FFmpeg output for specific error messages
- For quality issues:
  - Adjust the quality parameter (--quality) for different formats
  - Higher values generally mean better quality but larger file sizes

#### Audio File Issues

- If you get "Audio file not found":
  - Make sure the path to the audio file is correct
  - Verify that the file exists and has proper permissions
- For "Could not find downloaded audio file":
  - yt-dlp may have saved the file with a different name
  - Check the downloads folder manually for the correct file

### Platform-Specific Troubleshooting

#### FFmpeg Issues

- **Windows**:
  - Ensure FFmpeg is properly installed and in your PATH
  - Restart your terminal after installing FFmpeg
  - If using Git Bash or PowerShell, you may need to restart it after installation
- **macOS**:
  - If installed with Homebrew but not working, try: `brew link --overwrite ffmpeg`
  - You may need to restart your terminal after installation
- **Linux**:
  - For Ubuntu/Debian, you may need additional codecs:
    `sudo apt install ubuntu-restricted-extras`
  - For Fedora: `sudo dnf install ffmpeg-devel`

#### Python Environment Issues

- **Windows**:
  - Ensure Python is in PATH by running: `where python`
  - For virtual environment issues, try using absolute paths to the activate script
- **macOS/Linux**:
  - If you get "command not found: python", try using `python3` instead
  - For permission issues: `chmod +x ./venv/bin/activate`

#### SSL Certificate Errors

- Try updating with: `pip install --upgrade pip setuptools wheel`
- If problems persist, consider using: `yt-dlp --no-check-certificate URL` or adding
  the `--no-check-certificate` flag to your Python script arguments

## Technical Details

### Project Structure

```text
youtube_tools/
├── downloads/           # Directory where all downloads are saved
├── youtube_tools.py     # Main Python script with all functionality
├── download_demo.py     # Simple demo of video downloading functionality
├── video_to_text.py     # Script for transcribing videos to text
├── livestream_tools.py  # Live stream discovery and real-time transcription
├── requirements.txt     # Dependencies
└── README.md            # This documentation file
```

### Dependencies

- [yt-dlp](https://github.com/yt-dlp/yt-dlp): Advanced fork of youtube-dl for video
  downloading
- [SpeechRecognition](https://github.com/Uberi/speech_recognition): Library for
  performing speech recognition
- [pydub](https://github.com/jiaaro/pydub): Audio processing library for handling audio
  files
- [FFmpeg](https://ffmpeg.org/): Required for audio extraction and format conversion

## Future Development

- Support for more speech recognition engines
- Automatic subtitle generation from transcripts
- Multi-language transcription support
- Web interface
- Docker containerization

## Markdown Linting

This README follows Markdown linting standards. To check and fix Markdown linting issues,
you can use the following command:

```bash
npx markdownlint README.md
```

To automatically fix certain issues:

```bash
npx markdownlint-cli2-fix README.md
