#!/usr/bin/env python3
"""
Twelve Labs Tools - Video Analysis using Twelve Labs API

This script provides functionality to:
- Analyze video files using the Twelve Labs API
- Extract detailed information from video content

Required environment variable:
- TWELVE_LABS_API_KEY: Your Twelve Labs API key
"""

import os
import sys
import time
import argparse
import logging
import platform
from pathlib import Path
from typing import Dict, Optional, Any, List

# Import Twelve Labs SDK
try:
    from twelvelabs import TwelveLabs
    from twelvelabs.models.task import Task
    from twelvelabs.models.search import SearchData, GroupByVideoSearchData
except ImportError:
    print("Error: The Twelve Labs SDK is required. Install it with: pip install twelvelabs")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handlers with formatters
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", 
                                            "%Y-%m-%d %H:%M:%S"))
logger.addHandler(console_handler)

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
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: The path to the directory to ensure exists.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def normalize_path(path_str: str) -> str:
    """Normalize a path string to work on the current platform.
    
    Args:
        path_str: The path string to normalize
        
    Returns:
        Normalized path string for the current platform
    """
    # Convert to Path object and back to string to normalize slashes for the current OS
    path_obj = Path(path_str)
    return str(path_obj)


def initialize_client(api_key: Optional[str] = None) -> Optional[TwelveLabs]:
    """Initialize the Twelve Labs client using the provided API key or from environment variables.
    
    Args:
        api_key: Optional API key to use instead of environment variable
        
    Returns:
        TwelveLabs client if successful, None otherwise
    """
    # Use provided API key or get from environment
    if not api_key:
        api_key = os.getenv("TWELVE_LABS_API_KEY")
        
    if not api_key:
        logger.error("TWELVE_LABS_API_KEY environment variable is not set.")
        
        # Show platform-specific instructions
        if platform.system() == "Windows":
            logger.error("Set it with: set TWELVE_LABS_API_KEY=your_api_key")
            logger.error("Or use the --api-key parameter: python twelve_labs_tools.py get-video-info --api-key YOUR_KEY video_path")
        else:
            logger.error("Set it with: export TWELVE_LABS_API_KEY=your_api_key")
            logger.error("Or use the --api-key parameter: python twelve_labs_tools.py get-video-info --api-key YOUR_KEY video_path")
            
        return None
    
    try:
        client = TwelveLabs(api_key=api_key)
        logger.debug("Twelve Labs client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Error initializing Twelve Labs client: {e}")
        return None


def create_index(client: TwelveLabs, index_name: str = None) -> Optional[Dict[str, Any]]:
    """Create a new index in Twelve Labs.
    
    Args:
        client: Initialized Twelve Labs client
        index_name: Optional name for the index (default: auto-generated)
    
    Returns:
        Dictionary with index details if successful, None otherwise
    """
    try:
        # Use a timestamped name if none provided
        if not index_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            index_name = f"index_{timestamp}"
        
        # Define the models to use
        models = [
            {
                "name": "marengo2.7",
                "options": ["visual", "audio"]
            },
            {
                "name": "pegasus1.2",
                "options": ["visual", "audio"]
            }
        ]
        
        logger.info(f"Creating new Twelve Labs index: '{index_name}'")
        created_index = client.index.create(
            name=index_name,
            models=models,
            addons=["thumbnail"]
        )
        
        logger.info(f"Index created successfully:")
        logger.info(f"  ID: {created_index.id}")
        logger.info(f"  Name: {created_index.name}")
        
        if VERBOSE_OUTPUT:
            logger.info("Models:")
            for i, model in enumerate(created_index.models, 1):
                logger.info(f"  Model {i}:")
                logger.info(f"    Name: {model.name}")
                logger.info(f"    Options: {model.options}")
        
        return {
            "id": created_index.id,
            "name": created_index.name,
            "created_at": created_index.created_at,
            "models": [{"name": model.name, "options": model.options} for model in created_index.models]
        }
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        return None


def on_task_update(task: Task):
    """Callback function to print task status updates.
    
    Args:
        task: The task object with updated status
    """
    logger.info(f"  Task status: {task.status}")


def create_video_task(client: TwelveLabs, index_id: str, file_path: str) -> Optional[Dict[str, Any]]:
    """Create a video indexing task.
    
    Args:
        client: Initialized Twelve Labs client
        index_id: ID of the index to add the video to
        file_path: Path to the video file
    
    Returns:
        Dictionary with task details if successful, None otherwise
    """
    try:
        # Normalize the path for the current platform
        normalized_path = normalize_path(file_path)
        
        # Handle both absolute and relative paths
        if not os.path.isabs(normalized_path):
            # If it's a relative path, try multiple options
            # First check relative to current directory
            if os.path.isfile(normalized_path):
                absolute_path = os.path.abspath(normalized_path)
            else:
                # Try relative to script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                potential_path = os.path.join(script_dir, normalized_path)
                
                if os.path.isfile(potential_path):
                    absolute_path = potential_path
                else:
                    logger.error(f"Video file not found: {file_path}")
                    logger.error(f"Tried paths: {normalized_path} and {potential_path}")
                    return None
        else:
            # It's already an absolute path
            if os.path.isfile(normalized_path):
                absolute_path = normalized_path
            else:
                logger.error(f"Video file not found: {file_path}")
                return None
        
        logger.info(f"Creating video indexing task for: {absolute_path}")
        
        task = client.task.create(
            index_id=index_id,
            file=absolute_path
        )
        
        logger.info(f"Task created successfully with ID: {task.id}")
        logger.info("Waiting for task completion (this may take some time)...")
        
        # Wait for the task to complete
        task.wait_for_done(sleep_interval=5, callback=on_task_update)
        
        if task.status != "ready":
            logger.error(f"Indexing failed with status: {task.status}")
            return None
        
        logger.info(f"Video indexing complete!")
        logger.info(f"  Video ID: {task.video_id}")
        
        return {
            "task_id": task.id,
            "video_id": task.video_id,
            "status": task.status
        }
    except Exception as e:
        logger.error(f"Error creating video task: {e}")
        return None


def make_search_request(client: TwelveLabs, index_id: str, video_id: Optional[str] = None, query: str = "What is this video about?") -> Optional[Dict[str, Any]]:
    """Make a search request to analyze the video.
    
    Args:
        client: Initialized Twelve Labs client
        index_id: ID of the index containing the video
        video_id: Optional video ID to limit search to a specific video
        query: Search query to analyze the video content
    
    Returns:
        Dictionary with search results if successful, None otherwise
    """
    try:
        search_options = {
            "index_id": index_id,
            "options": ["visual", "audio"],
            "query_text": query,
            "group_by": "clip",
            "page_limit": 5,
            "sort_option": "score"
        }
        
        # Add video filter if specified
        if video_id:
            search_options["filter"] = {"video_id": video_id}
        
        logger.info(f"Making search request with query: '{query}'")
        
        # Add verbose logging for debugging
        if VERBOSE_OUTPUT:
            logger.debug(f"Search options: {search_options}")
        
        result = client.search.query(**search_options)
        
        # Log search result info for debugging
        if VERBOSE_OUTPUT:
            logger.debug(f"Search pool total count: {result.pool.total_count}")
            logger.debug(f"Search results count: {len(result.data) if result.data else 0}")
        
        search_data = {
            "pool": {
                "total_count": result.pool.total_count,
                "total_duration": result.pool.total_duration,
                "index_id": result.pool.index_id
            },
            "page_info": {
                "limit_per_page": result.page_info.limit_per_page,
                "total_results": result.page_info.total_results,
                "page_expires_at": result.page_info.page_expires_at
            },
            "results": []
        }
        
        # Process search results
        for item in result.data:
            if isinstance(item, GroupByVideoSearchData):
                video_result = {
                    "video_id": item.id,
                    "clips": []
                }
                
                if item.clips:
                    for clip in item.clips:
                        clip_data = {
                            "score": clip.score,
                            "start": clip.start,
                            "end": clip.end,
                            "confidence": clip.confidence,
                            "thumbnail_url": clip.thumbnail_url
                        }
                        video_result["clips"].append(clip_data)
                
                search_data["results"].append(video_result)
            else:
                clip_data = {
                    "score": item.score,
                    "start": item.start,
                    "end": item.end,
                    "video_id": item.video_id,
                    "confidence": item.confidence,
                    "thumbnail_url": item.thumbnail_url
                }
                search_data["results"].append(clip_data)
        
        return search_data
    except Exception as e:
        logger.error(f"Error making search request: {e}")
        return None


def generate_video_summary(client: TwelveLabs, video_id: str, api_key: Optional[str] = None) -> Optional[str]:
    """Generate a summary of the video content.
    
    Args:
        client: Initialized Twelve Labs client
        video_id: ID of the video to summarize
        api_key: Optional API key to use instead of extracting from client
    
    Returns:
        Summary text if successful, None otherwise
    """
    MAX_RETRIES = 2
    retry_count = 0
    
    try:
        import requests
        import json
        
        # If API key not provided, try to get it from environment
        if not api_key:
            api_key = os.getenv("TWELVE_LABS_API_KEY")
        
        if not api_key:
            logger.error("Could not access API key for summary generation")
            return None
            
        # Use the generate endpoint for summarization
        BASE_URL = "https://api.twelvelabs.io/v1.3"
        data = {
            "video_id": video_id,
            "prompt": "What is interesting about this video?",
            "temperature": 0.2
        }
        
        headers = {"x-api-key": api_key}
        
        while retry_count <= MAX_RETRIES:
            if retry_count > 0:
                logger.info(f"Retry {retry_count}/{MAX_RETRIES} for summary generation...")
                
            logger.info("Generating video summary...")
            response = requests.post(
                f"{BASE_URL}/generate",
                json=data,
                headers=headers
            )
            
            # For debug/verbose - log raw response info
            if VERBOSE_OUTPUT:
                logger.debug(f"Summary API Response Code: {response.status_code}")
                logger.debug(f"Summary API Response Headers: {response.headers}")
                logger.debug(f"Summary API Response Content Type: {response.headers.get('Content-Type', 'unknown')}")
                # Log the first 1000 chars of the response to avoid very large logs
                response_text = response.text[:1000]
                if len(response.text) > 1000:
                    response_text += "... (truncated)"
                logger.debug(f"Summary API Response (first 1000 chars): {response_text}")
            
            if response.status_code == 200:
                # First, try to determine the content type
                content_type = response.headers.get('Content-Type', '').lower()
                
                # For debug
                logger.info(f"Response received with content type: {content_type}")
                
                # Different parsing strategies based on content type
                if 'application/json' in content_type:
                    # Check if response might be JSON Lines (JSONL) format - multiple JSON objects separated by newlines
                    if '\n' in response.text and response.text.strip().startswith('{"event_type":'):
                        logger.info("Detected streaming JSONL response format")
                        # Process JSONL format (streaming response)
                        try:
                            lines = response.text.strip().split('\n')
                            combined_text = ""
                            
                            for line in lines:
                                if not line.strip():
                                    continue
                                    
                                try:
                                    json_obj = json.loads(line)
                                    # Most common pattern in streaming responses
                                    if json_obj.get('event_type') == 'text_generation' and 'text' in json_obj:
                                        combined_text += json_obj['text']
                                except json.JSONDecodeError:
                                    # Skip lines that aren't valid JSON
                                    continue
                            
                            if combined_text:
                                logger.info("Successfully extracted text from streaming response")
                                return combined_text
                        except Exception as jsonl_err:
                            logger.error(f"Error processing JSONL format: {jsonl_err}")
                    else:
                        # Try standard JSON parsing for single JSON object
                        try:
                            result = response.json()
                            
                            # Try different known response structures
                            if 'output' in result and 'text' in result['output']:
                                # Standard structure: {"output": {"text": "summary text"}}
                                summary = result['output']['text']
                                return summary
                            elif 'text' in result:
                                # Alternative structure: {"text": "summary text"}
                                return result['text']
                            elif 'summary' in result:
                                # Another alternative: {"summary": "summary text"}
                                return result['summary']
                            elif 'response' in result:
                                # Generic wrapper: {"response": "summary text"}
                                return result['response']
                            else:
                                # If we can't find known fields, log all top-level keys for debugging
                                if VERBOSE_OUTPUT:
                                    logger.debug(f"Available keys in response: {list(result.keys())}")
                                
                                # Check if any key has a string value that might be our summary
                                for key, value in result.items():
                                    if isinstance(value, str) and len(value) > 50:
                                        logger.info(f"Using value from key '{key}' as summary")
                                        return value
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Failed to parse single JSON response: {json_err}")
                            # Continue to text-based extraction fallback
                else:
                    # For text/plain or other content types, treat as direct text
                    logger.info("Response is not JSON, treating as plain text")
                
                # If we get here, JSON parsing didn't work - try text-based extraction
                text = response.text.strip()
                
                # Check again for JSONL format in raw text, regardless of content type
                if '\n' in text and text.strip().startswith('{"event_type":'):
                    logger.info("Detected JSONL format in text response")
                    try:
                        lines = text.strip().split('\n')
                        combined_text = ""
                        
                        for line in lines:
                            if not line.strip():
                                continue
                                
                            try:
                                json_obj = json.loads(line)
                                if json_obj.get('event_type') == 'text_generation' and 'text' in json_obj:
                                    combined_text += json_obj['text']
                            except json.JSONDecodeError:
                                # Skip lines that aren't valid JSON
                                continue
                        
                        if combined_text:
                            logger.info("Successfully extracted text from JSONL format")
                            return combined_text
                    except Exception as jsonl_err:
                        logger.error(f"Error processing JSONL in text: {jsonl_err}")
                
                # Other text-based extraction methods
                # Option 1: Look for chunks of text between quotes
                import re
                quotes_matches = re.findall(r'"([^"]{50,})"', text)
                if quotes_matches:
                    longest_match = max(quotes_matches, key=len)
                    logger.info("Extracted summary from quoted text in response")
                    return longest_match
                
                # Option 2: Look for JSON-like structures
                try:
                    # Find first '{' and all content up to matching '}'
                    if '{' in text:
                        start = text.find('{')
                        # Count braces to find matching end brace
                        brace_count = 1
                        for i in range(start + 1, len(text)):
                            if text[i] == '{':
                                brace_count += 1
                            elif text[i] == '}':
                                brace_count -= 1
                            
                            if brace_count == 0:
                                # Found matching closing brace
                                end = i + 1
                                json_str = text[start:end]
                                result = json.loads(json_str)
                                
                                # Check for known patterns in the extracted JSON
                                if 'output' in result and 'text' in result['output']:
                                    return result['output']['text']
                                elif 'text' in result:
                                    return result['text']
                                elif 'summary' in result:
                                    return result['summary']
                                elif 'response' in result:
                                    return result['response']
                                else:
                                    # Try any string value
                                    for key, value in result.items():
                                        if isinstance(value, str) and len(value) > 50:
                                            return value
                                
                                break
                except Exception as parse_err:
                    logger.error(f"Error in JSON extraction: {parse_err}")
                    
                # Option 3: If response is very short, just return it directly
                if len(text) < 2000 and len(text) > 50:
                    logger.info("Using raw response text as summary")
                    return text
                    
                # If we reach here, we couldn't extract a summary
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    logger.info(f"Summary extraction failed, retrying ({retry_count}/{MAX_RETRIES})...")
                    time.sleep(2)  # Brief delay before retry
                    continue
                else:
                    logger.error("Could not extract summary from API response after retries")
                    
                    # Log more details about the response in verbose mode
                    if VERBOSE_OUTPUT:
                        logger.debug(f"Response type: {type(response.text)}")
                        logger.debug(f"Response length: {len(response.text)} characters")
                    
                    # Fall back to a generic summary for short videos
                    return "This appears to be a short video clip. No detailed summary could be generated."
            elif response.status_code == 429:
                # Rate limiting - wait and retry
                logger.warning("Rate limited by API, waiting before retry...")
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    time.sleep(5)  # Wait longer for rate limit
                    continue
                else:
                    logger.error("Exceeded retry limit for rate limiting")
                    return None
            else:
                # Log the full error details
                logger.error(f"Error generating summary (HTTP {response.status_code}): {response.text}")
                
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    logger.info(f"Retrying after error ({retry_count}/{MAX_RETRIES})...")
                    time.sleep(2)
                    continue
                else:
                    logger.error("Failed to generate summary after retries")
                    return None
        
        # If we get here, we've exhausted retries
        return None
        
    except Exception as e:
        logger.error(f"Error generating video summary: {e}")
        return None


def get_video_info(video_path: str, verbose: bool = False, api_key: Optional[str] = None) -> bool:
    """Analyze a video file and extract information using Twelve Labs API.
    
    Args:
        video_path: Path to the video file
        verbose: Enable verbose output
        api_key: Optional API key to use instead of environment variable
    
    Returns:
        True if successful, False otherwise
    """
    # Configure logging based on verbosity
    configure_logging(verbose)
    
    # Log the file path being processed
    logger.info(f"Processing video file: {video_path}")
    
    # Initialize client
    client = initialize_client(api_key)
    if not client:
        return False
    
    # Create an index
    index_info = create_index(client)
    if not index_info:
        return False
    
    # Create a video indexing task
    task_info = create_video_task(client, index_info["id"], video_path)
    if not task_info:
        return False
    
    # Make a search request to analyze the video
    search_results = make_search_request(client, index_info["id"], task_info["video_id"])
    if not search_results:
        return False
    
    # Generate a summary of the video
    summary = generate_video_summary(client, task_info["video_id"], api_key)
    
    # Print the results
    print("\n" + "="*80)
    print(f"VIDEO ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nFile: {os.path.basename(video_path)}")
    print(f"Path: {os.path.abspath(video_path)}")
    
    print("\nAnalysis Details:")
    print(f"- Index ID: {index_info['id']}")
    print(f"- Video ID: {task_info['video_id']}")
    
    total_duration = search_results["pool"]["total_duration"]
    minutes, seconds = divmod(total_duration, 60)
    print(f"- Duration: {int(minutes)} minutes, {seconds:.1f} seconds")
    
    print("\nKey Moments:")
    if not search_results["results"]:
        print("  No key moments identified in this short video.")
    else:
        moment_count = 0
        for idx, result in enumerate(search_results["results"], 1):
            if "clips" in result and result["clips"]:
                for clip_idx, clip in enumerate(result["clips"], 1):
                    moment_count += 1
                    start_min, start_sec = divmod(clip["start"], 60)
                    end_min, end_sec = divmod(clip["end"], 60)
                    print(f"{idx}.{clip_idx} [{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] (Confidence: {clip['confidence']:.2f})")
            else:
                if "start" in result and "end" in result:
                    moment_count += 1
                    start_min, start_sec = divmod(result["start"], 60)
                    end_min, end_sec = divmod(result["end"], 60)
                    print(f"{idx}. [{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] (Confidence: {result['confidence']:.2f})")
        
        if moment_count == 0:
            print("  No meaningful key moments detected in this video.")
    
    if summary:
        print("\nSummary:")
        print(summary)
    else:
        print("\nSummary:")
        print("  No summary available for this short video clip.")
    
    print("\nAnalysis completed successfully!")
    return True


def main() -> int:
    """Main entry point for the command line interface.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Create a command line parser
    parser = argparse.ArgumentParser(description='Twelve Labs Video Analysis Tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Get video info command
    video_info_parser = subparsers.add_parser(
        'get-video-info', 
        help='Analyze a video file and extract information'
    )
    video_info_parser.add_argument(
        'video_path',
        help='Path to the video file to analyze'
    )
    video_info_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output with debugging information (default: False)'
    )
    video_info_parser.add_argument(
        '--api-key', '-k',
        help='Twelve Labs API key (overrides environment variable)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'get-video-info':
        success = get_video_info(args.video_path, args.verbose, args.api_key)
        return 0 if success else 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    # Handle the edge case for Windows command-line invocation
    if len(sys.argv) > 1 and sys.argv[1] == 'get-video-info':
        # Handle the case: python twelve_labs_tools get-video-info ...
        # This happens on Windows when the extension is not included
        args = sys.argv[2:]  # Skip script name and 'get-video-info'
        
        # Parse arguments manually
        verbose = False
        api_key = None
        video_path = None
        
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == '--verbose' or arg == '-v':
                verbose = True
                i += 1
            elif arg == '--api-key' or arg == '-k':
                if i + 1 < len(args):
                    api_key = args[i + 1]
                    i += 2
                else:
                    print("Error: --api-key requires a value")
                    sys.exit(1)
            else:
                # Assume it's the video path
                video_path = arg
                i += 1
        
        if video_path:
            print(f"Processing video: {video_path}")
            result = get_video_info(video_path, verbose, api_key)
            sys.exit(0 if result else 1)
        else:
            print("Error: Missing video path.")
            print("Usage: python twelve_labs_tools get-video-info [--verbose] VIDEO_PATH")
            sys.exit(1)
    
    # Set script file permissions to be executable on Unix-like systems
    if platform.system() != "Windows":
        try:
            import stat
            script_path = os.path.abspath(__file__)
            current_permissions = os.stat(script_path).st_mode
            os.chmod(script_path, current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception as e:
            # Non-critical error, just log it
            logger.debug(f"Could not set executable permissions: {e}")
    
    # Normal argparse-based execution
    sys.exit(main())
