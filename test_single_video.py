#!/usr/bin/env python3
"""
Test script to transcribe a single video.
Useful for testing before processing all videos.
"""

import sys
from pathlib import Path
from transcribe_channel import VideoTranscriber

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_video.py <video_url>")
        print("Example: python test_single_video.py https://www.youtube.com/watch?v=HN_hOuyXUkc")
        sys.exit(1)

    video_url = sys.argv[1]

    # Extract video ID from URL
    if 'v=' in video_url:
        video_id = video_url.split('v=')[1].split('&')[0]
    else:
        video_id = video_url.split('/')[-1]

    # Create a mock video object
    video = {
        'video_id': video_id,
        'url': video_url,
        'title': 'Test Video',
        'description': '',
        'published_at': ''
    }

    # Initialize transcriber
    print("Initializing transcriber...")
    transcriber = VideoTranscriber()

    # Transcribe
    result = transcriber.transcribe_video(video)

    if result:
        print(f"\nSuccess! Transcript saved to: {result}")
        print("\nFirst 500 characters:")
        print("-" * 60)
        with open(result, 'r') as f:
            print(f.read()[:500])
    else:
        print("\nFailed to transcribe video")


if __name__ == '__main__':
    main()
