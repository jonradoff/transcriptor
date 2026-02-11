#!/usr/bin/env python3
"""
Simplified transcription script that works without speaker diarization.
Just transcribes with Whisper - good for testing.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import yt_dlp
import whisper
from datetime import timedelta

load_dotenv()

DOWNLOADS_DIR = Path('downloads')
OUTPUT_DIR = Path('output')
DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def download_video(video_url, video_id):
    """Download video and extract audio."""
    print(f"Downloading video: {video_url}")

    audio_path = DOWNLOADS_DIR / f"{video_id}.wav"

    if audio_path.exists():
        print(f"Audio already exists: {audio_path}")
        return audio_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(DOWNLOADS_DIR / f"{video_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return audio_path


def transcribe_with_whisper(audio_path, model_size="base"):
    """Transcribe audio using Whisper."""
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    print("Transcribing...")
    result = model.transcribe(
        str(audio_path),
        task="transcribe",
        language="en",
        verbose=True
    )

    return result


def format_transcript(result):
    """Format transcript with timestamps."""
    lines = []

    for segment in result['segments']:
        start_time = str(timedelta(seconds=int(segment['start'])))
        text = segment['text'].strip()
        lines.append(f"[{start_time}] {text}")

    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_transcribe.py <video_url> [model_size]")
        print("Example: python simple_transcribe.py https://www.youtube.com/watch?v=Z-Ul7oFtQNw base")
        print("\nModel sizes: tiny, base, small, medium, large, large-v3")
        print("  tiny/base: Fast but less accurate")
        print("  small/medium: Good balance")
        print("  large/large-v3: Best accuracy but slow")
        sys.exit(1)

    video_url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"

    # Extract video ID
    if 'v=' in video_url:
        video_id = video_url.split('v=')[1].split('&')[0]
    else:
        video_id = video_url.split('/')[-1]

    print("=" * 60)
    print(f"Transcribing: {video_url}")
    print(f"Model: {model_size}")
    print("=" * 60)
    print()

    # Download
    audio_path = download_video(video_url, video_id)

    # Transcribe
    result = transcribe_with_whisper(audio_path, model_size)

    # Format
    transcript = format_transcript(result)

    # Save
    output_path = OUTPUT_DIR / f"{video_id}_simple.txt"
    header = f"""Video: {video_url}
Video ID: {video_id}
Model: whisper-{model_size}

---

"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + transcript)

    print()
    print("=" * 60)
    print(f"Transcript saved: {output_path}")
    print("=" * 60)
    print()
    print("First 1000 characters:")
    print("-" * 60)
    print(transcript[:1000])


if __name__ == '__main__':
    main()
