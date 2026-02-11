#!/usr/bin/env python3
"""
GPU-optimized batch transcription script.
Skips already-completed videos and uses GPU acceleration.
"""

import os
import sys
import re
import json
from pathlib import Path

# Set device to CUDA before imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_guest_from_title(title):
    """Extract guest name from video title."""

    # Pattern: "Topic with Guest Name"
    match = re.search(r'\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', title)
    if match:
        return match.group(1)

    # Pattern: "Guest Name and Jon Radoff" or "Guest Name and Jon"
    match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+and\s+Jon', title)
    if match:
        return match.group(1)

    # Pattern: "Guest Name + Jon Radoff"
    match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*\+\s*Jon', title)
    if match:
        return match.group(1)

    # Pattern: "Name | Company" at start or after separator
    match = re.search(r'[\|:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*[\|,]', title)
    if match:
        return match.group(1)

    # Pattern: "Topic | Guest Name | Company"
    match = re.search(r'\|\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*\|', title)
    if match:
        return match.group(1)

    # Pattern: "Topic - Guest Name" (name after dash)
    match = re.search(r'[\-–]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?:\s*$|\s*[\|,\-])', title)
    if match:
        name = match.group(1)
        if name.lower() not in ['artificial', 'generative', 'virtual', 'creative', 'decentralized', 'game', 'web3', 'the', 'and']:
            return name

    # Pattern: "Jon Radoff and Guest Name"
    match = re.search(r'Jon(?:\s+Radoff)?\s+and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', title)
    if match:
        return match.group(1)

    # Pattern: "Guest Name - Topic" at start
    match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*[\-\:]', title)
    if match:
        name = match.group(1)
        if name.lower() not in ['artificial', 'generative', 'virtual', 'creative', 'decentralized', 'game', 'web3']:
            return name

    return None


def load_completed_videos():
    """Load list of already-completed video IDs."""
    completed = set()

    # Check completed_videos.txt file
    completed_file = Path('completed_videos.txt')
    if completed_file.exists():
        with open(completed_file, 'r') as f:
            for line in f:
                video_id = line.strip()
                if video_id:
                    completed.add(video_id)

    # Also check output directory
    for txt_file in OUTPUT_DIR.glob('*.txt'):
        video_id = txt_file.stem
        completed.add(video_id)

    return completed


def transcribe_video(video_id, title, guest_name=None, transcriber=None):
    """Transcribe a single video with speaker identification."""
    output_path = OUTPUT_DIR / f"{video_id}.txt"

    # Skip if already transcribed
    if output_path.exists():
        print(f"  Already transcribed: {video_id}")
        return output_path

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Custom transcription with guest name
    result = transcriber.transcribe_video(video_url, video_title=title)

    # Post-process to replace "Guest" with actual guest name if known
    if result and guest_name:
        with open(result, 'r') as f:
            content = f.read()
        content = content.replace('] Guest:', f'] {guest_name}:')
        content = content.replace('] Unknown:', f'] {guest_name}:')
        with open(result, 'w') as f:
            f.write(content)
        print(f"  Labeled guest as: {guest_name}")

    return result


def main():
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load video list
    video_list_path = Path('video_list.json')
    if not video_list_path.exists():
        print("Error: video_list.json not found")
        sys.exit(1)

    with open(video_list_path, 'r') as f:
        videos = json.load(f)

    print(f"Total videos in list: {len(videos)}")

    # Load completed videos
    completed = load_completed_videos()
    print(f"Already completed: {len(completed)}")

    # Filter to only pending videos
    pending_videos = [v for v in videos if v['id'] not in completed]
    print(f"Pending videos: {len(pending_videos)}")

    if not pending_videos:
        print("All videos already transcribed!")
        return

    # Extract guest names
    for video in pending_videos:
        video['guest'] = extract_guest_from_title(video['title'])

    # Initialize transcriber with GPU
    print("\nInitializing GPU-accelerated transcriber...")
    from transcribe_with_speakers import SpeakerTranscriber
    transcriber = SpeakerTranscriber(whisper_model="base")

    # Move diarization pipeline to GPU if available
    if transcriber.diarization_pipeline and torch.cuda.is_available():
        transcriber.diarization_pipeline = transcriber.diarization_pipeline.to(torch.device("cuda"))
        print("Diarization pipeline moved to GPU")

    print("\n" + "=" * 60)
    print("Starting GPU-accelerated transcription...")
    print("=" * 60)

    completed_count = 0
    failed = []
    import time

    for i, video in enumerate(pending_videos):
        print(f"\n[{i+1}/{len(pending_videos)}] {video['title'][:50]}...")
        print(f"  ID: {video['id']}, Type: {video['type']}, Guest: {video.get('guest', 'Unknown')}")

        try:
            result = transcribe_video(video['id'], video['title'], video.get('guest'), transcriber)
            if result:
                completed_count += 1
                print(f"  ✓ Completed")
                # Add delay between successful downloads to avoid rate limiting
                if i < len(pending_videos) - 1:  # Don't delay after last video
                    print("  Waiting 10 seconds before next video...")
                    time.sleep(10)
            else:
                failed.append(video['id'])
                print(f"  ✗ Failed")
        except Exception as e:
            failed.append(video['id'])
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            # Add longer delay after errors
            if i < len(pending_videos) - 1:
                print("  Waiting 30 seconds after error...")
                time.sleep(30)

    print("\n" + "=" * 60)
    print(f"Transcription complete!")
    print(f"  Completed: {completed_count}/{len(pending_videos)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed}")


if __name__ == '__main__':
    main()
