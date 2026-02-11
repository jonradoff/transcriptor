#!/usr/bin/env python3
"""
Batch transcribe all videos from a YouTube channel with speaker identification.
Extracts guest names from video titles and uses them in transcripts.
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Common patterns for extracting guest names from titles
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

    # Pattern: "Guest Name + Jon Radoff" or "Guest Name + Jon"
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
        # Exclude common non-name words
        if name.lower() not in ['artificial', 'generative', 'virtual', 'creative', 'decentralized', 'game', 'web3', 'the', 'and']:
            return name

    # Pattern: "Topic: Guest Name and Jon" (name before "and Jon")
    match = re.search(r':\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+and\s+Jon', title)
    if match:
        return match.group(1)

    # Pattern: "Jon Radoff and Guest Name" (name after "and")
    match = re.search(r'Jon(?:\s+Radoff)?\s+and\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', title)
    if match:
        return match.group(1)

    # Pattern: "Guest Name - Topic" at start
    match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*[\-\:]', title)
    if match:
        name = match.group(1)
        # Exclude common non-name words
        if name.lower() not in ['artificial', 'generative', 'virtual', 'creative', 'decentralized', 'game', 'web3']:
            return name

    return None


def get_all_videos(channel_url):
    """Get all videos from channel including videos, streams, and shorts."""
    videos = []

    # Get regular videos
    print("Fetching regular videos...")
    result = subprocess.run(
        ['yt-dlp', '--flat-playlist', '--print', '%(id)s|||%(title)s|||%(duration)s',
         f'{channel_url}/videos'],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split('\n'):
        if line and '|||' in line:
            parts = line.split('|||')
            if len(parts) >= 2:
                video_id, title = parts[0], parts[1]
                duration = parts[2] if len(parts) > 2 else 'NA'
                videos.append({
                    'id': video_id,
                    'title': title,
                    'duration': duration,
                    'type': 'video'
                })

    # Get livestreams
    print("Fetching livestreams...")
    result = subprocess.run(
        ['yt-dlp', '--flat-playlist', '--print', '%(id)s|||%(title)s|||%(duration)s',
         f'{channel_url}/streams'],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split('\n'):
        if line and '|||' in line:
            parts = line.split('|||')
            if len(parts) >= 2:
                video_id, title = parts[0], parts[1]
                duration = parts[2] if len(parts) > 2 else 'NA'
                videos.append({
                    'id': video_id,
                    'title': title,
                    'duration': duration,
                    'type': 'livestream'
                })

    # Get shorts
    print("Fetching shorts...")
    result = subprocess.run(
        ['yt-dlp', '--flat-playlist', '--print', '%(id)s|||%(title)s|||%(duration)s',
         f'{channel_url}/shorts'],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split('\n'):
        if line and '|||' in line:
            parts = line.split('|||')
            if len(parts) >= 2:
                video_id, title = parts[0], parts[1]
                duration = parts[2] if len(parts) > 2 else 'NA'
                videos.append({
                    'id': video_id,
                    'title': title,
                    'duration': duration,
                    'type': 'short'
                })

    return videos


def transcribe_video(video_id, title, guest_name=None):
    """Transcribe a single video with speaker identification."""
    from transcribe_with_speakers import SpeakerTranscriber

    output_path = OUTPUT_DIR / f"{video_id}.txt"

    # Skip if already transcribed
    if output_path.exists():
        print(f"  Already transcribed: {video_id}")
        return output_path

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    # Initialize transcriber (reuse if possible)
    if not hasattr(transcribe_video, 'transcriber'):
        transcribe_video.transcriber = SpeakerTranscriber(whisper_model="base")

    transcriber = transcribe_video.transcriber

    # Custom transcription with guest name
    result = transcriber.transcribe_video(video_url, video_title=title)

    # Post-process to replace "Guest" with actual guest name if known
    if result and guest_name:
        with open(result, 'r') as f:
            content = f.read()
        content = content.replace('] Guest:', f'] {guest_name}:')
        content = content.replace('] Unknown:', f'] {guest_name}:')  # Also replace Unknown for guests
        with open(result, 'w') as f:
            f.write(content)
        print(f"  Labeled guest as: {guest_name}")

    return result


def main():
    channel_url = "https://www.youtube.com/@BuildingtheMetaverseRadoff"

    if len(sys.argv) > 1:
        channel_url = sys.argv[1]

    print(f"Fetching all videos from: {channel_url}")
    print("=" * 60)

    videos = get_all_videos(channel_url)

    print(f"\nTotal videos found: {len(videos)}")
    print(f"  - Regular videos: {len([v for v in videos if v['type'] == 'video'])}")
    print(f"  - Livestreams: {len([v for v in videos if v['type'] == 'livestream'])}")
    print(f"  - Shorts: {len([v for v in videos if v['type'] == 'short'])}")

    # Save video list
    with open('video_list.json', 'w') as f:
        json.dump(videos, f, indent=2)
    print(f"\nSaved video list to video_list.json")

    # Extract guest names and show them
    print("\n" + "=" * 60)
    print("Detected guests from titles:")
    print("=" * 60)

    for video in videos:
        guest = extract_guest_from_title(video['title'])
        video['guest'] = guest
        if guest:
            print(f"  {video['id']}: {guest}")
            print(f"    Title: {video['title'][:60]}...")

    # Start transcription
    print("\n" + "=" * 60)
    print("Starting transcription...")
    print("=" * 60)

    completed = 0
    failed = []

    for i, video in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {video['title'][:50]}...")
        print(f"  ID: {video['id']}, Type: {video['type']}, Guest: {video.get('guest', 'Unknown')}")

        try:
            result = transcribe_video(video['id'], video['title'], video.get('guest'))
            if result:
                completed += 1
                print(f"  ✓ Completed")
            else:
                failed.append(video['id'])
                print(f"  ✗ Failed")
        except Exception as e:
            failed.append(video['id'])
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print(f"Transcription complete!")
    print(f"  Completed: {completed}/{len(videos)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed IDs: {failed}")


if __name__ == '__main__':
    main()
