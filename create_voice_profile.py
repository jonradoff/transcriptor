#!/usr/bin/env python3
"""
Create a voice profile from a reference video.
This extracts your voice from the specified timestamp in the reference video
and creates an embedding that can be used to identify your voice in other videos.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import yt_dlp
import numpy as np
import pickle
from pydub import AudioSegment

# Load environment variables
load_dotenv()

REFERENCE_VIDEO_URL = os.getenv('REFERENCE_VIDEO_URL', 'https://www.youtube.com/watch?v=HN_hOuyXUkc')
REFERENCE_START_TIME = os.getenv('REFERENCE_START_TIME', '0:29')

# Directories
DOWNLOADS_DIR = Path('downloads')
VOICE_PROFILES_DIR = Path('voice_profiles')
DOWNLOADS_DIR.mkdir(exist_ok=True)
VOICE_PROFILES_DIR.mkdir(exist_ok=True)


def parse_time_to_seconds(time_str):
    """Convert time string like '0:29' or '1:23:45' to seconds."""
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        return int(parts[0])


def download_reference_video():
    """Download the reference video."""
    print(f"Downloading reference video: {REFERENCE_VIDEO_URL}")

    audio_path = DOWNLOADS_DIR / 'reference_video.wav'

    # Check if already downloaded
    if audio_path.exists():
        print(f"Reference audio already exists: {audio_path}")
        return audio_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(DOWNLOADS_DIR / 'reference_video.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([REFERENCE_VIDEO_URL])

    if not audio_path.exists():
        raise FileNotFoundError(f"Failed to download and convert reference video")

    return audio_path


def extract_voice_segment(audio_path, start_time_str, duration=30):
    """
    Extract a segment of audio starting from the specified time.
    Duration in seconds (default 30 seconds for voice profiling).
    """
    print(f"Extracting voice segment from {start_time_str} for {duration} seconds")

    start_seconds = parse_time_to_seconds(start_time_str)

    # Load audio using pydub
    audio = AudioSegment.from_wav(str(audio_path))

    # Extract segment (pydub uses milliseconds)
    start_ms = start_seconds * 1000
    end_ms = start_ms + (duration * 1000)
    segment = audio[start_ms:end_ms]

    # Convert to mono if stereo
    if segment.channels > 1:
        segment = segment.set_channels(1)

    # Resample to 16kHz (required by resemblyzer)
    segment = segment.set_frame_rate(16000)

    # Save segment
    segment_path = VOICE_PROFILES_DIR / 'jon_radoff_voice_segment.wav'
    segment.export(str(segment_path), format='wav')

    print(f"Saved voice segment to {segment_path}")
    return segment_path


def create_voice_embedding(audio_path):
    """
    Create a voice embedding using Resemblyzer.
    This embedding can be used to identify the speaker in other audio files.
    """
    print("Creating voice embedding using Resemblyzer...")

    from resemblyzer import VoiceEncoder, preprocess_wav

    # Load the voice encoder
    encoder = VoiceEncoder()

    # Load and preprocess audio
    wav = preprocess_wav(audio_path)

    # Generate embedding
    embedding = encoder.embed_utterance(wav)

    # Save embedding
    profile_path = VOICE_PROFILES_DIR / 'jon_radoff_voice_profile.pkl'
    with open(profile_path, 'wb') as f:
        pickle.dump({
            'embedding': embedding,
            'name': 'Jon Radoff',
            'source_video': REFERENCE_VIDEO_URL,
            'timestamp': REFERENCE_START_TIME
        }, f)

    print(f"Voice profile saved to {profile_path}")
    print(f"Embedding shape: {embedding.shape}")

    return embedding


def main():
    print("=" * 60)
    print("Creating Voice Profile for Jon Radoff")
    print("=" * 60)
    print()

    # Step 1: Download reference video
    audio_path = download_reference_video()

    # Step 2: Extract voice segment
    segment_path = extract_voice_segment(audio_path, REFERENCE_START_TIME)

    # Step 3: Create voice embedding
    embedding = create_voice_embedding(segment_path)

    print()
    print("=" * 60)
    print("Voice profile created successfully!")
    print("=" * 60)
    print(f"Profile location: {VOICE_PROFILES_DIR / 'jon_radoff_voice_profile.pkl'}")
    print()
    print("You can now run transcribe_channel.py to transcribe your videos")


if __name__ == '__main__':
    main()
