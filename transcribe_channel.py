#!/usr/bin/env python3
"""
Main script to download all videos from a YouTube channel and transcribe them
with speaker identification.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
import yt_dlp
import whisper
import torch
import torchaudio
from pyannote.audio import Pipeline
from speechbrain.pretrained import EncoderClassifier
import numpy as np
import pickle
from datetime import timedelta
import re

from youtube_fetcher import YouTubeFetcher

# Load environment variables
load_dotenv()

# Directories
DOWNLOADS_DIR = Path('downloads')
OUTPUT_DIR = Path('output')
VOICE_PROFILES_DIR = Path('voice_profiles')

DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration
YOUTUBE_CHANNEL_URL = os.getenv('YOUTUBE_CHANNEL_URL')
HF_TOKEN = os.getenv('HF_TOKEN')  # Hugging Face token for pyannote


class VideoTranscriber:
    def __init__(self):
        print("Initializing transcription system...")

        # Load Whisper model
        print("Loading Whisper model (large-v3 for best quality)...")
        self.whisper_model = whisper.load_model("large-v3")

        # Load speaker diarization pipeline
        print("Loading speaker diarization model...")
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not set. Speaker diarization requires a Hugging Face token.")
            print("Get one at: https://huggingface.co/settings/tokens")
            print("Accept the user agreement at: https://huggingface.co/pyannote/speaker-diarization")
            self.diarization_pipeline = None
        else:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN
                )
            except Exception as e:
                print(f"Warning: Could not load diarization model: {e}")
                self.diarization_pipeline = None

        # Load speaker recognition model
        print("Loading speaker recognition model...")
        self.speaker_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/spkrec-ecapa-voxceleb"
        )

        # Load Jon's voice profile
        print("Loading voice profile...")
        self.jon_profile = self.load_voice_profile()

        print("Initialization complete!\n")

    def load_voice_profile(self):
        """Load Jon Radoff's voice profile."""
        profile_path = VOICE_PROFILES_DIR / 'jon_radoff_voice_profile.pkl'
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Voice profile not found at {profile_path}\n"
                "Please run create_voice_profile.py first"
            )

        with open(profile_path, 'rb') as f:
            profile = pickle.load(f)

        print(f"Loaded voice profile: {profile['name']}")
        return profile

    def download_video(self, video_url, video_id):
        """Download video and extract audio."""
        print(f"Downloading video: {video_url}")

        audio_path = DOWNLOADS_DIR / f"{video_id}.wav"

        # Skip if already downloaded
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
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if not audio_path.exists():
            raise FileNotFoundError(f"Failed to download: {video_url}")

        print(f"Downloaded: {audio_path}")
        return audio_path

    def transcribe_with_whisper(self, audio_path):
        """Transcribe audio using Whisper with word-level timestamps."""
        print("Transcribing with Whisper...")

        result = self.whisper_model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            word_timestamps=True,
            verbose=False
        )

        return result

    def diarize_speakers(self, audio_path):
        """
        Perform speaker diarization to identify when different speakers talk.
        Returns segments with speaker labels and timestamps.
        """
        if not self.diarization_pipeline:
            print("Skipping diarization (no pipeline available)")
            return None

        print("Performing speaker diarization...")

        diarization = self.diarization_pipeline(str(audio_path))

        # Convert to list of segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })

        print(f"Found {len(segments)} speaker segments")
        return segments

    def identify_speakers(self, audio_path, diarization_segments):
        """
        Identify which speaker segments belong to Jon Radoff vs guests.
        Returns a mapping of speaker labels to names.
        """
        if not diarization_segments:
            return {}

        print("Identifying speakers...")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Get unique speakers
        unique_speakers = list(set(seg['speaker'] for seg in diarization_segments))
        print(f"Unique speakers detected: {unique_speakers}")

        # Calculate similarity scores for each speaker
        speaker_scores = {}

        for speaker_label in unique_speakers:
            # Get segments for this speaker
            speaker_segments = [s for s in diarization_segments if s['speaker'] == speaker_label]

            # Sample up to 3 segments (to get representative voice samples)
            sampled_segments = speaker_segments[:min(3, len(speaker_segments))]

            # Extract audio for these segments and compute embeddings
            embeddings = []

            for seg in sampled_segments:
                start_frame = int(seg['start'] * sample_rate)
                end_frame = int(seg['end'] * sample_rate)

                segment_audio = waveform[:, start_frame:end_frame]

                # Skip very short segments
                if segment_audio.shape[1] < sample_rate * 0.5:  # Less than 0.5 seconds
                    continue

                # Generate embedding
                with torch.no_grad():
                    embedding = self.speaker_classifier.encode_batch(segment_audio)
                    embeddings.append(embedding.squeeze().cpu().numpy())

            if not embeddings:
                continue

            # Average embeddings for this speaker
            avg_embedding = np.mean(embeddings, axis=0)

            # Calculate cosine similarity with Jon's profile
            jon_embedding = self.jon_profile['embedding']
            similarity = np.dot(avg_embedding, jon_embedding) / (
                np.linalg.norm(avg_embedding) * np.linalg.norm(jon_embedding)
            )

            speaker_scores[speaker_label] = similarity
            print(f"  {speaker_label}: similarity = {similarity:.3f}")

        # Identify Jon (highest similarity score)
        if speaker_scores:
            jon_speaker = max(speaker_scores, key=speaker_scores.get)
            jon_score = speaker_scores[jon_speaker]

            # Only identify as Jon if similarity is above threshold
            if jon_score > 0.6:  # Typical threshold for same speaker
                print(f"Identified Jon Radoff as: {jon_speaker} (score: {jon_score:.3f})")

                # Map speakers to names
                speaker_mapping = {}
                for speaker in unique_speakers:
                    if speaker == jon_speaker:
                        speaker_mapping[speaker] = "Jon Radoff"
                    else:
                        speaker_mapping[speaker] = f"Guest"

                return speaker_mapping

        # Fallback: couldn't identify Jon
        print("Could not confidently identify Jon Radoff")
        return {speaker: f"Speaker {i+1}" for i, speaker in enumerate(unique_speakers)}

    def extract_guest_name_from_metadata(self, video_title, video_description):
        """
        Try to extract guest name from video title or description.
        Common patterns: "Interview with John Doe", "John Doe on...", etc.
        """
        # Common patterns for guest appearances
        patterns = [
            r'(?:with|featuring|ft\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:on|discusses|talks)',
            r'(?:guest|interview):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]

        text = f"{video_title} {video_description}"

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                guest_name = match.group(1)
                # Basic validation (2-4 words, all capitalized)
                words = guest_name.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words):
                    return guest_name

        return None

    def merge_transcription_with_speakers(self, whisper_result, diarization_segments, speaker_mapping, guest_name=None):
        """
        Combine Whisper transcription with speaker diarization.
        Returns timestamped text with speaker labels.
        """
        if not diarization_segments or not speaker_mapping:
            # No diarization, return plain transcript
            return [{
                'start': seg['start'],
                'end': seg['end'],
                'speaker': 'Unknown',
                'text': seg['text']
            } for seg in whisper_result['segments']]

        # Map guest names
        final_speaker_mapping = speaker_mapping.copy()
        if guest_name:
            for speaker, name in final_speaker_mapping.items():
                if name.startswith("Guest"):
                    final_speaker_mapping[speaker] = guest_name

        merged = []

        for whisper_seg in whisper_result['segments']:
            seg_start = whisper_seg['start']
            seg_end = whisper_seg['end']
            seg_mid = (seg_start + seg_end) / 2

            # Find overlapping speaker segment
            speaker = "Unknown"
            for diar_seg in diarization_segments:
                if diar_seg['start'] <= seg_mid <= diar_seg['end']:
                    speaker = final_speaker_mapping.get(diar_seg['speaker'], diar_seg['speaker'])
                    break

            merged.append({
                'start': seg_start,
                'end': seg_end,
                'speaker': speaker,
                'text': whisper_seg['text'].strip()
            })

        return merged

    def format_transcript(self, segments):
        """Format transcript with timestamps and speaker labels."""
        lines = []

        current_speaker = None
        current_text = []
        current_start = None

        for seg in segments:
            # Group consecutive segments from the same speaker
            if seg['speaker'] == current_speaker:
                current_text.append(seg['text'])
            else:
                # Write previous speaker's text
                if current_speaker and current_text:
                    timestamp = str(timedelta(seconds=int(current_start)))
                    text = ' '.join(current_text)
                    lines.append(f"[{timestamp}] {current_speaker}: {text}")

                # Start new speaker
                current_speaker = seg['speaker']
                current_text = [seg['text']]
                current_start = seg['start']

        # Write last speaker's text
        if current_speaker and current_text:
            timestamp = str(timedelta(seconds=int(current_start)))
            text = ' '.join(current_text)
            lines.append(f"[{timestamp}] {current_speaker}: {text}")

        return '\n'.join(lines)

    def transcribe_video(self, video):
        """Complete transcription pipeline for a single video."""
        video_id = video['video_id']
        print("\n" + "=" * 60)
        print(f"Processing: {video['title']}")
        print("=" * 60)

        # Check if already transcribed
        output_path = OUTPUT_DIR / f"{video_id}.txt"
        if output_path.exists():
            print(f"Transcript already exists: {output_path}")
            return output_path

        try:
            # Download video
            audio_path = self.download_video(video['url'], video_id)

            # Transcribe with Whisper
            whisper_result = self.transcribe_with_whisper(audio_path)

            # Perform speaker diarization
            diarization_segments = self.diarize_speakers(audio_path)

            # Identify speakers
            speaker_mapping = {}
            if diarization_segments:
                speaker_mapping = self.identify_speakers(audio_path, diarization_segments)

            # Extract guest name from metadata
            guest_name = self.extract_guest_name_from_metadata(
                video['title'],
                video['description']
            )
            if guest_name:
                print(f"Detected guest name from metadata: {guest_name}")

            # Merge transcription with speakers
            segments = self.merge_transcription_with_speakers(
                whisper_result,
                diarization_segments,
                speaker_mapping,
                guest_name
            )

            # Format transcript
            transcript = self.format_transcript(segments)

            # Add metadata header
            header = f"""Video: {video['title']}
URL: {video['url']}
Published: {video['published_at']}
Transcribed: {Path(__file__).parent.name}

---

"""
            full_transcript = header + transcript

            # Save transcript
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_transcript)

            print(f"Saved transcript: {output_path}")

            # Clean up audio file to save space (optional)
            # audio_path.unlink()

            return output_path

        except Exception as e:
            print(f"Error transcribing video: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    if not YOUTUBE_CHANNEL_URL:
        print("Error: YOUTUBE_CHANNEL_URL not set in .env")
        print("Please add your YouTube channel URL to the .env file")
        sys.exit(1)

    print("YouTube Channel Video Transcriber")
    print("=" * 60)
    print()

    # Fetch all videos from channel
    fetcher = YouTubeFetcher()
    videos = fetcher.get_all_videos(YOUTUBE_CHANNEL_URL)

    if not videos:
        print("No videos found")
        return

    print(f"\nFound {len(videos)} videos")
    print()

    # Initialize transcriber
    transcriber = VideoTranscriber()

    # Process each video
    successful = 0
    failed = 0

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]")
        result = transcriber.transcribe_video(video)

        if result:
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("Transcription Complete!")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")


if __name__ == '__main__':
    main()
