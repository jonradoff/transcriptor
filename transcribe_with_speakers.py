#!/usr/bin/env python3
"""
Transcribe a video with speaker identification.
Uses Whisper for transcription, pyannote for diarization, and resemblyzer for speaker matching.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import yt_dlp
import whisper
import torch
import numpy as np
import pickle
from datetime import timedelta
from pydub import AudioSegment
import re

# Load environment variables
load_dotenv()

DOWNLOADS_DIR = Path('downloads')
OUTPUT_DIR = Path('output')
VOICE_PROFILES_DIR = Path('voice_profiles')

DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

HF_TOKEN = os.getenv('HF_TOKEN')


class SpeakerTranscriber:
    def __init__(self, whisper_model="base"):
        print("Initializing transcription system...")

        # Load Whisper model
        print(f"Loading Whisper model ({whisper_model})...")
        self.whisper_model = whisper.load_model(whisper_model)

        # Load speaker diarization pipeline
        print("Loading speaker diarization model...")
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not set. Speaker diarization will be disabled.")
            self.diarization_pipeline = None
        else:
            try:
                # Suppress the torchaudio warning and load pyannote
                import warnings
                warnings.filterwarnings("ignore", message=".*torchaudio.*")
                warnings.filterwarnings("ignore", message=".*torchcodec.*")

                # Monkey-patch torchaudio to avoid the issue
                import torchaudio
                if not hasattr(torchaudio, 'list_audio_backends'):
                    torchaudio.list_audio_backends = lambda: []

                from pyannote.audio import Pipeline
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=HF_TOKEN
                )
                print("Speaker diarization model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load diarization model: {e}")
                self.diarization_pipeline = None

        # Load voice encoder for speaker matching
        print("Loading voice encoder...")
        from resemblyzer import VoiceEncoder
        self.voice_encoder = VoiceEncoder()

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

        if audio_path.exists():
            print(f"Audio already exists: {audio_path}")
            return audio_path

        # Check for cookies file
        cookies_path = Path('cookies.txt')

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(DOWNLOADS_DIR / f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': False,
            'no_warnings': False,
            # Add retries
            'retries': 3,
            'fragment_retries': 3,
            # Add sleep to avoid rate limiting
            'sleep_interval': 2,
            'max_sleep_interval': 5,
            # Add user agent to appear more like a browser
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        }

        # Use cookies file if available
        if cookies_path.exists():
            ydl_opts['cookiefile'] = str(cookies_path)

        import time

        # Try downloading with retries
        last_error = None
        for attempt in range(3):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                return audio_path
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Don't retry if video has no audio formats available
                # These are typically live streams that ended or image-only content
                if any(phrase in error_str for phrase in [
                    'requested format is not available',
                    'only images are available',
                    'no formats found',
                    'this live event will begin'
                ]):
                    print(f"Video has no audio available (probably ended livestream or image-only), skipping: {video_id}")
                    return None

                # Don't retry bot detection errors - these need cookies or breaks
                if 'sign in to confirm' in error_str or 'not a bot' in error_str:
                    print(f"Bot detection triggered for: {video_id}")
                    raise

                # Don't retry on first attempt for format errors - just skip
                if attempt == 0 and 'requested format' in error_str:
                    print(f"Format error, skipping: {video_id}")
                    return None

                # Retry other errors
                if attempt < 2:
                    wait_time = (attempt + 1) * 30  # 30s, 60s
                    print(f"Download failed (attempt {attempt + 1}/3), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        # If we get here, all retries failed
        if last_error:
            raise last_error

        return audio_path

    def transcribe_with_whisper(self, audio_path):
        """Transcribe audio using Whisper."""
        print("Transcribing with Whisper...")

        result = self.whisper_model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            verbose=False
        )

        return result

    def diarize_speakers(self, audio_path):
        """Perform speaker diarization."""
        if not self.diarization_pipeline:
            print("Skipping diarization (no pipeline available)")
            return None

        print("Performing speaker diarization...")

        # Load audio using pydub and convert to format pyannote expects
        # This works around torchcodec compatibility issues
        audio = AudioSegment.from_wav(str(audio_path))
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        waveform = torch.from_numpy(samples).unsqueeze(0)  # Add channel dimension

        # Create audio dict for pyannote
        audio_dict = {"waveform": waveform, "sample_rate": 16000}

        diarization = self.diarization_pipeline(audio_dict)

        segments = []
        # Handle different pyannote API versions
        annotation = None

        if hasattr(diarization, 'itertracks'):
            # Old API (pyannote < 4.0) - diarization is an Annotation object directly
            annotation = diarization
        elif hasattr(diarization, 'speaker_diarization'):
            # New API (pyannote 4.x) - DiarizeOutput with speaker_diarization attribute
            annotation = diarization.speaker_diarization
        elif hasattr(diarization, 'annotation'):
            # Alternative structure
            annotation = diarization.annotation

        if annotation is not None and hasattr(annotation, 'itertracks'):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
        else:
            # Debug output if we still can't parse it
            print(f"Diarization type: {type(diarization)}")
            print(f"Diarization attributes: {[x for x in dir(diarization) if not x.startswith('_')]}")
            if annotation is not None:
                print(f"Annotation type: {type(annotation)}")
                print(f"Annotation attributes: {[x for x in dir(annotation) if not x.startswith('_')]}")

        print(f"Found {len(segments)} speaker segments")
        return segments

    def identify_speakers(self, audio_path, diarization_segments):
        """Identify which speaker is Jon vs guests using voice matching."""
        if not diarization_segments:
            return {}

        print("Identifying speakers...")
        from resemblyzer import preprocess_wav

        # Load audio
        audio = AudioSegment.from_wav(str(audio_path))

        # Get unique speakers
        unique_speakers = list(set(seg['speaker'] for seg in diarization_segments))
        print(f"Unique speakers detected: {unique_speakers}")

        speaker_scores = {}

        for speaker_label in unique_speakers:
            # Get segments for this speaker
            speaker_segments = [s for s in diarization_segments if s['speaker'] == speaker_label]

            # Sample segments
            sampled_segments = speaker_segments[:min(3, len(speaker_segments))]

            embeddings = []
            for seg in sampled_segments:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)

                segment_audio = audio[start_ms:end_ms]

                # Skip very short segments
                if len(segment_audio) < 500:  # Less than 0.5 seconds
                    continue

                # Convert to mono and 16kHz
                segment_audio = segment_audio.set_channels(1).set_frame_rate(16000)

                # Save temp file and load with resemblyzer
                temp_path = DOWNLOADS_DIR / f"temp_segment_{speaker_label}.wav"
                segment_audio.export(str(temp_path), format='wav')

                try:
                    wav = preprocess_wav(temp_path)
                    embedding = self.voice_encoder.embed_utterance(wav)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"  Warning: Could not process segment: {e}")

                # Clean up
                if temp_path.exists():
                    temp_path.unlink()

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
            if jon_score > 0.7:  # Threshold for same speaker
                print(f"Identified Jon Radoff as: {jon_speaker} (score: {jon_score:.3f})")

                speaker_mapping = {}
                for speaker in unique_speakers:
                    if speaker == jon_speaker:
                        speaker_mapping[speaker] = "Jon Radoff"
                    else:
                        speaker_mapping[speaker] = "Guest"

                return speaker_mapping

        print("Could not confidently identify Jon Radoff")
        return {speaker: f"Speaker {i+1}" for i, speaker in enumerate(unique_speakers)}

    def merge_transcription_with_speakers(self, whisper_result, diarization_segments, speaker_mapping):
        """Combine Whisper transcription with speaker diarization."""
        if not diarization_segments or not speaker_mapping:
            return [{
                'start': seg['start'],
                'end': seg['end'],
                'speaker': 'Unknown',
                'text': seg['text']
            } for seg in whisper_result['segments']]

        merged = []

        for whisper_seg in whisper_result['segments']:
            seg_start = whisper_seg['start']
            seg_end = whisper_seg['end']
            seg_mid = (seg_start + seg_end) / 2

            # Find overlapping speaker segment
            speaker = "Unknown"
            for diar_seg in diarization_segments:
                if diar_seg['start'] <= seg_mid <= diar_seg['end']:
                    speaker = speaker_mapping.get(diar_seg['speaker'], diar_seg['speaker'])
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
            if seg['speaker'] == current_speaker:
                current_text.append(seg['text'])
            else:
                if current_speaker and current_text:
                    timestamp = str(timedelta(seconds=int(current_start)))
                    text = ' '.join(current_text)
                    lines.append(f"[{timestamp}] {current_speaker}: {text}")

                current_speaker = seg['speaker']
                current_text = [seg['text']]
                current_start = seg['start']

        if current_speaker and current_text:
            timestamp = str(timedelta(seconds=int(current_start)))
            text = ' '.join(current_text)
            lines.append(f"[{timestamp}] {current_speaker}: {text}")

        return '\n'.join(lines)

    def transcribe_video(self, video_url, video_title=""):
        """Complete transcription pipeline for a single video."""
        # Extract video ID
        if 'v=' in video_url:
            video_id = video_url.split('v=')[1].split('&')[0]
        else:
            video_id = video_url.split('/')[-1]

        print("=" * 60)
        print(f"Processing: {video_title or video_url}")
        print("=" * 60)

        output_path = OUTPUT_DIR / f"{video_id}.txt"

        try:
            # Download video
            audio_path = self.download_video(video_url, video_id)

            # Skip if video has no audio
            if audio_path is None:
                print(f"Skipping video (no audio available): {video_id}")
                return None

            # Transcribe with Whisper
            whisper_result = self.transcribe_with_whisper(audio_path)

            # Perform speaker diarization
            diarization_segments = self.diarize_speakers(audio_path)

            # Identify speakers
            speaker_mapping = {}
            if diarization_segments:
                speaker_mapping = self.identify_speakers(audio_path, diarization_segments)

            # Merge transcription with speakers
            segments = self.merge_transcription_with_speakers(
                whisper_result,
                diarization_segments,
                speaker_mapping
            )

            # Format transcript
            transcript = self.format_transcript(segments)

            # Add metadata header
            header = f"""Video: {video_title or video_url}
URL: {video_url}
Video ID: {video_id}

---

"""
            full_transcript = header + transcript

            # Save transcript
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_transcript)

            print(f"\nSaved transcript: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error transcribing video: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_with_speakers.py <video_url> [model_size]")
        print("Example: python transcribe_with_speakers.py https://www.youtube.com/watch?v=abc123 base")
        print("\nModel sizes: tiny, base, small, medium, large, large-v3")
        sys.exit(1)

    video_url = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"

    transcriber = SpeakerTranscriber(whisper_model=model_size)
    result = transcriber.transcribe_video(video_url)

    if result:
        print("\n" + "=" * 60)
        print("Transcription complete!")
        print("=" * 60)

        # Show first 1500 chars
        with open(result, 'r') as f:
            content = f.read()
        print("\nFirst 1500 characters:")
        print("-" * 60)
        print(content[:1500])


if __name__ == '__main__':
    main()
