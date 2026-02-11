# Usage Guide

## Quick Start

### 1. Setup

```bash
# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` file:

```bash
# Required
YOUTUBE_API_KEY=your_key_here          # Already set
YOUTUBE_CHANNEL_URL=your_channel_url   # Add your channel
HF_TOKEN=your_huggingface_token        # Required for speaker diarization

# Optional
OPENAI_API_KEY=your_openai_key         # Only if using Whisper API (not needed for local)
```

**Important:** Get a Hugging Face token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token
3. Accept the user agreement at https://huggingface.co/pyannote/speaker-diarization

### 3. Create Voice Profile

This analyzes your voice from the reference video so the system can identify you vs guests:

```bash
source venv/bin/activate
python create_voice_profile.py
```

This will:
- Download the reference video (https://www.youtube.com/watch?v=HN_hOuyXUkc)
- Extract your voice from 0:29 onward
- Create a voice embedding/profile
- Save it to `voice_profiles/jon_radoff_voice_profile.pkl`

### 4. Test with a Single Video (Recommended)

Before processing all videos, test with one:

```bash
python test_single_video.py https://www.youtube.com/watch?v=VIDEO_ID
```

Check the output in `output/VIDEO_ID.txt`

### 5. Transcribe All Channel Videos

```bash
python transcribe_channel.py
```

This will:
1. Fetch all videos from your YouTube channel
2. For each video:
   - Download the video
   - Extract audio
   - Transcribe with Whisper (state-of-the-art accuracy)
   - Perform speaker diarization (detect when different people speak)
   - Identify which speaker is you vs guests
   - Try to extract guest names from video titles/descriptions
   - Generate formatted transcript with timestamps and speaker labels
3. Save transcripts to `output/` directory

## Output Format

Transcripts are saved as `output/VIDEO_ID.txt`:

```
Video: My Amazing Interview with Jane Doe
URL: https://www.youtube.com/watch?v=abc123
Published: 2024-01-15
Transcribed: transcriptor

---

[0:00:00] Jon Radoff: Welcome to the show everyone.
[0:00:15] Jon Radoff: Today we have a very special guest.
[0:00:29] Jane Doe: Thanks for having me!
[0:00:35] Jon Radoff: Let's dive right in...
```

## Technical Details

### Speaker Identification

The system uses multiple state-of-the-art models:

1. **Whisper (OpenAI)**: Speech-to-text transcription
   - Using `large-v3` model for maximum accuracy
   - Generates word-level timestamps

2. **pyannote.audio**: Speaker diarization
   - Detects when different speakers talk
   - Creates speaker segments with timestamps

3. **SpeechBrain ECAPA-TDNN**: Speaker verification
   - Generates voice embeddings
   - Compares speakers against your voice profile
   - Identifies "Jon Radoff" vs "Guest" speakers

4. **Guest Name Detection**:
   - Analyzes video titles and descriptions
   - Looks for patterns like "with [Name]", "[Name] on...", etc.
   - Automatically labels guests when detected

### Models Downloaded

First run will download several models (~5GB total):
- Whisper large-v3 (~3GB)
- pyannote diarization (~500MB)
- SpeechBrain speaker recognition (~500MB)

## Troubleshooting

### "HF_TOKEN not set"
- Get token at https://huggingface.co/settings/tokens
- Accept agreement at https://huggingface.co/pyannote/speaker-diarization
- Add to `.env` file

### "Voice profile not found"
- Run `python create_voice_profile.py` first

### Out of Memory
- Reduce Whisper model size in `transcribe_channel.py`:
  - Change `load_model("large-v3")` to `load_model("base")` or `load_model("medium")`
  - Accuracy will be slightly lower but memory usage much less

### Speaker identification not working well
- Ensure reference video has clear audio of your voice
- Try adjusting `REFERENCE_START_TIME` in `.env` to a section with clearer speech
- Re-run `create_voice_profile.py`

### Slow transcription
- Whisper large-v3 is very accurate but slow
- For faster processing, use a smaller model (medium, small, or base)
- Consider using a GPU if available (automatic with PyTorch + CUDA)

## File Structure

```
transcriptor/
├── .env                          # Configuration
├── requirements.txt              # Python dependencies
├── setup.sh                      # Setup script
├── create_voice_profile.py       # Step 1: Create voice profile
├── youtube_fetcher.py            # Fetch videos from channel
├── transcribe_channel.py         # Main transcription script
├── test_single_video.py          # Test single video
├── downloads/                    # Downloaded videos (can delete to save space)
├── output/                       # Transcription output files
└── voice_profiles/              # Your voice profile
    └── jon_radoff_voice_profile.pkl
```

## Tips

1. **Save Space**: Delete files in `downloads/` after transcription to save disk space
2. **Batch Processing**: The script processes videos sequentially. For faster processing with multiple GPUs, you could modify it to process in parallel
3. **Re-transcribe**: Delete the `.txt` file in `output/` to re-transcribe a specific video
4. **Custom Voice Segments**: Edit `.env` to use different reference video or timestamp for voice profiling
