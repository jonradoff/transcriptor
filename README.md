# YouTube Channel Transcription Pipeline

A complete pipeline for transcribing an entire YouTube channel with speaker diarization (identifying who said what), then converting the transcripts into clean, categorized HTML pages.

## What This Does

1. **Downloads audio** from all videos on a YouTube channel
2. **Transcribes speech to text** using OpenAI's Whisper model (locally)
3. **Identifies speakers** using voice diarization (pyannote) and voice matching (Resemblyzer)
4. **Fixes speaker names** and extracts guest information from video metadata
5. **Generates clean HTML** with YouTube embeds, descriptions, and formatted transcripts
6. **Creates a categorized index** organizing all videos by topic

## Requirements

### Software
- Python 3.10+
- ffmpeg (for audio processing)
- CUDA-compatible GPU recommended (CPU works but is ~20x slower)

### Python Packages
```bash
pip install openai-whisper pyannote.audio resemblyzer yt-dlp \
    python-dotenv google-api-python-client torch torchaudio
```

### API Keys
- **Hugging Face Token**: Required for pyannote speaker diarization model
  - Get one at: https://huggingface.co/settings/tokens
  - Accept the model agreement at: https://huggingface.co/pyannote/speaker-diarization
- **YouTube Data API Key**: Required for fetching video descriptions
  - Get one from Google Cloud Console

## Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/transcriptor.git
cd transcriptor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure your environment**:
```bash
cp .env.example .env
```
Then edit `.env` with your API keys:
- **HF_TOKEN**: Get from https://huggingface.co/settings/tokens (required)
- **YOUTUBE_API_KEY**: Get from Google Cloud Console (required)
- **YOUTUBE_CHANNEL_URL**: Your channel to transcribe (required)
- **REFERENCE_VIDEO_URL**: A video where you're speaking (optional, for voice ID)

4. **Create a voice profile** for the host (optional but improves accuracy):
```bash
python create_voice_profile.py
```
This extracts voice embeddings from a reference video where the host is speaking.

## How It Works

### Step 1: Build Video List

The system uses yt-dlp to enumerate all videos on the channel:

```python
ydl_opts = {
    'extract_flat': True,
    'quiet': True,
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    channel_info = ydl.extract_info(channel_url, download=False)
```

This creates `video_list.json` with video IDs, titles, and durations.

### Step 2: Download and Transcribe

For each video:

1. **Download audio** as WAV using yt-dlp with rate limiting to avoid bot detection:
```python
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
    'cookiefile': 'cookies.txt',  # Optional: helps avoid rate limiting
    'sleep_interval': 2,
    'http_headers': {'User-Agent': '...'}
}
```

2. **Transcribe with Whisper**:
```python
model = whisper.load_model("large-v2")
result = model.transcribe(audio_path, language="en")
```

3. **Diarize speakers** with pyannote:
```python
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
diarization = pipeline(audio_path)
```

4. **Match speakers to known voices** using Resemblyzer embeddings:
```python
encoder = VoiceEncoder()
embedding = encoder.embed_utterance(audio_segment)
similarity = np.dot(embedding, host_profile)
```

### Step 3: Generate HTML

The `create_html_transcripts.py` script:

1. **Reads each transcript** from the `output/` directory
2. **Fetches video description** from YouTube API
3. **Extracts guest names** from title patterns like:
   - "Topic | Guest Name | Company"
   - "Topic with Guest Name"
   - "Guest Name and Jon Radoff discuss..."
4. **Fixes name misspellings** (e.g., "John Raidoff" → "Jon Radoff")
5. **Generates SEO-friendly filenames** from titles
6. **Creates HTML** with:
   - YouTube video embed
   - Title and description
   - Transcript with bold speaker names and paragraph breaks

### Step 4: Create Index

The index.html organizes videos into categories based on content analysis:
- AI & Generative Technology
- Game Development
- Metaverse & Virtual Worlds
- Web3, Blockchain & NFTs
- DePIN & Decentralized Infrastructure
- Creator Economy & Digital Identity

## Usage

### Full Pipeline (GPU recommended)
```bash
# 1. Create voice profile for the host
python create_voice_profile.py

# 2. Transcribe all videos
python transcribe_channel_gpu.py

# 3. Generate HTML files
python create_html_transcripts.py
```

### CPU-Only (slower, ~1.5-2 hours per video)
```bash
python transcribe_channel_batch.py
```

### Generate HTML Only (if transcripts exist)
```bash
python create_html_transcripts.py
```

## Output Structure

```
transcriptor/
├── .env                          # API keys and config
├── video_list.json               # Channel video metadata
├── voice_profiles/
│   └── host_voice_profile.pkl    # Voice embedding for speaker matching
├── downloads/                    # Downloaded audio files (WAV)
├── output/                       # Raw transcripts (TXT)
│   ├── VIDEO_ID.txt
│   └── ...
└── html/                         # Final HTML output
    ├── index.html                # Categorized index page
    ├── video-title-slug.html     # Individual transcript pages
    └── ...
```

## Performance

| Hardware | Time per Video | Notes |
|----------|---------------|-------|
| NVIDIA L4 GPU | ~5-6 minutes | Recommended |
| NVIDIA T4 GPU | ~8-10 minutes | Good balance of cost/speed |
| CPU (M1 Mac) | ~90-120 minutes | Works but slow |

## Handling YouTube Rate Limiting

YouTube may block downloads after many requests. Mitigations:

1. **Rate limiting**: 10-second delays between videos
2. **User agent spoofing**: Appear as a normal browser
3. **Cookie authentication**: Export cookies from your browser
4. **Retry logic**: Exponential backoff on failures

```bash
# Extract cookies from Chrome
yt-dlp --cookies-from-browser chrome --cookies cookies.txt ...
```

## Customization

### Adding New Name Corrections
Edit the `JON_RADOFF_VARIANTS` list in `create_html_transcripts.py`:
```python
JON_RADOFF_VARIANTS = [
    "john radoff", "jon raidoff", "john raidoff", ...
]
```

### Changing Categories
Edit the category sections in `index.html` or create a script to auto-categorize based on keywords in titles/descriptions.

### Different Whisper Models
Available models (speed vs accuracy tradeoff):
- `tiny` - Fastest, least accurate
- `base` - Fast, decent accuracy
- `small` - Balanced
- `medium` - Good accuracy
- `large-v2` - Best accuracy (used here)

## Troubleshooting

### "Sign in to confirm you're not a bot"
- Add longer delays between downloads
- Use browser cookies: `--cookies-from-browser chrome`
- Try at different times of day

### PyTorch/pyannote version conflicts
Ensure compatible versions:
```bash
pip install torch==2.10.0 torchaudio==2.10.0 pyannote.audio==4.0.4
```

### Out of GPU memory
- Use a smaller Whisper model (`medium` instead of `large-v2`)
- Process shorter audio segments
- Use CPU for diarization, GPU for transcription

## Technical Stack

- **yt-dlp**: YouTube video downloading
- **OpenAI Whisper**: State-of-the-art speech-to-text transcription (local)
- **pyannote.audio**: Speaker diarization (who speaks when)
- **Resemblyzer**: Speaker recognition (voice matching)
- **YouTube Data API v3**: Channel video listing and descriptions
- **google-api-python-client**: YouTube API access

## License

MIT License - Copyright (c) 2025 Metavert LLC

See [LICENSE](LICENSE) for full details.

### Dependencies

This pipeline uses the following open source libraries:
- [OpenAI Whisper](https://github.com/openai/whisper) - MIT License
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - MIT License
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Apache 2.0
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Unlicense

## How to Create This from Scratch with Claude Code

This entire pipeline was built using [Claude Code](https://claude.ai/claude-code), Anthropic's AI coding assistant. Here's how you can replicate it for your own YouTube channel without writing any code yourself.

### Prerequisites
1. Install Claude Code CLI
2. Have a YouTube channel URL ready
3. Get API keys (Hugging Face, YouTube Data API)

### Prompt Sequence

**Prompt 1: Set up the project**
```
I want to transcribe all the videos on my YouTube channel. The channel is at
https://www.youtube.com/@MyChannelName. Create a Python project that:
- Downloads all videos from the channel
- Transcribes them using Whisper (locally, not API)
- Identifies different speakers (I'm the host, guests vary per video)
- Outputs timestamped transcripts with speaker labels

I have a Hugging Face token for pyannote speaker diarization.
```

**Prompt 2: Create voice profile for speaker identification**
```
Create a voice profile for me so the system can identify when I'm speaking vs guests.
Use this video as a reference where I'm the main speaker:
https://www.youtube.com/watch?v=VIDEO_ID (I start speaking at 0:29)
```

**Prompt 3: Run batch transcription**
```
Now transcribe all the videos on my channel. Start with a few to test, then do the rest.
Skip any videos that have already been transcribed.
```

**Prompt 4: Handle rate limiting (if needed)**
```
YouTube is blocking downloads with "Sign in to confirm you're not a bot".
Can you fix this? We should use the YouTube API or cookies to avoid detection.
```

**Prompt 5: Speed up with GPU (optional)**
```
The transcription is too slow on CPU. I have access to a cloud GPU
(GCP/AWS with NVIDIA T4 or L4). Can we run the transcription there instead?
```

**Prompt 6: Generate HTML output**
```
Create an html subdirectory with clean HTML versions of these transcripts:
- Fix any misspellings of my name (correct: Your Name)
- Identify guests from video titles and descriptions
- Add YouTube embed at top of each page
- Put each speaker's name in bold
- Create SEO-friendly filenames from titles
- Add the video description below the embed
```

**Prompt 7: Create categorized index**
```
Create an index.html that organizes all videos into logical categories
based on their content. Analyze the titles and descriptions to determine
appropriate categories.
```

### Tips for Success

1. **Be specific about your channel** - Give Claude the exact URL and any naming conventions you use in titles

2. **Provide reference content** - Point to a video where you're clearly the speaker for voice profiling

3. **Iterate on problems** - When you hit issues (rate limiting, GPU memory, etc.), just describe the error and Claude will fix it

4. **Let Claude choose tools** - Don't prescribe specific libraries; describe what you want and let Claude pick appropriate tools

5. **Review and refine** - Check the output after each step. If speaker identification is wrong or names are misspelled, tell Claude specifically what to fix

### Example Customization Prompts

**For different name corrections:**
```
In the transcripts, my name appears as "John Smith" and "Jon Smyth" sometimes.
The correct spelling is "John Smith". Fix all variations.
```

**For custom categories:**
```
I want to categorize my videos differently. My channel covers:
- Product tutorials
- Customer interviews
- Industry news
- Behind the scenes

Re-organize the index.html with these categories.
```

**For different output format:**
```
Instead of HTML, I want the transcripts as:
- Markdown files
- With YAML frontmatter containing title, date, guest name
- Suitable for a Jekyll or Hugo static site
```

**For subtitle generation:**
```
Generate SRT subtitle files from these transcripts that I can upload back to YouTube.
```

## Publishing to a CMS with LightCMS

Once you have your HTML transcripts generated, you can easily import them into a website using [LightCMS](https://github.com/jonradoff/lightcms), an AI-native content management system designed for agentic workflows.

LightCMS provides a Model Context Protocol (MCP) server with 38+ tools for managing your entire website through Claude Code or other AI assistants. Instead of manually uploading files through an admin interface, you can simply describe what you want and let the AI handle the import.

### Example: Bulk Import Transcripts

With the LightCMS MCP server connected to Claude Code, use this prompt to import all your transcripts:

```
On my website I want you to import all of the files from /Users/yourname/transcriptor/html
into the "videos" folder with one content page per html file. Preserve the original filename
as the slug. We will use the Standard Page template, where the ingested file comprises the
content field. Make the title of each page match the title of the video mentioned inside
each page.

IMPORTANT: it's OK to inspect these files using an LLM for things like the description.
But do NOT parse the content through an LLM to generate the page; we need to preserve the
exact original content of each html page into the content field of each content item in
the content management system.
```

Claude will then:
1. Read each HTML file from your `html/` directory
2. Extract the video title from the page content
3. Create a content page in LightCMS with the correct slug
4. Preserve the exact HTML content in the content field

This agentic approach makes the "last mile" of publishing incredibly simple—no manual copying, no form filling, just describe what you want and it happens.

## Credits

Copyright (c) 2025 Metavert LLC. Prompted by Jon Radoff with assistance from Claude (Anthropic).
