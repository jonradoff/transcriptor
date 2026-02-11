#!/usr/bin/env python3
"""
Convert transcripts to clean HTML files with:
- YouTube embed
- Title and description
- Fixed speaker names
- SEO-optimized filenames
"""

import os
import re
import json
import time
from pathlib import Path
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directories
TRANSCRIPT_DIR = Path(__file__).parent / "output"
HTML_DIR = Path(__file__).parent / "html"
VIDEO_LIST_PATH = Path(__file__).parent / "video_list.json"

# Create HTML directory
HTML_DIR.mkdir(exist_ok=True)

# YouTube API setup
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Name corrections - various misspellings to correct name
JON_RADOFF_VARIANTS = [
    "john radoff", "jon raidoff", "john raidoff", "jon radov", "john radov",
    "jon raydoff", "john raydoff", "jonradoff", "johnradoff", "j radoff",
    "j raidoff", "jon radioff", "john radioff", "jon radolph", "john radolph"
]

def load_video_list():
    """Load video metadata from JSON file."""
    with open(VIDEO_LIST_PATH, 'r') as f:
        videos = json.load(f)
    return {v['id']: v for v in videos}

def fetch_video_description(video_id, youtube):
    """Fetch video description from YouTube API."""
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        if response['items']:
            return response['items'][0]['snippet'].get('description', '')
    except Exception as e:
        print(f"  Error fetching description for {video_id}: {e}")
    return ""

def extract_guest_from_title(title):
    """Extract guest name from video title patterns."""
    # Common patterns:
    # "Topic with Guest Name"
    # "Topic | Guest Name | Company"
    # "Guest Name talks about Topic"
    # "Guest Name and Jon Radoff..."
    # "Topic - Guest Name"

    title_lower = title.lower()

    # Pattern: "Name and Jon Radoff" or "Jon Radoff and Name"
    match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+and\s+Jon\s+Radoff', title, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r'Jon\s+Radoff\s+and\s+([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', title, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern: "Topic | Guest Name | Company" or "Topic | Guest Name, Company"
    if '|' in title:
        parts = [p.strip() for p in title.split('|')]
        for part in parts[1:]:  # Skip first part (usually topic)
            # Check if it looks like a name (2-3 capitalized words, not a company)
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', part):
                return part
            # Check for "Name, Company" pattern
            name_match = re.match(r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),', part)
            if name_match:
                return name_match.group(1)

    # Pattern: "Topic - Guest Name" or "Topic - Guest Name - Company"
    if ' - ' in title:
        parts = [p.strip() for p in title.split(' - ')]
        for part in parts[1:]:  # Skip first part (usually topic)
            # Check if it looks like a name
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', part):
                if 'radoff' not in part.lower():
                    return part

    # Pattern: "Topic with Guest Name" - remove leading "with"
    match = re.search(r'with\s+([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', title)
    if match:
        guest = match.group(1)
        if 'radoff' not in guest.lower():
            return guest

    # Pattern: "Guest Name talks about" or "Guest Name discusses"
    match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:talks?|discusses?|on|about)', title)
    if match:
        guest = match.group(1)
        if 'radoff' not in guest.lower():
            return guest

    return None

def extract_guest_from_description(description, title):
    """Extract guest name from video description."""
    if not description:
        return None

    # Look for common patterns in descriptions
    # "In this episode, Jon Radoff talks with Guest Name"
    # "Guest: Name"
    # "Featuring Name"

    # Pattern: "with Guest Name" (not followed by common words)
    match = re.search(r'(?:talks?|speaks?|chats?|discusses?)\s+with\s+([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', description)
    if match:
        guest = match.group(1)
        if 'radoff' not in guest.lower():
            return guest

    # Pattern: "Guest: Name" or "Featuring: Name"
    match = re.search(r'(?:guest|featuring|with):\s*([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', description, re.IGNORECASE)
    if match:
        return match.group(1)

    return None

def fix_jon_radoff_name(text):
    """Fix all variants of Jon Radoff's name."""
    result = text
    for variant in JON_RADOFF_VARIANTS:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(variant), re.IGNORECASE)
        result = pattern.sub("Jon Radoff", result)
    return result

def identify_speakers(transcript_lines, title, description, guest_name):
    """
    Identify and replace speaker labels with actual names.
    Returns the transcript with corrected speaker names.
    """
    # Build speaker mapping
    speaker_map = {}

    # Look for speaker identification in transcript
    # Sometimes the transcript itself mentions names
    full_text = '\n'.join(transcript_lines)

    # Find which speaker numbers are used
    speakers_used = set(re.findall(r'Speaker (\d+)', full_text))

    # If we have a guest name, try to figure out who is who
    # Usually Jon Radoff is Speaker 2 (host) and guest is Speaker 1
    # But sometimes there's a narrator (Speaker 3)

    # Check if Jon Radoff is already identified somewhere
    jon_speaker = None
    guest_speaker = None

    for line in transcript_lines:
        lower_line = line.lower()
        if 'jon radoff' in lower_line or any(v in lower_line for v in JON_RADOFF_VARIANTS):
            # Check if this line identifies a speaker
            match = re.search(r'\[[\d:]+\]\s*(Speaker \d+|[^:]+):', line)
            if match:
                speaker = match.group(1)
                if 'speaker' in speaker.lower():
                    speaker_num = re.search(r'Speaker (\d+)', speaker)
                    if speaker_num:
                        jon_speaker = speaker_num.group(1)

    # If guest name found in a speaker label, map it
    if guest_name:
        for line in transcript_lines:
            if guest_name.lower() in line.lower():
                match = re.search(r'\[[\d:]+\]\s*(Speaker \d+):', line)
                if match:
                    guest_speaker = re.search(r'Speaker (\d+)', match.group(1)).group(1)

    # Default mapping if we couldn't identify
    # In interview format: usually alternating speakers, host often Speaker 2
    if not jon_speaker and not guest_speaker:
        # If title mentions guest name first, they're likely Speaker 1
        if guest_name and guest_name.lower() in title.lower()[:50]:
            guest_speaker = "1"
            jon_speaker = "2"
        else:
            jon_speaker = "2"
            guest_speaker = "1"
    elif jon_speaker and not guest_speaker:
        # Assign guest to other main speaker
        for s in speakers_used:
            if s != jon_speaker and s != "3":  # 3 is often narrator
                guest_speaker = s
                break
    elif guest_speaker and not jon_speaker:
        for s in speakers_used:
            if s != guest_speaker and s != "3":
                jon_speaker = s
                break

    # Build final speaker map
    if jon_speaker:
        speaker_map[f"Speaker {jon_speaker}"] = "Jon Radoff"
    if guest_speaker and guest_name:
        speaker_map[f"Speaker {guest_speaker}"] = guest_name

    # Speaker 3 is often narrator/intro - keep as is or label
    if "3" in speakers_used:
        speaker_map["Speaker 3"] = "Narrator"

    # "Unknown" should be replaced with best guess
    if guest_name:
        speaker_map["Unknown"] = guest_name

    return speaker_map

def create_seo_slug(title, guest_name):
    """Create SEO-optimized filename slug from title."""
    # Start with the title
    slug = title.lower()

    # Remove common filler words for SEO
    filler_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']

    # Extract key elements
    # Keep guest name, key topics

    # Replace special characters
    slug = re.sub(r'[|:,\-\(\)\[\]\"\']+', ' ', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    slug = re.sub(r'-+', '-', slug)
    slug = re.sub(r'[^a-z0-9\-]', '', slug)
    slug = slug.strip('-')

    # Limit length but keep meaningful
    if len(slug) > 80:
        # Try to cut at a word boundary
        slug = slug[:80]
        last_dash = slug.rfind('-')
        if last_dash > 50:
            slug = slug[:last_dash]

    return slug

def format_transcript_html(lines, speaker_map, guest_name):
    """Format transcript lines as HTML with paragraph breaks and bold speaker names."""
    html_parts = []

    for line in lines:
        if not line.strip():
            continue

        # Skip header lines (Video:, URL:, Video ID:, ---)
        if line.startswith('Video:') or line.startswith('URL:') or line.startswith('Video ID:') or line.strip() == '---':
            continue

        # Parse timestamp and speaker
        match = re.match(r'\[(\d+:\d+:\d+)\]\s*([^:]+):\s*(.*)', line)
        if match:
            timestamp, speaker, text = match.groups()

            # Fix Jon Radoff name variants in speaker label
            speaker = fix_jon_radoff_name(speaker)

            # Apply speaker mapping
            original_speaker = speaker.strip()
            if original_speaker in speaker_map:
                speaker = speaker_map[original_speaker]

            # Fix Jon Radoff name variants in text
            text = fix_jon_radoff_name(text)

            # If speaker is still "Unknown" and we have a guest, use guest name
            if speaker.strip() == "Unknown" and guest_name:
                speaker = guest_name

            # Format as HTML paragraph
            html_parts.append(f'<p><strong>{speaker}:</strong> {text}</p>')
        else:
            # Line without expected format - just add as paragraph
            text = fix_jon_radoff_name(line)
            html_parts.append(f'<p>{text}</p>')

    return '\n'.join(html_parts)

def create_html_content(video_id, title, description, transcript_html):
    """Create the full HTML content for a transcript."""
    # Escape HTML in title and description
    safe_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    safe_description = description.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # Convert description newlines to <br> tags
    safe_description = safe_description.replace('\n', '<br>\n')

    html = f'''<div class="video-embed">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}"
            title="{safe_title}" frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowfullscreen></iframe>
</div>

<h1>{safe_title}</h1>

<div class="video-description">
    <p>{safe_description}</p>
</div>

<hr>

<div class="transcript">
{transcript_html}
</div>
'''
    return html

def process_transcript(transcript_path, video_metadata, youtube):
    """Process a single transcript file."""
    video_id = transcript_path.stem

    print(f"Processing: {video_id}")

    # Read transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    # Get title from transcript or metadata
    title = ""
    if lines and lines[0].startswith('Video:'):
        title = lines[0].replace('Video:', '').strip()
    elif video_id in video_metadata:
        title = video_metadata[video_id].get('title', '')

    if not title:
        print(f"  Warning: No title found for {video_id}")
        title = f"Video {video_id}"

    # Fetch description from YouTube
    description = ""
    if youtube:
        description = fetch_video_description(video_id, youtube)
        time.sleep(0.1)  # Rate limiting

    # Extract guest name from title and description
    guest_name = extract_guest_from_title(title)
    if not guest_name:
        guest_name = extract_guest_from_description(description, title)

    # Clean up guest name - remove leading "with" if present
    if guest_name and guest_name.lower().startswith('with '):
        guest_name = guest_name[5:]

    print(f"  Title: {title[:60]}...")
    print(f"  Guest: {guest_name or 'Unknown'}")

    # Identify speakers
    speaker_map = identify_speakers(lines, title, description, guest_name)
    print(f"  Speaker map: {speaker_map}")

    # Format transcript as HTML
    transcript_html = format_transcript_html(lines, speaker_map, guest_name)

    # Create full HTML
    html_content = create_html_content(video_id, title, description, transcript_html)

    # Create SEO slug for filename
    slug = create_seo_slug(title, guest_name)

    # Include video_id in slug to ensure uniqueness
    base_slug = slug
    if base_slug in used_slugs:
        # Add video ID suffix to make unique
        slug = f"{base_slug}-{video_id[:8]}"

    used_slugs.add(slug)

    # Write HTML file
    output_path = HTML_DIR / f"{slug}.html"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  Output: {output_path.name}")

    return output_path

def main():
    # Load video metadata
    video_metadata = load_video_list()
    print(f"Loaded {len(video_metadata)} videos from metadata")

    # Initialize YouTube API
    youtube = None
    if YOUTUBE_API_KEY:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        print("YouTube API initialized")
    else:
        print("Warning: No YouTube API key found, descriptions will be empty")

    # Get all transcript files, excluding _simple versions (prefer full transcripts)
    all_files = list(TRANSCRIPT_DIR.glob("*.txt"))
    # Filter out _simple files and URLs-as-filenames
    transcript_files = []
    seen_ids = set()
    for f in all_files:
        stem = f.stem
        # Skip _simple versions
        if '_simple' in stem:
            continue
        # Skip files that look like URLs
        if stem.startswith('https-') or stem.startswith('http-'):
            continue
        # Avoid duplicates
        if stem in seen_ids:
            continue
        seen_ids.add(stem)
        transcript_files.append(f)

    print(f"Found {len(transcript_files)} transcript files (after filtering)")

    # Track slugs we've already used to avoid duplicate filenames
    global used_slugs
    used_slugs = set()

    # Process each transcript
    processed = 0
    errors = 0

    for transcript_path in transcript_files:
        try:
            process_transcript(transcript_path, video_metadata, youtube)
            processed += 1
        except Exception as e:
            print(f"  Error: {e}")
            errors += 1

    print(f"\nComplete! Processed {processed} files, {errors} errors")
    print(f"HTML files saved to: {HTML_DIR}")

if __name__ == "__main__":
    main()
