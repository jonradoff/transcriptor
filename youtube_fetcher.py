#!/usr/bin/env python3
"""
Fetch all videos from a YouTube channel using the YouTube Data API v3.
"""

import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')


class YouTubeFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError("YouTube API key not found. Set YOUTUBE_API_KEY in .env")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def get_channel_id_from_url(self, channel_url):
        """
        Extract channel ID from various YouTube channel URL formats:
        - https://www.youtube.com/channel/UC...
        - https://www.youtube.com/@username
        - https://www.youtube.com/c/channelname
        - https://www.youtube.com/user/username
        """
        channel_url = channel_url.strip()

        # Direct channel ID
        if '/channel/' in channel_url:
            return channel_url.split('/channel/')[-1].split('?')[0].split('/')[0]

        # Handle @username format
        if '/@' in channel_url:
            username = channel_url.split('/@')[-1].split('?')[0].split('/')[0]
            return self.get_channel_id_from_handle(username)

        # Handle /c/ or /user/ format
        if '/c/' in channel_url or '/user/' in channel_url:
            username = channel_url.split('/')[-1].split('?')[0]
            return self.get_channel_id_from_username(username)

        raise ValueError(f"Could not parse channel URL: {channel_url}")

    def get_channel_id_from_handle(self, handle):
        """Get channel ID from @handle."""
        try:
            request = self.youtube.search().list(
                part='snippet',
                q=handle,
                type='channel',
                maxResults=1
            )
            response = request.execute()

            if response['items']:
                return response['items'][0]['snippet']['channelId']
            else:
                raise ValueError(f"Channel not found for handle: @{handle}")
        except HttpError as e:
            raise Exception(f"YouTube API error: {e}")

    def get_channel_id_from_username(self, username):
        """Get channel ID from username or custom URL."""
        try:
            request = self.youtube.channels().list(
                part='id',
                forUsername=username
            )
            response = request.execute()

            if response['items']:
                return response['items'][0]['id']
            else:
                # Try searching
                return self.get_channel_id_from_handle(username)
        except HttpError as e:
            raise Exception(f"YouTube API error: {e}")

    def get_channel_info(self, channel_id):
        """Get channel information including uploads playlist ID."""
        try:
            request = self.youtube.channels().list(
                part='contentDetails,snippet',
                id=channel_id
            )
            response = request.execute()

            if not response['items']:
                raise ValueError(f"Channel not found: {channel_id}")

            channel = response['items'][0]
            uploads_playlist_id = channel['contentDetails']['relatedPlaylists']['uploads']
            channel_title = channel['snippet']['title']

            return {
                'channel_id': channel_id,
                'channel_title': channel_title,
                'uploads_playlist_id': uploads_playlist_id
            }
        except HttpError as e:
            raise Exception(f"YouTube API error: {e}")

    def get_all_videos(self, channel_url):
        """
        Get all videos from a channel.
        Returns a list of video dictionaries with id, title, description, publishedAt.
        """
        print(f"Fetching channel information from: {channel_url}")

        # Get channel ID
        channel_id = self.get_channel_id_from_url(channel_url)
        print(f"Channel ID: {channel_id}")

        # Get channel info
        channel_info = self.get_channel_info(channel_id)
        print(f"Channel: {channel_info['channel_title']}")
        print(f"Uploads playlist: {channel_info['uploads_playlist_id']}")

        # Fetch all videos from uploads playlist
        videos = []
        next_page_token = None

        while True:
            try:
                request = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=channel_info['uploads_playlist_id'],
                    maxResults=50,
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response['items']:
                    video = {
                        'video_id': item['snippet']['resourceId']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'published_at': item['snippet']['publishedAt'],
                        'thumbnail': item['snippet']['thumbnails']['high']['url'],
                        'url': f"https://www.youtube.com/watch?v={item['snippet']['resourceId']['videoId']}"
                    }
                    videos.append(video)

                next_page_token = response.get('nextPageToken')

                if not next_page_token:
                    break

                print(f"Fetched {len(videos)} videos so far...")

            except HttpError as e:
                print(f"Error fetching videos: {e}")
                break

        print(f"Total videos found: {len(videos)}")
        return videos


def main():
    """Test the fetcher."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python youtube_fetcher.py <channel_url>")
        print("Example: python youtube_fetcher.py https://www.youtube.com/@username")
        sys.exit(1)

    channel_url = sys.argv[1]
    fetcher = YouTubeFetcher()
    videos = fetcher.get_all_videos(channel_url)

    print("\nVideos:")
    for i, video in enumerate(videos[:10], 1):
        print(f"{i}. {video['title']}")
        print(f"   URL: {video['url']}")
        print(f"   Published: {video['published_at']}")
        print()

    if len(videos) > 10:
        print(f"... and {len(videos) - 10} more videos")


if __name__ == '__main__':
    main()
