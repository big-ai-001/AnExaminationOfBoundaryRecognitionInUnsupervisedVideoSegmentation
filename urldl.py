from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from pytube import YouTube

def dl(url):
    formatter = SRTFormatter()
    transcript = YouTubeTranscriptApi.get_transcript(YouTube(url).video_id)
    transcript = formatter.format_transcript(transcript)
    with open('./datapool/tmp.srt', 'w', encoding='utf-8') as f:
        f.write(transcript)