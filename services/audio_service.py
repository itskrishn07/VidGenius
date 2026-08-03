import os
import tempfile
from typing import List
import yt_dlp
from pydub import AudioSegment
import config


class AudioService:
    """Service handling media acquisition, format conversion, and chunking with auto-cleanup support."""

    @staticmethod
    def download_youtube_audio(url: str, output_dir: str) -> str:
        """Downloads audio from YouTube into output_dir as WAV."""
        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            # Correct extension post-processor replacement
            wav_path = os.path.splitext(filename)[0] + ".wav"
        return wav_path

    @staticmethod
    def convert_to_wav(input_path: str, output_dir: str) -> str:
        """Converts local audio/video file to 16kHz mono WAV format."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input media file not found: {input_path}")

        filename = os.path.splitext(os.path.basename(input_path))[0] + "_converted.wav"
        output_path = os.path.join(output_dir, filename)

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(config.AUDIO_SAMPLE_RATE)
        audio.export(output_path, format="wav")
        return output_path

    @staticmethod
    def chunk_audio(wav_path: str, output_dir: str, chunk_minutes: int = config.AUDIO_CHUNK_MINUTES) -> List[str]:
        """Chunks a WAV audio file into segment files of chunk_minutes length."""
        audio = AudioSegment.from_wav(wav_path)
        chunk_ms = chunk_minutes * 60 * 1000
        chunks = []

        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start: start + chunk_ms]
            chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)

        return chunks

    @classmethod
    def process_source(cls, source: str, temp_dir: str) -> List[str]:
        """
        Main pipeline entry for audio preparation.
        Converts/Downloads source media into temp_dir and returns list of chunk file paths.
        """
        source = source.strip()
        if source.startswith("http://") or source.startswith("https://"):
            wav_path = cls.download_youtube_audio(source, temp_dir)
        else:
            wav_path = cls.convert_to_wav(source, temp_dir)

        chunks = cls.chunk_audio(wav_path, temp_dir)
        return chunks
