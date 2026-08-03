import os
from typing import List, Optional
import requests
from pydub import AudioSegment
from groq import Groq
import config


class TranscriptionService:
    """Service handling multi-lingual speech-to-text using Groq (Whisper) and Sarvam AI."""

    _groq_client = None

    @classmethod
    def get_groq_client(cls) -> Groq:
        if cls._groq_client is None:
            if not config.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
            cls._groq_client = Groq(api_key=config.GROQ_API_KEY)
        return cls._groq_client

    @classmethod
    def transcribe_chunk_whisper(cls, chunk_path: str) -> str:
        """Sends audio chunk to Groq's cloud-hosted Whisper API."""
        client = cls.get_groq_client()
        with open(chunk_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), audio_file.read()),
                model=config.WHISPER_MODEL,
                response_format="text"
            )
        return str(transcription)

    @staticmethod
    def _send_to_sarvam(piece_path: str) -> str:
        """Sends a ≤30s audio piece to Sarvam AI STT Translate endpoint."""
        if not config.SARVAM_API_KEY:
            raise RuntimeError("SARVAM_API_KEY is not set in environment or .env file.")

        headers = {"api-subscription-key": config.SARVAM_API_KEY}
        url = "https://api.sarvam.ai/speech-to-text-translate"

        with open(piece_path, "rb") as f:
            files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
            data = {"model": config.SARVAM_MODEL, "with_diarization": "false"}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=120)

        if not response.ok:
            raise RuntimeError(f"Sarvam AI Error ({response.status_code}): {response.text}")

        return response.json().get("transcript", "")

    @classmethod
    def transcribe_chunk_sarvam(cls, chunk_path: str, temp_dir: str) -> str:
        """
        Splits chunk into 25s pieces to satisfy Sarvam API requirements,
        transcribes and translates Hinglish to English.
        """
        audio = AudioSegment.from_wav(chunk_path)
        piece_ms = config.SARVAM_PIECE_SECONDS * 1000
        full_text = []

        base_name = os.path.splitext(os.path.basename(chunk_path))[0]

        for i, start in enumerate(range(0, len(audio), piece_ms)):
            piece = audio[start: start + piece_ms]
            piece_path = os.path.join(temp_dir, f"{base_name}_sv_{i}.wav")
            piece.export(piece_path, format="wav")

            try:
                text = cls._send_to_sarvam(piece_path)
                if text:
                    full_text.append(text)
            finally:
                if os.path.exists(piece_path):
                    try:
                        os.remove(piece_path)
                    except OSError:
                        pass

        return " ".join(full_text)

    @classmethod
    def transcribe_all(cls, chunks: List[str], language: str = "english", temp_dir: str = None) -> str:
        """Routes audio chunks to Whisper (English) or Sarvam AI (Hinglish) and aggregates full transcript."""
        transcripts = []
        is_hinglish = language.lower() == "hinglish"

        for chunk_path in chunks:
            if is_hinglish:
                if temp_dir is None:
                    temp_dir = os.path.dirname(chunk_path)
                text = cls.transcribe_chunk_sarvam(chunk_path, temp_dir)
            else:
                text = cls.transcribe_chunk_whisper(chunk_path)

            if text:
                transcripts.append(text.strip())

        return " ".join(transcripts)
