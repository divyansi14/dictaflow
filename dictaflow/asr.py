import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from .config import Config
from .recorder import SAMPLE_RATE


@dataclass
class ASRResult:
    text: str
    audio_seconds: float
    seconds: float  # wall-clock transcription time

    @property
    def rtf(self) -> float:
        """Real-time factor = processing time / audio length (<1 is faster than real time)."""
        return self.seconds / self.audio_seconds if self.audio_seconds else 0.0


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model,
            device="cpu",
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads,
        )

    def transcribe(self, audio, vocab=None) -> ASRResult:
        """`audio` is a float32 numpy array (16 kHz mono) or a path to an audio file."""
        prompt = ("Vocabulary: " + ", ".join(vocab) + ".") if vocab else None
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,  # Silero VAD: trims silence, cuts hallucinations
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()  # generator runs here
        seconds = time.perf_counter() - t0
        if isinstance(audio, np.ndarray):
            audio_seconds = len(audio) / SAMPLE_RATE
        else:
            audio_seconds = info.duration
        return ASRResult(text=text, audio_seconds=audio_seconds, seconds=seconds)
