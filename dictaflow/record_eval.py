"""Record your own benchmark clips: python -m dictaflow.record_eval"""
import wave

import numpy as np

from .config import ROOT, load_config
from .recorder import SAMPLE_RATE, Recorder


def save_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main():
    lines = (ROOT / "eval" / "sentences.txt").read_text(encoding="utf-8").splitlines()
    sentences = [s.strip() for s in lines if s.strip()]
    out = ROOT / "eval" / "clips"
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(load_config().input_device)
    try:
        for i, sentence in enumerate(sentences, 1):
            name = f"clip_{i:02d}"
            if (out / f"{name}.wav").exists():
                continue  # already recorded; delete the file to re-record
            print(f"\n[{i}/{len(sentences)}] Read aloud:\n  {sentence}")
            input("  Press Enter to START...")
            rec.start()
            input("  Speak, then press Enter to STOP...")
            audio = rec.stop()
            save_wav(out / f"{name}.wav", audio)
            (out / f"{name}.txt").write_text(sentence, encoding="utf-8")
    finally:
        rec.close()
    print(f"\nSaved clips to {out}. Now run: python -m dictaflow.bench")


if __name__ == "__main__":
    main()