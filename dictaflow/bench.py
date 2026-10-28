"""Benchmark WER and latency: python -m dictaflow.bench --models base.en small.en [--llm]"""
import argparse
import dataclasses
import re
import statistics
import time
import wave
from datetime import date

import jiwer
import numpy as np

from .asr import Transcriber
from .cleanup import llm_cleanup, rule_cleanup
from .config import ROOT, load_config
from .dictionary import Dictionary


def load_wav(path) -> np.ndarray:
    """Read a 16 kHz mono 16-bit WAV (as saved by record_eval) into a float32 array."""
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def wer(refs, hyps) -> float:
    return jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps]) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--llm", action="store_true", help="also evaluate LLM cleanup")
    args = parser.parse_args()

    cfg = load_config()
    models = args.models or [cfg.model]
    clips = sorted((ROOT / "eval" / "clips").glob("*.wav"))
    if not clips:
        raise SystemExit("No clips found. Run: python -m dictaflow.record_eval")
    refs = [c.with_suffix(".txt").read_text(encoding="utf-8").strip() for c in clips]
    audios = [load_wav(c) for c in clips]
    vocab = Dictionary.load().words

    rows = []
    for model in models:
        print(f"== {model}")
        asr = Transcriber(dataclasses.replace(cfg, model=model))
        asr.transcribe(audios[0], vocab)  # warm-up, excluded from timings
        raws, rtfs, secs = [], [], []
        for audio in audios:
            res = asr.transcribe(audio, vocab)
            raws.append(res.text)
            rtfs.append(res.rtf)
            secs.append(res.seconds)
        ruled = [rule_cleanup(t) for t in raws]
        row = {
            "model": model,
            "wer_raw": wer(refs, raws),
            "wer_rules": wer(refs, ruled),
            "rtf": statistics.mean(rtfs),
            "asr_s": statistics.mean(secs),
            "wer_llm": None,
            "llm_s": None,
        }
        if args.llm:
            outs, times = [], []
            for t in raws:
                t0 = time.perf_counter()
                out = llm_cleanup(t, vocab, cfg.llm_model, cfg.llm_timeout_s) or rule_cleanup(t)
                times.append(time.perf_counter() - t0)
                outs.append(out)
            row["wer_llm"], row["llm_s"] = wer(refs, outs), statistics.mean(times)
        rows.append(row)

    lines = [
        f"# Benchmark ({date.today()}, {len(clips)} clips, CPU int8)\n",
        "| Model | WER raw % | WER +rules % | WER +LLM % | Mean RTF | Mean ASR s | Mean LLM s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
        lines.append(
            f"| {r['model']} | {f(r['wer_raw'])} | {f(r['wer_rules'])} | {f(r['wer_llm'])} | "
            f"{f(r['rtf'], 2)} | {f(r['asr_s'], 2)} | {f(r['llm_s'], 2)} |"
        )
    table = "\n".join(lines)
    print("\n" + table)
    (ROOT / "eval" / "results.md").write_text(table + "\n", encoding="utf-8")
    print("\nSaved eval/results.md")


if __name__ == "__main__":
    main()