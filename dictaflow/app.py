import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json
import platform
import threading
import time
from datetime import datetime

from pynput import keyboard

from .asr import Transcriber
from .cleanup import HALLUCINATIONS, cleanup
from .config import ROOT, load_config
from .dictionary import Dictionary
from .paster import paste
from .recorder import SAMPLE_RATE, Recorder

HISTORY = ROOT / "data" / "history.jsonl"


def parse_hotkey(name: str):
    key = getattr(keyboard.Key, name.lower(), None)
    return key if key is not None else keyboard.KeyCode.from_char(name)


class App:
    def __init__(self):
        self.cfg = load_config()
        self.dic = Dictionary.load()
        self.hotkey = parse_hotkey(self.cfg.hotkey)
        print(f"Loading Whisper model '{self.cfg.model}' (first run downloads it) ...")
        self.asr = Transcriber(self.cfg)
        self.rec = Recorder(self.cfg.input_device)
        self._busy = threading.Lock()  # one utterance at a time so pastes never interleave

    def _beep(self, freq: int):
        if self.cfg.beep and platform.system() == "Windows":
            import winsound

            threading.Thread(target=winsound.Beep, args=(freq, 50), daemon=True).start()

    def on_press(self, key):
        if key == self.hotkey and self.rec.start():
            self._beep(1000)

    def on_release(self, key):
        if key != self.hotkey:
            return
        t_release = time.perf_counter()
        audio = self.rec.stop()
        if audio is None:
            return
        self._beep(600)
        threading.Thread(target=self.process, args=(audio, t_release), daemon=True).start()

    def process(self, audio, t_release: float):
        with self._busy:
            duration = len(audio) / SAMPLE_RATE
            if duration < self.cfg.min_seconds:
                print(f"[{duration:.1f}s audio] too short, hold the key longer")
                return
            res = self.asr.transcribe(audio, vocab=self.dic.words)
            raw = res.text
            if not raw or (duration < 2.0 and raw.lower() in HALLUCINATIONS):
                print(f"[{duration:.1f}s audio] no speech detected")
                return

            t_clean = time.perf_counter()
            snippet = self.dic.expand_snippet(raw)
            if snippet:
                final, method = snippet, "snippet"
            else:
                final, method = cleanup(raw, self.cfg, self.dic.words)
                final = self.dic.replace(final)
            clean_s = time.perf_counter() - t_clean

            paste(final, self.cfg.restore_clipboard)
            total = time.perf_counter() - t_release
            print(
                f"[{duration:4.1f}s audio | asr {res.seconds:.2f}s (RTF {res.rtf:.2f}) | "
                f"{method} {clean_s:.2f}s | total {total:.2f}s] {final}"
            )
            if self.cfg.log_history:
                self._log(raw, final, method, duration, res.seconds, clean_s, total)

    def _log(self, raw, final, method, duration, asr_s, clean_s, total):
        HISTORY.parent.mkdir(exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "raw": raw, "final": final, "method": method,
            "audio_s": round(duration, 2), "asr_s": round(asr_s, 3),
            "clean_s": round(clean_s, 3), "total_s": round(total, 3),
        }
        with HISTORY.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def run(self):
        print(f"Ready. Hold [{self.cfg.hotkey.upper()}] to dictate, release to paste. Ctrl+C to quit.")
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        try:
            while listener.is_alive():
                listener.join(0.5)  # short timeout lets Ctrl+C get through on Windows
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
            self.rec.close()


def main():
    App().run()