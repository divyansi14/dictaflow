import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono


def resolve_device(spec):
    """Turn '' / '2' / 'Microphone Array' into a sounddevice input device index (or None)."""
    if spec in (None, ""):
        return None
    if str(spec).isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and str(spec).lower() in d["name"].lower():
            return i
    raise ValueError(f"No input device matching '{spec}'. Run diagnose.py to list devices.")


class Recorder:
    """Keeps the mic stream open so pressing the hotkey has zero startup delay."""

    def __init__(self, device=None):
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
            device=resolve_device(device),
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._frames.append(indata.copy())

    def start(self) -> bool:
        with self._lock:
            if self._recording:  # ignore key auto-repeat while held
                return False
            self._frames = []
            self._recording = True
            return True

    def stop(self):
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            frames, self._frames = self._frames, []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).flatten()

    def close(self):
        self._stream.stop()
        self._stream.close()