"""Run from the project folder with the venv active:  python diagnose.py"""
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard

print("== Microphones ==")
print(sd.query_devices())
print("Default input device index:", sd.default.device[0])

print("\n== Key test (10 s): press F9, then Fn+F9, then Right Ctrl, then Scroll Lock ==")


def on_press(key):
    print("  pressed:", key)


with keyboard.Listener(on_press=on_press):
    time.sleep(10)

print("\n== Mic test: speak normally for 4 seconds NOW ==")
audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype="float32")
sd.wait()
peak = float(np.abs(audio).max())
print(f"Peak level: {peak:.3f}")
if peak < 0.01:
    print("-> Almost silent: mic is muted, blocked by Windows privacy settings, or wrong device.")
else:
    print("-> Mic works.")