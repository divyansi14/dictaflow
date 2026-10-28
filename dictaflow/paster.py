import platform
import time

import pyperclip
from pynput import keyboard

_kbd = keyboard.Controller()
_MOD = keyboard.Key.cmd if platform.system() == "Darwin" else keyboard.Key.ctrl


def paste(text: str, restore_clipboard: bool = True) -> None:
    """Paste `text` into the focused app via the clipboard, then restore the old clipboard."""
    old = None
    if restore_clipboard:
        try:
            old = pyperclip.paste()
        except Exception:
            old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    with _kbd.pressed(_MOD):
        _kbd.tap("v")
    if old is not None:
        time.sleep(0.25)  # let the target app read the clipboard first
        pyperclip.copy(old)
