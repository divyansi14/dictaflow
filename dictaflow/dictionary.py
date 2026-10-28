"""Personal dictionary (biases Whisper), snippets (spoken shortcut -> text), replacements."""
import json
import re
from pathlib import Path
from typing import Optional

from .config import ROOT

DICT_PATH = ROOT / "data" / "dictionary.json"


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


class Dictionary:
    def __init__(self, words=None, snippets=None, replacements=None):
        self.words = list(words or [])
        self.snippets = {_norm(k): v for k, v in (snippets or {}).items()}
        self.replacements = dict(replacements or {})

    @classmethod
    def load(cls, path: Path = DICT_PATH) -> "Dictionary":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("words"), data.get("snippets"), data.get("replacements"))

    def expand_snippet(self, text: str) -> Optional[str]:
        """If the whole utterance is a snippet trigger, return its expansion."""
        return self.snippets.get(_norm(text))

    def replace(self, text: str) -> str:
        for wrong, right in self.replacements.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
        return text
