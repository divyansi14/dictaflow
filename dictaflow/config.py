import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


@dataclass
class Config:
    hotkey: str = "f9"
    input_device: str = ""  # name fragment or index; empty = Windows default mic
    model: str = "small.en"
    language: str = "en"
    compute_type: str = "int8"
    cpu_threads: int = 0
    min_seconds: float = 0.3
    beam_size: int = 1
    vad_filter: bool = True
    use_llm_cleanup: bool = False
    llm_model: str = "llama-3.1-8b-instant"
    llm_timeout_s: float = 3.0
    restore_clipboard: bool = True
    beep: bool = True
    log_history: bool = True


def load_config(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg