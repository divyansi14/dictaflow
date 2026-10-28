# dictaflow

Push-to-talk voice dictation for Windows (also runs on macOS/Linux-X11). Hold a hotkey, speak, release, and
clean text is pasted into whatever app has focus. Runs fully on CPU. Inspired by Wispr Flow.

## Pipeline

```
hotkey held -> mic stream (16 kHz) -> faster-whisper (int8, Silero VAD) -> snippet check
            -> cleanup (rules, optional LLM with fallback) -> dictionary replacements -> paste (clipboard restored)
```

| Module | Job |
|---|---|
| `recorder.py` | Always-open mic stream, zero start-up delay |
| `asr.py` | faster-whisper wrapper, VAD filter, vocabulary prompt, RTF measurement |
| `cleanup.py` | Filler/stutter removal, number self-corrections, casing; optional LLM pass with sanity guard + fallback |
| `dictionary.py` | Personal vocabulary, spoken snippets, text replacements |
| `paster.py` | Clipboard paste that restores your previous clipboard |
| `bench.py` / `record_eval.py` | Record your own clips, measure WER and latency |

## Setup (Windows + VS Code)

```powershell
python -m venv .venv
.venv\Scripts\activate          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python -m dictaflow
```

Hold **F9**, speak, release. First run downloads the Whisper model. Change hotkey/model in `config.json`
(for example `"hotkey": "ctrl_r"`, `"model": "base.en"`).

## Optional LLM cleanup

```powershell
$env:GROQ_API_KEY = "your-key"     # free key from console.groq.com
# then set "use_llm_cleanup": true in config.json
```

If the call fails, times out, or returns implausible output, dictaflow silently falls back to the rule-based result.

## Personalisation

Edit `data/dictionary.json`: `words` bias Whisper toward your jargon, `snippets` expand a spoken trigger
(say only "my email") into saved text, `replacements` fix recurring mis-hearings.

## Benchmark (your own voice)

```powershell
python -m dictaflow.record_eval                       # reads 15 sentences, saves clips
python -m dictaflow.bench --models base.en small.en   # add --llm to include LLM cleanup
python -m pytest -q                                   # unit tests
```

Results land in `eval/results.md`. Paste your table here:

| Model | WER raw % | WER +rules % | Mean RTF | Mean ASR s |
|---|---|---|---|---|
| _fill in_ | | | | |

## Known limitations

- Whisper is batch, not streaming: text appears after you release the key.
- Pasting into apps running as Administrator fails unless dictaflow is also run as Administrator.
- Bluetooth earbuds hurt accuracy; use the laptop mic or a wired headset.
- Rule-based self-correction only handles simple number/time cases; the LLM mode handles the rest.
- English only with the `.en` models.
