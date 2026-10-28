import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dictaflow.cleanup import _sane, rule_cleanup  # noqa: E402
from dictaflow.dictionary import Dictionary  # noqa: E402


def test_removes_fillers():
    assert rule_cleanup("um so I think uh we should go") == "So I think we should go."


def test_collapses_stutters():
    assert rule_cleanup("the the launch is is delayed") == "The launch is delayed."


def test_number_self_correction_ellipsis():
    assert rule_cleanup("let's meet at 5... actually 6pm") == "Let's meet at 6pm."


def test_number_self_correction_whisper_punctuation():
    assert rule_cleanup("Let's meet at 5. Actually, 6pm.") == "Let's meet at 6pm."


def test_capital_i_but_not_ie():
    assert rule_cleanup("i think i'm late") == "I think I'm late."
    assert "i.e." in rule_cleanup("use a tool, i.e. whisper, for this task")


def test_trailing_comma_becomes_period():
    assert rule_cleanup("we should ship it today, um") == "We should ship it today."


def test_short_utterance_gets_no_period():
    assert rule_cleanup("ok sounds good") == "Ok sounds good"


def test_snippet_and_replacement():
    d = Dictionary(snippets={"my email": "a@b.com"}, replacements={"git hub": "GitHub"})
    assert d.expand_snippet("My email.") == "a@b.com"
    assert d.expand_snippet("send my email now") is None
    assert d.replace("push to git hub") == "push to GitHub"


def test_llm_output_guard():
    raw = "um send the report to the team by friday"
    assert _sane(raw, "Send the report to the team by Friday.")
    assert not _sane(raw, "")
    assert not _sane(raw, "x" * 500)
