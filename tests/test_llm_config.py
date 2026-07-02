"""Per-phase LLM config resolution (llm._get_config).

Covers the per-phase API-key override (added 2026-07-02): a single phase (e.g.
MERGE) can target a cloud provider with a real token while the other phases
stay on a local endpoint whose key is a throwaway, so the secret never reaches
the local endpoint.
"""
import os

from mnemos.consolidation.llm import _get_config


def _clear(monkeypatch):
    for k in list(os.environ):
        if k.startswith("MNEMOS_LLM_"):
            monkeypatch.delenv(k, raising=False)


def test_per_phase_key_override(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-local")
    monkeypatch.setenv("MNEMOS_LLM_API_KEY_MERGE", "sk-cloud")
    assert _get_config(phase="MERGE")["key"] == "sk-cloud"
    assert _get_config(phase="WEAVE")["key"] == "sk-local"
    assert _get_config()["key"] == "sk-local"


def test_per_phase_key_falls_back_to_global(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "sk-local")
    assert _get_config(phase="MERGE")["key"] == "sk-local"


def test_per_phase_omit_temperature(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "k")
    monkeypatch.setenv("MNEMOS_LLM_OMIT_TEMPERATURE_MERGE", "1")
    assert _get_config(phase="MERGE")["omit_temperature"] is True
    assert _get_config(phase="WEAVE")["omit_temperature"] is False
    assert _get_config()["omit_temperature"] is False


def test_per_phase_url_and_model_still_resolve(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "k")
    monkeypatch.setenv("MNEMOS_LLM_API_URL", "http://local/v1/chat/completions")
    monkeypatch.setenv("MNEMOS_LLM_MODEL", "qwen-pool")
    monkeypatch.setenv("MNEMOS_LLM_API_URL_MERGE", "https://api.anthropic.com/v1/chat/completions")
    monkeypatch.setenv("MNEMOS_LLM_MODEL_MERGE", "claude-sonnet-5")
    cfg = _get_config(phase="MERGE")
    assert cfg["url"] == "https://api.anthropic.com/v1/chat/completions"
    assert cfg["model"] == "claude-sonnet-5"


class _FakeResp:
    def read(self):
        import json as _json
        return _json.dumps(
            {"choices": [{"message": {"content": "OK"}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture_chat(monkeypatch, **chat_kwargs):
    import json as _json
    import urllib.request
    from mnemos.consolidation.llm import chat
    _clear(monkeypatch)
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "k")
    monkeypatch.setenv("MNEMOS_LLM_MODEL", "m")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["payload"] = _json.loads(req.data.decode())
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = chat([{"role": "user", "content": "x"}], **chat_kwargs)
    return out, captured["payload"]


def test_chat_temperature_none_omits_param(monkeypatch):
    # Model families like Sonnet 5 reject temperature outright; callers pass
    # None to not send it at all, no env flag needed.
    out, payload = _capture_chat(monkeypatch, temperature=None)
    assert out == "OK"
    assert "temperature" not in payload


def test_chat_explicit_temperature_still_sent(monkeypatch):
    out, payload = _capture_chat(monkeypatch, temperature=0.0)
    assert out == "OK"
    assert payload["temperature"] == 0.0
