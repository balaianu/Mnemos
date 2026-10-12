"""10.42.3/10.42.4: one completion roof for every consolidation call, per-phase override, length-exhausted replies logged."""
import importlib
import io
import json
from unittest import mock


def _fake_response(payload):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return _Resp(json.dumps(payload).encode("utf-8"))


def _llm_env(monkeypatch):
    monkeypatch.setenv("MNEMOS_LLM_API_KEY", "x")
    monkeypatch.setenv("MNEMOS_LLM_MODEL", "claude-test")
    monkeypatch.setenv("MNEMOS_LLM_API_URL", "https://example.invalid/v1/chat/completions")


def _sent_payload(monkeypatch, call):
    """Run `call` against a fake endpoint and return the JSON payload it sent."""
    from mnemos.consolidation import llm
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _fake_response({"choices": [{"finish_reason": "stop",
                                            "message": {"role": "assistant", "content": "ok"}}]})

    with mock.patch.object(llm.urllib.request, "urlopen", side_effect=fake_urlopen):
        call(llm)
    return seen["payload"]


class TestRoof:
    def test_default_roof_is_10000(self, monkeypatch):
        for k in ("MNEMOS_LLM_MAX_TOKENS", "MNEMOS_LLM_MAX_TOKENS_SYNTHESIZE", "MNEMOS_LLM_MAX_TOKENS_WEAVE"):
            monkeypatch.delenv(k, raising=False)
        import mnemos.constants as c
        importlib.reload(c)
        assert c.LLM_MAX_TOKENS == 10000
        assert c.llm_max_tokens() == 10000
        assert c.llm_max_tokens("WEAVE") == 10000
        assert c.SYNTHESIS_MAX_TOKENS == 10000

    def test_global_and_phase_override(self, monkeypatch):
        monkeypatch.setenv("MNEMOS_LLM_MAX_TOKENS", "6000")
        monkeypatch.setenv("MNEMOS_LLM_MAX_TOKENS_CONTRADICT", "3000")
        from mnemos.constants import llm_max_tokens
        assert llm_max_tokens() == 6000
        assert llm_max_tokens("WEAVE") == 6000
        assert llm_max_tokens("CONTRADICT") == 3000
        assert llm_max_tokens("contradict") == 3000

    def test_chat_sends_the_roof_when_no_budget_given(self, monkeypatch):
        _llm_env(monkeypatch)
        monkeypatch.setenv("MNEMOS_LLM_MAX_TOKENS", "7777")
        monkeypatch.setenv("MNEMOS_LLM_MAX_TOKENS_WEAVE", "4444")
        p = _sent_payload(monkeypatch, lambda llm: llm.haiku_chat([{"role": "user", "content": "hi"}]))
        assert p["max_tokens"] == 7777
        p = _sent_payload(monkeypatch, lambda llm: llm.opus_chat([{"role": "user", "content": "hi"}], phase="WEAVE"))
        assert p["max_tokens"] == 4444
        p = _sent_payload(monkeypatch, lambda llm: llm.chat([{"role": "user", "content": "hi"}], max_tokens=12))
        assert p["max_tokens"] == 12

    def test_phase5_passes_roof_to_llm(self, monkeypatch):
        from mnemos.consolidation import phases
        seen = {}

        def fake_opus_chat(messages, max_tokens=None, **kw):
            seen["max_tokens"] = max_tokens

        monkeypatch.setattr(phases, "opus_chat", fake_opus_chat)
        monkeypatch.setattr(phases, "_select_nyx_packet", lambda conn, active, emb: list(active)[:3])
        mem_by_id = {i: {"status": "active", "project": "dev", "content": f"m{i}",
                         "type": "fact", "importance": 5} for i in range(12)}
        stats = phases.phase_synthesize(None, {}, mem_by_id, execute=True)
        assert seen["max_tokens"] == phases.SYNTHESIS_MAX_TOKENS
        assert stats["insights_generated"] == 0

    def test_no_call_site_pins_a_small_budget(self):
        import inspect

        from mnemos.consolidation import phases
        src = inspect.getsource(phases)
        assert "max_tokens=256" not in src and "max_tokens=1024" not in src


class TestLengthFinishIsLogged:
    def _run(self, monkeypatch, capsys, content):
        from mnemos.consolidation import llm
        _llm_env(monkeypatch)
        payload = {"choices": [{"finish_reason": "length",
                                "message": {"role": "assistant", "content": content}}]}
        with mock.patch.object(llm.urllib.request, "urlopen", return_value=_fake_response(payload)):
            out = llm.chat([{"role": "user", "content": "hi"}], max_tokens=7)
        return out, capsys.readouterr().out

    def test_empty_length_reply_returns_none_and_logs(self, monkeypatch, capsys):
        out, log = self._run(monkeypatch, capsys, "")
        assert out is None
        assert "max_tokens=7" in log and "no visible content" in log

    def test_truncated_reply_is_returned_and_logged(self, monkeypatch, capsys):
        out, log = self._run(monkeypatch, capsys, "partial text")
        assert out == "partial text"
        assert "output truncated" in log
