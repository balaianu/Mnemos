"""10.42.3: the synthesis budget is configurable and a budget-exhausted reply is logged."""
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


class TestSynthesisBudget:
    def test_default_budget_is_4096(self, monkeypatch):
        monkeypatch.delenv("MNEMOS_LLM_MAX_TOKENS_SYNTHESIZE", raising=False)
        import mnemos.constants as c
        importlib.reload(c)
        assert c.SYNTHESIS_MAX_TOKENS == 4096

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MNEMOS_LLM_MAX_TOKENS_SYNTHESIZE", "8192")
        import mnemos.constants as c
        importlib.reload(c)
        assert c.SYNTHESIS_MAX_TOKENS == 8192
        monkeypatch.delenv("MNEMOS_LLM_MAX_TOKENS_SYNTHESIZE")
        importlib.reload(c)

    def test_phase5_passes_budget_to_llm(self, monkeypatch):
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


class TestLengthFinishIsLogged:
    def _run(self, monkeypatch, capsys, content):
        from mnemos.consolidation import llm
        monkeypatch.setenv("MNEMOS_LLM_API_KEY", "x")
        monkeypatch.setenv("MNEMOS_LLM_MODEL", "claude-test")
        monkeypatch.setenv("MNEMOS_LLM_API_URL", "https://example.invalid/v1/chat/completions")
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
