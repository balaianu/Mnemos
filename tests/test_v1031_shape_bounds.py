"""Tests for v10.31.0: bounded tensor shapes and a reaper on every transport.

The ONNX Runtime CPU arena keeps the peak allocation of every tensor shape it
has served and never returns it while the session lives. Shape is (batch,
sequence length) and both axes were free-running: fastembed defaults to batch
64 for reranking and 256 for embedding, and nothing bounded sequence length at
all. Measured on a shared long-lived server, one limit=20 search (a
60-document rerank pool) claimed 1.6 GB permanently while an identical-shape
repeat cost 3% of that.

The second half is the reaper. Dropping an idle session is the only way
Mnemos can hand memory back, and it was started by the stdio entrypoint only,
which is precisely backwards: a stdio harness reclaims everything by exiting,
while the shared HTTP server (v10.27.0) never exits and had no reaper at all.
"""

import importlib
import pytest


def _reload(monkeypatch, **env):
    for k in ("MNEMOS_RERANK_BATCH", "MNEMOS_RERANK_MAX_CHARS",
              "MNEMOS_EMBED_BATCH", "MNEMOS_MODEL_IDLE_TTL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import mnemos.constants as c
    importlib.reload(c)
    return c


# --- sequence axis ---------------------------------------------------------

def test_clip_leaves_short_text_alone(monkeypatch):
    _reload(monkeypatch)
    import mnemos.rerank as r
    importlib.reload(r)
    assert r._clip("F: short") == "F: short"


def test_clip_bounds_long_text(monkeypatch):
    _reload(monkeypatch, MNEMOS_RERANK_MAX_CHARS="100")
    import mnemos.rerank as r
    importlib.reload(r)
    assert len(r._clip("x" * 5000)) == 100


def test_clip_disabled_at_zero(monkeypatch):
    _reload(monkeypatch, MNEMOS_RERANK_MAX_CHARS="0")
    import mnemos.rerank as r
    importlib.reload(r)
    assert len(r._clip("x" * 5000)) == 5000


def test_clip_does_not_mutate_the_stored_document(monkeypatch):
    """Truncation is on the scoring copy only."""
    _reload(monkeypatch, MNEMOS_RERANK_MAX_CHARS="10")
    import mnemos.rerank as r
    importlib.reload(r)
    docs = [{"id": 1, "text": "y" * 500}]
    r._clip(docs[0]["text"])
    assert len(docs[0]["text"]) == 500


# --- batch axis ------------------------------------------------------------

def test_rerank_passes_a_bounded_batch_size(monkeypatch):
    _reload(monkeypatch, MNEMOS_RERANK_BATCH="8", MNEMOS_RERANK_MAX_CHARS="50")
    import mnemos.rerank as r
    importlib.reload(r)
    seen = {}

    class FakeModel:
        def rerank(self, query, texts, batch_size=64):
            seen["batch_size"] = batch_size
            seen["max_len"] = max(len(t) for t in texts)
            return [1.0] * len(texts)

    monkeypatch.setattr(r, "_get_reranker", lambda: FakeModel())
    out = r.rerank("q", [{"id": i, "text": "z" * 400} for i in range(30)])
    assert seen["batch_size"] == 8
    assert seen["max_len"] == 50      # sequence axis bounded too
    assert len(out) == 30             # every document still scored


def test_defaults_are_bounded_not_fastembed_defaults(monkeypatch):
    c = _reload(monkeypatch)
    assert c.RERANK_BATCH < 64          # fastembed default
    assert c.EMBED_BATCH < 256          # fastembed default
    # The token pin is the real sequence bound; the char clip is a superseded
    # escape hatch and defaults off.
    assert 0 < c.RERANK_MAX_TOKENS <= 1024   # a sane pin; gte allows 8192, jina caps at 1024
    assert c.RERANK_MAX_CHARS == 0


# --- the reaper ------------------------------------------------------------

def test_reaper_is_a_noop_at_default_ttl(monkeypatch):
    _reload(monkeypatch)
    import mnemos._resource as res
    importlib.reload(res)
    assert res.start_idle_reaper() is False


def test_reaper_starts_once_when_ttl_set(monkeypatch):
    _reload(monkeypatch, MNEMOS_MODEL_IDLE_TTL="30")
    import mnemos._resource as res
    importlib.reload(res)
    assert res.start_idle_reaper(interval=3600) is True
    assert res.start_idle_reaper(interval=3600) is False   # not a second thread


def test_http_transport_starts_the_reaper(monkeypatch):
    """The regression this release exists for: the shared server had none."""
    import mnemos.http_server as h
    importlib.reload(h)
    called = []
    import mnemos._resource as res
    monkeypatch.setattr(res, "start_idle_reaper",
                        lambda **kw: called.append(True))
    # A non-local bind is refused, but only AFTER the reaper hook has run, so
    # this exercises the hook without ever opening a socket.
    with pytest.raises(ValueError):
        h.serve(http="8.8.8.8:65000", mnemos=object())
    assert called, "shared HTTP transport must start the idle reaper"


# --- the token pin (v10.31.0) ---------------------------------------------

def test_tokenizer_pin_fixes_both_truncation_and_padding(monkeypatch):
    """fastembed truncates at 1024 but pads to longest-in-batch, so one long
    memory drags a whole batch up. Pinning both collapses that to one shape."""
    _reload(monkeypatch, MNEMOS_RERANK_MAX_TOKENS="512")
    import mnemos.rerank as r
    importlib.reload(r)

    class FakeTok:
        def __init__(self):
            self.truncation = {"max_length": 1024}
            self.padding = {"length": None, "pad_id": 1,
                            "pad_token": "<pad>", "pad_type_id": 0,
                            "direction": "right"}
        def enable_truncation(self, max_length): self.truncation["max_length"] = max_length
        def enable_padding(self, **kw): self.padding.update(kw)

    class FakeEnc:
        class model:
            tokenizer = FakeTok()

    assert r._pin_tokenizer_shape(FakeEnc) is True
    assert FakeEnc.model.tokenizer.truncation["max_length"] == 512
    assert FakeEnc.model.tokenizer.padding["length"] == 512
    assert FakeEnc.model.tokenizer.padding["pad_id"] == 1   # preserved


def test_tokenizer_pin_is_a_noop_at_zero(monkeypatch):
    _reload(monkeypatch, MNEMOS_RERANK_MAX_TOKENS="0")
    import mnemos.rerank as r
    importlib.reload(r)
    assert r._pin_tokenizer_shape(object()) is False


def test_tokenizer_pin_degrades_instead_of_failing_the_load(monkeypatch):
    """It reaches through a private attribute; a fastembed refactor must not
    take the reranker down with it."""
    _reload(monkeypatch, MNEMOS_RERANK_MAX_TOKENS="512")
    import mnemos.rerank as r
    importlib.reload(r)
    assert r._pin_tokenizer_shape(object()) is False   # no .model.tokenizer


# --- NLI joins the reaper --------------------------------------------------

def test_nli_exposes_maybe_unload():
    """It had none, so a quiet night left both DeBERTa sessions resident."""
    import mnemos.nli as n
    assert hasattr(n, "maybe_unload")


def test_nli_unload_is_a_noop_when_nothing_is_loaded():
    import mnemos.nli as n
    n._scorers = {}
    assert n.maybe_unload(force=True) is False


def test_nli_force_unload_drops_scorers():
    import mnemos.nli as n
    n._scorers = {"en": object()}
    assert n.maybe_unload(force=True) is True
    assert n._scorers == {}


def test_reaper_ticks_nli_too():
    import inspect
    import mnemos._resource as res
    src = inspect.getsource(res.start_idle_reaper)
    assert "_nli_mod.maybe_unload()" in src


# --- custom reranker registration (v10.33.0) --------------------------------

def test_register_if_custom_adds_unknown_model(monkeypatch):
    import mnemos.rerank as r
    importlib.reload(r)
    calls = {}

    class FakeTCE:
        @staticmethod
        def add_custom_model(**kw):
            calls.update(kw)

    monkeypatch.setattr(r._c, "RERANKER_MODEL", "acme/unknown-reranker-9000")
    r._register_if_custom(FakeTCE)
    assert calls["model"] == "acme/unknown-reranker-9000"
    assert calls["model_file"] == "onnx/model.onnx"


def test_register_if_custom_is_noop_for_catalogue_models(monkeypatch):
    import mnemos.rerank as r
    importlib.reload(r)

    class FakeTCE:
        @staticmethod
        def add_custom_model(**kw):
            raise ValueError("Model x is already registered in CrossEncoderModel")

    monkeypatch.setattr(r._c, "RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    r._register_if_custom(FakeTCE)   # must not raise


def test_register_if_custom_reraises_other_valueerrors(monkeypatch):
    import mnemos.rerank as r
    importlib.reload(r)

    class FakeTCE:
        @staticmethod
        def add_custom_model(**kw):
            raise ValueError("something else entirely")

    monkeypatch.setattr(r._c, "RERANKER_MODEL", "x/y")
    with pytest.raises(ValueError):
        r._register_if_custom(FakeTCE)


# --- the probe catches byte-BPE mojibake (v10.33.0) --------------------------

def test_probe_flags_byte_fallback_fragmentation():
    """gte-modernbert never emits [UNK]: operators fragment into UTF-8 byte
    mojibake instead. A model represents an operator only if some produced
    piece still contains the character."""
    import mnemos.rerank as r

    class MojibakeEnc:
        def __init__(self, toks): self._t = toks
        def encode(self, sym, add_special_tokens=False):
            class E: tokens = self._t
            return E()

    class Enc:
        class model:
            tokenizer = MojibakeEnc(["â", "ľĵ"])   # byte pieces
    assert r._probe_cml_support(Enc)

    class Enc2:
        class model:
            tokenizer = None
        def __init__(self): pass
    # a tokenizer whose pieces contain the real symbol AND assign it a real
    # id is judged native
    class NativeEnc:
        class model:
            class tokenizer:
                @staticmethod
                def token_to_id(t):
                    return 3 if t == "<unk>" else None
                @staticmethod
                def encode(sym, add_special_tokens=False):
                    class E:
                        tokens = ["▁", sym]
                        ids = [6, 2299]
                    return E()
    assert not r._probe_cml_support(NativeEnc)


# --- the probe trusts ids over token strings (v10.35.1) -----------------------

def test_probe_catches_unk_echoed_as_surface_char():
    """Unigram/XLM-R tokenizers echo an unrepresentable character back as its
    own surface piece: jina-v2-multilingual encodes the because-operator to
    tokens ['▁','∵'] with ids [6,3] where 3 IS <unk>. The string test alone
    judges that native; the id is the ground truth."""
    import mnemos.rerank as r

    class EchoUnkEnc:
        class model:
            class tokenizer:
                @staticmethod
                def token_to_id(t):
                    return 3 if t == "<unk>" else None
                @staticmethod
                def encode(sym, add_special_tokens=False):
                    class E:
                        tokens = ["▁", sym]   # surface echo, looks native
                        ids = [6, 3]          # ...but 3 = <unk>
                    return E()

    missing = r._probe_cml_support(EchoUnkEnc)
    from mnemos.embed import CML_EMBED_MAP
    assert missing == frozenset(s for s in CML_EMBED_MAP if not s.isspace())


def test_probe_is_per_symbol_and_map_respects_it():
    """A model missing only the because-operator gets exactly that one
    spelled out; natively represented operators stay untouched in the
    reranker prep."""
    import mnemos.rerank as r
    from mnemos.embed import apply_cml_map

    class PartialEnc:
        class model:
            class tokenizer:
                @staticmethod
                def token_to_id(t):
                    return 3 if t == "<unk>" else None
                @staticmethod
                def encode(sym, add_special_tokens=False):
                    class E:
                        tokens = ["▁", sym]
                        ids = [6, 3 if sym == "∵" else 2299]
                    return E()

    missing = r._probe_cml_support(PartialEnc)
    assert missing == frozenset({"∵"})

    out = apply_cml_map("a ∵ b → c", missing)
    assert "∵" not in out and "because" in out
    assert "→" in out                      # native operator untouched
    assert apply_cml_map("x → y", frozenset()) == "x → y"
