"""NLI scoring layer for the store decision path and the Nyx phase-4 finder.

Replaces the cross-encoder reranker for dedup confirmation and contradiction
detection. A reranker answers "are these about the same topic?"; an NLI model
answers "do these say the same / opposite things?", which is the question the
store decision layer actually asks (benchmarks/nli-bench, 2026-07-02:
contradiction AUC 0.939 vs 0.69 for the reranker; dedup AUC 0.983 vs 0.95).

Language routing is agnostic, not tied to any specific language: content that
reads as English goes to an English checkpoint (ANLI+FEVER-hardened, the
strongest benched); everything else goes to a multilingual XNLI checkpoint
(~100 languages). Routing uses a cheap English-stopword heuristic; text with
no prose signal at all (paths, versions, numbers) defaults to English.

Optional dependency: transformers + torch (install extra: mnemos[nli]).
Every entry point degrades gracefully (returns None) when unavailable.
"""

import re
import threading

from .embed import embed
from .constants import NLI_EN_MODEL, NLI_MULTI_MODEL, NLI_MAX_LENGTH

_EN_STOPWORDS = re.compile(
    r"\b(the|and|is|are|was|were|of|to|in|for|with|on|at|by|from|that|this|"
    r"it|as|be|has|have|had|not|but|or|an|when|which|will|would|should|"
    r"there|their|then|than|these|those|its|into|about|after|before|only)\b",
    re.IGNORECASE,
)
_NON_ASCII_LETTER = re.compile(r"[^\x00-\x7f]")

_scorers = {}
_scorer_lock = threading.Lock()


def is_available() -> bool:
    """True when the optional NLI runtime (transformers + torch) importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def is_english(text: str) -> bool:
    """Cheap routing heuristic: does this text read as English prose?

    English function words present -> English. No function words but
    non-ASCII letters present -> not English (covers diacritics and
    non-Latin scripts). No prose signal at all -> default English, which
    is harmless: such texts are identifiers and numbers either model
    reads the same way.
    """
    words = text.split()
    if not words:
        return True
    hits = len(_EN_STOPWORDS.findall(text))
    if hits >= 2 or hits / len(words) > 0.15:
        return True
    if _NON_ASCII_LETTER.search(text):
        return False
    return True


class _TorchNliScorer:
    """Lazy transformers-backed scorer. score() returns (P(entail), P(contra))."""

    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.idx = {}
        for i, label in self.model.config.id2label.items():
            label = label.lower()
            if "entail" in label:
                self.idx["entail"] = i
            elif "contra" in label:
                self.idx["contra"] = i

    def score(self, premise, hypothesis):
        with self._torch.no_grad():
            enc = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                 truncation=True, max_length=NLI_MAX_LENGTH)
            probs = self._torch.softmax(self.model(**enc).logits[0], dim=-1)
        return float(probs[self.idx["entail"]]), float(probs[self.idx["contra"]])


def _get_scorer(multilingual=False):
    key = "multi" if multilingual else "en"
    with _scorer_lock:
        if key not in _scorers:
            model_id = NLI_MULTI_MODEL if multilingual else NLI_EN_MODEL
            _scorers[key] = _TorchNliScorer(model_id)
        return _scorers[key]


def _score_pair(a, b):
    """Both directions through the routed scorer. Returns ((e1,c1),(e2,c2))."""
    multilingual = not (is_english(a) and is_english(b))
    scorer = _get_scorer(multilingual=multilingual)
    return scorer.score(a, b), scorer.score(b, a)


def p_contradiction(a, b):
    """Max-direction P(contradiction). None when the runtime is unavailable.

    Max over both directions is mandatory: real contradictions can score
    asymmetrically (benched 0.44 one direction, 0.99 the other).
    """
    if not is_available():
        return None
    (_, c1), (_, c2) = _score_pair(a, b)
    return max(c1, c2)


def bidirectional_entailment(a, b):
    """Min-direction P(entailment): duplicate = each side entails the other.

    None when the runtime is unavailable.
    """
    if not is_available():
        return None
    (e1, _), (e2, _) = _score_pair(a, b)
    return min(e1, e2)


def _lines(text):
    return [ln.strip() for ln in text.split("\n")
            if ln.strip() and ln.strip() != "---"]


def line_max_contradiction(a, b, top_k=8):
    """Line-level contradiction finder: max P(contra) over the top_k
    cosine-preselected line pairs of two records.

    Recall-first: isolating conflicting statements from surrounding lines
    rescues contradictions that blob-level scoring buries (benched: the
    diagnosis conflict scored 0.58 blob-level, 0.9956 line-level). None
    when the runtime is unavailable or either record has no lines.
    """
    if not is_available():
        return None
    lines_a, lines_b = _lines(a), _lines(b)
    if not lines_a or not lines_b:
        return None
    vecs = embed(lines_a + lines_b, prefix="passage")
    if not vecs or len(vecs) != len(lines_a) + len(lines_b):
        pairs = [(la, lb) for la in lines_a for lb in lines_b][:top_k]
    else:
        va, vb = vecs[:len(lines_a)], vecs[len(lines_a):]
        scored = []
        for i, la in enumerate(lines_a):
            for j, lb in enumerate(lines_b):
                cos = sum(x * y for x, y in zip(va[i], vb[j]))
                scored.append((cos, la, lb))
        scored.sort(key=lambda t: t[0], reverse=True)
        pairs = [(la, lb) for _, la, lb in scored[:top_k]]
    best = 0.0
    for la, lb in pairs:
        multilingual = not (is_english(la) and is_english(lb))
        scorer = _get_scorer(multilingual=multilingual)
        _, c1 = scorer.score(la, lb)
        _, c2 = scorer.score(lb, la)
        best = max(best, c1, c2)
    return best
