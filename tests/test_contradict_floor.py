"""Phase-4 contradiction floor calibration (linnuc bug report 2026-08-20).

A fixed cosine floor is a property of the embedding model, not the corpus.
0.60 was tuned on a space that is not e5-large's: measured on a 746-row
e5 store the same-project cosines span [0.74, 0.97], so 0.60 admitted
100.0% of 65,960 pairs and the nominator degenerated into an exhaustive
scan -- which is why a fleet host disabled its nightly cycle entirely.
"""
import numpy as np
import pytest

from mnemos.consolidation.phases import (
    calibrate_contradict_floor, select_contradict_candidates,
)
from mnemos.constants import CONTRADICT_MIN_SIM


def _e5_like(n=400, seed=0):
    """Similarities shaped like a real e5-large store: high and narrow."""
    rng = np.random.default_rng(seed)
    return list(np.clip(rng.normal(0.86, 0.04, n), 0.74, 0.974))


def test_absolute_floor_is_inert_on_an_e5_shaped_distribution():
    """The bug, stated as a test: 0.60 selects nothing in this space."""
    sims = _e5_like()
    assert min(sims) > CONTRADICT_MIN_SIM
    assert sum(1 for v in sims if v >= CONTRADICT_MIN_SIM) == len(sims)


def test_calibration_restores_selectivity():
    sims = _e5_like()
    floor = calibrate_contradict_floor(sims)
    admitted = sum(1 for v in sims if v >= floor)
    assert floor > CONTRADICT_MIN_SIM, "must rise above the inert constant"
    assert admitted < len(sims) * 0.10, f"still admitting {admitted}/{len(sims)}"


def test_calibration_never_goes_below_the_absolute_floor():
    """A low-similarity space must not be made MORE permissive than before."""
    sims = [0.10 + i * 0.001 for i in range(400)]
    assert calibrate_contradict_floor(sims) >= CONTRADICT_MIN_SIM


def test_small_stores_keep_a_usable_candidate_count():
    """A high percentile of few pairs is noise, not selection."""
    sims = _e5_like(n=40, seed=3)
    floor = calibrate_contradict_floor(sims)
    admitted = sum(1 for v in sims if v >= floor)
    assert admitted >= 1


def test_an_explicit_floor_turns_calibration_off(monkeypatch):
    """Operators who pinned a floor must keep getting exactly that floor."""
    import mnemos.consolidation.phases as ph
    monkeypatch.setattr(ph, "CONTRADICT_MIN_SIM_EXPLICIT", True)
    monkeypatch.setattr(ph, "CONTRADICT_MIN_SIM", 0.42)
    assert ph.calibrate_contradict_floor(_e5_like()) == 0.42


def test_selection_uses_the_calibrated_floor_end_to_end():
    """The whole point: fewer pairs reach the expensive NLI pass."""
    n = 60
    rng = np.random.default_rng(1)
    vecs = rng.normal(0, 1, (n, 32))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    # Compress into an e5-like band: everything similar to a shared centre.
    centre = vecs[0]
    vecs = vecs * 0.15 + centre * 0.985
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = list(range(n))
    sim = vecs @ vecs.T
    mem_by_id = {i: {"project": "dev", "type": "fact"} for i in ids}

    calibrated = select_contradict_candidates(ids, sim, mem_by_id, mode="nli")
    absolute = select_contradict_candidates(ids, sim, mem_by_id, mode="nli",
                                            min_sim=CONTRADICT_MIN_SIM)
    assert len(absolute) == n * (n - 1) // 2, "precondition: 0.60 admits all"
    assert len(calibrated) < len(absolute)
