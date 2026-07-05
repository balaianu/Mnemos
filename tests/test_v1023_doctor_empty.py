"""Tests for v10.23.0: doctor flags an empty store instead of blessing it.

A doctor run against a store with zero active memories for the resolved
namespace almost always means a misconfigured MNEMOS_DB path or a
MNEMOS_NAMESPACE mismatch, not a healthy system. Reporting "healthy" on
such a store is a vacuous green (2026-07-05 incident: it helped a false
all-clear survive). Doctor now surfaces it as an issue with a config hint;
a genuinely fresh store sees the same hint once, which the message allows
for.
"""

from mnemos.core import Mnemos
from mnemos.storage.sqlite_store import SQLiteStore


def _mnemos(tmp_path, namespace="t"):
    store = SQLiteStore(db_path=str(tmp_path / "m.db"), namespace=namespace)
    return Mnemos(store=store, namespace=namespace,
                  enable_contradiction_detection=False, enable_rerank=False)


def _seed(m, namespace, mid=1, status="active"):
    conn = m.store._get_conn()
    conn.execute(
        "INSERT INTO memories (id, namespace, project, content, tags, type, "
        "layer, status) VALUES (?, ?, 'dev', 'F: seeded', '', 'fact', "
        "'semantic', ?)", (mid, namespace, status))
    conn.commit()


class TestDoctorEmptyStore:
    def test_completely_empty_store_is_flagged(self, tmp_path):
        m = _mnemos(tmp_path)
        report = m.doctor()
        flagged = [i for i in report["issues"] if "empty" in i.lower()]
        assert flagged, f"no empty-store issue in {report['issues']}"
        assert "MNEMOS_DB" in flagged[0]
        assert report["status"] == "issues_detected"

    def test_namespace_mismatch_hint_lists_other_namespaces(self, tmp_path):
        m = _mnemos(tmp_path)
        _seed(m, namespace="other", mid=1)
        _seed(m, namespace="other", mid=2)
        report = m.doctor()
        flagged = [i for i in report["issues"]
                   if "namespace" in i.lower() and "'t'" in i]
        assert flagged, f"no namespace-hint issue in {report['issues']}"
        assert "other" in flagged[0]
        assert "MNEMOS_NAMESPACE" in flagged[0]

    def test_populated_store_is_not_flagged(self, tmp_path):
        m = _mnemos(tmp_path)
        _seed(m, namespace="t")
        report = m.doctor()
        assert not any("empty" in i.lower() for i in report["issues"])
        assert not any("MNEMOS_NAMESPACE" in i for i in report["issues"])

    def test_archived_only_namespace_is_flagged(self, tmp_path):
        m = _mnemos(tmp_path)
        _seed(m, namespace="t", status="archived")
        report = m.doctor()
        flagged = [i for i in report["issues"]
                   if "0 active" in i or "no active" in i.lower()]
        assert flagged, f"no active-memories issue in {report['issues']}"
