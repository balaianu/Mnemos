# Memory correctness in 10.39.0

This release fixes retrieval and metadata consistency without changing the
embedding models, consolidation configuration, or database schema.

## Link direction

Each link retains `linked_id`, `relation`, and `strength`, and adds `source_id`,
`target_id`, and `direction`. Interpret the relation as:

```text
source_id --relation--> target_id
```

For `10 --superseded_by--> 20`, memory 10 has an outgoing link to 20 and
memory 20 has an incoming link from 10. Both retain the same stored relation
name. Do not interpret the name as a statement from the current search hit to
its neighbour without checking direction. Linked summaries carry the same
fields. At depth > 1, direction is relative to `via`, the preceding node.

## Namespace boundaries

SQLite ID operations now enforce the store's namespace, including direct
get/update/delete, bulk retrieval, snippets, archive moves, and linked/merged
source reads. Missing and foreign IDs behave as not found. Creating a link
with a missing or foreign endpoint raises `ValueError`. Existing cross-namespace
links are filtered on read; no historical rows are rewritten by this release.

Use a separate store instance for each namespace. These boundaries complement
transport access control; namespaces do not authenticate HTTP clients.
The reference Qdrant backend delegates these metadata operations to SQLite.

## Filtered retrieval and validity

The SQLite active and archived vector indexes now apply eligibility filters
inside the KNN candidate selection. A closer vector in another project or
namespace cannot crowd an eligible result out of a fixed global candidate pool.
Both implicit-rowid and explicit-id vec0 schemas remain supported.

`valid_only=True` excludes content before `valid_from` and at or after
`valid_until`, using the server's local date, as before. Linked content now
obeys the same rule, and invalid/archived nodes cannot bridge a multi-hop
traversal. `valid_only=False` still allows historical active content. Raw link
metadata may reference expired records; it is a record of the relationship,
not an assertion that those records are current.

## Access and confirmation

`memory_get` still increments `access_count`, updates `last_accessed`, and
applies the existing importance thresholds. It no longer updates
`last_confirmed`: looking up a claim is not evidence that it remains true.

After independently checking a source or receiving explicit confirmation:

```text
memory_update(id=123, confirmed=true)
mnemos update 123 --confirm
```

Python callers use `m.update(123, confirmed=True)`. The server records the
current local timestamp. Confirmation can accompany a content correction;
it does not change the separate `verified` flag. Do not automatically confirm
every result or every read. Existing `last_confirmed` values are left alone
because this release cannot reconstruct which old values came from evidence
versus access.

## Embedding consistency

Changes to any input of `prep_memory_text` (project, content, tags, type, layer)
trigger re-embedding. Project-only updates previously missed that step.
If embedding fails, the update reports `embedded=false` and a stale-vector
warning, retaining the existing repair workflow. This release does not reindex
existing stores automatically.

## Upgrade and verification

- No migration, automatic data repair, or model download is introduced.
- Update the package and restart long-lived MCP processes to load the code.
  Refresh cached client tool schemas to expose the `confirmed` argument.
- New regressions use synthetic temporary databases and mocked embeddings or
  small deterministic vectors. Run `python -m pytest tests/` in an environment
  that permits localhost sockets for the HTTP transport tests.
- Regression coverage verifies both successful in-namespace behavior and
  rejection of foreign IDs, active and archived filtered KNN, current versus
  historical linked content, and confirmation independent of reads.
