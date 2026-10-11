"""Load a versioned corpus into PostgreSQL, replacing the act's previous data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import psycopg

from appalti_qa.ingest.versioning import VersionedCorpus

logger = logging.getLogger(__name__)

# Embeddings of the versions about to be replaced, keyed by what was embedded.
_KEEP_EMBEDDINGS = """
CREATE TEMP TABLE kept_embeddings ON COMMIT DROP AS
SELECT DISTINCT ON (v.text_hash, v.heading) v.text_hash, v.heading, v.embedding, v.embedding_model
FROM unit_versions v JOIN units u ON u.id = v.unit_id
WHERE u.act_id = %s AND v.embedding IS NOT NULL
ORDER BY v.text_hash, v.heading, v.id DESC
"""
_RESTORE_EMBEDDINGS = """
UPDATE unit_versions v
SET embedding = k.embedding, embedding_model = k.embedding_model
FROM kept_embeddings k, units u
WHERE u.id = v.unit_id AND u.act_id = %s
  AND v.text_hash = k.text_hash AND v.heading IS NOT DISTINCT FROM k.heading
"""
# Passages (and their embeddings) depend only on the heading and text of a version: keep them too.
_KEEP_PASSAGES = """
CREATE TEMP TABLE kept_passages ON COMMIT DROP AS
SELECT DISTINCT ON (v.text_hash, v.heading, p.ordinal) v.text_hash AS version_hash, v.heading,
    p.ordinal, p.text, p.text_hash, p.scheme, p.embedding, p.embedding_model, p.hub_score
FROM unit_passages p JOIN unit_versions v ON v.id = p.version_id JOIN units u ON u.id = v.unit_id
WHERE u.act_id = %s
ORDER BY v.text_hash, v.heading, p.ordinal, p.id DESC
"""
_RESTORE_PASSAGES = """
INSERT INTO unit_passages (version_id, ordinal, text, text_hash, scheme, embedding,
    embedding_model, hub_score)
SELECT v.id, k.ordinal, k.text, k.text_hash, k.scheme, k.embedding, k.embedding_model, k.hub_score
FROM unit_versions v JOIN units u ON u.id = v.unit_id
JOIN kept_passages k ON k.version_hash = v.text_hash AND k.heading IS NOT DISTINCT FROM v.heading
WHERE u.act_id = %s
"""


@dataclass(frozen=True, slots=True)
class LoadStats:
    snapshots: int
    events: int
    units: int
    versions: int
    embeddings_kept: int


def load_corpus(corpus: VersionedCorpus, conn: psycopg.Connection) -> LoadStats:
    """Replace the act's snapshots, events, units and versions in one transaction; commits.

    Idempotent: loading the same corpus twice gives the same rows. Embeddings already computed
    for a (text_hash, heading) pair, and the passages split from it (`unit_passages`, with their
    embeddings), are restored on the new rows, so re-ingesting after an amendment only leaves the
    changed texts to embed.
    """
    act = corpus.act
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "INSERT INTO acts (code, urn, title, short_title, issued_on)"
            " VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (code) DO UPDATE SET urn = EXCLUDED.urn, title = EXCLUDED.title,"
            " short_title = EXCLUDED.short_title, issued_on = EXCLUDED.issued_on"
            " RETURNING id",
            (act.code, act.urn, act.title, act.short_title, act.issued_on),
        )
        act_id = cur.fetchone()[0]
        cur.execute(_KEEP_EMBEDDINGS, (act_id,))
        cur.execute(_KEEP_PASSAGES, (act_id,))
        for table in ("units", "events", "snapshots"):  # unit_versions cascade from units
            cur.execute(f"DELETE FROM {table} WHERE act_id = %s", (act_id,))

        cur.executemany(
            "INSERT INTO snapshots (act_id, vigenza_date, effective_date, source_file, sha256)"
            " VALUES (%s, %s, %s, %s, %s)",
            [
                (act_id, s.vigenza_date, effective, s.source_file, s.sha256)
                for s, effective in corpus.snapshots
            ],
        )
        event_ids: dict[tuple[str, date], int] = {}
        for event in corpus.events:
            cur.execute(
                "INSERT INTO events (act_id, event_date, kind, amending_urn, amending_label)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (act_id, event.date, event.kind, event.amending_urn, event.label),
            )
            event_ids[(event.amending_urn, event.date)] = cur.fetchone()[0]

        cur.executemany(
            "INSERT INTO units (act_id, key, kind, article, comma, allegato, heading, ordinal)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (act_id, u.key, u.kind, u.article, u.comma, u.allegato, u.heading, u.ordinal)
                for u in corpus.units
            ],
        )
        cur.execute("SELECT key, id FROM units WHERE act_id = %s", (act_id,))
        unit_ids = dict(cur.fetchall())

        cur.executemany(
            "INSERT INTO unit_versions (unit_id, heading, text, text_hash, valid_from, valid_to,"
            " changed_by_event, change_kind, repealed)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    unit_ids[v.unit_key],
                    v.heading,
                    v.text,
                    v.text_hash,
                    v.valid_from,
                    v.valid_to,
                    event_ids[(v.changed_by.amending_urn, v.changed_by.date)]
                    if v.changed_by
                    else None,
                    v.change_kind,
                    v.repealed,
                )
                for v in corpus.versions
            ],
        )
        cur.execute(_RESTORE_EMBEDDINGS, (act_id,))
        embeddings_kept = cur.rowcount
        cur.execute(_RESTORE_PASSAGES, (act_id,))
    if not conn.autocommit:
        conn.commit()
    stats = LoadStats(
        snapshots=len(corpus.snapshots),
        events=len(corpus.events),
        units=len(corpus.units),
        versions=len(corpus.versions),
        embeddings_kept=embeddings_kept,
    )
    logger.info("loaded %s: %s", act.short_title, stats)
    return stats
