"""Point-in-time retrieval: lexical, semantic and hybrid search, and lookups by reference.

Every function reads the rows in force at a date: `valid_from <= D < valid_to`. Searches skip
repealed versions; lookups and history return them, since "abrogato" is itself an answer.

A row of `unit_versions` is one text under one heading. When only the article heading changes,
ingestion closes the row and opens one with the same text and `change_kind = 'heading'`, so the
heading, the full-text vector and the embedding of a row are those in force over its interval.
Such rows continue the text version before them: hits report the validity of the text, and
`history` folds them into that version as heading changes.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date
from typing import get_args

import psycopg
from psycopg.rows import class_row, dict_row

from appalti_qa.config import RetrievalMode, Settings, get_settings
from appalti_qa.db import get_pool
from appalti_qa.keys import is_valid_key
from appalti_qa.models import Event, HeadingChange, Hit, UnitVersion
from appalti_qa.retrieval.embedding import Embedder, get_embedder
from appalti_qa.retrieval.fusion import reciprocal_rank_fusion
from appalti_qa.retrieval.refs import extract_refs
from appalti_qa.retrieval.urls import normattiva_url

logger = logging.getLogger(__name__)

FUSION_DEPTH = 4  # hybrid mode fuses a lexical and a semantic list of FUSION_DEPTH * k each
PINNED_SCORE = 1.0  # base score of units the query cites explicitly (above any RRF score)
HNSW_EF_SEARCH = (100, 1000)  # bounds for hnsw.ef_search; pgvector rejects values above 1000

_IN_FORCE = "v.valid_from <= %(at)s AND (v.valid_to IS NULL OR v.valid_to > %(at)s)"
_SEARCHABLE = f"{_IN_FORCE} AND NOT v.repealed"
_MATCHES_REFS = """EXISTS (
    SELECT 1 FROM unnest(%(refs)s::text[]) AS r(ref)
    WHERE u.key = r.ref OR starts_with(u.key, r.ref || '/') OR starts_with(r.ref, u.key || '/'))"""
_FROM = "FROM unit_versions v JOIN units u ON u.id = v.unit_id"
_COLUMNS = """v.id AS version_id, v.unit_id, u.key, u.ordinal,
    v.heading, v.text, v.valid_from, v.valid_to"""

# Natural-language questions rarely contain every term of a unit, so the AND query produced by
# websearch_to_tsquery is relaxed to OR and ts_rank_cd is weighted by the share of query terms
# the unit contains. Queries with a negation ("-lavori") are kept as written.
_LEXICAL_SQL = f"""
    SELECT {_COLUMNS},
        ts_rank_cd(v.tsv, %(tsq)s::tsquery, 1) * coalesce(
            cardinality(ARRAY(SELECT unnest(%(terms)s::text[])
                              INTERSECT SELECT unnest(tsvector_to_array(v.tsv))))::float8
            / nullif(cardinality(%(terms)s::text[]), 0),
            1) AS score
    {_FROM}
    WHERE v.tsv @@ %(tsq)s::tsquery AND {_SEARCHABLE}
    ORDER BY score DESC, u.ordinal
    LIMIT %(limit)s
"""
_SEMANTIC_SQL = f"""
    SELECT {_COLUMNS}, 1 - (v.embedding <=> %(vector)s) AS score
    {_FROM}
    WHERE v.embedding IS NOT NULL AND v.embedding_model = %(model)s AND {_SEARCHABLE}
    ORDER BY v.embedding <=> %(vector)s
    LIMIT %(limit)s
"""
_PINNED_SQL = f"""
    SELECT {_COLUMNS}, 0::float8 AS score {_FROM}
    WHERE {_MATCHES_REFS} AND {_SEARCHABLE}
"""
_LOOKUP_SQL = f"""
    SELECT {_COLUMNS}, 1::float8 AS score {_FROM}
    WHERE {_MATCHES_REFS} AND {_IN_FORCE}
    ORDER BY u.ordinal
"""
_AMENDMENTS_SQL = """
    SELECT DISTINCT v.unit_id, e.id, e.event_date AS date, e.kind, e.amending_urn,
        e.amending_label AS label
    FROM unit_versions v JOIN events e ON e.id = v.changed_by_event
    WHERE v.unit_id = ANY(%(units)s) AND v.valid_from <= %(at)s
    ORDER BY e.event_date, e.id
"""
_ROWS_SQL = """
    SELECT id, unit_id, valid_from, valid_to, change_kind FROM unit_versions
    WHERE unit_id = ANY(%(units)s)
    ORDER BY unit_id, valid_from
"""
_HISTORY_SQL = f"""
    SELECT v.id, v.unit_id, u.key AS unit_key, u.kind, u.article, u.comma, u.allegato,
        v.heading, v.text, v.text_hash, v.valid_from,
        v.valid_to, v.change_kind, v.repealed, e.id AS event_id, e.event_date, e.kind AS event_kind,
        e.amending_urn, e.amending_label
    {_FROM} LEFT JOIN events e ON e.id = v.changed_by_event
    WHERE {_MATCHES_REFS}
    ORDER BY u.ordinal, v.valid_from
"""
_LEXEME_RE = re.compile(r"'((?:[^']|'')+)'")


@dataclass(frozen=True, slots=True)
class _Candidate:
    version_id: int
    unit_id: int
    key: str
    ordinal: int
    heading: str | None
    text: str
    valid_from: date
    valid_to: date | None
    score: float


def search(
    query: str,
    at_date: date,
    k: int = 8,
    mode: RetrievalMode = "hybrid",
    *,
    conn: psycopg.Connection | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> list[Hit]:
    """The `k` units in force at `at_date` that best match `query`, best first.

    Scores: lexical = ts_rank_cd weighted by query-term coverage, semantic = cosine similarity,
    hybrid = reciprocal rank fusion, with units cited in the query ("art. 50, comma 1") first.
    Semantic search uses only embeddings made by the embedder's model.
    """
    if mode not in get_args(RetrievalMode):
        raise ValueError(f"unknown retrieval mode: {mode!r}")
    if k < 1:
        raise ValueError("k must be positive")
    query = query.strip()
    if not query:
        return []
    settings = settings or get_settings()
    with _session(conn) as c:
        if mode == "lexical":
            ranked = _lexical(c, query, at_date, k)
        elif mode == "semantic":
            ranked = _semantic(c, query, at_date, k, embedder or get_embedder(settings))
        else:
            active = embedder or get_embedder(settings)
            ranked = _hybrid(c, query, at_date, k, active, settings.rrf_k)
        return _to_hits(c, ranked[:k], at_date)


def lookup(
    refs: Sequence[str], at_date: date, *, conn: psycopg.Connection | None = None
) -> list[Hit]:
    """Units matching any of `refs` (any granularity) in force at `at_date`, in document order.

    Invalid keys match nothing. Every hit has score 1.0.
    """
    valid = [ref for ref in dict.fromkeys(refs) if is_valid_key(ref)]
    if not valid:
        return []
    with _session(conn) as c:
        candidates = _fetch(c, _LOOKUP_SQL, {"refs": valid, "at": at_date})
        return _to_hits(c, candidates, at_date)


def history(ref: str, *, conn: psycopg.Connection | None = None) -> list[UnitVersion]:
    """Every version of the units matching `ref`, by document order then date.

    A version's `heading` is the one in force when it starts; rows that only change the heading
    are folded into the version as `heading_changes`.
    """
    if not is_valid_key(ref):
        return []
    with _session(conn) as c, c.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(_HISTORY_SQL, {"refs": [ref]}).fetchall()
    versions: list[UnitVersion] = []
    previous: dict | None = None
    for row in rows:
        if previous is not None and _continues_text(previous, row):
            version = versions[-1]
            change = HeadingChange(
                date=row["valid_from"], heading=row["heading"], changed_by=_event(row)
            )
            versions[-1] = version.model_copy(
                update={
                    "valid_to": row["valid_to"],
                    "text": row["text"],
                    "text_hash": row["text_hash"],
                    "repealed": row["repealed"],
                    "heading_changes": [*version.heading_changes, change],
                }
            )
        else:
            versions.append(_unit_version(row))
        previous = row
    return versions


def available_range(*, conn: psycopg.Connection | None = None) -> tuple[date, date] | None:
    """(earliest valid_from, latest snapshot vigenza date), or None when nothing is loaded."""
    with _session(conn) as c:
        first, last = c.execute(
            "SELECT (SELECT min(valid_from) FROM unit_versions),"
            " (SELECT max(vigenza_date) FROM snapshots)"
        ).fetchone()
    return None if first is None or last is None else (first, last)


@contextmanager
def _session(conn: psycopg.Connection | None) -> Iterator[psycopg.Connection]:
    """A transaction on the given connection (a savepoint if one is open) or on a pooled one."""
    if conn is not None:
        with conn.transaction():
            yield conn
    else:
        with get_pool().connection() as pooled, pooled.transaction():
            yield pooled


def _fetch(conn: psycopg.Connection, sql: str, params: dict) -> list[_Candidate]:
    with conn.cursor(row_factory=class_row(_Candidate)) as cur:
        return cur.execute(sql, params).fetchall()


def _rank_order(candidate: _Candidate) -> tuple[float, int]:
    return (-candidate.score, candidate.ordinal)


def _lexical(conn: psycopg.Connection, query: str, at_date: date, limit: int) -> list[_Candidate]:
    row = conn.execute("SELECT websearch_to_tsquery('italian', %s)::text", (query,)).fetchone()
    strict = row[0] if row else ""
    if not strict:  # only stop words
        return []
    if "!" in strict:
        tsquery, terms = strict, []
    else:
        tsquery = strict.replace(" & ", " | ")
        terms = sorted({lexeme.replace("''", "'") for lexeme in _LEXEME_RE.findall(strict)})
    params = {"tsq": tsquery, "terms": terms, "at": at_date, "limit": limit}
    return _fetch(conn, _LEXICAL_SQL, params)


def _semantic(
    conn: psycopg.Connection, query: str, at_date: date, limit: int, embedder: Embedder
) -> list[_Candidate]:
    model = embedder.model_name
    if not conn.execute(
        "SELECT EXISTS (SELECT 1 FROM unit_versions WHERE embedding_model = %s)", (model,)
    ).fetchone()[0]:
        logger.warning("No embeddings from %s: semantic search skipped (run `embed`)", model)
        return []
    vector = embedder.embed_queries([query])[0]
    # Iterative HNSW scans (pgvector 0.8) keep returning neighbours until enough rows pass the
    # date filter; relaxed order may swap close neighbours, hence the final sort.
    conn.execute(
        "SELECT set_config('hnsw.ef_search', %s, true),"
        " set_config('hnsw.iterative_scan', 'relaxed_order', true)",
        (str(min(max(HNSW_EF_SEARCH[0], limit), HNSW_EF_SEARCH[1])),),
    )
    params = {"vector": vector, "model": model, "at": at_date, "limit": limit}
    return sorted(_fetch(conn, _SEMANTIC_SQL, params), key=_rank_order)


def _hybrid(
    conn: psycopg.Connection,
    query: str,
    at_date: date,
    k: int,
    embedder: Embedder,
    rrf_k: int,
) -> list[_Candidate]:
    depth = FUSION_DEPTH * k
    lexical = _lexical(conn, query, at_date, depth)
    semantic = _semantic(conn, query, at_date, depth, embedder)
    fused = reciprocal_rank_fusion(
        [[c.version_id for c in lexical], [c.version_id for c in semantic]], k=rrf_k
    )
    by_id = {c.version_id: c for c in (*lexical, *semantic)}
    ranked = {vid: replace(by_id[vid], score=score) for vid, score in fused.items()}
    refs = extract_refs(query)
    if refs:
        for cited in _fetch(conn, _PINNED_SQL, {"refs": refs, "at": at_date}):
            score = PINNED_SCORE + fused.get(cited.version_id, 0.0)
            ranked[cited.version_id] = replace(cited, score=score)
    return sorted(ranked.values(), key=_rank_order)


def _to_hits(conn: psycopg.Connection, candidates: list[_Candidate], at_date: date) -> list[Hit]:
    amendments: dict[int, list[Event]] = {}
    spans: dict[int, tuple[date, date | None]] = {}
    if candidates:
        units = list({c.unit_id for c in candidates})
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_AMENDMENTS_SQL, {"units": units, "at": at_date})
            for row in cur:
                amendments.setdefault(row.pop("unit_id"), []).append(Event(**row))
            spans = _text_spans(cur.execute(_ROWS_SQL, {"units": units}).fetchall())
    hits = []
    for c in candidates:
        valid_from, valid_to = spans.get(c.version_id, (c.valid_from, c.valid_to))
        hits.append(
            Hit(
                unit_key=c.key,
                version_id=c.version_id,
                heading=c.heading,
                text=c.text,
                score=c.score,
                valid_from=valid_from,
                valid_to=valid_to,
                normattiva_url=normattiva_url(c.key, at_date),
                modified_by=amendments.get(c.unit_id, []),
            )
        )
    return hits


def _continues_text(previous: dict, row: dict) -> bool:
    """True for a row that only changes the heading of the text version of `previous`."""
    return (
        row["change_kind"] == "heading"
        and row["unit_id"] == previous["unit_id"]
        and row["valid_from"] == previous["valid_to"]
    )


def _text_spans(rows: list[dict]) -> dict[int, tuple[date, date | None]]:
    """Row id -> validity of the text version the row belongs to (rows sorted by unit, date)."""
    spans: dict[int, tuple[date, date | None]] = {}
    group: list[dict] = []
    for row in [*rows, None]:
        if row is not None and group and _continues_text(group[-1], row):
            group.append(row)
            continue
        for member in group:
            spans[member["id"]] = (group[0]["valid_from"], group[-1]["valid_to"])
        group = [row] if row is not None else []
    return spans


def _event(row: dict) -> Event | None:
    if row["event_id"] is None:
        return None
    return Event(
        id=row["event_id"],
        date=row["event_date"],
        kind=row["event_kind"],
        amending_urn=row["amending_urn"],
        label=row["amending_label"],
    )


def _unit_version(row: dict) -> UnitVersion:
    return UnitVersion(
        id=row["id"],
        unit_key=row["unit_key"],
        kind=row["kind"],
        article=row["article"],
        comma=row["comma"],
        allegato=row["allegato"],
        heading=row["heading"],
        text=row["text"],
        text_hash=row["text_hash"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        changed_by=_event(row),
        change_kind=row["change_kind"],
        repealed=row["repealed"],
    )
