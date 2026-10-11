"""Passage-level embeddings: long units split into sub-passages that keep their context.

A single vector for a long comma averages its parts: a comma with five lettere, or an allegato
table of 30,000 characters (cut by the model at 512 tokens), is far from a question about any one
of them. Each unit version is therefore also embedded as smaller passages, and semantic search
scores a version by its best passage (max-sim):

- ordinal 0 is the whole unit, the same string as the unit embedding (`passage_text`);
- a paragraph with two or more lettere gives its intro alone and one passage per lettera made of
  heading + intro + that lettera (the intro is clipped when it is long);
- other text longer than `SPLIT_MIN_CHARS` gives overlapping windows of about `WINDOW_CHARS`,
  cut at line or sentence boundaries, each prefixed with the heading.

Passages are stored in `unit_passages`, one row per (version, ordinal), and embedded once per
distinct text. `PASSAGE_SCHEME` names the splitting rules: rows built with other rules are rebuilt.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import psycopg

from appalti_qa.keys import LATIN_SUFFIXES
from appalti_qa.retrieval.embedding import Embedder, passage_text

logger = logging.getLogger(__name__)

PASSAGE_SCHEME = "v1"  # bump when the splitting rules change: existing passages are rebuilt
WINDOW_CHARS = 1000  # target size of a window (and of a lettera passage's own text)
WINDOW_OVERLAP = 200  # at most this many characters repeated between consecutive windows
SPLIT_MIN_CHARS = 1200  # text without lettere is windowed only above this length
INTRO_MAX_CHARS = 500  # the intro repeated in each lettera passage is clipped to this
ELLIPSIS = " […]"

_SUFFIX = rf"(?:-(?:{'|'.join(LATIN_SUFFIXES)}))?"
_LETTERA_RE = re.compile(rf"^\s*[a-z]{{1,2}}{_SUFFIX}\)\s")
_PARAGRAPH_RE = re.compile(rf"^\s*\d+{_SUFFIX}\.\s")
# Break points for windows: line ends, and sentence or clause ends followed by whitespace.
_BREAK_RE = re.compile(r"\n+|(?<=[.;:])\s+")


@dataclass(frozen=True, slots=True)
class _Group:
    """A paragraph: the lines from one numbered line ("2. ...") to the next."""

    intro: list[str]
    lettere: list[list[str]]  # each lettera: its line and the lines that continue it


def split_passages(heading: str | None, text: str) -> list[str]:
    """The passages embedded for a unit version, whole unit first, without duplicates."""
    passages = [passage_text(heading, text)]
    groups = _paragraphs(text)
    if len(text) <= SPLIT_MIN_CHARS and not any(len(g.lettere) >= 2 for g in groups):
        return passages
    plain: list[str] = []  # consecutive text without lettere, windowed together

    def flush() -> None:
        if plain:
            passages.extend(passage_text(heading, w) for w in windows("\n".join(plain)))
            plain.clear()

    for group in groups:
        if len(group.lettere) < 2:
            plain.extend(group.intro)
            plain.extend(line for lettera in group.lettere for line in lettera)
            continue
        flush()
        intro = "\n".join(group.intro)
        if intro:
            passages.extend(passage_text(heading, w) for w in windows(intro))
        context = passage_text(heading, clip(intro, INTRO_MAX_CHARS))
        for lettera in group.lettere:
            for piece in windows("\n".join(lettera)):
                passages.append(f"{context}\n{piece}" if context else piece)
    flush()
    return list(dict.fromkeys(passages))


def _paragraphs(text: str) -> list[_Group]:
    """Lines grouped into paragraphs, each with its intro and its lettere."""
    groups: list[_Group] = []
    for line in (line for line in text.split("\n") if line.strip()):
        if not groups or (_PARAGRAPH_RE.match(line) and (groups[-1].intro or groups[-1].lettere)):
            groups.append(_Group([], []))
        group = groups[-1]
        if _LETTERA_RE.match(line):
            group.lettere.append([line])
        elif group.lettere:
            group.lettere[-1].append(line)
        else:
            group.intro.append(line)
    return groups


def clip(text: str, limit: int) -> str:
    """`text` cut at a word boundary to at most `limit` characters (plus an ellipsis)."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip() + ELLIPSIS


def windows(text: str, size: int = WINDOW_CHARS, overlap: int = WINDOW_OVERLAP) -> list[str]:
    """Overlapping windows of at most `size` characters, cut at line or sentence boundaries.

    A sentence longer than `size` is cut at word boundaries. Consecutive windows share at most
    `overlap` characters (whole sentences), so every sentence is in some window.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    spans = _pieces(text, size)
    out: list[str] = []
    start = 0
    while start < len(spans):
        end = start + 1
        while end < len(spans) and spans[end][1] - spans[start][0] <= size:
            end += 1
        out.append(text[spans[start][0] : spans[end - 1][1]])
        if end == len(spans):
            break
        # The next window starts at the earliest piece that keeps the overlap within bounds.
        nxt = next(
            (i for i in range(start + 1, end) if spans[end - 1][1] - spans[i][0] <= overlap), end
        )
        start = nxt
    return out


def _pieces(text: str, size: int) -> list[tuple[int, int]]:
    """(start, end) spans of the sentences of `text`; longer ones cut at whitespace."""
    spans: list[tuple[int, int]] = []
    position = 0
    for match in [*_BREAK_RE.finditer(text), None]:
        end = match.start() if match else len(text)
        if end > position:
            spans.extend(_cut(text, position, end, size))
        if match:
            position = match.end()
    return spans


def _cut(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    spans = []
    while end - start > size:
        cut = text.rfind(" ", start + 1, start + size)
        cut = cut if cut > start else start + size
        spans.append((start, cut))
        start = cut + 1 if text[cut] == " " else cut
    spans.append((start, end))
    return spans


# --- storage ----------------------------------------------------------------------------------

Progress = Callable[[int, int], None]

_STALE_SQL = """
    SELECT v.id, v.heading, v.text FROM unit_versions v
    WHERE NOT EXISTS (SELECT 1 FROM unit_passages p WHERE p.version_id = v.id)
       OR EXISTS (SELECT 1 FROM unit_passages p WHERE p.version_id = v.id AND p.scheme <> %s)
    ORDER BY v.id
"""
_INSERT_SQL = """
    INSERT INTO unit_passages (version_id, ordinal, text, text_hash, scheme)
    VALUES (%s, %s, %s, %s, %s)
"""
# The whole-unit passage is the string the unit embedding was made from: reuse that vector.
_REUSE_UNIT_SQL = """
    UPDATE unit_passages p SET embedding = v.embedding, embedding_model = v.embedding_model
    FROM unit_versions v
    WHERE p.version_id = v.id AND p.ordinal = 0 AND v.embedding_model = %(model)s
      AND (p.embedding IS NULL OR p.embedding_model IS DISTINCT FROM %(model)s)
"""
_PENDING_SQL = """
    SELECT text_hash, min(text) AS text FROM unit_passages
    WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %(model)s
    GROUP BY text_hash ORDER BY min(id)
    LIMIT %(limit)s
"""
_UPDATE_SQL = """
    UPDATE unit_passages SET embedding = %(embedding)s, embedding_model = %(model)s
    WHERE text_hash = %(text_hash)s
      AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %(model)s)
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_passages(conn: psycopg.Connection) -> int:
    """Split the versions that have no passages, or passages of another scheme; commits.

    Returns the number of versions (re)built. Embeddings are left empty for `embed_passages`.
    """
    rows = conn.execute(_STALE_SQL, (PASSAGE_SCHEME,)).fetchall()
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM unit_passages WHERE version_id = ANY(%s)", ([row[0] for row in rows],)
        )
        params = [
            (version_id, ordinal, passage, _hash(passage), PASSAGE_SCHEME)
            for version_id, heading, text in rows
            for ordinal, passage in enumerate(split_passages(heading, text))
        ]
        cur.executemany(_INSERT_SQL, params)
    conn.commit()
    if rows:
        logger.info("Split %d versions into %d passages", len(rows), len(params))
    return len(rows)


def embed_passages(
    conn: psycopg.Connection,
    embedder: Embedder,
    *,
    batch_size: int = 32,
    limit: int | None = None,
    progress: Progress | None = None,
) -> int:
    """Embed passages with no embedding, or one from another model; commits after each batch.

    Whole-unit passages take the unit embedding when it comes from the same model. Every other
    distinct passage text is embedded once. `limit` caps the texts embedded. Returns the number
    of distinct texts embedded.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = embedder.model_name
    conn.execute(_REUSE_UNIT_SQL, {"model": model})
    pending = conn.execute(_PENDING_SQL, {"model": model, "limit": limit}).fetchall()
    conn.commit()
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_passages([text for _, text in batch])
        params = [
            {"embedding": vector, "model": model, "text_hash": text_hash}
            for (text_hash, _), vector in zip(batch, vectors, strict=True)
        ]
        with conn.cursor() as cur:
            cur.executemany(_UPDATE_SQL, params)
        conn.commit()
        if progress:
            progress(start + len(batch), len(pending))
    logger.info("Embedded %d passage texts with %s", len(pending), model)
    return len(pending)


def max_sim(scores: Sequence[tuple[int, float]]) -> dict[int, float]:
    """Version id -> best passage score, best first (ties keep first-seen order)."""
    best: dict[int, float] = {}
    for version_id, score in scores:
        if version_id not in best or score > best[version_id]:
            best[version_id] = score
    return dict(sorted(best.items(), key=lambda pair: -pair[1]))


# --- hubness -----------------------------------------------------------------------------------

HUB_NEIGHBOURS = 10  # probes averaged per passage (the K of CSLS)
_PROBES_SQL = """
    SELECT DISTINCT btrim(heading) FROM unit_versions
    WHERE heading IS NOT NULL AND btrim(heading) <> '' ORDER BY 1
"""


def hub_scores(passages: np.ndarray, probes: np.ndarray, k: int = HUB_NEIGHBOURS) -> np.ndarray:
    """Mean cosine similarity of each passage to its `k` most similar probe queries.

    This is the r(p) term of cross-domain similarity local scaling (CSLS, Conneau et al., 2018):
    a "hub" is close to every query, so its neighbourhood similarity is high. Inputs are
    L2-normalised rows; the result has one value per passage.
    """
    if len(passages) == 0 or len(probes) == 0:
        return np.zeros(len(passages), dtype=np.float32)
    k = min(k, len(probes))
    sims = passages @ probes.T
    nearest = np.partition(sims, len(probes) - k, axis=1)[:, len(probes) - k :]
    return nearest.mean(axis=1).astype(np.float32)


def update_hub_scores(conn: psycopg.Connection, embedder: Embedder) -> int:
    """Recompute `hub_score` of every passage embedded by `embedder`'s model; commits.

    The probes are a fixed, dataset-independent sample of generic queries: the distinct headings
    of the corpus, embedded as queries. Returns the number of passages scored.
    """
    model = embedder.model_name
    headings = [row[0] for row in conn.execute(_PROBES_SQL).fetchall()]
    rows = conn.execute(
        "SELECT id, embedding FROM unit_passages WHERE embedding_model = %s ORDER BY id", (model,)
    ).fetchall()
    conn.commit()
    if not rows or not headings:
        return 0
    matrix = np.stack([np.asarray(row[1].to_numpy(), dtype=np.float32) for row in rows])
    scores = hub_scores(matrix, embedder.embed_queries(headings))
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE unit_passages SET hub_score = %s WHERE id = %s",
            [(float(score), row[0]) for score, row in zip(scores, rows, strict=True)],
        )
    conn.commit()
    logger.info("Hub scores of %d passages from %d probe queries", len(rows), len(headings))
    return len(rows)
