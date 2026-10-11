import argparse
from collections.abc import Sequence

import numpy as np
import psycopg
import pytest
from pgvector import Vector

from appalti_qa.retrieval import FakeEmbedder, cli_embed, embed_missing, passage_text

pytestmark = pytest.mark.db

SEED_VERSIONS = 5


class _CountingEmbedder(FakeEmbedder):
    def __init__(self, model_name: str = "fake") -> None:
        super().__init__()
        self.model_name = model_name
        self.passages: list[str] = []

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.passages.extend(texts)
        return super().embed_passages(texts)


def _add_duplicate_of_1_1(conn: psycopg.Connection) -> None:
    """A second unit whose only version has the same heading and text as 1/1."""
    conn.execute(
        """
        WITH source AS (
            SELECT u.act_id, v.heading, v.text, v.text_hash, v.valid_from
            FROM units u JOIN unit_versions v ON v.unit_id = u.id WHERE u.key = '1/1'
        ), unit AS (
            INSERT INTO units (act_id, key, kind, article, comma, heading, ordinal)
            SELECT act_id, '2/1', 'comma', '2', '1', heading, 99 FROM source RETURNING id
        )
        INSERT INTO unit_versions (unit_id, heading, text, text_hash, valid_from)
        SELECT unit.id, heading, text, text_hash, valid_from FROM unit, source
        """
    )
    conn.commit()


def _embedding_state(conn: psycopg.Connection) -> list[tuple[str, str | None, Vector | None]]:
    return conn.execute(
        "SELECT u.key, v.embedding_model, v.embedding FROM unit_versions v"
        " JOIN units u ON u.id = v.unit_id ORDER BY v.id"
    ).fetchall()


def test_embeds_each_distinct_passage_once(seeded_conn: psycopg.Connection) -> None:
    _add_duplicate_of_1_1(seeded_conn)
    embedder = _CountingEmbedder()

    updated = embed_missing(seeded_conn, embedder, batch_size=2)

    assert updated == SEED_VERSIONS + 1
    assert len(embedder.passages) == SEED_VERSIONS  # the duplicate shares 1/1's embedding
    heading, text = seeded_conn.execute(
        "SELECT v.heading, v.text FROM unit_versions v JOIN units u ON u.id = v.unit_id"
        " WHERE u.key = '1/1'"
    ).fetchone()
    assert passage_text(heading, text) in embedder.passages

    state = _embedding_state(seeded_conn)
    assert all(model == "fake" and vector is not None for _, model, vector in state)
    vectors = {key: vector.to_numpy() for key, _, vector in state}
    assert np.array_equal(vectors["1/1"], vectors["2/1"])
    assert vectors["1/1"].shape == (768,)

    assert embed_missing(seeded_conn, embedder) == 0
    assert len(embedder.passages) == SEED_VERSIONS


def test_reembeds_versions_from_another_model(seeded_conn: psycopg.Connection) -> None:
    embed_missing(seeded_conn, _CountingEmbedder())
    other = _CountingEmbedder("another-model")
    assert embed_missing(seeded_conn, other) == SEED_VERSIONS
    assert {model for _, model, _ in _embedding_state(seeded_conn)} == {"another-model"}


def test_limit_and_progress(seeded_conn: psycopg.Connection) -> None:
    calls: list[tuple[int, int]] = []
    progress = lambda done, total: calls.append((done, total))  # noqa: E731
    assert embed_missing(seeded_conn, FakeEmbedder(), limit=3, batch_size=2, progress=progress) == 3
    assert calls == [(2, 3), (3, 3)]
    pending = [key for key, model, _ in _embedding_state(seeded_conn) if model is None]
    assert len(pending) == SEED_VERSIONS - 3
    with pytest.raises(ValueError):
        embed_missing(seeded_conn, FakeEmbedder(), batch_size=0)


def test_cli_embed_with_the_fake_model(
    seeded_conn: psycopg.Connection,
    db_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")

    assert cli_embed(argparse.Namespace(batch_size=4, limit=None)) == 0
    output = capsys.readouterr().out
    assert "4/5 passages" in output and "5/5 passages" in output
    assert f"Done: {SEED_VERSIONS} versions updated." in output

    assert cli_embed(argparse.Namespace(batch_size=4, limit=None)) == 0
    assert "Nothing to embed." in capsys.readouterr().out


def test_cli_embed_reports_an_unreachable_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")
    assert cli_embed(argparse.Namespace(batch_size=4, limit=None)) == 1
    assert "Database not reachable" in capsys.readouterr().out
