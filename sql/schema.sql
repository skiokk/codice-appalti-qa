-- Schema for codice-appalti-qa. Idempotent: safe to apply repeatedly.
-- {{EMBEDDING_DIM}} is replaced by appalti_qa.db.schema_sql() with Settings.embedding_dim.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- The consolidated act (one row: D.Lgs. 31 marzo 2023, n. 36).
CREATE TABLE IF NOT EXISTS acts (
    id          serial PRIMARY KEY,
    code        text NOT NULL UNIQUE,              -- Normattiva codice redazionale, "23G00044"
    urn         text NOT NULL UNIQUE,              -- "urn:nir:stato:decreto.legislativo:2023-03-31;36"
    title       text NOT NULL,
    short_title text,                              -- "D.Lgs. 36/2023"
    issued_on   date NOT NULL
);

-- Ingested Normattiva snapshots (one per distinct consolidated text).
CREATE TABLE IF NOT EXISTS snapshots (
    id              serial PRIMARY KEY,
    act_id          integer NOT NULL REFERENCES acts (id) ON DELETE CASCADE,
    vigenza_date    date NOT NULL,                 -- date from the file name (dataVigenza)
    effective_date  date NOT NULL,                 -- date used as valid_from for its changes
    source_file     text NOT NULL,
    sha256          char(64) NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (act_id, vigenza_date)
);

-- Amendment events from meta/lifecycle (the original act is an event with kind 'original').
CREATE TABLE IF NOT EXISTS events (
    id             serial PRIMARY KEY,
    act_id         integer NOT NULL REFERENCES acts (id) ON DELETE CASCADE,
    event_date     date NOT NULL,
    kind           text NOT NULL DEFAULT 'amendment' CHECK (kind IN ('original', 'amendment')),
    amending_urn   text NOT NULL,                  -- AKN href, "/akn/it/act/DECRETO-LEGGE/stato/2023-05-10/51/!main"
    amending_label text NOT NULL,                  -- "D.L. 10 maggio 2023, n. 51"
    UNIQUE (act_id, amending_urn, event_date)
);

-- Retrieval units: commi, articles without commi, allegato blocks. Key grammar in keys.py.
CREATE TABLE IF NOT EXISTS units (
    id        serial PRIMARY KEY,
    act_id    integer NOT NULL REFERENCES acts (id) ON DELETE CASCADE,
    key       text NOT NULL,                       -- "50/1", "225-bis", "all:I.1/2/1", "all:I.3/p3"
    kind      text NOT NULL CHECK (kind IN ('article', 'comma', 'allegato_block')),
    article   text,                                -- "50", "225-bis"; article inside the allegato for annex units
    comma     text,                                -- "1", "1-bis"
    allegato  text,                                -- "I.1"; NULL for units of the code body
    heading   text,                                -- latest article (or allegato) heading
    ordinal   integer NOT NULL,                    -- document order across the whole act
    UNIQUE (act_id, key)
);
CREATE INDEX IF NOT EXISTS units_article_idx ON units (act_id, allegato, article);

-- Point-in-time text of each unit. In force at D: valid_from <= D AND (valid_to IS NULL OR valid_to > D).
CREATE TABLE IF NOT EXISTS unit_versions (
    id               bigserial PRIMARY KEY,
    unit_id          integer NOT NULL REFERENCES units (id) ON DELETE CASCADE,
    heading          text,                         -- article heading as of this version
    text             text NOT NULL,
    text_hash        char(64) NOT NULL,            -- sha256 hex of text (UTF-8)
    valid_from       date NOT NULL,
    valid_to         date,                         -- exclusive; NULL = still in force
    changed_by_event integer REFERENCES events (id) ON DELETE SET NULL,  -- NULL for the original text
    repealed         boolean NOT NULL DEFAULT false,
    tsv              tsvector GENERATED ALWAYS AS (
                         setweight(to_tsvector('italian'::regconfig, coalesce(heading, '')), 'A')
                         || setweight(to_tsvector('italian'::regconfig, text), 'B')
                     ) STORED,
    embedding        vector({{EMBEDDING_DIM}}),
    embedding_model  text,                         -- model that produced the embedding
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    EXCLUDE USING gist (unit_id WITH =, daterange(valid_from, valid_to, '[)') WITH &&)
);
CREATE INDEX IF NOT EXISTS unit_versions_unit_idx ON unit_versions (unit_id, valid_from);
CREATE INDEX IF NOT EXISTS unit_versions_hash_idx ON unit_versions (text_hash);
CREATE INDEX IF NOT EXISTS unit_versions_tsv_idx ON unit_versions USING gin (tsv);
CREATE INDEX IF NOT EXISTS unit_versions_embedding_idx
    ON unit_versions USING hnsw (embedding vector_cosine_ops);

-- Answered questions (public demo usage, rate limiting, latency and cost tracking).
CREATE TABLE IF NOT EXISTS qa_log (
    id             bigserial PRIMARY KEY,
    created_at     timestamptz NOT NULL DEFAULT now(),
    client_hash    char(64),                       -- salted sha256 of the client IP, never the IP itself
    question       text NOT NULL,
    at_date        date NOT NULL,
    provider       text NOT NULL,
    model          text NOT NULL,
    abstained      boolean NOT NULL,
    cited_keys     text[] NOT NULL DEFAULT '{}',
    input_tokens   integer NOT NULL DEFAULT 0,
    output_tokens  integer NOT NULL DEFAULT 0,
    latency_ms     integer NOT NULL DEFAULT 0,
    error          text
);
CREATE INDEX IF NOT EXISTS qa_log_created_idx ON qa_log (created_at);
CREATE INDEX IF NOT EXISTS qa_log_client_idx ON qa_log (client_hash, created_at);
