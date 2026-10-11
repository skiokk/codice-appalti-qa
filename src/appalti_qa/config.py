"""Application settings, read from environment variables and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["fake", "anthropic", "groq"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
RetrievalMode = Literal["lexical", "semantic", "hybrid"]
SemanticGranularity = Literal["unit", "passage"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://appalti:appalti@localhost:5433/appalti"
    db_pool_min_size: int = Field(default=1, ge=0)
    db_pool_max_size: int = Field(default=10, ge=1)

    # Corpus
    raw_dir: Path = Path("data/raw/akn")

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = Field(default=768, gt=0)
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)

    # Answer generation
    llm_provider: ProviderName = "fake"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Effort = "medium"
    llm_max_tokens: int = Field(default=16000, gt=0)  # a ceiling; reasoning tokens count too
    llm_timeout_s: float = Field(default=60.0, gt=0)
    groq_model: str = "openai/gpt-oss-120b"
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # LLM judge used by the evaluation
    judge_provider: ProviderName = "fake"
    judge_model: str = "claude-opus-5-5"

    # Retrieval
    default_top_k: int = Field(default=8, gt=0, le=50)
    retrieval_mode: RetrievalMode = "hybrid"
    rrf_k: int = Field(default=60, gt=0)
    fusion_depth: int = Field(default=4, gt=0)  # hybrid fuses two lists of fusion_depth * k
    lexical_weight: float = Field(default=1.0, ge=0)  # weight of the lexical list in RRF
    semantic_granularity: SemanticGranularity = "unit"  # "passage": best sub-passage (max-sim)
    hubness_weight: float = Field(default=0.0, ge=0)  # passage score -= weight * hub_score
    min_relevance_score: float = 0.0

    # Public demo limits
    max_question_chars: int = Field(default=500, gt=0)
    rate_limit_per_ip_per_hour: int = Field(default=20, ge=0)
    rate_limit_global_per_day: int = Field(default=500, ge=0)
    # GET /api/v1/search embeds each query on CPU: limited per client, more generously
    rate_limit_search_per_ip_per_hour: int = Field(default=120, ge=0)
    client_hash_salt: SecretStr | None = (
        None  # salt for hashing client IPs; random per process if unset
    )

    # Comma-separated IPs/CIDRs of trusted reverse proxies; empty = ignore X-Forwarded-For.
    # "*" trusts any peer but then only the rightmost X-Forwarded-For hop is used.
    forwarded_allow_ips: str = ""

    log_level: str = "INFO"

    def api_key(self, provider: ProviderName) -> str | None:
        """The configured key for a provider, or None when unset or blank."""
        secret = {"anthropic": self.anthropic_api_key, "groq": self.groq_api_key}.get(provider)
        value = secret.get_secret_value().strip() if secret else ""
        return value or None

    def effective_provider(self, provider: ProviderName | None = None) -> ProviderName:
        """The provider actually used: falls back to "fake" when the requested one has no key."""
        requested = provider or self.llm_provider
        if requested == "fake" or self.api_key(requested) is None:
            return "fake"
        return requested


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
