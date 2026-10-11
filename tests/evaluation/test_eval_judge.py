from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from appalti_qa.config import Settings
from appalti_qa.evaluation.dataset import EvalItem
from appalti_qa.evaluation.judge import (
    JudgeError,
    JudgeVerdict,
    LLMJudge,
    OverlapJudge,
    content_tokens,
    get_judge,
    parse_verdict,
)


@dataclass
class StubProvider:
    """Records `complete` calls and returns a fixed reply."""

    reply: str
    name: str = "anthropic"
    model: str = "claude-opus-5-5"
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> SimpleNamespace:
        self.calls.append((system, prompt, max_tokens))
        return SimpleNamespace(
            text=self.reply,
            provider=self.name,
            model=self.model,
            input_tokens=2000,
            output_tokens=100,
        )


def test_content_tokens() -> None:
    tokens = content_tokens("La tempestivita' e la qualità [1] per 140.000 euro.")
    assert tokens == {"tempes", "qualit", "140000", "euro"}


def test_overlap_judge_scores(items: list[EvalItem], make_answer: Callable) -> None:
    judge = OverlapJudge()
    item = items[0]
    full = make_answer(item, item.expected_answer, ["50/1"])
    partial = make_answer(item, "Affidamento diretto di servizi.", ["50/1"])
    wrong = make_answer(item, "Si applica il principio di rotazione.", ["49/1"])
    assert judge.grade(item, full).score == 1.0
    assert judge.grade(item, partial).score == 0.5
    verdict = judge.grade(item, wrong)
    assert verdict.score == 0.0 and verdict.label == "wrong"
    assert "Copertura lessicale" in verdict.rationale


def test_abstentions(items: list[EvalItem], make_answer: Callable) -> None:
    judge = OverlapJudge()
    answerable, unanswerable = items[0], items[2]
    assert judge.grade(unanswerable, make_answer(unanswerable, "-", abstained=True)).score == 1.0
    assert judge.grade(answerable, make_answer(answerable, "-", abstained=True)).score == 0.0
    answered = make_answer(unanswerable, unanswerable.expected_answer, ["1/1"])
    assert judge.grade(unanswerable, answered).score == 0.0


@pytest.mark.parametrize(
    ("reply", "score", "reason"),
    [
        ('{"verdict": "correct", "reason": "Completa."}', 1.0, "Completa."),
        ('```json\n{"verdict": "partial", "reason": "Manca la soglia."}\n```', 0.5, "Manca"),
        ('Ecco il giudizio: {"verdict": "Errata", "reason": "No."}', 0.0, "No."),
    ],
)
def test_parse_verdict(reply: str, score: float, reason: str) -> None:
    parsed_score, parsed_reason = parse_verdict(reply)
    assert parsed_score == score and parsed_reason.startswith(reason)


@pytest.mark.parametrize("reply", ["nessun JSON", '{"verdict": "maybe"}', "{not json}", "[1]"])
def test_parse_verdict_errors(reply: str) -> None:
    with pytest.raises(JudgeError):
        parse_verdict(reply)


def test_llm_judge(items: list[EvalItem], make_answer: Callable) -> None:
    provider = StubProvider('{"verdict": "partial", "reason": "Omette la soglia."}')
    judge = LLMJudge(provider)
    item = items[0]
    verdict = judge.grade(item, make_answer(item, "Risposta del sistema [1].", ["50/1"]))
    assert verdict == JudgeVerdict(0.5, "Omette la soglia.", 2000, 100, pytest.approx(0.01))
    system, prompt, max_tokens = provider.calls[0]
    assert '"verdict": "correct" | "partial" | "wrong"' in system
    assert "Data di riferimento: 01/01/2025" in prompt
    assert item.expected_answer in prompt and "Risposta del sistema [1]." in prompt
    assert max_tokens >= 1024
    assert judge.model == "claude-opus-5-5"


def test_llm_judge_skips_abstentions(items: list[EvalItem], make_answer: Callable) -> None:
    provider = StubProvider("unused")
    item = items[2]
    verdict = LLMJudge(provider).grade(item, make_answer(item, "-", abstained=True))
    assert verdict.score == 1.0 and provider.calls == []


def test_get_judge_offline(settings: Settings) -> None:
    assert isinstance(get_judge(settings), OverlapJudge)
    without_key = settings.model_copy(update={"judge_provider": "anthropic"})
    assert get_judge(without_key).name == "overlap"


def test_get_judge_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict] = []

    def fake_get_provider(settings: Settings, *, provider: str, model: str | None) -> StubProvider:
        requests.append({"provider": provider, "model": model})
        return StubProvider("{}", name=provider, model=model or "default")

    monkeypatch.setattr("appalti_qa.llm.get_provider", fake_get_provider)
    groq = Settings(_env_file=None, judge_provider="groq", groq_api_key="placeholder")
    judge = get_judge(groq)
    assert isinstance(judge, LLMJudge) and judge.name == "llm"
    assert requests == [{"provider": "groq", "model": None}]  # Claude model name not sent to Groq
