"""Answer correctness judges: an LLM with an Italian rubric, or a deterministic overlap score."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from appalti_qa.config import Settings, get_settings
from appalti_qa.evaluation.dataset import EvalItem
from appalti_qa.evaluation.pricing import cost_usd
from appalti_qa.models import Answer

if TYPE_CHECKING:
    from appalti_qa.llm import LLMProvider

SCORES: dict[str, float] = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
_ALIASES = {"corretta": "correct", "parziale": "partial", "errata": "wrong", "sbagliata": "wrong"}


class JudgeError(ValueError):
    """The judge did not return a usable verdict."""


@dataclass(frozen=True)
class JudgeVerdict:
    score: float  # 1.0 correct, 0.5 partial, 0.0 wrong
    rationale: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0

    @property
    def label(self) -> str:
        return min(SCORES, key=lambda name: abs(SCORES[name] - self.score))


class Judge(Protocol):
    name: str

    def grade(self, item: EvalItem, answer: Answer) -> JudgeVerdict: ...


def abstention_verdict(item: EvalItem, answer: Answer) -> JudgeVerdict | None:
    """Verdicts that need no grading: abstentions, right or wrong."""
    if not answer.abstained:
        return None
    if item.kind == "unanswerable":
        return JudgeVerdict(1.0, "Astensione corretta: il codice non risponde alla domanda.")
    return JudgeVerdict(0.0, "Il sistema si è astenuto, ma la risposta è nel testo del codice.")


# --- deterministic judge ----------------------------------------------------------------------

_STOPWORD_TEXT = """
a ad agli ai al alla alle allo anche che chi ci con cui da dagli dai dal dalla dalle dallo deve
devono degli dei del della delle dello di e ed essere gli ha hanno i il in la le lo ma nei nel
nella nelle nello non o per piu puo possono quale quali quando se si sia sono su sui sul sulla
sulle tra un una uno dell dall nell sull all
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_STEM_CHARS = 6


def content_tokens(text: str) -> set[str]:
    """Accent-folded, stopword-free word stems (first six letters); citation markers dropped."""
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"(?<=\d)\.(?=\d{3})", "", text)  # 140.000 -> 140000
    folded = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return {
        word if word.isdigit() else word[:_STEM_CHARS]
        for word in re.findall(r"[a-z0-9]+", folded)
        if word.isdigit() or (len(word) > 2 and word not in _STOPWORDS)
    }


class OverlapJudge:
    """Share of the expected answer's content words found in the answer.

    A rough, offline proxy for correctness, used when no judge model is available: answers that
    paraphrase correctly can score low and answers that copy the source text can score high.
    """

    name = "overlap"

    def __init__(self, correct_at: float = 0.5, partial_at: float = 0.25) -> None:
        self.correct_at = correct_at
        self.partial_at = partial_at

    def grade(self, item: EvalItem, answer: Answer) -> JudgeVerdict:
        verdict = abstention_verdict(item, answer)
        if verdict is not None:
            return verdict
        if item.kind == "unanswerable":
            return JudgeVerdict(
                0.0, "Il sistema ha risposto a una domanda a cui il codice non risponde."
            )
        expected = content_tokens(item.expected_answer)
        if not expected:
            return JudgeVerdict(0.0, "Risposta attesa vuota.")
        overlap = len(expected & content_tokens(answer.text)) / len(expected)
        if overlap >= self.correct_at:
            score = 1.0
        elif overlap >= self.partial_at:
            score = 0.5
        else:
            score = 0.0
        return JudgeVerdict(score, f"Copertura lessicale della risposta attesa: {overlap:.0%}.")


# --- LLM judge --------------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
Sei un valutatore esperto di contratti pubblici italiani (D.Lgs. 31 marzo 2023, n. 36). \
Ricevi una domanda, la data di riferimento, la risposta attesa scritta da un esperto, un estratto \
del testo normativo vigente a quella data e la risposta di un sistema automatico. Giudica se la \
risposta del sistema è corretta rispetto alla risposta attesa.

Criteri:
- "correct": contiene gli elementi essenziali della risposta attesa e nessuna affermazione in \
contrasto con essa o con l'estratto;
- "partial": è in parte corretta ma omette un elemento essenziale o è imprecisa su un dettaglio \
rilevante (importi, termini, soggetti, condizioni, eccezioni);
- "wrong": è errata, contraddice la risposta attesa, si basa su un testo non vigente alla data \
indicata, oppure non risponde alla domanda.
Se la risposta attesa dice che il codice non disciplina la questione, la risposta del sistema è \
"correct" solo se lo dichiara senza inventare contenuti.
Non premiare la lunghezza o lo stile. Ignora i marcatori di citazione come [1].

Rispondi solo con un oggetto JSON, senza altro testo:
{"verdict": "correct" | "partial" | "wrong", "reason": "una o due frasi in italiano"}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> tuple[float, str]:
    """(score, reason) from the judge's JSON reply; raises JudgeError when it is unusable."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise JudgeError(f"no JSON object in the judge reply: {text[:200]!r}")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON in the judge reply: {exc}") from None
    if not isinstance(data, dict):
        raise JudgeError("the judge reply is not a JSON object")
    verdict = str(data.get("verdict", "")).strip().lower()
    verdict = _ALIASES.get(verdict, verdict)
    if verdict not in SCORES:
        raise JudgeError(f"unknown verdict {verdict!r}")
    return SCORES[verdict], str(data.get("reason", "")).strip()


def judge_prompt(item: EvalItem, answer: Answer) -> str:
    return (
        f"Data di riferimento: {item.vigenza:%d/%m/%Y}\n\n"
        f"<domanda>\n{item.question}\n</domanda>\n\n"
        f"<risposta_attesa>\n{item.expected_answer}\n</risposta_attesa>\n\n"
        f"<estratto_normativo>\n{item.source_excerpt or '(nessuno)'}\n</estratto_normativo>\n\n"
        f"<risposta_sistema>\n{answer.text}\n</risposta_sistema>"
    )


class LLMJudge:
    """Grades with an LLM (`LLMProvider.complete`) and the Italian rubric above."""

    name = "llm"

    def __init__(self, provider: LLMProvider, *, max_tokens: int = 4096) -> None:
        self.provider = provider
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self.provider.model

    def grade(self, item: EvalItem, answer: Answer) -> JudgeVerdict:
        verdict = abstention_verdict(item, answer)
        if verdict is not None:
            return verdict
        generation = self.provider.complete(
            JUDGE_SYSTEM_PROMPT, judge_prompt(item, answer), max_tokens=self.max_tokens
        )
        score, reason = parse_verdict(generation.text)
        return JudgeVerdict(
            score,
            reason,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            cost_usd=cost_usd(
                generation.provider,
                generation.model,
                generation.input_tokens,
                generation.output_tokens,
            ),
        )


def get_judge(settings: Settings | None = None) -> Judge:
    """LLM judge for `settings.judge_provider`, or OverlapJudge when it resolves to fake."""
    settings = settings or get_settings()
    provider = settings.effective_provider(settings.judge_provider)
    if provider == "fake":
        return OverlapJudge()
    from appalti_qa.llm import get_provider

    # judge_model names a Claude model by default; Groq then uses its own default model.
    model: str | None = settings.judge_model
    if provider != "anthropic" and settings.judge_model.startswith("claude-"):
        model = None
    return LLMJudge(get_provider(settings, provider=provider, model=model))
