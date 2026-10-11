# LLM and answers

Package `appalti_qa.llm` generates cited answers from retrieved units; `appalti_qa.answer`
orchestrates retrieval, generation and abstention.

| Module | Content |
| --- | --- |
| `llm/base.py` | `Generation`, `LLMProvider`, `LLMError` |
| `llm/prompts.py` | Italian system prompt, `ABSTAIN_TOKEN = "NON_TROVATO"`, source titles and context |
| `llm/anthropic_provider.py` | Claude with native citations |
| `llm/groq_provider.py` | open-weight models on Groq, numbered-marker citations |
| `llm/fake.py` | deterministic offline provider |
| `llm/spans.py` | sentence spans that are exact substrings (abbreviation-aware: "art. 50", "n. 36") |
| `llm/factory.py` | `get_provider` |
| `answer/service.py` | `answer_question`, `ABSTAIN_MESSAGE` |

## Providers

- **Anthropic** (`LLM_PROVIDER=anthropic`, default model `claude-opus-5-5`): each retrieved unit
  is a plain-text `document` block with citations enabled, titled with its label and the
  reference date; its context carries the heading, the validity interval and the amending acts.
  The API returns each cited span with its document index, so citations are exact by
  construction. `output_config.effort` comes from `LLM_EFFORT`; the system prompt is a stable,
  cache-marked block. The request opts in to server-side refusal fallback, and a refusal that
  survives it is an abstention.
- **Groq** (`LLM_PROVIDER=groq`, default `openai/gpt-oss-120b`): sources are numbered `[1]..[n]`
  in the prompt; `[n]` markers (and variants such as `[1, 3]` or `【1†...】`) are parsed into
  citations, out-of-range markers are dropped, `<think>` blocks are stripped. The quoted span is
  best effort: the source sentence with the most content words in common with the answer
  sentence carrying the marker.
- **Fake** (default, and whenever the selected provider has no key): quotes the first sentence of
  the top one or two sources with markers; abstains with no sources.

SDK errors (timeouts, connection, auth, rate limits, 4xx/5xx) become `LLMError` with messages that
never contain keys or prompts. `LLM_MAX_TOKENS` (default 16000) is a ceiling: reasoning tokens
count against it on both providers.

## Answer flow

1. Validate the question (Italian `ValueError` if empty or longer than `MAX_QUESTION_CHARS`).
2. Retrieve `DEFAULT_TOP_K` units in force at the date (`RETRIEVAL_MODE`, hybrid by default).
3. No hits, or best score below `MIN_RELEVANCE_SCORE`: abstain without calling the model.
4. Generate; drop citations to unknown sources and stray markers; abstain when the model answers
   `NON_TROVATO` or no valid citation remains.
5. Abstained answers keep their sources for transparency. `latency_ms` covers retrieval and
   generation.
