# Evaluation report `20261011T161819Z_retrieval`

Created 2026-10-11T16:18:19Z. Dataset `eval/dataset.yaml`: 81 questions evaluated (selection: all); 0 of 81 questions in the file are verified.

> **Draft questions included.** 81 of the evaluated questions are not verified yet (status `bozza`): the numbers are indicative and not for publication.

## Configuration

| Setting | Value |
| --- | --- |
| retrieval_mode | hybrid |
| modes | ['lexical', 'semantic', 'hybrid'] |
| k_values | [1, 3, 5, 10] |
| embedding_model | intfloat/multilingual-e5-base |

## Retrieval

Questions with expected references only, each at its own reference date. Comma level: an expected ref is found when a retrieved unit matches it (`50/1/b` is found by `50/1`). Article level: a retrieved unit of the same article is enough.

| Mode | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical | 73 | 19.2% | 40.4% | 54.6% | 75.1% | 0.355 | 61.0% | 78.8% | 0.387 |
| semantic | 73 | 53.4% | 71.9% | 79.9% | 90.0% | 0.684 | 88.4% | 93.2% | 0.783 |
| hybrid | 73 | 66.0% | 81.7% | 83.6% | 88.6% | 0.786 | 87.0% | 91.8% | 0.827 |

### Retrieval by kind (hybrid)

| Kind | n | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| answerable | 52 | 88.5% | 92.3% | 0.784 | 90.4% | 92.3% | 0.826 |
| temporal | 12 | 83.3% | 83.3% | 0.833 | 83.3% | 91.7% | 0.844 |
| multi | 9 | 55.6% | 74.1% | 0.732 | 72.2% | 88.9% | 0.806 |

### Retrieval by category (hybrid)

| Category | n | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| affidamento_diretto | 4 | 58.3% | 66.7% | 0.750 | 62.5% | 75.0% | 0.750 |
| concessioni | 2 | 100.0% | 100.0% | 0.667 | 100.0% | 100.0% | 0.667 |
| contenzioso | 3 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| criteri_aggiudicazione | 3 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| digitalizzazione | 4 | 75.0% | 100.0% | 0.531 | 75.0% | 100.0% | 0.656 |
| esecuzione | 15 | 82.2% | 82.2% | 0.783 | 86.7% | 93.3% | 0.792 |
| garanzie | 4 | 75.0% | 91.7% | 0.708 | 87.5% | 100.0% | 1.000 |
| incentivi | 1 | 100.0% | 100.0% | 0.333 | 100.0% | 100.0% | 1.000 |
| principi | 5 | 80.0% | 80.0% | 0.800 | 80.0% | 80.0% | 0.800 |
| procedure | 9 | 88.9% | 88.9% | 0.759 | 88.9% | 88.9% | 0.778 |
| progettazione | 2 | 50.0% | 50.0% | 0.250 | 50.0% | 50.0% | 0.250 |
| programmazione | 1 | 0.0% | 0.0% | 0.000 | 0.0% | 0.0% | 0.000 |
| qualificazione | 1 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| requisiti | 6 | 100.0% | 100.0% | 0.917 | 100.0% | 100.0% | 0.917 |
| rotazione | 2 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| rup | 5 | 73.3% | 100.0% | 0.829 | 90.0% | 100.0% | 0.900 |
| soccorso_istruttorio | 1 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| soglie | 2 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| subappalto | 3 | 88.9% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
