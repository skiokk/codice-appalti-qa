# Evaluation report `20261011T205828Z_anthropic_claude-opus-5-5`

Created 2026-10-11T20:58:28Z. Dataset `eval/dataset.yaml`: 81 questions evaluated (selection: all); 0 of 81 questions in the file are verified.

> **Draft questions included.** 81 of the evaluated questions are not verified yet (status `bozza`): the numbers are indicative and not for publication.

## Configuration

| Setting | Value |
| --- | --- |
| provider | anthropic |
| model | claude-opus-5-5 |
| retrieval_mode | hybrid |
| k | 8 |
| embedding_model | intfloat/multilingual-e5-base |
| judge | llm |
| judge_model | claude-opus-5-5 |
| effort | medium |
| rerank_model | claude-haiku-5-5 |

## Retrieval

Questions with expected references only, each at its own reference date. Comma level: an expected ref is found when a retrieved unit matches it (`50/1/b` is found by `50/1`). Article level: a retrieved unit of the same article is enough.

| Mode | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | 73 | 86.1% | 95.9% | 96.4% | 97.7% | 0.958 | 99.3% | 99.3% | 0.986 |

### Retrieval by kind (hybrid)

| Kind | n | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| answerable | 52 | 100.0% | 100.0% | 0.990 | 100.0% | 100.0% | 1.000 |
| temporal | 12 | 91.7% | 100.0% | 0.830 | 100.0% | 100.0% | 0.917 |
| multi | 9 | 81.5% | 81.5% | 0.944 | 94.4% | 94.4% | 1.000 |

### Retrieval by category (hybrid)

| Category | n | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| affidamento_diretto | 4 | 91.7% | 91.7% | 1.000 | 100.0% | 100.0% | 1.000 |
| concessioni | 2 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| contenzioso | 3 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| criteri_aggiudicazione | 3 | 100.0% | 100.0% | 0.778 | 100.0% | 100.0% | 1.000 |
| digitalizzazione | 4 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| esecuzione | 15 | 88.9% | 95.6% | 0.942 | 100.0% | 100.0% | 0.967 |
| garanzie | 4 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| incentivi | 1 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| principi | 5 | 100.0% | 100.0% | 0.900 | 100.0% | 100.0% | 1.000 |
| procedure | 9 | 92.6% | 92.6% | 0.944 | 94.4% | 94.4% | 1.000 |
| progettazione | 2 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| programmazione | 1 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| qualificazione | 1 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| requisiti | 6 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| rotazione | 2 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| rup | 5 | 100.0% | 100.0% | 0.900 | 100.0% | 100.0% | 0.900 |
| soccorso_istruttorio | 1 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| soglie | 2 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |
| subappalto | 3 | 100.0% | 100.0% | 1.000 | 100.0% | 100.0% | 1.000 |

## Answers

Correctness is the judge score (correct 1, partial 0.5, wrong 0). Citation precision and recall compare cited units with the expected refs (questions with refs only); abstention accuracy checks that exactly the unanswerable questions are declined. Costs use list prices without caching discounts; Groq models count as free tier.

| Metric | Value |
| --- | --- |
| Questions | 81 |
| Errors (answer / judge) | 0 / 0 |
| Correctness | 95.7% |
| Citation precision | 53.8% |
| Citation recall | 97.3% |
| Abstention accuracy | 96.3% |
| Abstention rate | 6.2% |
| Latency p50 / p95 (ms) | 7,817 / 13,495 |
| Tokens in / out | 509,464 / 45,279 |
| Cost of answers | $3.0851 |
| Cost of judging | $1.0139 |

### Answers by kind

| Kind | n | Correctness | Citation precision | Citation recall | Abstention accuracy | Latency p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| answerable | 52 | 97.1% | 52.1% | 100.0% | 100.0% | 7,368 |
| temporal | 12 | 95.8% | 58.3% | 100.0% | 100.0% | 7,271 |
| unanswerable | 8 | 100.0% | n/a | n/a | 62.5% | 5,325 |
| multi | 9 | 83.3% | 57.6% | 77.8% | 100.0% | 11,468 |

### Answers by category

| Category | n | Correctness | Citation precision | Citation recall | Abstention accuracy | Latency p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| affidamento_diretto | 5 | 100.0% | 54.2% | 91.7% | 80.0% | 9,483 |
| concessioni | 2 | 100.0% | 37.5% | 100.0% | 100.0% | 7,279 |
| contenzioso | 4 | 87.5% | 36.1% | 88.9% | 75.0% | 11,162 |
| criteri_aggiudicazione | 3 | 100.0% | 50.0% | 100.0% | 100.0% | 8,777 |
| digitalizzazione | 5 | 100.0% | 43.8% | 100.0% | 80.0% | 7,477 |
| esecuzione | 16 | 96.9% | 59.4% | 95.6% | 100.0% | 6,760 |
| garanzie | 4 | 100.0% | 52.1% | 100.0% | 100.0% | 11,468 |
| incentivi | 1 | 50.0% | 50.0% | 100.0% | 100.0% | 7,372 |
| principi | 6 | 100.0% | 60.0% | 100.0% | 100.0% | 6,087 |
| procedure | 11 | 90.9% | 50.0% | 92.6% | 100.0% | 7,213 |
| progettazione | 2 | 100.0% | 33.3% | 100.0% | 100.0% | 5,222 |
| programmazione | 1 | 100.0% | 50.0% | 100.0% | 100.0% | 12,539 |
| qualificazione | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 5,169 |
| requisiti | 6 | 100.0% | 72.2% | 100.0% | 100.0% | 6,237 |
| rotazione | 2 | 100.0% | 26.7% | 100.0% | 100.0% | 7,817 |
| rup | 5 | 80.0% | 40.3% | 100.0% | 100.0% | 9,933 |
| soccorso_istruttorio | 1 | 100.0% | 33.3% | 100.0% | 100.0% | 7,995 |
| soglie | 3 | 100.0% | 66.7% | 100.0% | 100.0% | 4,842 |
| subappalto | 3 | 100.0% | 77.8% | 100.0% | 100.0% | 6,440 |

### Questions to look at

| ID | Kind | Score | Expected | Cited | Note |
| --- | --- | --- | --- | --- | --- |
| Q018 | answerable | 0.500 | 45/2 | 45/2, 45/4 | La risposta indica correttamente il limite del 2 per cento dell'importo posto a base delle procedure di affidamento. Omette però la condizione per cui, negli appalti di servizi e forniture, la regola si applica solo quando è nominato il direttore dell'esecuzione. |
| Q023 | answerable | 0.500 | 50/2 | 50/2, all:II.1/3/4 | La risposta riporta correttamente il divieto di sorteggio salvo situazioni particolari e specificamente motivate in cui nessun altro metodo di selezione è praticabile. Omette però l'obbligo di pubblicare sul sito istituzionale i nominativi degli operatori consultati, elemento presente nella risposta attesa. |
| Q050 | answerable | 0.500 | 212/2 | 212/2, 212/1 | La risposta conferma correttamente che serve il parere, perché per le forniture la soglia è 100.000 euro, e riporta la regola per le amministrazioni sub centrali. Però non qualifica l'azienda sanitaria come amministrazione sub centrale e dichiara di non poter stabilire chi debba rendere il parere, omettendo questo elemento essenziale della risposta attesa (legale interno o, in mancanza, funzionario più elevato in grado competente per il contenzioso). |
| Q055 | temporal | 0.500 | 15/2 | 15/2, all:I.2/2/3, 140/11, 62/14 | La risposta concorda nella sostanza: in via ordinaria il RUP si nomina tra i dipendenti, anche a tempo determinato, della stazione appaltante. Però omette che, in mancanza di nomina, l'incarico spetta al responsabile dell'unità organizzativa competente, e introduce un'eccezione emergenziale che la risposta attesa non contempla, la quale nega qualsiasi previsione espressa. |
| Q077 | multi | 0.500 | 55/1, 55/2, 18/3 | 18/3, 90/1 | La risposta indica correttamente che sopra soglia vale lo stand still di trentadue giorni, con le relative eccezioni, e che sotto soglia il termine dilatorio non si applica. Omette però l'elemento, presente nella risposta attesa, che sotto soglia il contratto va stipulato entro trenta giorni dall'aggiudicazione. |
| Q078 | multi | 0.500 | 120/1/c, 120/2, 120/3 | 120/1, all:II.18/21/2, all:II.14/5/12 | La risposta individua correttamente l'ammissibilità della variante senza nuova gara per i rinvenimenti imprevisti e la condizione della struttura del contratto inalterata. Omette però il limite essenziale del 50 per cento del valore iniziale per ciascuna modifica, presente anche nell'estratto, e la soglia generale del 15 per cento per i lavori, sostenendo invece che non risulti alcun limite di importo. |
| Q081 | multi | 0.500 | 15/2, all:I.2/4/1, all:I.2/4/2 | 15/2, all:I.2/2/3, all:I.2/4/1, all:I.2/4/2, all:I.2/4/4 | La risposta individua correttamente i soggetti nominabili, il requisito di tecnico abilitato e i cinque anni per i non abilitati. Però non conclude che per 2 milioni di euro servono tre anni: la soglia europea per i lavori (circa 5,5 milioni) è nota, e la risposta lascia indeterminato proprio il requisito di esperienza richiesto dalla domanda. |
| Q065 | unanswerable | 1.000 | - | 25/2, 25/3, 62/6 | La risposta dichiara che il codice non nomina EmPULIA né le ASL pugliesi e non impone una specifica piattaforma regionale. Richiama solo l'obbligo generale di usare piattaforme digitali e rinvia alla normativa regionale non coperta, in linea con la risposta attesa e senza inventare contenuti. L'omissione del requisito della certificazione delle piattaforme è un dettaglio marginale. |
| Q066 | unanswerable | 1.000 | - | all:II.1/1/2 | Come la risposta attesa, il sistema afferma che il codice non individua chi firma la determina a contrarre e rinvia all'ordinamento interno della stazione appaltante. Le indicazioni sul contenuto dell'atto non contrastano con la risposta attesa né con la normativa. |
| Q068 | unanswerable | 1.000 | - | 70/4, 33/8, 98/3 | La risposta dichiara che non è indicata alcuna pena penale, senza inventarla, e descrive correttamente le sole conseguenze amministrative previste dal codice (inammissibilità dell'offerta, illecito professionale), in linea con la risposta attesa. Non rinvia esplicitamente al codice penale, ma non dice nulla in contrasto. |
