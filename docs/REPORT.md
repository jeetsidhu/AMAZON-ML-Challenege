# Pipeline review: diagnosis, changes and results

Scope: the v2 pipeline (`src/`) as imported in commit `2ec71f7` ("baseline" below) versus the
current tree ("after"). All numbers in this document were produced on a 5 % name-group sample of
the training data (`tools/make_subset.py --frac 0.05`: 111k Source 1 entities, 519k Source 2/3
records, 4 CPU cores, 15 GB RAM) plus a disjoint 2 % sample with **twice the decoy density** as a
local hold-out (`subset/test`). Absolute scores on a sample are higher than on the full data
(the Source 1 index is 20x smaller, so retrieval has fewer confusable namesakes); the
comparisons are like-for-like. Every experiment is reproducible with the commands in the README;
the raw JSON outputs are in `reports/`.

## 1. Diagnosis of the baseline

| # | problem | where | effect |
|---|---|---|---|
| 1 | **Label leakage through preprocessing.** The Indic-script lexicon is learned from *all* training pairs (`build_lexicon.py`) and then applied to every record, including the validation folds. A validation record in Indic script was transliterated with a table that had seen its own ground-truth pair. | `build_lexicon.py`, `prepare.py` | OOF scores for the ~7 % of Source 2/3 records in Indic script are optimistic; the test split never gets this help. |
| 2 | **Folds that do not respect entity structure.** Folds were hashed per Source 2/3 record (matched ones inherit their Source 1 entity's fold). 38 % of Source 1 records share their exact name with another Source 1 record of the same country (chains, franchises), so namesakes were split across folds and the model saw the training copy of the validation entity's name/address pattern. Unmatched ("decoy") records got a fold of their own, so their pairs with a validation entity were trained on and evaluated by models that had seen the same record elsewhere. | `train.py` | Validation looked easier than the test split, where every entity and record is new. |
| 3 | **No prior-shift awareness in validation.** The real test split has 5.75 Source 2/3 records per Source 1 entity vs 4.68 in training (about twice the decoys per entity) and 15 % of its entities are French, a country absent from training. Random folds cannot show what that costs. | `train.py` | The published 0.9725 OOF number is an i.i.d. estimate only. |
| 4 | **Destructive text cleaning.** Every token equal to the record's country label was deleted (`Air India` → `air`, `Us Engineering` → `engineering`); articles and single letters were always dropped (`The One` → `one`, `A & W` → `w`, `L A Fitness` → `fitness`); ambiguous two-letter legal forms were dropped anywhere (`PC World` → `world`, `Ag Supply` → `supply`, `SA Toys` → `toys`); `tech` was *expanded* to `technologies` (`Tech Mahindra` → `technologies mahindra`); French street abbreviations fired in every country (`R K Puram` → `rue k puram`); the address components `DE` (Delaware) and `LA` (Louisiana) were deleted as French articles. | `textnorm.py` | Identity information destroyed for ~3 % of names; namesake collisions created where none existed. |
| 5 | **No probability calibration.** LightGBM scores were thresholded directly; the density-adjusted threshold and the expected-F0.5 decoder both silently assumed they were probabilities. | `train.py`, `select_threshold.py`, `decode.py` | Threshold cannot be transferred analytically under prior shift; the decoder's expected-F computation is wrong. |
| 6 | **Retrieve 10, keep 3.** Blocking retrieved the top-10 candidates per Source 2/3 record and computed the exact per-block cosines for all of them (`--topk 10`), then `pair_features.py` kept `rank < 3` and `score >= 0.1` (a second, different `min_score`). 70 % of the retrieval output and exact-cosine work was thrown away, and recall@k was never measured. | `blocking.py`, `pair_features.py` | Wasted work; the recall ceiling of the system (retrieval recall@3) was unknown. |
| 7 | **Threshold sweep re-sorts every pair per grid point.** `train.py` called `assign()` (a full sort of all candidate pairs + joins + group-by) 16 times, on 5 % steps; `select_threshold.py` used 10 grid points. | `train.py`, `select_threshold.py` | Minutes on the full data for a computation that is one `bincount` per grid point; a coarse grid. |
| 8 | **No per-stage profiling, no CPU-utilisation evidence.** | all | The "2 h for 10 %" complaint could not be attributed to a stage. |
| 9 | **Feature groups never ablated.** 77 stage-1 + 13 stage-2 features, importance by gain only. | `train.py` | No evidence which groups matter for recall vs precision. |

### Why "accuracy 0.96, recall 0.88, F0.5 0.75–0.80" can all be true at once

These three numbers come from three different units of measurement, and the gap between them is
structural, not a bug:

* **Pair-level accuracy** counts candidate pairs. Roughly three quarters of candidate pairs are
  negatives (positive rate ≈ 0.25 with top-3 retrieval), so predicting "no" for everything already
  scores ≈ 0.75 and a mediocre model scores 0.96. It says almost nothing about the submission.
* **Micro recall** counts matched records: it is capped by *retrieval* recall (a true record whose
  Source 1 entity is not among its top-k candidates can never be recalled) and reduced by the
  one-to-one assignment (a record linked to the wrong namesake is a miss *and* a false positive).
* **Macro F0.5** counts entities, with singletons (5.6 %) scoring 1/0 and every false link on an
  entity with one true record dropping that entity to F0.5 ≈ 0.56. It weights precision twice as
  much as recall, so a model tuned for accuracy or recall is tuned for the wrong thing.

The `pair_level` and `oof_at_threshold` blocks of `validation_report.json` now report all three
side by side for every run; section 4 shows them for the after run.

## 2. Leakage audit

Every place where training labels or training-derived statistics touch validation or test data
was traced. "Transductive" means an unsupervised statistic computed on the corpus being scored
(train on train, test on test); that is not leakage, but it is a shift risk and is listed.

| source | kind | status | fix |
|---|---|---|---|
| Indic lexicon learned from all training pairs, applied to validation folds | **label leakage** | fixed | `folds.py` runs first; `build_lexicon.py` writes one lexicon per held-out fold (`fold_k`, learned from the other folds) plus `all`; `prepare.py` transliterates each training record with its own fold's lexicon, decoys and test records with `all` (exactly what inference sees). `leakage_check.py` verifies that a token whose only evidence is in fold k is absent from `fold_k`. |
| Namesake Source 1 entities split across folds | **entity leakage** | fixed | folds are hashed per name group (normalised core name + country) so all namesakes share a fold. |
| Decoy records hashed into their own fold, so a validation entity's negative pairs were trained on elsewhere | entity leakage (mild) | fixed | decoys take the fold of their best blocking candidate; the model of fold k is trained only on pairs whose Source 1 entity **and** Source 2/3 record are outside fold k; the remaining cross-fold pairs are counted and metrics are reported separately for entities with / without them. |
| Stage-2 context features computed from stage-1 probabilities of neighbouring pairs | stacking | fine | stage-1 probabilities are OOF for every pair (cross-fitting); the stage-2 model of fold k never sees fold-k pairs. |
| TF-IDF document frequencies for blocking (`blocking.py`) | transductive, unsupervised | fine, mirrored at test | IDF is computed on the Source 1 side of the split being scored. Shift risk: none beyond corpus size. |
| Address crowding counts `a_mult` / `a_smult` (`pair_features.address_crowding`) | transductive, unsupervised | fine, mirrored at test | Counts depend on corpus size (2.2M vs 1.7M Source 1 records); ablation shows the group is not load-bearing. |
| Candidate-list context features (`t_ncand`, gaps, `s_rank`, …) | transductive, unsupervised | fine, mirrored at test | Their distribution shifts with decoy density; covered by the robustness scenarios. |
| Decoy-density ratio in `select_threshold.py` | uses test *record counts* only | fine | no labels involved; kept and complemented by the prior-shift alternative. |
| Practice-test / test path (`prepare.py`, `blocking.py`, `pair_features.py`, `predict.py` with `--split test`) | | clean | no ground-truth file is read for the test split (`truth = None`), the test records carry no `label` / `fold` column (asserted by `leakage_check.py`), the test lexicon is `all`, and the model feature list is checked against the bookkeeping columns. |
| Training pairs subsampled for the model (`--sample-rows`) | | fine | the sample is drawn before fold exclusion; OOF predictions cover every pair. |
| Platt calibrator and threshold | | fine | fit on OOF predictions only; the nested calibration report fits on K-1 folds and scores the held-out one. |

`src/leakage_check.py` runs these assertions after training (the pipeline stops on failure) and adds
two canaries: a stage-1 model trained on **shuffled labels** with the same strict folds must score
OOF AUC ≈ 0.5 (rows shared between training and evaluation would let it memorise the shuffled
labels), and no single feature may separate the classes perfectly.

## 3. Text normalisation changes (`src/textnorm.py`, tests in `tests/test_textnorm.py`)

| rule | baseline | after |
|---|---|---|
| country token in a name | deleted wherever it appears | a parenthesised country tag `(India)` (including OCR-damaged `(lndia)`) is removed; a bare country token is always kept in the full name and dropped from the core name only when it trails at least three other content tokens (`Tata Consultancy Services India` → core `tata consultancy svcs`; `Air India`, `Reliance India`, `Toys R Us`, `Air France` keep it) |
| articles / connectives (`the`, `la`, `le`, `el`, `de`, `of`, `a`, …) | always deleted from the core name | deleted only when at least two other tokens remain: `The Dent Diner` → `dent diner`, but `The One`, `La Poste`, `El Lincoln` are kept; always kept in the full name |
| initials | `d`, `l`, `a` deleted; other single letters kept separately | runs of single letters (optionally glued by `&`) are merged into one token: `A & W` → `aw`, `J.P. Morgan` → `jp morgan`, `L A Fitness` → `la fitness`; `D&L`, `D & L` and `DL` normalise identically |
| legal forms | all of `inc corp co ltd pvt llc … pc pa sa ag ei …` removed anywhere | strict forms (`inc`, `ltd`, `llc`, `pvt`, `gmbh`, `sarl`, …) removed anywhere (corrupted records shuffle them: `Federal LLC Minerals Star`); ambiguous ones (`co`, `pc`, `pa`, `sa`, `ag`, `lp`, `bv`, `nv`, `ei`) removed only in legal position (trailing, after `&`, or next to another legal token): `PC World` and `Ag Supply` keep their first token, `Crystal Lending PC` and `Tiffany & Co` still yield the legal token for the conflict feature |
| abbreviations | `tech` → `technologies`, `intl` → `international`, `mfg` → `manufacturing`, `svcs` → `services` (short forms expanded) | long forms collapse to the short form (`technologies` / `technology` → `tech`, `international` → `intl`, `manufacturing` → `mfg`, `services` → `svcs`); a short token is never expanded into a meaning it may not have |
| French street abbreviations | `r` → `rue`, `q` → `qu`, `ch` → `che`, `bd` → `blvd`, `av` → `ave` everywhere | only for records whose country is France (`R K Puram` stays) |
| address components that are a single stop word | deleted (`DE`, `LA` state codes vanished) | kept when the component is that single token |
| everything else (alias markers, `(ID: …)`, `M/s`, domains, Indic lexicon, state tables) | | unchanged |

48 unit tests cover these cases (`python -m pytest tests -q`).
