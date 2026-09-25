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

## 4. Before vs after

Same subset, same hyper-parameters, same hardware. "Baseline" = commit `2ec71f7` with its random per-record folds and the lexicon fit on all labels; "after" = the current tree with namesake-grouped strict folds and per-fold lexicons. The baseline's OOF numbers are therefore *optimistic* by construction, the after numbers are not, so a tie is a gain in validity, not a null result.

### 4.1 Out-of-fold, training subset (111k entities), each at its own tuned threshold

| metric | baseline (leaky folds) | after (strict folds) | delta |
|---|---|---|---|
| macro F0.5 | 0.99080 | 0.99088 | +0.00009 |
| micro precision | 0.99808 | 0.99852 | +0.00044 |
| micro recall | 0.97681 | 0.97562 | -0.00119 |
| singleton accuracy | 0.99478 | 0.99493 | +0.00016 |
| tuned threshold | 0.75 (raw score, 0.05 grid) | 0.79 (calibrated, 0.01 grid) | |
| pair-level accuracy | n/a | 0.99650 | |
| pair-level recall / precision | n/a | 0.98727 / 0.99851 | |
| retrieval recall@3 (ceiling of micro recall) | not measured | 0.9882 (@1 0.9792, @5 0.9906, @10 0.9931) | |
| per-fold macro F0.5 std / min | not reported | 0.00030 / 0.99039 | |

The three headline numbers of the after run, side by side: pair-level accuracy **0.9965**, micro recall **0.9756**, macro F0.5 **0.9909**. The gap between accuracy and recall is the negative-pair majority; the gap between recall and F0.5 is (a) the retrieval ceiling (recall@3 = 0.9882) and (b) the precision weighting, which makes the tuned threshold deliberately sacrifice ~1.5 points of recall for ~0.1 points of precision (see the sweep below).

### 4.2 Shifted local hold-out (44k entities, 2x decoy density, disjoint sample), full pipeline incl. threshold transfer

| metric | baseline | after | delta |
|---|---|---|---|
| macro F0.5 | 0.99197 | 0.99191 | -0.00006 |
| macro F0.5, non-singletons | 0.99161 | 0.99155 | -0.00006 |
| micro precision | 0.99914 | 0.99913 | -0.00001 |
| micro recall | 0.97808 | 0.97775 | -0.00033 |
| mean per-entity precision | 0.99706 | 0.99708 | +0.00002 |
| mean per-entity recall | 0.97770 | 0.97749 | -0.00021 |
| singleton accuracy | 0.99797 | 0.99797 | +0.00000 |
| entities with F0.5 = 1 | 0.92753 | 0.92663 | -0.00091 |
| entities with F0.5 = 0 | 0.00215 | 0.00211 | -0.00005 |
| macro F0.5, India | 0.99128 | 0.99123 | -0.00005 |
| macro F0.5, US | 0.99242 | 0.99236 | -0.00006 |
| threshold used | 0.85 (raw) | 0.79 (calibrated) | |

On this subset the two pipelines are tied within fold noise (std 0.0003). That is the expected outcome: precision is already 0.999 and the remaining recall loss is dominated by name-only records among namesake chains (section 8), which no cleaning or calibration change can recover. What changed is the *validity* of the estimate and the machinery for measuring shift; on the full data, where the Source 1 index is 20x larger and namesake collisions far more frequent, the leaky folds have more to hide (the team's full-data OOF number, 0.9725, was produced with them).

### 4.3 Threshold sweep (after, OOF, calibrated scale)

| thr | macro F0.5 | precision | recall | links |
|---|---|---|---|---|
| 0.10 | 0.98384 | 0.98663 | 0.98272 | 383492 |
| 0.20 | 0.98631 | 0.99008 | 0.98218 | 381948 |
| 0.30 | 0.98778 | 0.99226 | 0.98151 | 380849 |
| 0.40 | 0.98889 | 0.99412 | 0.98050 | 379748 |
| 0.50 | 0.98982 | 0.99576 | 0.97932 | 378665 |
| 0.60 | 0.99040 | 0.99697 | 0.97809 | 377726 |
| 0.70 | 0.99077 | 0.99792 | 0.97681 | 376876 |
| 0.79 | 0.99088 | 0.99852 | 0.97562 | 376189 |
| 0.80 | 0.99086 | 0.99855 | 0.97546 | 376116 |
| 0.90 | 0.99018 | 0.99904 | 0.97288 | 374940 |

Macro F0.5 is flat (within 0.0005) between 0.6 and 0.9: the precision/recall trade is nearly balanced there, which is why a 0.05-step grid and a 0.01-step grid pick different but equivalent thresholds. F0.5 is what is optimised; accuracy would pick a much lower threshold (pair accuracy peaks near 0.5).

## 5. Fold-level validation (after)

| fold | entities | macro F0.5 | precision | recall | singleton acc | fold-optimal thr | F0.5 at own thr |
|---|---|---|---|---|---|---|---|
| 0 | 27606 | 0.99039 | 0.99857 | 0.97541 | 0.9943 | 0.77 | 0.99048 |
| 1 | 27588 | 0.99119 | 0.99838 | 0.97696 | 0.9949 | 0.79 | 0.99119 |
| 2 | 27872 | 0.99094 | 0.99858 | 0.97543 | 0.9949 | 0.79 | 0.99094 |
| 3 | 28263 | 0.99100 | 0.99855 | 0.97470 | 0.9956 | 0.70 | 0.99106 |

mean 0.99088, std 0.00030, min 0.99039. Fold-optimal thresholds range [0.7, 0.79]; using each fold's own threshold instead of the global one gains at most 0.00009, i.e. the threshold is stable across folds.

Per country (OOF): India macro F0.5 0.98959 (P 0.9981, R 0.9721), US macro F0.5 0.99175 (P 0.9988, R 0.9780). India is consistently ~0.2 points below the US: Indic-script records and address-less records are both more frequent there.

Cross-fold audit: 36.8% of candidate pairs join a Source 2/3 record and a Source 1 entity from different folds (never a positive pair). Entities whose candidate lists are entirely within their fold score macro F0.5 0.99086; entities with at least one cross-fold pair score 0.99089. The two agree, so the residual cross-fold exposure does not inflate the estimate.

### Leave-one-country-out (train on one country, validate on the other)

<!-- COUNTRY -->

## 6. Leakage check results (after)

`leakage_check.py`: **OK**.

| check | result |
|---|---|
| Source 1 name groups split across folds | 0 of 75796 |
| matched Source 2/3 records outside their entity's fold | 0 |
| positive candidate pairs across folds | 0 |
| duplicated pair ids in the OOF file | 0 |
| fold-only Indic tokens present in their own fold's lexicon | 0 of 80 checked |
| label-shuffle canary, OOF AUC (expect 0.5) | 0.4999 |
| strongest single features (AUC) | gap_second 0.9926, score 0.9797, cos_addr 0.9595, ad_jacc 0.959 |
| test split carries label / fold columns | False |
| bookkeeping columns among model features | none |

For comparison, the baseline lexicon had 1307 name entries learned from every pair; the per-fold lexicons have 1277/1279/1281/1276 entries each, i.e. ~2 % of the transliterations a validation record used to receive came from its own label.

## 7. Calibration and threshold analysis

Nested (fit on 3 folds, scored on the 4th) calibration of the stage-2 OOF probabilities:

| probabilities | log loss | Brier | ECE (15 bins) |
|---|---|---|---|
| uncalibrated | 0.00828 | 0.00238 | 0.00086 |
| platt | 0.00802 | 0.00234 | 0.00032 |
| isotonic | 0.00803 | 0.00234 | 0.00025 |

Platt parameters fit on all OOF predictions: a = 0.805, b = -0.065 (a < 1: the raw scores are over-confident and are pulled towards 0.5). Platt removes ~60 % of the calibration error; isotonic is not better enough to justify its extra flexibility, so Platt is what `predict.py` applies. Reliability table (Platt): [0.00-0.07] n=1153284 pred 0.000 obs 0.000; [0.07-0.13] n=3479 pred 0.097 obs 0.109; [0.13-0.20] n=2208 pred 0.165 obs 0.193; [0.20-0.27] n=1806 pred 0.233 obs 0.244; [0.27-0.33] n=1504 pred 0.299 obs 0.293; [0.33-0.40] n=1347 pred 0.366 obs 0.372; [0.40-0.47] n=1109 pred 0.433 obs 0.411; [0.47-0.53] n=973 pred 0.498 obs 0.464; [0.53-0.60] n=731 pred 0.564 obs 0.512; [0.60-0.67] n=605 pred 0.633 obs 0.572; [0.67-0.73] n=557 pred 0.698 obs 0.614; [0.73-0.80] n=520 pred 0.766 obs 0.689; [0.80-0.87] n=693 pred 0.836 obs 0.831; [0.87-0.93] n=1321 pred 0.906 obs 0.899; [0.93-1.00] n=374107 pred 0.999 obs 0.999.

Threshold transfer to the test split (record counts only, no test labels):

decoy-density ratio r = 2.019 (test has 2.02x the unmatched records per entity); implied pair-level odds ratio 0.742.

| method | threshold | OOF macro F0.5 at training mix | at test-like mix (decoy FPs weighted r) | precision | recall |
|---|---|---|---|---|---|
| train | 0.790 | 0.99088 | 0.99042 | 0.99741 | 0.97562 |
| density | 0.790 | 0.99088 | 0.99042 | 0.99741 | 0.97562 |
| prior | 0.835 | 0.99070 | 0.99028 | 0.99772 | 0.97479 |

The density-adjusted optimum coincides with the training optimum on this subset (the F0.5 curve is flat there); the analytic prior-shift threshold is more conservative and costs 0.0002. `select_threshold.py --method` switches between them; `density` stays the default. The threshold stability under *simulated* shift is in section 9.

## 8. Error analysis (after, OOF at the tuned threshold)

Of 385020 true pairs, 97.56% are recovered. The misses split into three causes:

| cause | n | share of true pairs | Indic script | no address | Source 1 has namesakes | Source 3 |
|---|---|---|---|---|---|---|
| retrieval_miss | 4541 | 0.0118 | 0.089 | 0.717 | 0.812 | 0.562 |
| outranked | 1879 | 0.0049 | 0.006 | 0.947 | 0.907 | 0.514 |
| below_threshold | 2968 | 0.0077 | 0.031 | 0.569 | 0.667 | 0.521 |
| (base rate over all true pairs) | | | 0.074 | 0.044 | 0.400 | |

| false positives | n | share of links | Indic | no address | S1 has namesakes | mean p |
|---|---|---|---|---|---|---|
| decoy_linked | 412 | 0.0011 | 0.029 | 0.068 | 0.308 | 0.935 |
| wrong_entity | 145 | 0.0004 | 0.145 | 0.510 | 0.552 | 0.896 |

Reading: recall is lost almost entirely on **records without an address whose Source 1 entity has namesakes** (72 % / 95 % of retrieval misses / out-ranked matches have no address, against a 4 % base rate; 81 % / 91 % belong to namesake chains against 40 %). A name-only `ONE SUMMIT LP` against several `ONE Summit LP` entities in different cities is a coin toss, and under F0.5 the right decision is to abstain, which is what the threshold does. Retrieval misses are also where Indic-script records concentrate (8.9 % vs 7.4 %), the one lever that is still available (a better lexicon / phonetic key). The false positives at p > 0.999 are near-copies at the same address that the ground truth labels as unrelated (`Hashmi, Clotilda W., MD, DDS PC` vs `Hashmi, Clotilda W., MMD, DDS` at 7813 Shady Banks Terrace): synthetic decoys generated from the entity itself. They are label noise for any matcher and put the precision ceiling below 1.

## 9. Robustness under simulated distribution shift

Scenarios are re-samplings of the OOF candidate pairs (`tools/robustness.py`); nothing is fit on them. "regret" = macro F0.5 at the scenario's own best threshold minus macro F0.5 at the training threshold (0.79); the prior-corrected threshold shifts the calibrated odds by the change in positive/negative pair ratio.

| scenario | F0.5 @ training thr | own best thr | F0.5 @ own thr | regret | F0.5 @ prior-corrected thr | recall | precision |
|---|---|---|---|---|---|---|---|
| train_distribution | 0.99088 | 0.79 | 0.99088 | +0.00000 | 0.99088 (thr 0.79) | 0.9756 | 0.9985 |
| decoys_x2 | 0.99043 | 0.79 | 0.99043 | +0.00000 | 0.99029 (thr 0.83) | 0.9756 | 0.9974 |
| decoys_x3 | 0.99011 | 0.79 | 0.99011 | +0.00000 | 0.98992 (thr 0.86) | 0.9756 | 0.9963 |
| matched_x0.7 | 0.98870 | 0.77 | 0.98873 | +0.00002 | 0.98860 (thr 0.81) | 0.9756 | 0.9981 |
| matched_x0.5 | 0.98705 | 0.77 | 0.98710 | +0.00005 | 0.98688 (thr 0.83) | 0.9761 | 0.9974 |
| singletons_x2 | 0.93610 | 0.79 | 0.93610 | +0.00000 | 0.93610 (thr 0.79) | 0.9757 | 0.9398 |
| source_2_only | 0.98677 | 0.62 | 0.98741 | +0.00064 | 0.98677 (thr 0.79) | 0.9770 | 0.9986 |
| source_3_only | 0.98480 | 0.56 | 0.98546 | +0.00066 | 0.98480 (thr 0.79) | 0.9748 | 0.9984 |
| indic_script_only | 0.99610 | 0.51 | 0.99618 | +0.00008 | 0.99610 (thr 0.79) | 0.9821 | 0.9988 |

| calibration under prior shift | ECE Platt | ECE Platt + prior correction | Brier Platt | Brier corrected |
|---|---|---|---|---|
| decoys_x1 | 0.0003 | 0.0003 | 0.00234 | 0.00234 |
| decoys_x2 | 0.0005 | 0.0004 | 0.00216 | 0.00214 |
| decoys_x3 | 0.0009 | 0.0005 | 0.00204 | 0.00198 |

Reading:

* **The threshold is stable.** Under 2x and 3x decoy density, 30-50 % fewer true matches per entity, a single source only, or Indic-script records only, the regret of keeping the training threshold is at most 0.0007. Macro F0.5 moves because the *problem* gets harder (3x decoys: -0.0008; half the matches: -0.004), not because the threshold is wrong.
* **Prior-shift correction is not needed for the decision** and is slightly harmful here (-0.0001 to -0.0002): the F0.5-vs-threshold curve is flat between 0.6 and 0.9, so moving the threshold buys nothing while the correction assumes the shift is purely in the class prior. It does improve *calibration* under shift (ECE 0.0009 → 0.0005 at 3x decoys), which matters only for the expected-F decoder.
* **What does hurt is label structure, not density**: turning 6 % of entities into singletons by relabelling their matches as decoys ("singletons_x2") costs 5.5 points, all in precision. Those relabelled records are exact copies of the entity, so no matcher can reject them; this bounds what a hidden test with a different singleton rate can do to the score and is the main argument for the conservative threshold.
* Source 3 records are harder than Source 2 (0.985 vs 0.987), and their optimal thresholds are lower (0.56 / 0.62): with only one source the decoys per entity halve. The mixed-source training threshold remains within 0.0007 of optimal for either.


## 10. Runtime profile and optimisations

Subset, 4 cores. `cpu/wall` is process + children CPU seconds over wall seconds: 4.0 would be perfect use of the 4 cores.

| stage | baseline wall s | after wall s | after cpu/wall | note |
|---|---|---|---|---|
| folds |  | 2.4 | 1.11 | new |
| lexicon | 1.4 | 1.3 | 1.26 | 4 leave-one-fold-out lexicons + all |
| prepare_train | 16.4 | 17.9 | 3.37 |  |
| blocking_train | 27.8 | 18.2 | 2.40 | top-10 retrieval kept only for recall@k; exact cosines for top-3 only |
| features_train | 32.3 | 27.2 | 2.34 | 3 instead of up to 10 candidates per record were already scored; less I/O |
| prepare_test | 9.8 | 10.3 | 2.82 |  |
| blocking_test | 11.1 | 6.6 | 2.66 | top-3 retrieved directly |
| features_test | 11.9 | 11.2 | 2.65 |  |
| train | 285.2 | 265.6 | 3.58 | vectorised sweep on a 0.01 grid, strict folds, calibration + report |
| leakage_check |  | 19.9 | 2.58 | new (label-shuffle canary trains 4 small models) |
| predict | 12.4 | 11.4 | 2.97 | calibration applied |
| **total** | 409.5 | 391.9 (+ 1.2 s select_threshold) | | after includes two new stages (folds, leakage check) |

Findings:

* **CPU is used.** Every parallel stage shows cpu/wall between 2.3 and 3.6 on 4 cores (normalisation 3.4, LightGBM 3.6). No GPU is used anywhere (LightGBM CPU build; Kaggle CPU sessions have none), and none is needed: the matcher is 80-feature tabular boosting.
* **Training dominates** (68 % of wall time): 8 cross-fitted LightGBM models + 2 final models with 300/200 rounds of 127-leaf trees. It is compute-bound and parallel; the only way down is fewer rounds/leaves or early stopping. The ablation run (section 11) trains with 150/100 rounds and shows the cost of that trade.
* **Threshold sweep**: the baseline re-sorted all candidate pairs 16 times (0.4 s each here, ~10 s each on the full 30M-pair set, so ~3 min of a 10 min stage); `thresholds.py` does 94 grid points in under a second.
* **Blocking**: exact cosines are now computed for 3 candidates per record instead of 10 (`blocking_test` 11.1 s → 6.6 s; the training split keeps the deeper retrieval only to measure recall@k, and even there the cosine step runs on the kept 3).
* **Feature generation** already used multiprocessing (`_py_features` over 50k-row jobs) and rapidfuzz's C++ `cpdist(workers=-1)`; the remaining Python loop (`_hn_relation`, `_tok_unmatched`) is inside the worker pool. It is not the bottleneck here (27 s).
* **The "2 h for 10 %" number belongs to the notebook, not to this code base.** On this subset the whole pipeline is 6.5 min; the team's own full-data timings (README) put blocking at ~12 min per split on 10 cores with 14 GB peak, and training at ~10 min. If a run takes hours, `work/profile.json` now says which stage, and cpu/wall says whether it ran in parallel.

## 11. Feature engineering: what is generated and what it is worth

### 11.1 Inventory

Stage-1 features (82 on this data) are generated in `pair_features.py`; nothing is selected or removed automatically - LightGBM sees all of them and `train.py --drop-features` / `tools/ablation.py` remove groups explicitly. The earlier notebook's AI-suggested feature reduction was not carried into this code base; the table below is the measured basis for any reduction.

| group | features | what it encodes |
|---|---|---|
| retrieval (21) | `score`, `rank`, `cos_name/addr/cross`, `t_best*`, `t_second`, `t_ncand`, `gap_*`, `s_ncand`, `s_rank`, `s_best`, `s_ntop1`, `s_gap_best` | blocking similarity and the *competition* around the pair: how far the candidate is from the best one for this record, and how many records compete for this entity |
| name_string (13) | `nm_ratio`, `nm_core_ratio`, `nm_tsort`, `nm_tset`, `nm_partial`, `nm_jw`, `nm_part_max/min`, `nm_first_ratio`, `cmp_ratio`, `cmp_jw` | rapidfuzz similarities on full / core names, alias parts and the compact (space-less) name; typos, token order, concatenation |
| name_tokens (9) | `t_ntok`, `s_ntok`, `nm_common`, `nm_jacc`, `nm_cov_s/t`, `nm_xt`, `nm_xs`, `t_n_namesake` | token overlap, typo-tolerant extra tokens on each side, how many candidates of the record carry the same name (namesakes) |
| domain (3) | `dom_prefix`, `dom_full`, `t_domain` | domain-style names (`laborerslocal207.com`) against the compact name |
| address_string (8) | `ad_ratio/tsort/tset/partial`, `street_ratio/tset`, `st_ratio`, `st_eq` | address and street-line similarity |
| address_tokens (4) | `t_natok`, `s_natok`, `ad_common`, `ad_jacc` | address token overlap |
| house_numbers (12) | `hn_rel`, `hn_logdiff`, `hn_reldiff`, `hn_s_in_t`, `hn_t_in_s`, `hn_lendiff`, `num_common`, `num_frac`, `num_first_eq`, `t_nnum`, `s_nnum`, `t_n_addr_exact` | relation between the house numbers (equal / truncated / one digit off / transposed / a few doors away), digit-group overlap |
| legal_form (4) | `lg_t`, `lg_s`, `lg_common`, `lg_conflict` | legal suffixes on each side and whether they conflict (LLC vs Inc, SARL vs SAS) - now computed from the contextual legal extraction of section 3 |
| record_flags (8) | `t_src`, `t_indic`, `t_alias`, `t_noaddr`, `s_noaddr`, `t_noname`, `t_nlen`, `s_nlen` | source, script, alias, missing fields, lengths |
| crowding (2) | `s_addr_mult`, `s_street_mult` | how many Source 1 records share this entity's exact address / street (an address match is weaker evidence in a business tower) |
| stage 2 (13, on top of stage 1 + `p1`) | `c_t_pmax/psum/prank/gap/vs_rest`, `c_s_pmax/psum/prank/n05/n/psum_other/gap`, `c_hn_same_t/s` | stage-1 probabilities of the competing pairs around this pair (the other candidates of the record, the other records of the entity) and house-number consensus among them |

Country is deliberately **not** a feature (open label set: France is unseen); it only partitions retrieval. Multilingual handling is in normalisation (Indic lexicon + unidecode fallback, French tables), not in features, so the model cannot learn country-specific shortcuts that would not transfer.

### 11.2 Ablation (OOF, strict folds, 150/100 boosting rounds, one group removed at a time)

<!-- ABLATION_TABLE -->

## 12. Remaining risks and recommended next experiments

Risks, in the order I would worry about them for the hidden test:

1. **France is unseen and 15 % of the test.** Everything French relies on hand-written tables (`FR_REGIONS`, French street abbreviations, `sarl/sas/eurl`) and on the model's ability to generalise from US/India string statistics. The leave-one-country-out result (section 5) is the only evidence of what an unseen country costs, and it is a lower bound on the damage (India→US and US→India share the Latin script and address conventions more than either shares with France). The public leaderboard, which includes France, is the only French signal you have; use it for *that* question only.
2. **Namesake chains without addresses are the recall ceiling** (section 8) and the retrieval index is 20x larger on the full data than on the subset, so the full-data retrieval recall@3 is lower than 0.988. Measure it: `blocking_recall.json` is written on every full training run. If recall@5 - recall@3 is material there, raise `--topk` (it is one knob now) and re-tune.
3. **Decoys that are near-copies of the entity** put a floor on false positives that no threshold removes (`Hashmi ... MD, DDS PC` vs `... MMD, DDS`). If the hidden test generates decoys differently (more or fewer of these), precision moves and the threshold with it; the `singletons_x2` scenario bounds the damage.
4. **Transductive statistics** (IDF, address crowding, candidate-list context) depend on corpus size and decoy density. They are computed on the test split itself, which is the right thing to do, but their training distribution differs from the test one. The ablation says how much the model leans on them.
5. **The subset over-states absolute scores.** Every number here is relative. Run the full pipeline once with the new code before submitting and read `validation_report.json` (per fold, per country, cross-fold audit) and `leakage_check.json` for the full data; the runtime profile will tell you where the hours go.
6. **Stage-2 context is a stacking layer**; it is cross-fitted correctly, but it makes the model depend on the candidate-list structure (top-3 per record), so changing `--topk` requires retraining both stages.

Next experiments, cheapest first:

* Full-data run of `run_pipeline.sh` with the current code; compare its `validation_report.json` with the leaderboard score. If OOF and leaderboard disagree by more than the fold std, the gap is France + decoy density, and the leave-one-country-out number tells you how much of it is France.
* `blocking.py --topk 5` on the full training split (recall@k is measured anyway) and the ablation on the full data with `--rounds1 150 --rounds2 100`, which the subset shows costs nothing.
* Better Indic transliteration for retrieval misses: a phonetic key on the unidecode fallback (`phonetic.py` exists and is unused in blocking) as an extra blocking feature family for Indic-script records.
* A French dry run: normalise `test_source*.tsv` France records and inspect `n_core` / `a_comp` for the 50 most frequent tokens; the tables in `textnorm.py` were written blind.
* Country-conditional threshold: the per-country OOF thresholds are within 0.05 of each other on the subset, so a single threshold is fine, but France will not have an OOF estimate. Consider the more conservative of the two known thresholds for France.
* Isotonic vs Platt is a wash here; revisit only if the expected-F decoder (`decode.py`) is switched on, in which case calibration quality is what its guarantee rests on.


