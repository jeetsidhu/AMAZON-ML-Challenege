# Business Entity Resolution — Team ICE

Multi-channel blocking + two-stage LightGBM matcher with **many-to-one** assignment (each Source 2/3
record goes to at most one Source 1 entity, an entity receives any number of records), one global
acceptance threshold tuned on out-of-fold entity-level macro F0.5 and adjusted for the test split's decoy
density (optional confidence-based rejection: margin over the runner-up + contradiction rule, selected by
nested macro F0.5), leak-free cross-fitted validation, Platt-calibrated probabilities and training checkpoints.
Only the provided challenge data is used; there are no external lookups, APIs or pretrained
language models. The only learned model is LightGBM (MIT licence), with far fewer than 8B parameters.

`docs/REPORT.md` holds the diagnosis of the previous version, the leakage audit, the before/after
metrics, fold-level results, calibration / threshold analysis, the runtime profile, the feature
ablation, the robustness tests and the list of remaining risks. Section 13 is the per-class
threshold study (negative result; the machinery was removed in v5.1); section 14 is the review of the
matching logic, validation, candidate generation, decision layer and error analysis (what was verified,
what was changed, before/after numbers) and 14.7 the v5.1 simplifications.

## Layout

```
├── README.md
├── requirements.txt
├── docs/REPORT.md             # analysis + results of the review
├── reports/                   # JSON / markdown outputs of the experiments quoted in the report
├── tests/                     # python -m pytest tests: text normalisation, calibration, checkpoints,
│                              # threshold policies, blocking channels + audit, assignment / decision rule
├── tools/
│   ├── make_subset.py         # name-group-sampled subset of the training data for fast experiments
│   ├── error_analysis.py      # entity-level error report: categories ranked by macro F0.5 points lost
│   ├── ablation.py            # feature-group ablation (OOF macro F0.5 / recall / precision)
│   └── robustness.py          # threshold + calibration under simulated distribution shift
└── src/
    ├── run_pipeline.sh        # end-to-end driver (all steps below, then validation)
    ├── common.py              # paths, TSV reader, logging, per-stage profiler (-> work/profile.json)
    ├── textnorm.py            # name/address normalisation + blocking-feature generation
    ├── folds.py               # step 0: validation folds BEFORE any label-derived preprocessing
    ├── build_lexicon.py       # step 0b: Indic-script -> Latin lexicon, one per held-out fold + "all"
    ├── prepare.py             # step 1: normalise every record (multiprocess) -> records.parquet
    ├── blocking.py            # step 2: multi-channel TF-IDF retrieval per country (union of channels) + candidate audit
    ├── pair_features.py       # step 3: ~100 pairwise features (incl. channel ranks, contradictions) -> pairs/part_*.parquet
    ├── model.py               # LightGBM helpers, stage-2 context / competition features, many-to-one assignment + rejection
    ├── thresholds.py          # vectorised macro-F0.5 sweep; thresholds, margin, contradiction penalty; nested rule selection
    ├── threshold_policy.py    # the policy file: global threshold + decision rule (JSON)
    ├── checkpoint.py          # training checkpoints: manifest, fold models, OOF, calibration, policies, experiment log
    ├── calibrate.py           # Platt / isotonic calibration, prior-shift correction, ECE / Brier
    ├── train.py               # step 4: strict K-fold cross-fitted 2-stage model -> checkpoint (OOF, calibration, report, models)
    ├── leakage_check.py       # step 4a: automated leakage audit of the finished training run
    ├── select_threshold.py    # step 4b: global test threshold under the expected decoy-density / prior shift -> threshold_policy.json
    ├── predict.py             # step 5: score test pairs with a checkpoint + policy, write output/*.tsv
    ├── evaluate.py            # score a matching_results.tsv against a ground truth (local hold-outs)
    ├── decode.py              # optional expected-F0.5 decoder
    ├── metrics.py             # macro F0.5 exactly as the challenge defines it
    └── validate_submission.py # official validator (copied from student_resource/utils)
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

## Reproduce end-to-end

On a fresh Linux machine, `bash run.sh` does everything: a Python 3.12 virtualenv via `uv` (independent of
the system Python), the data download from the challenge repository, the unit tests, a **smoke test** (the
whole pipeline on a 0.3 % slice of the real data, scored on a labelled hold-out, ~2 minutes), then every
pipeline step with one log file each under `logs/`, the validator and a summary. It is resumable:
re-running it skips the steps that already finished. `bash run.sh --smoke-only` runs just the smoke test.
See the header of `run.sh` for `--from <step>`, `--fresh`, `--skip-download`, `--skip-smoke`, `--no-venv`
and the `ROUNDS1` / `ROUNDS2` overrides.

`DATA` is the challenge `dataset/` directory, which contains `train/` and `test/`.

```bash
bash src/run_pipeline.sh /path/to/student_resource/dataset work output
```

This writes `output/matching_results.tsv` (the leaderboard file) and
`output/candidate_pairs.tsv` (the exact candidate set scored by the model), then runs the
official validator. Diagnostics land next to the models in `work/`:

| file | content |
| --- | --- |
| `work/validation_report.json` | OOF metrics (macro F0.5, singleton false positives, FP by kind), nested macro F0.5, the decision-rule search, per-fold / per-country / per-source metrics, pair-level metrics, calibration (nested), cross-fold audit, candidate-generation audit |
| `work/train/blocking_recall.json` | candidate audit: true-pair recall (union and per channel), complete-entity coverage, candidate oracle macro F0.5, ambiguous vs recoverable misses |
| `work/error_analysis.json` / `.md` | entity-level error analysis under the selected policy: categories ranked by macro F0.5 points lost (`tools/error_analysis.py`) |
| `work/leakage_check.json` | result of the automated leakage audit (the pipeline stops if it fails) |
| `work/threshold_analysis.json` | training-optimal vs density-adjusted vs prior-shift global thresholds |
| `work/threshold_policy.json` | the shipped policy (global density-adjusted threshold + decision rule), read by `predict.py` |
| `work/checkpoints/<name>/` | the training checkpoint: manifest, fold + final models, calibration, OOF predictions, report, `thresholds/*.json`, `experiments.jsonl` (see below) |
| `work/profile.json` | wall / CPU seconds per stage (cpu/wall shows the parallelism actually achieved) |
| `work/train/oof.parquet` | out-of-fold stage-1 / stage-2 / calibrated probabilities of every training pair (link to the latest checkpoint) |

The steps can also be run one at a time (same order as in `run_pipeline.sh`):

```bash
D=/path/to/student_resource/dataset; W=work
python src/folds.py          --data-dir $D --work-dir $W            # --scheme country for leave-one-country-out
python src/build_lexicon.py  --data-dir $D --work-dir $W
for S in train test; do
  python src/prepare.py       --data-dir $D --work-dir $W --split $S
  python src/blocking.py      --data-dir $D --work-dir $W --split $S   # --channels "combined=5,nchar=3,addr=2,rev=2" (see below)
  python src/pair_features.py --data-dir $D --work-dir $W --split $S
done
python src/train.py          --data-dir $D --work-dir $W --checkpoint-name run1   # --resume continues an interrupted run; --no-rule-search
python src/leakage_check.py  --data-dir $D --work-dir $W
python src/select_threshold.py --data-dir $D --work-dir $W           # --method density|prior|train (global threshold) -> threshold_policy.json
python tools/error_analysis.py --data-dir $D --work-dir $W           # entity-level error report (OOF, selected policy)
python src/predict.py        --data-dir $D --work-dir $W --out-dir output   # --checkpoint run1 --policy <file> --reuse-scores
```

### Candidate generation (blocking channels)

`blocking.py` retrieves candidates through several channels per country and scores the **union**:
`combined` (name + address + name x place TF-IDF cosine, the previous single channel), `name`, `nchar`
(character 4-grams of the compact name: typos, concatenations, transliterations), `addr`, `cross`, `rare`
(shared rare name tokens), `hn` (house number + street key) and `rev` (bidirectional: every Source 1 record
retrieves its top-k Source 2/3 records). `--channels "combined=5,nchar=3,addr=2,rev=2"` sets the channels
and their k; `combined=3` reproduces the previous candidate set, `combined=5,nchar=3,addr=2,rev=2` is the
default (validated end to end: `docs/REPORT.md` 14.1 / 14.4) and `combined=5` the cheaper fallback (half the
blocking time). Every pair keeps its rank in every channel
and the number of channels that proposed it (retrieval agreement) as features. On the training split the
step writes a **candidate audit** (`work/train/blocking_recall.json`): true-pair recall of the union and per
channel, the share of Source 1 entities whose complete record set was retrieved, the candidate *oracle*
macro F0.5 (a perfect classifier on this candidate set) and the split of the misses into *ambiguous* (an
address-less record whose entity has exact-name namesakes: no retrieval can single it out) and
*recoverable*. The channel configuration is chosen on end-to-end macro F0.5, not on recall alone
(`docs/REPORT.md` section 14).

`pair_features.py --token-idf` adds Source 1 IDF-weighted rare-token features; they are off by default because
they raise the out-of-fold score but lower the shifted hold-out score (they do not transfer between corpora;
`docs/REPORT.md` 14.4).

### Decision rule

`model.assign` links a Source 2/3 record to its best Source 1 candidate only when (1) the calibrated
probability clears the threshold of the pair's class, (2) the margin over the record's runner-up candidate
is at least `margin`, and (3) the pair carries no strong contradiction (different unit numbers, different
postal-like codes, identical names at contradicting addresses, or a clearly different house number on the
same street) or its probability also clears the threshold raised by `contra_penalty` (>= 1 is a hard veto).
`train.py` chooses `margin` / `contra_penalty` by *nested* macro F0.5 (thresholds fit on K-1 folds, scored
on the held-out fold) and keeps the plain threshold unless a richer rule wins by `--rule-min-gain`; the rule
is stored in every policy file and applied unchanged by `predict.py`.

### Checkpoints and the threshold policy

`train.py` writes every artefact of a run to `work/checkpoints/<name>/` (default name `ckpt_<timestamp>`):
the four cross-fitted fold models of each stage as soon as they are trained, the stage-1 OOF
probabilities, the final models, the Platt calibrator, `oof.parquet`, `validation_report.json`, a
`manifest.json` (arguments, data fingerprint, git commit, stages done) and `thresholds/global_train.json`,
the OOF-optimal global threshold as a policy file. `--resume` picks an interrupted run up at the last
saved fold model (same arguments required; `--overwrite` replaces a checkpoint). `work/checkpoints/LATEST`
names the newest checkpoint and the top-level `work/` files are links into it, so older tools keep working.

A **threshold policy** (`src/threshold_policy.py`) is a JSON file with one global threshold on the calibrated
probability plus the decision rule (`margin`, `contra_penalty`, both 0 unless `train.py --rule-search` selected
them). `select_threshold.py` writes the density-adjusted global threshold as `<checkpoint>/thresholds/selected.json`
and links it as `work/threshold_policy.json`, which `predict.py` applies; `predict.py --checkpoint <name> --policy
<file> --reuse-scores` scores any checkpoint under any policy file (the `train` / `density` / `prior` candidates are
all written) without re-running the model. Per-class thresholds (per country / source / script) were studied in
`docs/REPORT.md` section 13 and never beat the global threshold outside the fold noise; that code path was removed.

### Running on Kaggle (CPU notebook, 30 GB RAM)

**Easiest: import `kaggle/entity_resolution_kaggle.ipynb`** (*Create → Import Notebook*), attach the challenge
dataset, set *Accelerator = None*, and *Save Version → Save & Run All*. The notebook embeds every source file
(works with Internet off), finds the data files by name, runs the unit tests, a smoke test on a 0.3 % slice with a
scored hold-out, then the full pipeline (~2-2.5 h), the validator and the summary. Outputs land in
`/kaggle/working/output` (submission), `/kaggle/working/diagnostics` and `/kaggle/working/logs`. It is generated
from the sources by `python tools/build_kaggle_notebook.py`; regenerate after any code change.

Alternatively, attach the challenge data as a dataset and use one code cell with internet on:

```bash
%%bash
git clone -q https://github.com/jeetsidhu/AMAZON-ML-Challenege.git /kaggle/working/er
cd /kaggle/working/er
# build the expected layout from wherever the attached dataset put the files
mkdir -p /kaggle/tmp/dataset/train /kaggle/tmp/dataset/test
for f in train_source1 train_source2 train_source3 train_ground_truth; do ln -sf "$(find /kaggle/input -name $f.tsv | head -1)" /kaggle/tmp/dataset/train/$f.tsv; done
for f in test_source1 test_source2 test_source3; do ln -sf "$(find /kaggle/input -name $f.tsv | head -1)" /kaggle/tmp/dataset/test/$f.tsv; done
DATA=/kaggle/tmp/dataset WORK=/kaggle/tmp/work OUT=/kaggle/working/output LOGS=/kaggle/working/logs \
  bash run.sh --no-venv --skip-download
```

`--no-venv` keeps Kaggle's preinstalled LightGBM / polars / rapidfuzz. `unidecode` is installed when the
notebook has internet; without it the code falls back to accent folding (Indic tokens missing from the
learned lexicon are then dropped instead of transliterated, a small recall cost on Indian records).
Without internet, upload a zip of this repository as a dataset and copy it instead of `git clone`. `/kaggle/tmp` holds the
multi-GB intermediates outside the 20 GB `/kaggle/working` limit; the submission and logs land in
`/kaggle/working`, which is what the notebook keeps. Use *Save Version -> Save & Run All* for a run that
survives the browser closing (12 h CPU limit).

### Fast local experiments

```bash
python tools/make_subset.py --data-dir $D --out-dir subset --frac 0.05      # ~110k S1 entities, 8 min end-to-end on 4 cores
bash src/run_pipeline.sh subset work_subset out_subset
python src/evaluate.py --pred out_subset/matching_results.tsv --truth subset/test/subset_ground_truth.tsv --source1 subset/test/test_source1.tsv
python tools/ablation.py   --data-dir subset --work-dir work_subset
python tools/robustness.py --data-dir subset --work-dir work_subset
```

The subset's `test/` folder is a disjoint sample of the *training* data with twice the decoy density
(the shift observed between the real train and test splits) and its own ground truth, so the full
pipeline including threshold transfer can be scored locally. The validator is not run for it.

## Method in one paragraph

Names and addresses are normalised with rule tables: legal suffixes (strict ones anywhere, ambiguous
ones such as `PC` / `Co` / `AG` only in legal position), street-type, state and French abbreviations
(the ambiguous French ones only for French records), alias markers (`DBA:`, `f/k/a`, `t/a`, `| www…`),
domains, `(ID: …)` tags, honorifics and parenthesised country tags. Country names, articles and
initials that are part of the business name are kept (`Air India`, `La Poste`, `A&W`). Indic-script
names and state names are mapped to Latin script with a lexicon learned by aligning training pairs,
fit per validation fold. Blocking works on TF-IDF vectors (IDF from Source 1) over name tokens,
order-free name token pairs, compact-name prefixes, character 4-grams, address tokens and address
bigrams. Retrieval runs per country label (an open set) through several channels (combined cosine,
name, char n-grams, address, rare tokens, house-number keys, and the reverse direction from each
Source 1 record), each a parallel chunked sparse matrix product; the union of the channels is the exact
list the matcher scores, and its recall / entity coverage / oracle F0.5 are measured on the training
split. Each candidate gets ~100 string-similarity, overlap, number-agreement, contradiction,
retrieval-channel and candidate-context features. Stage-1 LightGBM scores the pairs; stage-2 LightGBM re-scores them
using the stage-1 probabilities of the competing candidates. Folds are assigned per Source 1
*name group* (namesakes share a fold; matched Source 2/3 records inherit their entity's fold) and the
model of fold k never sees a pair touching fold k. Stage-2 OOF probabilities are Platt-calibrated (fit on
OOF only). Assignment is many-to-one: each Source 2/3 record is linked to its best Source 1 candidate
when the calibrated probability clears the global threshold (tuned for out-of-fold macro F0.5 and
adjusted for the test split's higher decoy density) and, when a decision rule was selected, its margin
over the runner-up candidate is large enough and no strong contradiction vetoes it; the links are then
aggregated per Source 1 entity.
