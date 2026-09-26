# Business Entity Resolution — Team ICE

Blocking + two-stage LightGBM matcher with one-to-one assignment, leak-free cross-fitted
validation, Platt-calibrated probabilities, training checkpoints and configurable (global or
per-class) acceptance thresholds selected on out-of-fold predictions.
Only the provided challenge data is used; there are no external lookups, APIs or pretrained
language models. The only learned model is LightGBM (MIT licence), with far fewer than 8B parameters.

`docs/REPORT.md` holds the diagnosis of the previous version, the leakage audit, the before/after
metrics, fold-level results, calibration / threshold analysis, the runtime profile, the feature
ablation, the robustness tests and the list of remaining risks. Section 13 is the per-class
threshold study (global vs per-class thresholds, nested validation, hold-out results, recommendation).

## Layout

```
├── README.md
├── requirements.txt
├── docs/REPORT.md             # analysis + results of the review
├── reports/                   # JSON / markdown outputs of the experiments quoted in the report
├── tests/test_textnorm.py     # edge cases of the text normalisation (python -m pytest tests)
├── tools/
│   ├── make_subset.py         # name-group-sampled subset of the training data for fast experiments
│   ├── ablation.py            # feature-group ablation (OOF macro F0.5 / recall / precision)
│   └── robustness.py          # threshold + calibration under simulated distribution shift
└── src/
    ├── run_pipeline.sh        # end-to-end driver (all steps below, then validation)
    ├── common.py              # paths, TSV reader, logging, per-stage profiler (-> work/profile.json)
    ├── textnorm.py            # name/address normalisation + blocking-feature generation
    ├── folds.py               # step 0: validation folds BEFORE any label-derived preprocessing
    ├── build_lexicon.py       # step 0b: Indic-script -> Latin lexicon, one per held-out fold + "all"
    ├── prepare.py             # step 1: normalise every record (multiprocess) -> records.parquet
    ├── blocking.py            # step 2: TF-IDF sparse top-k retrieval per country -> candidates_raw.parquet
    ├── pair_features.py       # step 3: ~75 pairwise features -> pairs/part_*.parquet
    ├── model.py               # LightGBM helpers, stage-2 context features, assignment
    ├── thresholds.py          # vectorised macro-F0.5 sweep; scalar or per-link thresholds (shared by train / tuning)
    ├── threshold_policy.py    # global / per-class threshold policies: class keys, coordinate-ascent fit, nested comparison, JSON
    ├── checkpoint.py          # training checkpoints: manifest, fold models, OOF, calibration, policies, experiment log
    ├── calibrate.py           # Platt / isotonic calibration, prior-shift correction, ECE / Brier
    ├── train.py               # step 4: strict K-fold cross-fitted 2-stage model -> checkpoint (OOF, calibration, report, models)
    ├── leakage_check.py       # step 4a: automated leakage audit of the finished training run
    ├── select_threshold.py    # step 4b: global test threshold under the expected decoy-density / prior shift
    ├── tune_thresholds.py     # step 4c: global vs per-class policies on one checkpoint, nested comparison, selection
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
| `work/validation_report.json` | OOF metrics, full threshold sweep, per-fold metrics + variance, per-country metrics, pair-level metrics, calibration (nested), cross-fold audit, retrieval recall@k |
| `work/leakage_check.json` | result of the automated leakage audit (the pipeline stops if it fails) |
| `work/threshold_analysis.json` | training-optimal vs density-adjusted vs prior-shift global thresholds |
| `work/threshold_experiments.json` / `.md` | global vs per-class threshold policies: per-class thresholds, per-class metrics before/after, nested (leak-free) comparison, the selected policy |
| `work/threshold_policy.json` | the selected policy, read by `predict.py` |
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
  python src/blocking.py      --data-dir $D --work-dir $W --split $S   # --topk 3 = the scored candidate list
  python src/pair_features.py --data-dir $D --work-dir $W --split $S
done
python src/train.py          --data-dir $D --work-dir $W --checkpoint-name run1   # --resume continues an interrupted run
python src/leakage_check.py  --data-dir $D --work-dir $W
python src/select_threshold.py --data-dir $D --work-dir $W           # --method density|prior|train (global threshold)
python src/tune_thresholds.py --data-dir $D --work-dir $W            # global vs per-class policies -> threshold_policy.json
python src/predict.py        --data-dir $D --work-dir $W --out-dir output   # --checkpoint run1 --policy <file> --reuse-scores
```

### Checkpoints and threshold policies

`train.py` writes every artefact of a run to `work/checkpoints/<name>/` (default name `ckpt_<timestamp>`):
the four cross-fitted fold models of each stage as soon as they are trained, the stage-1 OOF
probabilities, the final models, the Platt calibrator, `oof.parquet`, `validation_report.json`, a
`manifest.json` (arguments, data fingerprint, git commit, stages done) and `thresholds/global_train.json`,
the OOF-optimal global threshold as a policy file. `--resume` picks an interrupted run up at the last
saved fold model (same arguments required; `--overwrite` replaces a checkpoint). `work/checkpoints/LATEST`
names the newest checkpoint and the top-level `work/` files are links into it, so older tools keep working.

A **threshold policy** (`src/threshold_policy.py`) is a JSON file: one global threshold, or one threshold
per *class*, where a class is a value of record attributes known at prediction time without labels -
`country` (of the Source 1 entity), `src` (2/3), `indic` (script), `noaddr`, `alias`, `domain` of the Source 2/3
record, or any combination (`country,src`). Classes unseen when the policy was fit (France) get the
policy's `default` threshold (`--unseen max|min|mean` for other choices). `tune_thresholds.py` fits the
global threshold and every per-class configuration on the checkpoint's OOF predictions (decoy-density
weighted like `select_threshold.py`; `--density per_class` estimates one ratio per class from record
counts), reports per-class metrics before and after, and compares the configurations **nested**: thresholds
refit on K-1 folds, scored on the held-out fold, so the reported gain is on entities the thresholds were
never tuned on. `--select auto` keeps the global policy unless a per-class one wins by `--min-gain` (1e-4);
`--select country` forces one. Every policy lands in `<checkpoint>/thresholds/`, the chosen one in
`work/threshold_policy.json`, and `predict.py --checkpoint <name> --policy <file> --reuse-scores` scores any
checkpoint under any policy without re-running the model (test scores are cached per checkpoint).

```bash
python src/tune_thresholds.py --data-dir $D --work-dir $W --class-by country,src      # one configuration
python src/tune_thresholds.py --data-dir $D --work-dir $W --configs my_configs.json  # [{"name":..,"class_by":[..],"objective":"macro_f05"|"fbeta"|"precision_floor","beta":..,"floor":..,"min_support":..}]
python src/predict.py --data-dir $D --work-dir $W --out-dir out_country --policy $W/checkpoints/run1/thresholds/country.json --reuse-scores
```

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
python tools/policy_holdout_eval.py --data-dir subset --work-dir work_subset --out-dir out_subset/policies   # every policy of the checkpoint on the labelled hold-out
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
order-free name token pairs, compact-name prefixes, address tokens and address bigrams. Retrieval
runs per country label (an open set), from each Source 2/3 record to its top-3 Source 1 records
(the exact list the matcher scores; recall@k is measured on the training split), using a parallel
chunked sparse matrix product. Each candidate gets ~75 string-similarity, overlap, number-agreement
and retrieval-context features. Stage-1 LightGBM scores the pairs; stage-2 LightGBM re-scores them
using the stage-1 probabilities of the competing candidates. Folds are assigned per Source 1
*name group* (namesakes share a fold) and the model of fold k never sees a pair touching fold k.
Stage-2 OOF probabilities are Platt-calibrated (fit on OOF only). Each Source 2/3 record is linked
to its best Source 1 candidate when the calibrated probability clears the threshold of its class
under the selected policy: a global threshold tuned for out-of-fold macro F0.5 and adjusted for the
test split's higher decoy density, or per-class thresholds when they win the nested comparison.
