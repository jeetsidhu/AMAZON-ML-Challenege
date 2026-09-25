# Business Entity Resolution — Team ICE

Blocking + two-stage LightGBM matcher with one-to-one assignment, leak-free cross-fitted
validation, Platt-calibrated probabilities and a shift-aware threshold.
Only the provided challenge data is used; there are no external lookups, APIs or pretrained
language models. The only learned model is LightGBM (MIT licence), with far fewer than 8B parameters.

`docs/REPORT.md` holds the diagnosis of the previous version, the leakage audit, the before/after
metrics, fold-level results, calibration / threshold analysis, the runtime profile, the feature
ablation, the robustness tests and the list of remaining risks.

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
    ├── thresholds.py          # vectorised macro-F0.5 threshold sweep (shared by train / select_threshold)
    ├── calibrate.py           # Platt / isotonic calibration, prior-shift correction, ECE / Brier
    ├── train.py               # step 4: strict K-fold cross-fitted 2-stage model -> OOF, calibration, report
    ├── leakage_check.py       # step 4a: automated leakage audit of the finished training run
    ├── select_threshold.py    # step 4b: test threshold under the expected decoy-density / prior shift
    ├── predict.py             # step 5: score test pairs, write output/*.tsv
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
| `work/threshold_analysis.json` | training-optimal vs density-adjusted vs prior-shift thresholds |
| `work/profile.json` | wall / CPU seconds per stage (cpu/wall shows the parallelism actually achieved) |
| `work/train/oof.parquet` | out-of-fold stage-1 / stage-2 / calibrated probabilities of every training pair |

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
python src/train.py          --data-dir $D --work-dir $W
python src/leakage_check.py  --data-dir $D --work-dir $W
python src/select_threshold.py --data-dir $D --work-dir $W           # --method density|prior|train
python src/predict.py        --data-dir $D --work-dir $W --out-dir output
```

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
order-free name token pairs, compact-name prefixes, address tokens and address bigrams. Retrieval
runs per country label (an open set), from each Source 2/3 record to its top-3 Source 1 records
(the exact list the matcher scores; recall@k is measured on the training split), using a parallel
chunked sparse matrix product. Each candidate gets ~75 string-similarity, overlap, number-agreement
and retrieval-context features. Stage-1 LightGBM scores the pairs; stage-2 LightGBM re-scores them
using the stage-1 probabilities of the competing candidates. Folds are assigned per Source 1
*name group* (namesakes share a fold) and the model of fold k never sees a pair touching fold k.
Stage-2 OOF probabilities are Platt-calibrated (fit on OOF only). Each Source 2/3 record is linked
to its best Source 1 candidate when the calibrated probability clears a threshold tuned for
out-of-fold macro F0.5 and adjusted for the test split's higher decoy density.
