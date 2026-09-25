# Business Entity Resolution — Team ICE

Blocking + two-stage LightGBM matcher with one-to-one assignment.
Only the provided challenge data is used; there are no external lookups, APIs or pretrained
language models. The only learned model is LightGBM (MIT licence), with far fewer than 8B parameters.

## Layout

```
business_entity_resolution/
├── README.md
├── requirements.txt
└── src/
    ├── run_pipeline.sh        # end-to-end driver (all steps below, then validation)
    ├── common.py              # paths, TSV reader, logging
    ├── textnorm.py            # name/address normalisation + blocking-feature generation
    ├── build_lexicon.py       # step 0: learn Indic-script -> Latin lexicon from TRAIN ground truth
    ├── prepare.py             # step 1: normalise every record (multiprocess) -> records.parquet
    ├── blocking.py            # step 2: TF-IDF sparse top-k retrieval per country -> candidates_raw.parquet
    ├── pair_features.py       # step 3: candidate selection + ~60 pairwise features -> pairs/part_*.parquet
    ├── model.py               # LightGBM helpers, stage-2 context features, assignment
    ├── train.py               # step 4: K-fold cross-fitted 2-stage model -> OOF predictions + final models
    ├── select_threshold.py    # step 4b: threshold maximising (test-decoy-density adjusted) OOF macro F0.5
    ├── predict.py             # step 5: score test pairs, write output/*.tsv
    ├── metrics.py             # macro F0.5 exactly as the challenge defines it
    └── validate_submission.py # official validator (copied from student_resource/utils)
```

## Environment

Tested on macOS (Apple M4, 10 cores, 16 GB RAM) with Python 3.14.0.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce end-to-end

`DATA` is the challenge `dataset/` directory, which contains `train/` and `test/`.

```bash
cd code/business_entity_resolution
bash src/run_pipeline.sh /path/to/student_resource/dataset work output
```

This writes `output/matching_results.tsv` (the leaderboard file) and
`output/candidate_pairs.tsv` (the exact candidate set scored by the model). It then runs the
official validator. The steps can also be run one at a time:

```bash
D=/path/to/student_resource/dataset; W=work
python src/build_lexicon.py  --data-dir $D --work-dir $W
python src/prepare.py        --data-dir $D --work-dir $W --split train
python src/blocking.py       --data-dir $D --work-dir $W --split train
python src/pair_features.py  --data-dir $D --work-dir $W --split train
python src/prepare.py        --data-dir $D --work-dir $W --split test
python src/blocking.py       --data-dir $D --work-dir $W --split test
python src/pair_features.py  --data-dir $D --work-dir $W --split test
python src/train.py          --data-dir $D --work-dir $W
python src/select_threshold.py --data-dir $D --work-dir $W
python src/predict.py        --data-dir $D --work-dir $W --out-dir output
```

Approximate wall-clock times on the machine above:

| step | train | test |
| --- | --- | --- |
| prepare (normalisation) | ~2 min | ~2 min |
| blocking | ~12 min | ~12 min |
| pair features | ~3 min | ~3 min |
| train (8 CV models + 2 final) | ~10 min | – |
| select threshold | ~10 s | – |
| predict + write | – | ~3 min |

Peak memory is about 14 GB, during blocking. All randomness is seeded: fold hashing uses a fixed seed and row
sampling uses `--seed 2026`.

## Method in one paragraph

Names and addresses are normalised with rule tables: legal suffixes, street-type, state and
French abbreviations, alias markers (`DBA:`, `f/k/a`, `t/a`, `| www…`), domains, `(ID: …)` tags and
honorifics. Indic-script names and state names are mapped to Latin script with a lexicon learned
by aligning training pairs. Blocking works on TF-IDF vectors (IDF from Source 1) over name tokens,
order-free name token pairs, compact-name prefixes, address tokens and address bigrams. Retrieval
runs per country label (an open set), from each Source 2/3 record to its top 3 Source 1 records,
using a parallel chunked sparse matrix product. Each candidate gets about 60 string-similarity,
overlap, number-agreement and retrieval-context features. Stage-1 LightGBM scores the pairs.
Stage-2 LightGBM re-scores them using the stage-1 probabilities of the competing candidates.
Since a Source 2/3 record belongs to at most one entity, each one is linked only to its best
Source 1 candidate, and only when the probability clears a threshold tuned for out-of-fold
macro F0.5 (θ = 0.75, adjusted for the test split's higher density of decoy records).

Results: macro F0.5 is 0.9725 out-of-fold over all 2.2M training Source 1 entities (θ=0.70),
and 0.9723 at the submitted θ=0.75.
