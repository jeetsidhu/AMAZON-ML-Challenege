# Amazon ML Challenge 2026 — Business Entity Resolution

`amazon_ml_challenge_2026_entity_resolution.ipynb` is a single, self-contained Kaggle notebook (Python 3, CPU,
no internet) that resolves every test Source-1 business entity against the Source-2 / Source-3 records and writes
the two submission files:

* `output/matching_results.tsv` — `source1_entity_id<TAB>matched_entity_ids`
* `output/candidate_pairs.tsv` — `source1_entity_id<TAB>candidate_entity_ids` (exactly the candidate set the model scored)

Pipeline: Load → Clean → Normalize → Block (nine hashed key families, capped, ≤ 50 candidates per entity) →
77 pair features → LightGBM → macro-F0.5 threshold → outputs → official-rule validation. Model and preprocessing
state are saved to `artifacts/er_artifacts.joblib` together with `feature_importance.csv`, `threshold_grid.csv`
and `validation_metrics.json`.

## Running on Kaggle

1. Attach the challenge data as a dataset (any folder layout; prefixed uploads such as
   `1790278321851_train_source1.tsv` are discovered automatically). Required: `train_source1/2/3.tsv` and
   `train_ground_truth.tsv`; `test_source1/2/3.tsv` are needed for a submission (without them the validation split is
   written in the challenge format); `utils/validate_submission.py` is optional and run when present.
2. CPU session, internet disabled, *Run All*. Full run ≈ 1.5–2 h (`N_TRAIN_S1` / `N_VAL_S1` in `CONFIG` trade time
   for accuracy; the S2 + S3 pool is always used in full). `ER_QUICK=1` runs a 3,000 / 1,500-entity smoke test.

Dependencies (all preinstalled on Kaggle): pandas ≥ 2.2, numpy, scikit-learn, lightgbm, rapidfuzz ≥ 3.6, pyarrow,
joblib, matplotlib. Optional libraries have working fallbacks (LightGBM → HistGradientBoosting, RapidFuzz →
pure-Python scorers, pyarrow → object dtype, unidecode → skipped). No external data, APIs or pretrained models are used.
