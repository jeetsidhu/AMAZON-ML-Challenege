"""Generates kaggle/entity_resolution_kaggle.ipynb: a self-contained Kaggle notebook.

The notebook embeds every source file as a %%writefile cell (so it works with Internet OFF and
so that multiprocessing's spawn start method finds the modules on disk), locates the attached
dataset by file name, runs the unit tests, a smoke test on a 0.3 % slice with a scored hold-out,
then the full pipeline, the validator and the summary. Regenerate after any source change:

    python tools/build_kaggle_notebook.py
"""
import os

import nbformat as nbf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ER = "/kaggle/working/er"
FILES = ["src/common.py", "src/textnorm.py", "src/phonetic.py", "src/metrics.py", "src/folds.py", "src/build_lexicon.py",
         "src/prepare.py", "src/blocking.py", "src/pair_features.py", "src/model.py", "src/thresholds.py", "src/calibrate.py",
         "src/train.py", "src/leakage_check.py", "src/select_threshold.py", "src/decode.py", "src/predict.py",
         "src/evaluate.py", "src/validate_submission.py", "tools/make_subset.py", "tools/summarize_reports.py",
         "tests/test_textnorm.py", "tests/test_calibrate.py"]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# Business entity resolution - full pipeline (Kaggle, CPU)

Self-contained: every source file of the repository is written to `/kaggle/working/er` by the cells
below (no internet needed), the attached dataset is located by file name, then the notebook runs the
unit tests, a **smoke test** (whole pipeline on a 0.3 % slice with a scored hold-out, ~3 min) and the
**full pipeline** (~2-2.5 h on the 4 Kaggle CPU threads). Outputs:

* `/kaggle/working/output/matching_results.tsv`, `candidate_pairs.tsv` - the submission
* `/kaggle/working/diagnostics/` - validation report, leakage check, threshold analysis, runtime profile, summary
* `/kaggle/working/logs/` - one log per step

Settings: *Accelerator = None*, *Internet* not required. Use **Save Version -> Save & Run All** for the full run.""")

code('''# ---------------------------------------------------------------- settings
SMOKE_ONLY = False          # True: stop after the smoke test (~3 min)
SKIP_SMOKE = False          # True: go straight to the full run
ROUNDS1, ROUNDS2 = 150, 100 # LightGBM boosting rounds (stage 1 / stage 2); 300 / 200 = original, same score, 2x slower
DATASET_ROOT = "/kaggle/input"   # searched recursively for train_source1.tsv etc.

import os, sys, subprocess, time, json, shutil, glob
ER = "/kaggle/working/er"
OUT = "/kaggle/working/output"
LOGS = "/kaggle/working/logs"
DIAG = "/kaggle/working/diagnostics"
WORK = "/kaggle/temp/er_work" if os.access("/kaggle/temp", os.W_OK) else "/kaggle/working/er_work"
for d in (ER, OUT, LOGS, DIAG, WORK, f"{ER}/src", f"{ER}/tools", f"{ER}/tests"):
    os.makedirs(d, exist_ok=True)
print("python", sys.version.split()[0], "| cpus", os.cpu_count(), "| work dir", WORK)''')

code('''# ---------------------------------------------------------------- environment check
import importlib
need = {"polars": "1.0", "lightgbm": "4.0", "rapidfuzz": "3.0", "scipy": "1.10", "numpy": "1.24", "pyarrow": "10.0"}
for mod, minv in need.items():
    try:
        m = importlib.import_module(mod)
        v = tuple(int(x) for x in m.__version__.split(".")[:2])
        ok = v >= tuple(int(x) for x in minv.split("."))
        print(f"{mod:10s} {m.__version__:10s} {'ok' if ok else 'TOO OLD (need >= ' + minv + ')'}")
        assert ok
    except ImportError:
        print(f"{mod:10s} MISSING -> pip install {mod}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=True)
try:
    import unidecode; print("unidecode  ", unidecode.__version__ if hasattr(unidecode, "__version__") else "ok")
except ImportError:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unidecode"])
    print("unidecode   installed" if r.returncode == 0 else "unidecode   not available (no internet): accent-folding fallback is used")
import pytest; print("pytest      ok")''')

md("## Source files")
for f in FILES:
    src = open(os.path.join(ROOT, f), encoding="utf-8").read()
    code(f"%%writefile {ER}/{f}\n{src}")

code(f'''# ---------------------------------------------------------------- locate the dataset
DATA = f"{{WORK}}/dataset"
for split in ("train", "test"):
    os.makedirs(f"{{DATA}}/{{split}}", exist_ok=True)
names = {{"train": ["train_source1", "train_source2", "train_source3", "train_ground_truth"],
         "test": ["test_source1", "test_source2", "test_source3"]}}
missing = []
for split, fs in names.items():
    for f in fs:
        hits = glob.glob(f"{{DATASET_ROOT}}/**/{{f}}.tsv", recursive=True)
        if not hits:
            missing.append(f); continue
        dst = f"{{DATA}}/{{split}}/{{f}}.tsv"
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(hits[0], dst)
        print(f"{{f}}.tsv  <-  {{hits[0]}}  ({{os.path.getsize(hits[0]) / 1e6:.0f}} MB)")
assert not missing, f"not found under {{DATASET_ROOT}}: {{missing}} - attach the challenge dataset to the notebook"''')

code(f'''# ---------------------------------------------------------------- step runner
os.chdir(ER)
def run_step(name, args, cwd=ER):
    """Runs one pipeline script as a subprocess with its own log; raises on failure."""
    log = f"{{LOGS}}/{{name}}.log"
    t0 = time.time()
    print(f"[{{time.strftime('%H:%M:%S')}}] start {{name}}", flush=True)
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable] + args, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(open(log).read()[-4000:])
        raise RuntimeError(f"{{name}} failed after {{dt:.0f}}s (log: {{log}})")
    print(f"[{{time.strftime('%H:%M:%S')}}] done  {{name}} in {{dt:.0f}}s", flush=True)

print(subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ER, capture_output=True, text=True).stdout[-400:])''')

md("## Smoke test\nThe whole pipeline on a 0.3 % name-group slice of the training data, scored on a disjoint labelled hold-out. Catches environment problems in minutes instead of hours.")
code('''if not SKIP_SMOKE:
    SD, SW, SO = f"{WORK}/smoke/data", f"{WORK}/smoke/work", f"{WORK}/smoke/out"
    shutil.rmtree(f"{WORK}/smoke", ignore_errors=True)
    run_step("smoke_00_subset", ["tools/make_subset.py", "--data-dir", DATA, "--out-dir", SD, "--frac", "0.003", "--test-frac", "0.002"])
    run_step("smoke_01_folds", ["src/folds.py", "--data-dir", SD, "--work-dir", SW])
    run_step("smoke_02_lexicon", ["src/build_lexicon.py", "--data-dir", SD, "--work-dir", SW])
    for sp in ("train", "test"):
        run_step(f"smoke_03_prepare_{sp}", ["src/prepare.py", "--data-dir", SD, "--work-dir", SW, "--split", sp])
        run_step(f"smoke_04_blocking_{sp}", ["src/blocking.py", "--data-dir", SD, "--work-dir", SW, "--split", sp])
        run_step(f"smoke_05_features_{sp}", ["src/pair_features.py", "--data-dir", SD, "--work-dir", SW, "--split", sp])
    run_step("smoke_06_train", ["src/train.py", "--data-dir", SD, "--work-dir", SW, "--rounds1", "30", "--rounds2", "20"])
    run_step("smoke_07_leakage_check", ["src/leakage_check.py", "--data-dir", SD, "--work-dir", SW, "--canary-rows", "50000", "--canary-rounds", "10"])
    run_step("smoke_08_select_threshold", ["src/select_threshold.py", "--data-dir", SD, "--work-dir", SW])
    run_step("smoke_09_predict", ["src/predict.py", "--data-dir", SD, "--work-dir", SW, "--out-dir", SO])
    run_step("smoke_10_validate", ["src/validate_submission.py", "--matching", f"{SO}/matching_results.tsv", "--candidate", f"{SO}/candidate_pairs.tsv", "--test-dir", f"{SD}/test"])
    run_step("smoke_11_evaluate", ["src/evaluate.py", "--pred", f"{SO}/matching_results.tsv", "--truth", f"{SD}/test/subset_ground_truth.tsv", "--out", f"{SW}/smoke_eval.json"])
    e = json.load(open(f"{SW}/smoke_eval.json"))["overall"]
    print(f"SMOKE hold-out: macro F0.5 {e['macro_f05']:.4f}  precision {e['micro_precision']:.4f}  recall {e['micro_recall']:.4f}")
    print("smoke test OK: every stage ran end to end in this environment")
if SMOKE_ONLY:
    print("SMOKE_ONLY=True: the full-pipeline cells below are skipped")''')

md("## Full pipeline")
code('''if not SMOKE_ONLY:
    run_step("01_folds", ["src/folds.py", "--data-dir", DATA, "--work-dir", WORK])
    run_step("02_lexicon", ["src/build_lexicon.py", "--data-dir", DATA, "--work-dir", WORK])
    for sp in ("train", "test"):
        run_step(f"03_prepare_{sp}", ["src/prepare.py", "--data-dir", DATA, "--work-dir", WORK, "--split", sp])
        run_step(f"04_blocking_{sp}", ["src/blocking.py", "--data-dir", DATA, "--work-dir", WORK, "--split", sp])
        run_step(f"05_features_{sp}", ["src/pair_features.py", "--data-dir", DATA, "--work-dir", WORK, "--split", sp])
    run_step("06_train", ["src/train.py", "--data-dir", DATA, "--work-dir", WORK, "--rounds1", str(ROUNDS1), "--rounds2", str(ROUNDS2)])
    run_step("07_leakage_check", ["src/leakage_check.py", "--data-dir", DATA, "--work-dir", WORK])
    run_step("08_select_threshold", ["src/select_threshold.py", "--data-dir", DATA, "--work-dir", WORK])
    run_step("09_predict", ["src/predict.py", "--data-dir", DATA, "--work-dir", WORK, "--out-dir", OUT])
    run_step("10_validate", ["src/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv", "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", f"{DATA}/test"])''')

md("## Results")
code('''if not SMOKE_ONLY:
    for f in glob.glob(f"{WORK}/*.json") + glob.glob(f"{WORK}/train/*.json") + [f"{WORK}/stage1.txt", f"{WORK}/stage2.txt", f"{WORK}/lexicon.json"]:
        if os.path.exists(f):
            shutil.copy(f, DIAG)
    summary = subprocess.run([sys.executable, "tools/summarize_reports.py", "--work-dir", WORK], capture_output=True, text=True).stdout
    open(f"{DIAG}/summary.md", "w").write(summary)
    print(summary)
    print(open(f"{LOGS}/10_validate.log").read()[-600:])
    for f in sorted(glob.glob(f"{OUT}/*")):
        print(f"{os.path.getsize(f) / 1e6:8.1f} MB  {f}")
    print("FULL RUN COMPLETE: submission in", OUT)''')

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                  "language_info": {"name": "python"}})
out = os.path.join(ROOT, "kaggle", "entity_resolution_kaggle.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
