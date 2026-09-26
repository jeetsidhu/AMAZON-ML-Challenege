"""Training checkpoints: everything one trained model needs to be re-evaluated later.

Layout: <work>/checkpoints/<name>/
    manifest.json        what was trained, on which data, with which arguments (+ git commit, timing)
    stage1_fold<k>.txt   cross-fitted fold models (written as soon as each one is trained: a crashed
    stage2_fold<k>.txt   run resumes from them with train.py --resume)
    oof_stage1.npy       out-of-fold stage-1 probabilities (resume point for stage 2)
    stage1.txt           final models (refit on the whole sample)
    stage2.txt
    calibration.json     Platt calibrator fit on the OOF predictions
    model_meta.json      feature lists, the OOF-optimal global threshold, feature importances
    oof.parquet          OOF predictions of every training pair (p1, p2, p2_cal, folds, label)
    validation_report.json
    thresholds/          policies fit on this checkpoint by tune_thresholds.py (one JSON each)
    experiments.jsonl    one line per threshold-tuning / hold-out evaluation run on this checkpoint

<work>/checkpoints/LATEST names the most recent checkpoint. The top-level files that older
tools read (<work>/stage1.txt, model_meta.json, calibration.json, train/oof.parquet, ...) are
links to the latest checkpoint, so every existing script keeps working.
"""
import datetime as dt
import json
import os
import shutil
import subprocess

FINAL_FILES = ["stage1.txt", "stage2.txt", "calibration.json", "model_meta.json", "validation_report.json"]


def _git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def data_fingerprint(data_dir):
    """Sizes + mtimes of the challenge files: enough to notice that a checkpoint was trained on other data."""
    fp = {}
    for split in ("train", "test"):
        d = os.path.join(data_dir, split)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".tsv"):
                    st = os.stat(os.path.join(d, f))
                    fp[f"{split}/{f}"] = {"bytes": st.st_size, "mtime": int(st.st_mtime)}
    return fp


def link_or_copy(src, dst):
    """Relative symlink dst -> src (copy where symlinks are unavailable); replaces whatever was at dst."""
    if os.path.lexists(dst):
        os.remove(dst)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    try:
        os.symlink(os.path.relpath(src, os.path.dirname(os.path.abspath(dst))), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


class Checkpoint:
    def __init__(self, work_dir, name):
        self.work_dir, self.name = work_dir, name
        self.dir = os.path.join(work_dir, "checkpoints", name)

    # ---- lookup
    @staticmethod
    def default_name():
        return dt.datetime.now().strftime("ckpt_%Y%m%d_%H%M%S")

    @classmethod
    def resolve(cls, work_dir, name=None):
        """name: a checkpoint name, a path to a checkpoint directory, or None for LATEST."""
        if name and os.path.isdir(name) and os.path.exists(os.path.join(name, "manifest.json")):
            return cls(os.path.dirname(os.path.dirname(os.path.abspath(name))), os.path.basename(os.path.normpath(name)))
        if name is None:
            latest = os.path.join(work_dir, "checkpoints", "LATEST")
            if not os.path.exists(latest):
                return None
            with open(latest) as f:
                name = f.read().strip()
        return cls(work_dir, name)

    @classmethod
    def list(cls, work_dir):
        root = os.path.join(work_dir, "checkpoints")
        if not os.path.isdir(root):
            return []
        out = []
        for n in sorted(os.listdir(root)):
            c = cls(work_dir, n)
            if c.exists():
                out.append(c)
        return out

    def exists(self):
        return os.path.exists(self.path("manifest.json"))

    def path(self, fname):
        return os.path.join(self.dir, fname)

    def has(self, fname):
        return os.path.exists(self.path(fname))

    # ---- manifest
    def manifest(self):
        if not self.exists():
            return {}
        with open(self.path("manifest.json")) as f:
            return json.load(f)

    def update_manifest(self, **fields):
        os.makedirs(self.dir, exist_ok=True)
        m = self.manifest()
        if not m:
            m = {"name": self.name, "created": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": _git_commit(), "stages_done": []}
        m.update(fields)
        m["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        m["files"] = {f: os.path.getsize(self.path(f)) for f in sorted(os.listdir(self.dir)) if os.path.isfile(self.path(f))}
        tmp = self.path("manifest.json.tmp")
        with open(tmp, "w") as f:
            json.dump(m, f, indent=1)
        os.replace(tmp, self.path("manifest.json"))
        return m

    def mark_stage(self, stage):
        m = self.manifest()
        done = list(m.get("stages_done", []))
        if stage not in done:
            done.append(stage)
        self.update_manifest(stages_done=done)

    def stage_done(self, stage):
        return stage in self.manifest().get("stages_done", [])

    # ---- models
    def save_model(self, booster, fname):
        os.makedirs(self.dir, exist_ok=True)
        tmp = self.path(fname + ".tmp")
        booster.save_model(tmp)
        os.replace(tmp, self.path(fname))

    def load_model(self, fname):
        import lightgbm as lgb
        return lgb.Booster(model_file=self.path(fname))

    # ---- experiments log
    def log_experiment(self, record):
        os.makedirs(self.dir, exist_ok=True)
        record = {"time": dt.datetime.now().isoformat(timespec="seconds"), "checkpoint": self.name, **record}
        with open(self.path("experiments.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

    def experiments(self):
        if not self.has("experiments.jsonl"):
            return []
        with open(self.path("experiments.jsonl")) as f:
            return [json.loads(line) for line in f if line.strip()]

    # ---- publish as the current model
    def set_latest(self):
        """Marks this checkpoint as LATEST and points the legacy top-level files at it."""
        os.makedirs(os.path.join(self.work_dir, "checkpoints"), exist_ok=True)
        with open(os.path.join(self.work_dir, "checkpoints", "LATEST"), "w") as f:
            f.write(self.name + "\n")
        for fname in FINAL_FILES:
            if self.has(fname):
                link_or_copy(self.path(fname), os.path.join(self.work_dir, fname))
        if self.has("oof.parquet"):
            link_or_copy(self.path("oof.parquet"), os.path.join(self.work_dir, "train", "oof.parquet"))
        if self.has(os.path.join("thresholds", "selected.json")):
            link_or_copy(self.path(os.path.join("thresholds", "selected.json")), os.path.join(self.work_dir, "threshold_policy.json"))

    def summary(self):
        m = self.manifest()
        return {"name": self.name, "created": m.get("created"), "git_commit": m.get("git_commit"), "stages_done": m.get("stages_done"),
                "train_args": m.get("train_args"), "oof_macro_f05": m.get("oof_macro_f05"), "global_threshold": m.get("global_threshold")}
