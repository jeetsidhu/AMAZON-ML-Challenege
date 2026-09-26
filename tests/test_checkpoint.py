"""Checkpoint directory: manifest, stages, LATEST pointer and the legacy links."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from checkpoint import Checkpoint, link_or_copy  # noqa: E402


def test_manifest_stages_and_latest(tmp_path):
    work = str(tmp_path / "work")
    c = Checkpoint(work, "ckpt_a")
    assert not c.exists() and Checkpoint.resolve(work) is None
    c.update_manifest(train_args={"rounds1": 3})
    assert c.exists() and c.manifest()["train_args"] == {"rounds1": 3}
    c.mark_stage("stage1")
    c.mark_stage("stage1")
    assert c.stage_done("stage1") and not c.stage_done("stage2")
    assert c.manifest()["stages_done"] == ["stage1"]
    for f in ("stage1.txt", "model_meta.json", "calibration.json"):
        with open(c.path(f), "w") as fh:
            fh.write("{}")
    os.makedirs(c.path("thresholds"))
    with open(c.path("thresholds/selected.json"), "w") as fh:
        json.dump({"default": 0.5}, fh)
    c.set_latest()
    assert Checkpoint.resolve(work).name == "ckpt_a"
    assert Checkpoint.resolve(work, c.dir).name == "ckpt_a"
    assert os.path.exists(os.path.join(work, "model_meta.json"))
    assert json.load(open(os.path.join(work, "threshold_policy.json")))["default"] == 0.5
    assert [x.name for x in Checkpoint.list(work)] == ["ckpt_a"]
    c.log_experiment({"kind": "test", "x": 1})
    assert c.experiments()[0]["x"] == 1 and c.experiments()[0]["checkpoint"] == "ckpt_a"
    assert "files" in c.manifest() and "manifest.json" in c.manifest()["files"]


def test_link_or_copy_replaces(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "sub" / "b.txt"
    a.write_text("1")
    link_or_copy(str(a), str(b))
    assert b.read_text() == "1"
    a2 = tmp_path / "a2.txt"
    a2.write_text("2")
    link_or_copy(str(a2), str(b))
    assert b.read_text() == "2"
