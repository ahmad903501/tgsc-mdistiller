"""Generate per-run EP-KD notebooks, short end-to-end verifier, configs and portable source ZIP."""
import argparse
import copy
import json
from pathlib import Path
import zipfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, index):
    c = {"cell_type": kind, "id": f"epkd-{index}", "metadata": {}, "source": source.strip()+"\n"}
    if kind == "code":
        compile(c["source"], "EPKD notebook", "exec")
        c.update(execution_count=None, outputs=[])
    return c


GITHUB_BOOTSTRAP = '''from pathlib import Path
import json, subprocess, sys

# Clone the published implementation; no runtime ZIP/Dataset is needed.
def complete(root):
    return all((root/path).is_file() for path in REQUIRED_SOURCE_FILES)

cloned_now = False
if not (REPO/".git").exists():
    if REPO.exists() and any(REPO.iterdir()):
        raise RuntimeError("REPO contains a non-Git runtime. Select a new writable checkout directory.")
    REPO.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(["git","clone","--quiet","--no-checkout",REPO_URL,str(REPO)],check=True)
    cloned_now = True
dirty = subprocess.check_output(["git","-C",str(REPO),"status","--porcelain","--untracked-files=no"],text=True)
if dirty.strip() and not cloned_now:
    raise RuntimeError("Tracked source changes in REPO; select a fresh directory instead of overwriting them.")
available = subprocess.run(["git","-C",str(REPO),"rev-parse","--verify",REVISION+"^{commit}"],capture_output=True)
if available.returncode:
    subprocess.run(["git","-C",str(REPO),"fetch","--quiet","origin",REVISION],check=True)
subprocess.run(["git","-C",str(REPO),"checkout","--quiet","--detach",REVISION],check=True)
actual = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
if len(REVISION) != 40 or actual != REVISION:
    raise RuntimeError("Notebook must select the exact published40-character source commit.")
if not complete(REPO):
    raise RuntimeError("Selected source commit does not contain the required implementation.")
print("Source:",REPO_URL,"commit:",actual)
'''

SETUP = GITHUB_BOOTSTRAP + '''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO/"requirements-tgsc.txt")], check=True)
import torch
assert torch.cuda.is_available(), "Enable a GPU runtime; notebook training does not silently fall back to CPU."
print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name(GPU))
OUTPUT.mkdir(parents=True, exist_ok=True)
'''

INPUTS = '''# Reuse only the teacher from an attached old Dataset; do not read its crop bank.
if TEACHER_CHECKPOINT is None:
    matches = [p for p in Path("/kaggle/input").rglob("ckpt_epoch_240.pth") if p.parent.name == TEACHER+"_vanilla"]
    if len(matches) > 1:
        raise RuntimeError("Multiple teacher copies found; set TEACHER_CHECKPOINT explicitly.")
    if matches:
        TEACHER_CHECKPOINT = matches[0]
    else:
        subprocess.run([sys.executable, str(REPO/"tools/download_epkd_teacher.py"), "--teacher", TEACHER,
                        "--output-dir", str(OUTPUT/"teacher_weights")], check=True)
        TEACHER_CHECKPOINT = OUTPUT/"teacher_weights/cifar_teachers"/(TEACHER+"_vanilla")/"ckpt_epoch_240.pth"
assert Path(TEACHER_CHECKPOINT).is_file()
# Set DATA_ROOT to a folder containing cifar-100-python/, including an attached read-only Dataset.
# Otherwise torchvision downloads the original archive once, shared by all methods/seeds here.
from torchvision.datasets import CIFAR100
if not (DATA_ROOT/"cifar-100-python/train").exists():
    CIFAR100(str(DATA_ROOT), train=True, download=True)
print("Teacher:", TEACHER_CHECKPOINT, "Original CIFAR:", DATA_ROOT)
'''


def config_cell(teacher, configs, verify, revision=None):
    revision = revision or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    return f'''from pathlib import Path
REPO = Path("/kaggle/working/mdistiller_epkd_github")
REPO_URL = "https://github.com/ahmad903501/tgsc-mdistiller.git"
REVISION = {revision!r}
REQUIRED_SOURCE_FILES = ["tools/run_epkd.py", "mdistiller/epkd/runtime.py"]
DATA_ROOT = Path("/kaggle/working/data")
OUTPUT = Path("/kaggle/working/epkd_{'verification' if verify else 'runs'}")
TEACHER = {teacher!r}
TEACHER_CHECKPOINT = None  # Optional attached checkpoint; otherwise discover or download it.
CONFIGS = {configs!r}
GPU = 0  # Each process uses one GPU. Run another independent notebook on the other GPU if desired.
SEEDS = [27]
EVAL_EVERY = 5
TEST_DATA = None  # Optional verified cifar100-test.npz.
'''


RUN = '''# Explicit full-run launcher: no synthetic data, no stored crop banks.
def run(command, config, seed, output, extra=()):
    cmd = [sys.executable, str(REPO/"tools/run_epkd.py"), command,
           "--config", str(config), "--teacher-checkpoint", str(TEACHER_CHECKPOINT),
           "--data-root", str(DATA_ROOT), "--output", str(output), "--gpu", str(GPU),
           "--seed", str(seed), "--eval-every", str(EVAL_EVERY), "--download", *extra]
    if TEST_DATA:
        cmd.extend(["--test-data", str(TEST_DATA)])
    subprocess.run(cmd, cwd=REPO, check=True)

for name in CONFIGS:
    config = REPO/"configs/epkd/cifar100"/name
    for seed in SEEDS:
        output = OUTPUT/(config.stem+f"_seed{seed}")
        latest = output/"latest.pt"
        run("check", config, seed, output, ["--limit", "16"])
        run("train", config, seed, output, ["--resume", str(latest)] if latest.exists() else [])
        run("evaluate", config, seed, output, ["--export", str(output/"inference_best.pt")])
        from IPython.display import display, Image
        display(Image(filename=str(output/"training_curves.png")))
'''

VERIFY = '''# Short real-data smoke: one epoch per crop size, all three methods, then resume and export inference.
# These are verification settings, NOT the240-epoch benchmark. Test accuracy is not a research result.
for name in CONFIGS:
    original = json.loads((REPO/"configs/epkd/cifar100"/name).read_text())
    original["data"]["stages"] = [[r,1] for r in (16,20,24,28,32)]
    original["train"]["epochs"] = 5
    original["data"]["batch_size"] = 32
    original["epkd"]["passes"] = [1,1,1,1,1]  # Smoke budget only.
    path = OUTPUT/(Path(name).stem+"_SMOKE.json")
    path.write_text(json.dumps(original, indent=2))
    for seed in SEEDS:
        output = OUTPUT/(path.stem+f"_seed{seed}")
        run("check", path, seed, output, ["--limit", "16"])
        latest = output/"latest.pt"
        if not latest.exists():
            run("train", path, seed, output, ["--limit", "128", "--stop-after-epoch", "2"])
        run("train", path, seed, output, ["--limit", "128", "--resume", str(latest)])
        run("evaluate", path, seed, output, ["--limit", "128", "--export", str(output/"inference_latest.pt")])
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 5
        history = checkpoint["metrics_history"]
        assert [r["resolution"] for r in history] == [16,20,24,28,32]
        assert all(r["source_draws"] == 128 for r in history)
        assert (output/"metrics.csv").is_file() and (output/"training_curves.png").is_file()
        print("PASS:", name, "all stages, resume, logging, export and evaluation")
'''


def notebook(path, teacher, names, verify=False, revision=None):
    intro = f'''# {'End-to-end verification' if verify else 'EP-KD run'}: {teacher}

Original EP-KD top3 CAM curriculum wrapped around the **official method objectives**.
DKD keeps CE + decoupled KD; ReviewKD keeps CE + ABF/HCL; SimKD keeps feature-only
alignment and the reused teacher head (no added CE or vanilla-KD loss).
Full-image and crop objectives are summed. No TGSC quality gate or pixel cache is used.

Default full recipe:16/20/24/28/32 at48epochs each, stride8, passes3/3/3/1/1,
full-image anchor, full-image pad4/flip and crop pad2/flip. Lower-ranked28px cached
members repeat because only one grid window exists; the separate difficulty experiment
is deliberately not mixed into this integration. Trainable crop-path BatchNorm running
statistics are held fixed. FP32 implementation; original alignment and warmup preserved.

Setup clones the published code from GitHub at a pinned commit; no runtime Dataset or ZIP is needed.
On Kaggle enable Internet and a GPU.
Original CIFAR training data is required; you can attach cifar-100-python/ or download it.
Existing teacher-only inputs work, but original TGSC crop banks are not consumed.
The verification notebook uses explicit short settings; full notebooks retain240epochs.
Logs/CSV/PNG/SVG, latest/best checkpoints and teacher-free exports are saved in OUTPUT.
Evaluation is every5epochs plus final/requested stop. Resume rejects changed science/data/code.
'''
    run_code = RUN.split("for name in CONFIGS:")[0] + VERIFY if verify else RUN
    cells = [cell("markdown", intro, 0), cell("code", config_cell(teacher, names, verify, revision), 1),
             cell("code", SETUP, 2), cell("code", INPUTS, 3), cell("code", run_code, 4)]
    nb = {"cells": cells, "metadata": {
          "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "codemirror_mode": {"name": "ipython", "version": 3},
                            "file_extension": ".py", "mimetype": "text/x-python",
                            "nbconvert_exporter": "python", "pygments_lexer": "ipython3"}},
          "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(ROOT.parent/"kaggle_notebooks/EPKD_RUNTIME.zip"))
    parser.add_argument("--revision", help="Published implementation commit; defaults to current HEAD")
    args = parser.parse_args()
    revision = args.revision or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    target = ROOT/"configs/epkd/cifar100"
    target.mkdir(parents=True, exist_ok=True)
    runs = ROOT/"notebooks/runs"
    old = json.loads((runs/"run_manifest.json").read_text())
    manifest = []
    for entry in old:
        if not entry["runnable"]:
            continue
        raw = json.loads((ROOT/"configs/tgsc/cifar100"/entry["config"]).read_text())
        raw["data"]["stages"] = [[r,48] for r in (16,20,24,28,32)]
        raw["optim"]["milestones"] = [48,144,192,210]
        raw["epkd"] = {"passes": [3,3,3,1,1], "strides": [8]*5, "anchor_weight": 1., "view_weight": 1., "proposal_batch_size": 256}
        raw["profile_source"] = "Original EP-KD1e75cd1 curriculum + existing official-method options"
        raw.pop("tgsc", None)
        raw["train"]["run_name"] = "epkd_"+Path(entry["config"]).stem
        raw["train"]["save_path"] = "epkd_runs/"+Path(entry["config"]).stem+"/model.pth"
        (target/entry["config"]).write_text(json.dumps(raw, indent=2), encoding="utf-8")
        name = "epkd_"+Path(entry["config"]).stem+".ipynb"
        notebook(runs/name, entry["teacher"], [entry["config"]], revision=revision)
        manifest.append({"notebook": name, "config": entry["config"], "method": raw["method"],
                         "teacher": entry["teacher"], "student": entry["student"], "training_input": "original_cifar100",
                         "pipeline": "epkd", "runtime_revision": revision, "runnable": True})
    (runs/"epkd_run_manifest.json").write_text(json.dumps(manifest, indent=2))
    names = [f"{m}_resnet56_to_resnet20.json" for m in ("dkd", "simkd", "reviewkd")]
    notebook(runs/"epkd_verify_end_to_end.ipynb", "resnet56", names, verify=True, revision=revision)
    bundle = Path(args.bundle)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        paths = [*ROOT.glob("mdistiller/**/*.py"), *ROOT.glob("tools/*.py"), *target.glob("*.json"), ROOT/"requirements-tgsc.txt", ROOT/"LICENSE"]
        for path in sorted(set(paths)):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
    print(f"Generated {len(manifest)} full-run notebooks, one verification notebook, and {bundle}")


if __name__ == "__main__":
    main()
