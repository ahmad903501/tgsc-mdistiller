"""Clone the shared method notebooks into Dataset-specific Kaggle run notebooks."""
import argparse
import ast
import copy
import json
from pathlib import Path
import re


DATASETS = (
    ("resnet32x4", "resnet8x4", "aj903501", "tgsc-resnet32x4-try1"),
    ("vgg13", "vgg8", "ahmadjawwad903", "tgsc-vgg13-try1"),
    ("resnet56", "resnet20", "toxica001", "tgsc-resnet56-try1"),
    ("resnet110", "resnet32", "ahmadjawwad903", "tgsc-resnet110-try1"),
)

RESOLVE_CODE = '''import json
from importlib.metadata import version

def resolve_inputs(roots):
    matches = set()
    for root in roots:
        if not root.is_dir():
            continue
        checkpoints = [p for p in root.rglob("ckpt_epoch_240.pth")
                       if p.parent.name == TEACHER_NAME + "_vanilla"]
        banks = []
        for path in root.rglob("manifest.json"):
            identity = json.loads(path.read_text())
            if identity.get("schema") == 2 and identity.get("teacher") == TEACHER_NAME:
                if all((path.parent / f"res{r}.pt").is_file() for r in (16,20,24,28,32)):
                    banks.append(path)
        for checkpoint in checkpoints:
            for bank in banks:
                matches.add((checkpoint.resolve(), bank.resolve()))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one complete {TEACHER_NAME} teacher/bank pair; found {len(matches)}. "
            f"Attach {DATASET_URL} using Add Input, then rerun. "
            "The Dataset must contain extracted teacher_weights/ and tgsc_cache/ folders. "
            "If multiple versions/banks are attached, keep only the intended version.")
    return next(iter(matches))

TEACHER_CHECKPOINT, BANK_MANIFEST = resolve_inputs([
    INPUT_ROOT,
    Path("/kaggle/input") / DATASET_SLUG,
])
CACHE_ROOT = BANK_MANIFEST.parent.parent
identity = json.loads(BANK_MANIFEST.read_text())
for package_name, field in [("torch", "torch"), ("torchvision", "torchvision")]:
    installed = version(package_name)
    if installed != identity[field]:
        raise RuntimeError(
            f"{package_name}={installed}, but this cache was prepared with {identity[field]}. "
            "Use the preparation environment; cache identity checks must not be bypassed.")
print("Dataset:", DATASET_URL)
print("Teacher checkpoint:", TEACHER_CHECKPOINT)
print("Crop bank:", BANK_MANIFEST.parent)
print("Runtime revision:", REVISION)
'''


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def replace_assignment(source, key, value):
    source, count = re.subn(r"^" + key + r" = .*?$", lambda _: key + " = " + value,
                            source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Missing template setting: {key}")
    return source


def generate(root, revision):
    output = root / "notebooks" / "runs"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for teacher, student, owner, slug in DATASETS:
        dataset_url = f"https://www.kaggle.com/datasets/{owner}/{slug}"
        for method in ("dkd", "simkd", "reviewkd"):
            config_name = f"{method}_{teacher}_to_{student}.json"
            ready = (root / "configs/tgsc/cifar100" / config_name).is_file()
            if not ready and (teacher, method) != ("vgg13", "reviewkd"):
                raise FileNotFoundError(config_name)
            template = root / "notebooks" / f"tgsc_{method}_cifar100.ipynb"
            nb = copy.deepcopy(json.loads(template.read_text(encoding="utf-8")))
            title = f"{method.upper()} + TGSC: {teacher} -> {student}"
            intro = (f"# {title}\n\n"
                     f"Attach [{owner}/{slug}]({dataset_url}) with **Add Input** before running. "
                     "Enable GPU and Internet. No GitHub token is required. "
                     "Training uses only the supplied crop bank. Evaluation downloads only the verified 28 MB test split. "
                     "The notebook checks the teacher/cache, trains for 240 epochs, evaluates every five epochs, and saves logs/plots. "
                     "Use the same Dataset version and preparation environment across methods.\n")
            nb["cells"][0] = cell("markdown", intro)
            settings = "".join(nb["cells"][1]["source"])
            replacements = {
                "REVISION": repr(revision), "CONFIG_NAME": repr(config_name),
                "PRIVATE_REPO": "False", "GPU": "0", "SEED": "27",
                "CIFAR_SOURCE": repr("download"), "DATA_ROOT": 'Path("/kaggle/working/data")',
                "TEACHER_SOURCE": repr("kaggle"),
                "TEACHER_CHECKPOINT": f'Path("/kaggle/input/datasets/{owner}/{slug}/teacher_weights/cifar_teachers/{teacher}_vanilla/ckpt_epoch_240.pth")',
                "CACHE_ROOT": f'Path("/kaggle/input/datasets/{owner}/{slug}/tgsc_cache")',
                "OUTPUT_DIR": f'Path("/kaggle/working/tgsc_runs/{method}_{teacher}_to_{student}_seed27")',
                "RESUME": "None",
            }
            for key, value in replacements.items():
                settings = replace_assignment(settings, key, value)
            settings += (f"\nTEACHER_NAME = {teacher!r}\nDATASET_SLUG = {slug!r}\n"
                         f"DATASET_URL = {dataset_url!r}\n"
                         f'INPUT_ROOT = Path("/kaggle/input/datasets/{owner}/{slug}")\n')
            nb["cells"][1] = cell("code", settings)
            nb["cells"].insert(2, cell("code", RESOLVE_CODE))
            if not ready:
                nb["cells"][0] = cell("markdown", f"# PENDING: {title}\n\n"
                    "This run is blocked: the official VGG13 -> VGG8 ReviewKD loss weight "
                    "has not been established and no verified config exists. "
                    "The input settings are prefilled for later use, but Run All stops immediately. "
                    "Do not substitute another pair's coefficient.\n")
                nb["cells"].insert(1, cell("code", 'raise RuntimeError("VGG13 ReviewKD is pending verified pair settings; this notebook cannot train yet.")\n'))
            name = f"tgsc_{method}_{teacher}_to_{student}" + ("" if ready else "_PENDING") + ".ipynb"
            for block in nb["cells"]:
                if block["cell_type"] == "code":
                    ast.parse("".join(block["source"]))
                    block.update(execution_count=None, outputs=[])
            (output / name).write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
            records.append({"notebook": name, "teacher": teacher, "student": student,
                            "method": method.upper(), "config": config_name if ready else None,
                            "runtime_revision": revision, "dataset": f"{owner}/{slug}",
                            "training_input": "crop_bank", "evaluation_every": 5,
                            "dataset_url": dataset_url, "runnable": ready,
                            "output_dir": f"/kaggle/working/tgsc_runs/{method}_{teacher}_to_{student}_seed27"})
    (output / "run_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    lines = ["# Dataset-specific Kaggle runs", "",
             "Import one notebook into Kaggle, attach its linked Dataset with Add Input, enable GPU and Internet, then Run All.",
             "The public repository needs no token. Input paths resolve both owner-qualified and standard Kaggle mounts, including a single nested extraction folder.",
             f"Runtime/config revision: `{revision}`. Defaults: GPU 0, seed 27, 240 epochs, DKD unified gating, evaluation every 5 epochs plus final/requested stop.",
             "Training/check use only the prepared bank. Evaluation downloads the verified 28 MB test-only NPZ; no original training images are downloaded. Checkpoints and JSONL/CSV logs are saved each epoch; PNG/SVG plots update with evaluation.",
             "These notebooks have local CPU/config checks; actual Kaggle execution is still pending. Keep Dataset versions fixed across methods.",
             "", "| Notebook | Dataset | Status |", "|---|---|---|"]
    lines += [f"| [{r['notebook']}]({r['notebook']}) | [{r['dataset']}]({r['dataset_url']}) | {'Ready for Kaggle preflight' if r['runnable'] else 'BLOCKED: unverified VGG ReviewKD settings'} |" for r in records]
    lines += ["", "Keep this Dataset's preparation torch/torchvision versions. If the early environment check fails, do not edit cache manifests to bypass it.",
              "For an interrupted run, attach latest.pt and set RESUME to that path before running. Use a fresh output directory for a fresh run.",
              "If you want a one-epoch smoke run, add --stop-after-epoch 1 to the training arguments, then resume latest.pt for the remaining epochs."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Published runtime revision containing all runnable configs")
    args = parser.parse_args()
    records = generate(Path(__file__).resolve().parents[1], args.revision)
    print(f"Generated {sum(r['runnable'] for r in records)} runnable notebooks and 1 explicitly blocked notebook")
