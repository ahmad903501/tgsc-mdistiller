# Running official distillation methods with TGSC

Supported first delivery: DKD, SimKD, ReviewKD. All runtime code is in this checkout. Sibling reference repos/PDFs are not needed on Kaggle or a local GPU machine. FitNet remains deferred.

## Environment

Use a compatible torch/torchvision GPU installation supplied by Kaggle or install the matched pair for your local CUDA environment. Do not blindly install the old torch pins from the upstream `requirements.txt`.

From the repository root:

```bash
python -m pip install -e . --no-deps
python -m pip install -r requirements-tgsc.txt
python tools/run_tgsc.py --help
```

The TGSC runner uses JSON profiles and does not require YACS, TensorBoard, or W&B. Upstream `tools/train.py` still needs its original auxiliary dependencies. Verified here: Python 3.13, torch 2.6 CPU, torchvision 0.21 CPU; CUDA/Kaggle execution is not yet verified. No CPU training fallback is enabled by the CLI. `check --cpu-check` is available for offline checks with supplied weights/data/cache.

## Minimal upload packages (existing preparation ZIPs)

To avoid uploading CIFAR and duplicated preparation downloads, package existing saved-output ZIPs with:

```bash
python tools/package_tgsc_upload.py --source-dir ../datasets_cache --output-dir ../datasets_cache/kaggle_upload
```

The tool preserves originals and creates one teacher ZIP containing only its checkpoint, bank manifest and res16/res20/res24/res28/res32 crop files. It omits CIFAR (extracted and tar.gz), the combined teacher tar, repository/.git, logs, and reference.pt (used only to continue preparation). Info-ZIP, when installed, copies compressed entries without recompression; otherwise Python uses a portable streaming fallback. It verifies bank identity, teacher SHA-256 and output contents; copy mode also verifies compressed payload bytes. It refuses to overwrite existing outputs. The upload_manifest.json records exact inputs, identities and sizes.

On 2026-10-28 the supplied four ZIPs (ResNet32x4, VGG13, ResNet56, ResNet110) yielded about 6.1 GiB of upload packages from 9.8 GiB of originals. WRN40-2 and ResNet50 were not present. Upload the four generated tgsc_*.zip files; copy actual attached paths from Kaggle after expansion. A Dataset containing all four packages can share teacher_weights/ and tgsc_cache/ because checkpoint/bank directories are unique.

Training notebooks now default to CIFAR_SOURCE="download" and DATA_ROOT=/kaggle/working/data. Keep TEACHER_SOURCE="kaggle" and set TEACHER_CHECKPOINT/CACHE_ROOT to attached inputs. This removes CIFAR from the upload, not from the runtime: the current runner still downloads original train files for cache identity/label checks and uses test files for evaluation. No cache-only training bypass or loss/augmentation change was introduced.

The inspected real cache manifests record torch 2.11.0+cu128 / torchvision 0.26.0+cu128; strict reuse requires that preparation environment and the same normalized source identity. Linux source identity matches the archived sources; Windows Path sorting orders mixed-case filenames differently, so do not reinterpret that platform-specific hash difference as changed crop/model code or rewrite cache manifests.

## Inputs and configs

Provide CIFAR-100 in torchvision format: the data root contains the extracted `cifar-100-python/` directory. `--download` explicitly enables torchvision downloading into the selected data root; use a writable root for downloads. Teacher checkpoints accept `model`, `state_dict`, or a direct state dict and load strictly. Use trusted teacher artifacts compatible with the selected architecture; the runner never assumes personal checkpoint paths.

All four notebooks support two teacher sources through `TEACHER_SOURCE`:

- `"download"`: download the official combined teacher archive (about 348 MiB), extract only the teacher selected by `CONFIG_NAME`, compare its SHA-256 against the archive, then verify strict loading and a finite 100-class CPU forward. The official release does not expose individual teacher assets. Existing archives are reused.
- `"kaggle"`: use the attached single-model checkpoint at `TEACHER_CHECKPOINT`. Preflight loads it strictly into the configured architecture; cache identity also uses the checkpoint hash.

Preparation defaults to downloading both inputs. Method notebooks attach teacher weights and download CIFAR to writable storage; an attached CIFAR Dataset remains optional. No `gdown` dependency or nonexistent `download_ckpts.sh` is used.

For CIFAR-100, `CIFAR_SOURCE="download"` uses `torchvision.datasets.CIFAR100`, matching TGSC, with a writable `DATA_ROOT` (default `/kaggle/working/data` in preparation). `CIFAR_SOURCE="kaggle"` reads an attached Dataset with `download=False`; set `DATA_ROOT` to the parent of `cifar-100-python/`. CSV/PNG-only Kaggle copies need conversion and are not accepted directly. The same original files/order/labels must be used for preparation and training. See [torchvision CIFAR100](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html).

Equivalent local command:

```bash
python tools/download_teachers.py --config configs/tgsc/cifar100/dkd_resnet32x4_to_resnet8x4.json --output-dir download_ckpts
```

To prepare a small upload folder locally from an already downloaded archive:

```bash
python tools/download_teachers.py --teacher resnet32x4 --archive download_ckpts/cifar_teachers.tar --output-dir download_ckpts/kaggle/resnet32x4
```

Upload **the contents of `download_ckpts/kaggle/resnet32x4/`** as a Kaggle Dataset (e.g. `tgsc-teacher-resnet32x4`). It contains only the selected checkpoint under `cifar_teachers/resnet32x4_vanilla/` and `teacher_manifest.json`, which records its portable relative path, teacher name, official source and SHA-256. Do not include the 348 MiB archive. For this layout set `TEACHER_SOURCE="kaggle"` and `TEACHER_CHECKPOINT=/kaggle/input/tgsc-teacher-resnet32x4/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth`; adjust to the actual mounted path. Change `--teacher` for another model. No weights are committed to Git, and no Dataset upload is automatic.

For CIFAR, preserve `DATA_ROOT/cifar-100-python/` as a separate Dataset after the first download; attach it in later sessions with `CIFAR_SOURCE="kaggle"`. Keep the crop bank in its own reusable Dataset.

The verified archive directory convention is `cifar_teachers/<teacher>_vanilla/ckpt_epoch_240.pth`. Case matters (`ResNet50`, versus `resnet32x4`). Actual archive entries and strict-load results are recorded in `TGSC_TEACHER_ARCHIVE_CHECKS.json`. The legacy official checkpoints include NumPy scalar metadata; the loader allows only the required scalar/dtype constructors while retaining `weights_only=True`. This compatibility fix requires updated code, so previously pinned revisions should be advanced before first real-checkpoint runs.

Profiles are in `configs/tgsc/cifar100/`:

- DKD and SimKD: existing TGSC ResNet32x4 -> ResNet8x4, WRN40-2 -> WRN40-1, VGG13 -> VGG8 profiles.
- ReviewKD: the first two existing profiles; their official script review weight is 5.
- All three methods also have an explicitly derived ResNet32x4 -> ShuffleV2 profile, reusing the ResNet teacher's crop settings with official student LR .01. Review weight is 8.
- VGG ReviewKD is intentionally not supplied: its official pair coefficient has not been established from the active released scripts. Other pairs need explicit verified profiles before an official-default claim.

The original profile source is recorded in JSON. Crop selection/formulas, batch 128, five 48-epoch stages, default augmentation .5/2, and each existing pair's LR schedule are preserved. Method objectives/weights come from audited official implementations. `method_options.gate_mode` can be `unified` or `decoupled` for DKD. No separate CE is added to SimKD.

### Additional teacher preparation configs

The following derived DKD configs extend the existing ResNet32x4 TGSC profile without changing its crop schedule, scoring, selection, multiplicities or augmentation. Published teacher/student pairs and DKD beta are taken from the official DKD YAMLs. MobileNetV2 uses the official pair LR .01; the other new students retain .05. These are derived TGSC profiles, not previously supplied pair JSONs.

| Teacher | CONFIG_NAME | Suggested OUTPUT_DIR suffix |
|---|---|---|
| ResNet56 | dkd_resnet56_to_resnet20.json | prepare_resnet56 |
| ResNet110 | dkd_resnet110_to_resnet32.json | prepare_resnet110 |
| ResNet50 | dkd_ResNet50_to_MobileNetV2.json | prepare_ResNet50 |

Use the preparation notebook with TEACHER_SOURCE="download" and the same TEACHER_DOWNLOAD_ROOT and CACHE_WRITE_ROOT. The actual registry/checkpoint capitalization matters for ResNet50. Run each teacher sequentially, rerunning all dependent notebook cells after changing CONFIG_NAME. Refresh the checkout to a revision containing these configs after any active preparation finishes; earlier pinned revisions do not contain them. Reuse the same original CIFAR files and keep all generated bank hash directories in the final Dataset.

These additions are checked for crop preparation and DKD, not a claim that corresponding SimKD/ReviewKD configs have been provided. CPU checks with official teacher weights and two synthetic images covered oracle/cache creation and DKD backward at all five crop sizes; evidence is in TGSC_ADDITIONAL_TEACHER_CHECKS.json. Full-data Kaggle runs remain unverified.

## Prepare a bank once per teacher

Example (replace paths for your system):

```bash
python tools/run_tgsc.py prepare --config configs/tgsc/cifar100/dkd_resnet32x4_to_resnet8x4.json --gpu 0 --teacher-checkpoint /weights/teacher.pth --data-root /data/cifar --cache-root /cache/tgsc --cache-write-root /cache/tgsc
```

The bank is `<cache-root>/<identity-hash>/res16.pt` etc., plus a manifest and full-image teacher reference. The hash includes dataset content/order, teacher checkpoint, oracle/model source, normalization and score/selection settings. It excludes method/student. Different methods/students can reuse the same teacher bank. Cache schema 2 stores raw r1/r2/r3, q, labels, boxes, original indices, rank flags and FP32 crop pixels. Legacy q-only caches need regeneration.

Preparation validates existing entries, writes missing resolutions atomically, and uses an exclusive writer lock. After a hard interruption, inspect/remove a stale `.prepare.lock` only after ensuring no writer remains. Completed files survive interruption. `--limit 6` can prepare a small diagnostic bank; it has a different dataset identity and is not a full-data experiment.

Persistent crops are the unaugmented TGSC base views. Training retains TGSC's random padded crop/flip; cached quality describes the base view, as in the supplied implementation. Teacher/student use the same augmented input. Teacher targets/features are computed online; oracle caching is not dense-feature caching.

## Check, train, resume and evaluate

Preflight now defaults to a quick cache check: it memory-maps each crop file and checks up to 32 evenly spaced crop records per size, while retaining complete shape/identity, metadata, quality-score and original-source label checks. Model forward/backward checks still cover every curriculum size. It does not change which records are used for training. Training continues to fully read/validate every active stage once when loading it, rather than exhaustively scanning all stages again at startup.

For exhaustive preflight, add `--full-cache-check`, e.g. in a notebook `run_tgsc("check", "--full-cache-check")`. Default `run_tgsc("check")` uses the quick mode. Pixel errors outside the quick sample are deferred to full stage loading; quick success is not a claim that every pixel has been scanned. Per-size progress/timings distinguish model checks, cache preflight and actual training-cache loading. Original CIFAR downloading and identity hashing still occur when required; Kaggle's UI estimate is not a measured duration or a guarantee.


```bash
python tools/run_tgsc.py check --config configs/tgsc/cifar100/dkd_resnet32x4_to_resnet8x4.json --gpu 0 --teacher-checkpoint /weights/teacher.pth --data-root /data/cifar --cache-root /cache/tgsc
python tools/run_tgsc.py train --config configs/tgsc/cifar100/dkd_resnet32x4_to_resnet8x4.json --gpu 0 --teacher-checkpoint /weights/teacher.pth --data-root /data/cifar --cache-root /cache/tgsc --output-dir /runs/dkd_seed27
```

Use another method's JSON for SimKD/ReviewKD. `--seed` overrides the seed. One selected GPU is used per process; `--gpu 1` selects index 1 within inherited `CUDA_VISIBLE_DEVICES` (or the host CUDA-visible list when unrestricted). The child sees that selected device as `cuda:0`. A second independent process can train a different method on another GPU with a distinct output directory and the same read-only bank. DDP for one run is outside this release.

Resume by adding `--resume /previous/latest.pt`; a new `--output-dir` is allowed. Dataset/cache/teacher paths may relocate while their identities and scientific settings stay compatible. The latest completed epoch is resumed, including LR/optimizer, curriculum, RNG, and best metric. There is no exact mid-batch recovery guarantee. `--stop-after-epoch 1` saves/stops without changing the full scientific config.

Evaluate an inference export:

```bash
python tools/run_tgsc.py evaluate --config configs/tgsc/cifar100/simkd_resnet32x4_to_resnet8x4.json --gpu 0 --data-root /data/cifar --export /runs/simkd/inference_best.pt
```

Evaluation requires neither the oracle cache nor teacher-backbone weights. It uses saved normalization and the method's actual inference graph. SimKD retains its projector, frozen head, and teacher grid geometry; unsupported resolutions fail explicitly.

Outputs: `run.json` (config/cache/environment), `metrics.jsonl`, `latest.pt`, `best.pt`, periodic `epoch_40.pt` etc., `inference_latest.pt`, and `inference_best.pt`. Best means highest observed epoch test top-1, following mdistiller's convention; it is not a separate held-out-validation selection claim. Do not overwrite an existing run without explicit resume. A final singleton crop is merged into the preceding batch to keep BatchNorm valid without dropping examples; this can produce a last batch one sample larger than the configured batch size.

## Kaggle notebooks

For a private GitHub repository, add a Kaggle Secret named `GITHUB_TOKEN` and enable access to it for the notebook. A fine-grained token needs read access to repository contents for this repository. Leave `PRIVATE_REPO=True` in the settings cell. The clone helper uses temporary Git askpass code and a child-process environment; it does not put the token in the repository URL or print/store its value. For a public repository, set `PRIVATE_REPO=False` instead. These notebook cells target Kaggle's Linux runtime; local users can clone with their normal Git credentials.

Open `notebooks/tgsc_prepare_cifar100.ipynb`, edit its settings cell, enable a GPU, and prepare the teacher bank. Save its outputs or create a Kaggle Dataset from `tgsc_cache/`. Attach that artifact to each method notebook:

- `tgsc_dkd_cifar100.ipynb`
- `tgsc_simkd_cifar100.ipynb`
- `tgsc_reviewkd_cifar100.ipynb`

Set `CACHE_ROOT` to the parent of the bank-hash folders, not the hash folder itself. Attached paths under `/kaggle/input/` are treated as read-only; new preparation outputs/checkpoints go under `/kaggle/working/`. Configure `--cache-write-root` to place missing entries in a writable overlay. Each notebook invokes the same Python runner in a subprocess using its kernel Python, streams output, and saves a log. No torch import in notebook setup is required for GPU selection.

The notebooks now point to the public repository `https://github.com/ahmad903501/tgsc-mdistiller.git` on `main`, with `PRIVATE_REPO=False`; no GitHub token is needed for this repository. The previous mdistiller-tgsc repository is preserved separately. Set a fixed commit revision to keep preparation and training reproducible. Cloning/installing/downloading needs Internet; offline sessions need those inputs supplied beforehand. Saving files locally in one session does not automatically persist/share them across sessions. Preserve notebook output before continuing elsewhere. No automatic upload or publication is performed.

### Dataset-specific notebooks for the uploaded four caches

Use [notebooks/runs/README.md](../notebooks/runs/README.md) for the per-run notebook index. There are 11 runnable notebooks with exact Dataset owner/slug, teacher/student/config, writable CIFAR root, separate outputs and pinned runtime/config revision 5c01799. The twelfth VGG13 ReviewKD file is explicitly PENDING and stops immediately because its official pair loss weight has not been established. No coefficient was guessed.

Import a run notebook, attach its linked public Dataset using Add Input, enable GPU and Internet, then Run All. Dataset attachment cannot be performed just by setting a URL in Python. The input cell resolves both /kaggle/input/datasets/<owner>/<slug> and /kaggle/input/<slug>, handles one nested extraction folder, and requires one matching teacher plus a complete five-resolution bank. It checks preparation torch/torchvision versions early; the runner performs full identity checks later. Distinct outputs prevent cross-method overwrite. Fresh runs use RESUME=None; interrupted runs must attach latest.pt and set RESUME explicitly.

The source notebooks were cloned from the existing three method templates. ResNet56/ResNet110 method config checks use official teachers and synthetic pixels; local notebook checks do not establish real Kaggle training success. All Dataset version numbers remain user-selected and should stay fixed across comparisons.

### Stage the three training notebooks while preparation runs

Import each method notebook into a separate Kaggle notebook and save it as a draft: DKD + TGSC, SimKD + TGSC, and ReviewKD + TGSC. The distributed method notebooks pin runtime revision `3d350da`; use the actual preparation revision if different. Keep GPU=0, SEED=27, PRIVATE_REPO=False and both input sources set to "kaggle". Each method already has its own config and output directory.

Do not start Run All while preparation is incomplete. Once preparation finishes, preserve its outputs and attach either that saved notebook output directly, or the separate CIFAR/teacher/cache Datasets. When attaching the full preparation output, let `PREP` be its actual mounted root (copy it from Kaggle's input browser), then set:

```python
PREP = Path("/kaggle/input/<actual-preparation-output-root>")
DATA_ROOT = PREP / "data"
TEACHER_CHECKPOINT = PREP / "teacher_weights/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth"
CACHE_ROOT = PREP / "tgsc_cache"
```

These paths assume preparation used the documented writable defaults. Separate Datasets may have a different layout; set the three paths individually in that case. Keep RESUME=None for a fresh run. Enable Internet for public clone/install and an available GPU before execution; no GitHub secret is required. Run the preflight and one-epoch check on DKD first, then resume it for the full schedule and run the other notebooks in sequence. A fresh Run All performs preflight, full training and final evaluation automatically. Each separate Kaggle session uses its own GPU 0; this does not assign the two GPUs of one session across notebooks. Save each completed run's output.

### Recommended T4 x2 workflow

Start with one GPU for cache preparation (`GPU=0`). Preparing one teacher's bank is not distributed across both GPUs by this runner. Different teachers can be prepared by independent processes if desired, but do not launch two writers for the same bank.

For training, use one T4 per experiment. Using two T4s for one small model is not automatically faster because distributed training introduces communication; this release does not implement DDP. To use both GPUs now, launch two independent jobs in the same notebook/session, e.g. DKD on GPU 0 and SimKD on GPU 1, then ReviewKD when a device becomes free. This improves experiment throughput in principle; actual speed depends on CPU/RAM/I/O contention and needs measurement. Keep batch size and learning rate unchanged.

Selecting T4 x2 does not itself use both devices: a single method notebook launches one selected GPU. Separate Kaggle notebooks run in separate sessions; setting GPU 0/1 across them does not pair their hardware. To use both devices from one session, launch both subprocesses before waiting. Example after the common setup/settings cells:

```python
jobs = []
log_streams = []
try:
    for method, gpu in [("dkd", 0), ("simkd", 1)]:
        config = REPO_DIR / "configs/tgsc/cifar100" / f"{method}_resnet32x4_to_resnet8x4.json"
        output = Path("/kaggle/working/tgsc_runs") / f"{method}_seed{SEED}"
        output.mkdir(parents=True, exist_ok=True)
        log = (output / "train.log").open("a", encoding="utf-8")
        log_streams.append(log)
        args = [sys.executable, "-u", str(RUNNER), "train", "--config", str(config),
                "--gpu", str(gpu), "--seed", str(SEED), "--data-root", str(DATA_ROOT),
                "--teacher-checkpoint", str(TEACHER_CHECKPOINT), "--cache-root", str(CACHE_ROOT),
                "--output-dir", str(output)]
        jobs.append(subprocess.Popen(args, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT))
    exit_codes = [job.wait() for job in jobs]
    if any(exit_codes):
        raise RuntimeError(f"Training exit codes: {exit_codes}; inspect each train.log")
finally:
    for job in jobs:
        if job.poll() is None:
            job.terminate()
            try:
                job.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.kill()
                job.wait()
    for log in log_streams:
        log.close()
```

Run each selected config's preflight before this concurrent launch. The two jobs share the cache on disk but load their own active-resolution tensors into host RAM; check RAM as well as GPU utilization. Logs are in separate files; the concurrent example does not stream both into the notebook output. This example starts fresh runs; use separate explicit resume commands for interrupted runs.

### Preparation, dataset upload, and first training run

1. Import `tgsc_prepare_cifar100.ipynb` into Kaggle. Choose an available T4 x2 accelerator and turn Internet on for the clone/install. Add the GitHub Secret if the repository is private.
2. For the first preparation, leave `TEACHER_SOURCE="download"` and `CIFAR_SOURCE="download"` with writable roots. Alternatively set either source to `"kaggle"` and provide its attached path. Preserve the selected weights and original CIFAR files as reusable Datasets, in addition to the crop bank.
3. Use one teacher/profile first: ResNet32x4 -> ResNet8x4. Set `GPU=0`, the repository revision, and `CACHE_WRITE_ROOT=/kaggle/working/tgsc_cache`. Run preparation. The DKD-named config selects the teacher's TGSC profile; the resulting bank is not DKD-specific.
4. Preserve the completed `tgsc_cache/` output, then create a Kaggle Dataset containing that directory (or attach the saved notebook output directly). Keep the bank hash directory and filenames unchanged.
5. Import the DKD method notebook and attach **all three** inputs: original CIFAR-100, teacher weights, and the prepared cache. Original training data is still used to validate dataset identity; test data supplies evaluation. Cached oracle values do not replace online teacher features/logits.
6. Set `CACHE_ROOT` to the directory directly containing the bank hash folders, e.g. `/kaggle/input/<your-cache-dataset>/tgsc_cache`, using the actual mounted path. Keep the same code revision, torch/torchvision versions, teacher weights, and TGSC preprocessing used during preparation, because these identify the cache.
7. Run preflight, then a one-epoch GPU smoke run by adding `--stop-after-epoch 1` to the training arguments. Check finite losses/checkpoints and `metrics.jsonl`, then resume from `latest.pt` for the remaining epochs. The stop flag does not shorten the saved scientific configuration.
8. After one method works, run the other method notebooks against the same teacher cache, or use the concurrent two-GPU example. Save each notebook's output/checkpoints before ending its session. Repeat preparation only when changing the teacher checkpoint or cache-identifying settings.

Current platform references: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [datasets](https://www.kaggle.com/docs/datasets), and [PyTorch distributed training](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html). GPU/Kaggle speed and real-data acceptance have not been measured locally.

## Development checks

```bash
python -m unittest discover -s tests -q
python tools/validate_tgsc_delivery.py
python tools/validate_tgsc_references.py --reference-root .. --output docs/TGSC_REFERENCE_CHECKS.json
```

The first suite is self-contained and uses synthetic pixels/random model weights. The delivery validator checks an isolated copy, notebook parsing, and CLI error handling. The reference audit explicitly requires sibling repos for development only. None establishes final CIFAR accuracy or real-GPU/Kaggle validation. See `TGSC_IMPLEMENTATION_PROGRESS.md`, `TGSC_VALIDATION.json`, and `TGSC_REFERENCE_CHECKS.json` for recorded evidence.
