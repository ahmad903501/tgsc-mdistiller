# EP-KD with official distillation methods

Branch: `experiment/epkd-official-methods` in mdistiller. The standalone WRN difficulty
experiment in epkd-vision is separate. Start with `notebooks/runs/epkd_verify_end_to_end.ipynb`.

## Objectives and inputs

The new `tools/run_epkd.py` uses ground-truth-class teacher CAM on clean full images.
On entering a stage it builds an in-memory coordinate cache. Original CIFAR pixels are
cropped and augmented on demand; no teacher backward, persistent pixel bank or r1/r2/r3 gate.

| Method | Existing official objective on each input | Export |
|---|---|---|
| DKD | CE + decoupled TCKD/NCKD, existing coefficients/warmup | Student |
| SimKD | Projector feature MSE only, no new CE or logit KD | Student + projector + frozen head |
| ReviewKD | CE + ABF/HCL, existing coefficient/warmup | Student |

Full and crop objectives are summed with weight1 each. Teacher and student see the
same input within a branch; augmentations are independent across branches. All teacher
parameters remain frozen. Crop forwards preserve BN running means/variances for the
student and trainable alignment modules, while still using batch statistics. Full-image
forwards update the statistics. Official loss, projector, head and ABF code are reused.

## Explicit initial defaults

- Original EP-KD:16/20/24/28/32,48epochs each; top3 CAM windows; stride8 throughout.
- Passes3/3/3/1/1; at28px onlyone stride8 window exists, repeated in the cache. The single
  pass uses its first member. Final32px stage is full-image-only, with no extra crop loss.
- Full-image reflect pad4/random crop/flip; crop reflect pad2/random crop/flip.
- LR milestones48/144/192/210; preserve each pair's base LR and method options.
- FP32, seed27, one GPU per process; AMP and DDP are not implemented here.
- Drop incomplete training batches, matching EP-KD (49,920draws/pass at50k/batch128).
- Evaluation every5epochs plus final/requested stop; logging/checkpointing every epoch.
- Finer strides and the difficulty curriculum remain separately named experiments.
- VGG13->VGG8 ReviewKD remains unavailable; no unaudited coefficient/alignment was invented.

## End-to-end verification notebook

Open `notebooks/runs/epkd_verify_end_to_end.ipynb` in Kaggle, Colab or Jupyter with a GPU.
Enable Internet and GPU, then Run All. Setup clones `ahmad903501/tgsc-mdistiller`
from GitHub at the immutable implementation commit specified in REVISION. No runtime
ZIP or runtime Dataset is required. The publication branch is `experiment/epkd-official-methods`.
Existing local tracked edits are protected; a fresh writable checkout directory is used.
Original TGSC notebooks remain for earlier experiment reproduction.

Provide an attached teacher path or allow discovery/download. Old TGSC Kaggle Datasets
can supply teacher files, but their crop banks are unused. The downloader first fetches
the individual official RepDistiller checkpoint with an audited SHA256, then falls back
to the official mdistiller archive. Attached original CIFAR data is supported: DATA_ROOT
must contain `cifar-100-python/`. Otherwise torchvision downloads the original archive
once, shared by all runs here. An optional verified test-only NPZ can also be provided.

The verifier uses ResNet56->ResNet20 for all3methods,128training and128evaluation images,
batch32, one epoch and one pass per size. It stops afterepoch2, resumes through all5stages,
loads teacher-free exports and checks metrics/figures/history/draw counts. These explicit
smoke settings are not benchmark results. Full-run notebooks keep240epochs.

## Full runs and local GPUs

Use the11 `epkd_<method>_<teacher>_to_<student>.ipynb` notebooks in `notebooks/runs`.
SEEDS defaults to[27] and is editable. Each process uses one GPU; separate method/seed
jobs can select different GPU indices. Existing run output is resumed explicitly.

```powershell
python tools/run_epkd.py check --config configs/epkd/cifar100/dkd_resnet56_to_resnet20.json --teacher-checkpoint download_ckpts/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth --data-root data --gpu 0
python tools/run_epkd.py train --config configs/epkd/cifar100/dkd_resnet56_to_resnet20.json --teacher-checkpoint download_ckpts/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth --data-root data --output epkd_runs/dkd_resnet56_seed27 --gpu 0 --download
```

Use `--cpu-check` explicitly for small CPU verification, `--limit` for an explicit subset,
`--stop-after-epoch` to stop without changing the intended schedule, and `--resume <path>/latest.pt`
to continue. At least one complete batch is required for training. Recovery is at epoch
boundaries, not mid-batch. `evaluate --export` uses the method-aware teacher-free export.

## Reporting and provenance

Outputs: run.json, metrics.jsonl/CSV, curvesPNG/SVG, latest/best checkpoints and inference
exports. Logs separate branch losses, passes, stride, steps, source draws and teacher crop
accuracy. SimKD exports require the projector/head, but not a full teacher forward.

Resume rejects different data, teacher, method/config, source code, torch/torchvision or
evaluation interval; paths may relocate. It restores optimizer/scheduler, Python/NumPy/
Torch/CUDA RNG, metric history and best export. Stage coordinates regenerate with RNG
preserved. Completed runs can be reopened safely. Different hardware need not give
bitwise-identical results even with identical provenance.

Generate with `python tools/generate_epkd_notebooks.py`. Local CPU evidence is recorded
in EPKD_CHECKS.json and the progress report. GPU verification/full accuracy remain pending.
