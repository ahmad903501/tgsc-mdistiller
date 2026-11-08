# Two VRM + CAM options

Both options use ResNet32x4->ResNet8x4, with seeds0/1/2, full-image/CAM-crop pairs
below32px, then the exact original VRM weak/strong32px view pipeline.

| Size | Epochs | Stride | Option1 passes | Option2 passes | Inputs per image |
|---|---|---|---|---|---|
|16|1-48|8|3|3|Weak full32 + weak crop16|
|20|49-96|8|3|3|Weak full32 + weak crop20|
|24|97-144|8|3|3|Weak full32 + weak crop24|
|28|145-192|2|1|3|Weak full32 + weak crop28|
|32|193-240|N/A|1|1|Original VRM weak32 + strong32|

Stride2 at28px was explicitly confirmed by the user. Nine positions are scored; top3
CAM windows are distinct. Option1 visits the highest-ranked window each epoch;
option2 rotates through top1/top2/top3 across the three passes. Crop proposals are
rebuilt on stage entry, kept as coordinates in memory, and extracted on demand.
24px remains stride8; this experiment does not adopt the separate middle/worst curriculum.

## Loss and paper/code fidelity

Use exactly the cloned VISION-SJTU/VRM@56d02ce loss code:

`CE(student_full,y) + CE(student_second,y) + 64*isv_loss + 32*icv_loss`.

The relation helpers include their original internal edge_weight50, sample-wise logit
standardization, temperature4, L2-normalized relation vectors and mean SmoothL1.
There are no additional independent KD, KL, CE terms or quality gates.
The two inputs jointly produce one VRM loss, not two standalone method losses.

Important: the cloned code omits the paper's unreliable-edge entropy masks. These
experiments are explicitly labelled the **published-code variant**, not a completed
reproduction of the full paper pruning procedure. No guessed pruning formula is added.
See VRM_INTEGRATION_REVIEW.md for the discrepancy and paper page references.

At16-28px, weak augmentation follows our existing EP-KD tensor pipeline (full reflect
pad4/flip, crop reflect pad2/flip). Teacher and student receive the identical inputs.
Crop-path student BN running mean/variance updates are suppressed, as EP-KD does.

At32px, the selected crop branch is replaced by the **exact original VRM PIL transforms**:
weak RandomCrop32 with constant padding4 + flip; strong RandomCrop32 with constant
padding4 + flip + custom RandAugment(2,10) + Cutout; both normalized identically.
Both32px student forwards update BN, matching VRM. The original custom implementation
draws operation magnitudes across its listed ranges; torchvision RandAugment is not substituted.
CPU PIL processing at32px prioritizes faithful behavior; no GPU-augmentation speed claim is made.

## Files to run

- Option1: `notebooks/runs/vrm_cam_resnet32x4_to_resnet8x4_28p1.ipynb`.
- Option2: `notebooks/runs/vrm_cam_resnet32x4_to_resnet8x4_28p3.ipynb`.
- Source: `https://github.com/ahmad903501/tgsc-mdistiller`, branch `experiment/epkd-official-methods`.

Enable GPU and Internet, import a notebook and Run All. Setup clones the published code
from GitHub at the exact commit in REVISION. No runtime ZIP or runtime Dataset is needed.
The old runtime-mine input is unused. A fresh writable checkout directory avoids collisions
with earlier extracted runtimes. Original CIFAR data and a teacher can be attached or downloaded.
No original VRM checkout, old TGSC crop bank or paper PDF is required to run the bundle.

`SMOKE_TEST=False` runs the full240epoch experiments. Set it toTrue first for128images,
one epoch per size and the first seed only. The smoke keeps each option's actual pass
counts, stops afterepoch2, resumes throughstage5, evaluates exports and checks logs.
Smoke outputs have a separate suffix and cannot be silently resumed as a full run.

Seeds run sequentially on the selected GPU. GPU0 is default; use GPU1 for the other
option in a separate process/session if the execution environment supports that setup.
To resume an interrupted run, retain its output folder and rerun. On a fresh Kaggle
session attach/copy the saved run folders back under OUTPUT before continuing.
Runtime files are read-only inputs; writable checkpoints/data are under /kaggle/working.

## Explicit defaults and interpretation

- Batch64, FP32, SGD LR.05, momentum.9, weight decay.0005,48epochs per stage.
- EP-KD LR milestones48/144/192/210. The VRM paper's baseline uses150/180/210 instead;
  we retain the current integration schedule rather than silently mixing baseline schedules.
- Evaluation every5epochs plus final/requested stop. Losses, teacher second-view accuracy,
  step/draw counts, test metrics, CSV/PNG/SVG and latest/best exports are saved.
- Option2 adds96dataset passes over the whole run (624versus528). Its performance
  comparison includes more training compute, not just a change in view diversity.
- No test-set accuracy or speed improvement is claimed from local correctness checks.

## Local verification

The tests compare official/local losses and student gradients at batch64/classes100;
compare model-level loss/backward/BN behavior; compare exact original32px transform
pixels across30seeds plus batch RNG ordering; check both real configured pairs with
the official ResNet32x4 checkpoint at every size; verify distinct28px windows; and
exercise both pass budgets, exact CPU stop/resume across32px augmentation, teacher-free
exports, notebook syntax and source ZIP contents. Results are in VRM_CAM_CHECKS.json.

Regenerate delivery with `python tools/generate_vrm_cam_notebooks.py`.
