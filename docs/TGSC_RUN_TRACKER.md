# TGSC Kaggle run tracker

User reports four teacher/cache packages uploaded (2026-10-28). Dataset owner/slug URLs and expected mount paths are now recorded in notebooks/runs/run_manifest.json; Dataset version numbers have not been supplied. Cache upload does not mean a training run or Kaggle preflight has passed.

Initial comparison: seed 27, 240 epochs, same uploaded Dataset version and matching preparation dependency versions for every run. DKD uses unified gating for the first comparison; decoupled gating is a later ablation. Config availability is distinct from real-GPU validation.

| ID | Teacher | Student | Method | Notebook | Setup | Run status | Test top-1 (%) | Kaggle output / checkpoint |
|---|---|---|---|---|---|---|---|---|
| 01 | resnet32x4 | resnet8x4 | DKD + TGSC | [tgsc_dkd_resnet32x4_to_resnet8x4.ipynb](../notebooks/runs/tgsc_dkd_resnet32x4_to_resnet8x4.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 02 | resnet32x4 | resnet8x4 | SIMKD + TGSC | [tgsc_simkd_resnet32x4_to_resnet8x4.ipynb](../notebooks/runs/tgsc_simkd_resnet32x4_to_resnet8x4.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 03 | resnet32x4 | resnet8x4 | REVIEWKD + TGSC | [tgsc_reviewkd_resnet32x4_to_resnet8x4.ipynb](../notebooks/runs/tgsc_reviewkd_resnet32x4_to_resnet8x4.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 04 | vgg13 | vgg8 | DKD + TGSC | [tgsc_dkd_vgg13_to_vgg8.ipynb](../notebooks/runs/tgsc_dkd_vgg13_to_vgg8.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 05 | vgg13 | vgg8 | SIMKD + TGSC | [tgsc_simkd_vgg13_to_vgg8.ipynb](../notebooks/runs/tgsc_simkd_vgg13_to_vgg8.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 06 | vgg13 | vgg8 | REVIEWKD + TGSC | [tgsc_reviewkd_vgg13_to_vgg8_PENDING.ipynb](../notebooks/runs/tgsc_reviewkd_vgg13_to_vgg8_PENDING.ipynb) | BLOCKED: verify VGG ReviewKD weight/alignment | Not started | - | - |
| 07 | resnet56 | resnet20 | DKD + TGSC | [tgsc_dkd_resnet56_to_resnet20.ipynb](../notebooks/runs/tgsc_dkd_resnet56_to_resnet20.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 08 | resnet56 | resnet20 | SIMKD + TGSC | [tgsc_simkd_resnet56_to_resnet20.ipynb](../notebooks/runs/tgsc_simkd_resnet56_to_resnet20.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 09 | resnet56 | resnet20 | REVIEWKD + TGSC | [tgsc_reviewkd_resnet56_to_resnet20.ipynb](../notebooks/runs/tgsc_reviewkd_resnet56_to_resnet20.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 10 | resnet110 | resnet32 | DKD + TGSC | [tgsc_dkd_resnet110_to_resnet32.ipynb](../notebooks/runs/tgsc_dkd_resnet110_to_resnet32.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 11 | resnet110 | resnet32 | SIMKD + TGSC | [tgsc_simkd_resnet110_to_resnet32.ipynb](../notebooks/runs/tgsc_simkd_resnet110_to_resnet32.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |
| 12 | resnet110 | resnet32 | REVIEWKD + TGSC | [tgsc_reviewkd_resnet110_to_resnet32.ipynb](../notebooks/runs/tgsc_reviewkd_resnet110_to_resnet32.ipynb) | Config and per-run notebook checked; Kaggle preflight pending | Not started | - | - |

There are 12 planned primary runs: 11 configs and per-run notebooks now exist; VGG13 ReviewKD remains blocked. ResNet56/ResNet110 SimKD runs are additional pair evaluations rather than exact published SimKD pair reproductions. ReviewKD ResNet56/ResNet110 weights .6/1.0 and all five crop sizes have now been checked with official-weight CPU forward/backward probes. VGG13 ReviewKD coefficient verification remains pending; do not invent a default.

Use one-epoch DKD ResNet32x4 smoke/resume as the first GPU acceptance step. For every full run, retain metrics.jsonl, run.json, latest.pt, inference_best.pt and the Kaggle output URL. Record whether the reported accuracy is final-epoch or best-epoch; do not mix them in comparison columns.

Optional additional pair: ResNet32x4 -> ShuffleV2 has configs for all three methods and reuses the existing ResNet32x4 teacher bank. These 3 runs are outside the 12 above. WRN40-2 and ResNet50 complete banks have not been supplied; FitNet is deferred. Adding another seed multiplies the primary run count by 12.
