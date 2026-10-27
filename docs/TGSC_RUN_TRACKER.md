# TGSC Kaggle run tracker

User reports the four teacher/cache packages uploaded (2026-10-28). Dataset URL/version and mounted paths are not recorded yet. Cache upload does not mean a training run or Kaggle preflight has passed.

Initial comparison: seed 27, 240 epochs, same uploaded Dataset version and matching preparation dependency versions for every run. DKD uses unified gating for the first comparison; decoupled gating is a later ablation. Config availability is distinct from real-GPU validation.

| ID | Teacher | Student | Method | CONFIG_NAME | Setup | Run status | Test top-1 (%) | Kaggle output / checkpoint |
|---|---|---|---|---|---|---|---|---|
| 01 | resnet32x4 | resnet8x4 | DKD + TGSC | `dkd_resnet32x4_to_resnet8x4.json` | Config available; Kaggle preflight pending | Not started | - | - |
| 02 | resnet32x4 | resnet8x4 | SIMKD + TGSC | `simkd_resnet32x4_to_resnet8x4.json` | Config available; Kaggle preflight pending | Not started | - | - |
| 03 | resnet32x4 | resnet8x4 | REVIEWKD + TGSC | `reviewkd_resnet32x4_to_resnet8x4.json` | Config available; Kaggle preflight pending | Not started | - | - |
| 04 | vgg13 | vgg8 | DKD + TGSC | `dkd_vgg13_to_vgg8.json` | Config available; Kaggle preflight pending | Not started | - | - |
| 05 | vgg13 | vgg8 | SIMKD + TGSC | `simkd_vgg13_to_vgg8.json` | Config available; Kaggle preflight pending | Not started | - | - |
| 06 | vgg13 | vgg8 | REVIEWKD + TGSC | Not added | Verify ReviewKD weight; add/test config | Not started | - | - |
| 07 | resnet56 | resnet20 | DKD + TGSC | `dkd_resnet56_to_resnet20.json` | Config available; Kaggle preflight pending | Not started | - | - |
| 08 | resnet56 | resnet20 | SIMKD + TGSC | Not added | Add config; check all crop sizes | Not started | - | - |
| 09 | resnet56 | resnet20 | REVIEWKD + TGSC | Not added | Add config; check all crop sizes | Not started | - | - |
| 10 | resnet110 | resnet32 | DKD + TGSC | `dkd_resnet110_to_resnet32.json` | Config available; Kaggle preflight pending | Not started | - | - |
| 11 | resnet110 | resnet32 | SIMKD + TGSC | Not added | Add config; check all crop sizes | Not started | - | - |
| 12 | resnet110 | resnet32 | REVIEWKD + TGSC | Not added | Add config; check all crop sizes | Not started | - | - |

There are 12 planned primary runs: 7 configs currently exist; 5 require setup. Missing ResNet56/ResNet110 SimKD configs are additional pair evaluations rather than exact published SimKD pair reproductions. ReviewKD ResNet56/ResNet110 official weights are .6/1.0, but local configs and variable-size checks are still missing. VGG13 ReviewKD coefficient verification remains pending; do not invent a default.

Use one-epoch DKD ResNet32x4 smoke/resume as the first GPU acceptance step. For every full run, retain metrics.jsonl, run.json, latest.pt, inference_best.pt and the Kaggle output URL. Record whether the reported accuracy is final-epoch or best-epoch; do not mix them in comparison columns.

Optional additional pair: ResNet32x4 -> ShuffleV2 has configs for all three methods and reuses the existing ResNet32x4 teacher bank. These 3 runs are outside the 12 above. WRN40-2 and ResNet50 complete banks have not been supplied; FitNet is deferred. Adding another seed multiplies the primary run count by 12.
