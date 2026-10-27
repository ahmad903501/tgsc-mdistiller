# Dataset-specific Kaggle runs

Import one notebook into Kaggle, attach its linked Dataset with Add Input, enable GPU and Internet, then Run All.
The public repository needs no token. Input paths resolve both owner-qualified and standard Kaggle mounts, including a single nested extraction folder.
Runtime/config revision: `5c01799`. Defaults: GPU 0, seed 27, 240 epochs, DKD unified gating.
These notebooks have local CPU/config checks; actual Kaggle execution is still pending. Keep Dataset versions fixed across methods.

| Notebook | Dataset | Status |
|---|---|---|
| [tgsc_dkd_resnet32x4_to_resnet8x4.ipynb](tgsc_dkd_resnet32x4_to_resnet8x4.ipynb) | [aj903501/tgsc-resnet32x4-try1](https://www.kaggle.com/datasets/aj903501/tgsc-resnet32x4-try1) | Ready for Kaggle preflight |
| [tgsc_simkd_resnet32x4_to_resnet8x4.ipynb](tgsc_simkd_resnet32x4_to_resnet8x4.ipynb) | [aj903501/tgsc-resnet32x4-try1](https://www.kaggle.com/datasets/aj903501/tgsc-resnet32x4-try1) | Ready for Kaggle preflight |
| [tgsc_reviewkd_resnet32x4_to_resnet8x4.ipynb](tgsc_reviewkd_resnet32x4_to_resnet8x4.ipynb) | [aj903501/tgsc-resnet32x4-try1](https://www.kaggle.com/datasets/aj903501/tgsc-resnet32x4-try1) | Ready for Kaggle preflight |
| [tgsc_dkd_vgg13_to_vgg8.ipynb](tgsc_dkd_vgg13_to_vgg8.ipynb) | [ahmadjawwad903/tgsc-vgg13-try1](https://www.kaggle.com/datasets/ahmadjawwad903/tgsc-vgg13-try1) | Ready for Kaggle preflight |
| [tgsc_simkd_vgg13_to_vgg8.ipynb](tgsc_simkd_vgg13_to_vgg8.ipynb) | [ahmadjawwad903/tgsc-vgg13-try1](https://www.kaggle.com/datasets/ahmadjawwad903/tgsc-vgg13-try1) | Ready for Kaggle preflight |
| [tgsc_reviewkd_vgg13_to_vgg8_PENDING.ipynb](tgsc_reviewkd_vgg13_to_vgg8_PENDING.ipynb) | [ahmadjawwad903/tgsc-vgg13-try1](https://www.kaggle.com/datasets/ahmadjawwad903/tgsc-vgg13-try1) | BLOCKED: unverified VGG ReviewKD settings |
| [tgsc_dkd_resnet56_to_resnet20.ipynb](tgsc_dkd_resnet56_to_resnet20.ipynb) | [toxica001/tgsc-resnet56-try1](https://www.kaggle.com/datasets/toxica001/tgsc-resnet56-try1) | Ready for Kaggle preflight |
| [tgsc_simkd_resnet56_to_resnet20.ipynb](tgsc_simkd_resnet56_to_resnet20.ipynb) | [toxica001/tgsc-resnet56-try1](https://www.kaggle.com/datasets/toxica001/tgsc-resnet56-try1) | Ready for Kaggle preflight |
| [tgsc_reviewkd_resnet56_to_resnet20.ipynb](tgsc_reviewkd_resnet56_to_resnet20.ipynb) | [toxica001/tgsc-resnet56-try1](https://www.kaggle.com/datasets/toxica001/tgsc-resnet56-try1) | Ready for Kaggle preflight |
| [tgsc_dkd_resnet110_to_resnet32.ipynb](tgsc_dkd_resnet110_to_resnet32.ipynb) | [ahmadjawwad903/tgsc-resnet110-try1](https://www.kaggle.com/datasets/ahmadjawwad903/tgsc-resnet110-try1) | Ready for Kaggle preflight |
| [tgsc_simkd_resnet110_to_resnet32.ipynb](tgsc_simkd_resnet110_to_resnet32.ipynb) | [ahmadjawwad903/tgsc-resnet110-try1](https://www.kaggle.com/datasets/ahmadjawwad903/tgsc-resnet110-try1) | Ready for Kaggle preflight |
| [tgsc_reviewkd_resnet110_to_resnet32.ipynb](tgsc_reviewkd_resnet110_to_resnet32.ipynb) | [ahmadjawwad903/tgsc-resnet110-try1](https://www.kaggle.com/datasets/ahmadjawwad903/tgsc-resnet110-try1) | Ready for Kaggle preflight |

Keep this Dataset's preparation torch/torchvision versions. If the early environment check fails, do not edit cache manifests to bypass it.
For an interrupted run, attach latest.pt and set RESUME to that path before running. Use a fresh output directory for a fresh run.
If you want a one-epoch smoke run, add --stop-after-epoch 1 to the training arguments, then resume latest.pt for the remaining epochs.
