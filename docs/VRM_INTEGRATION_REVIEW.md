# VRM paper/code review before integration

Reviewed supplied `../../VRM_Paper.pdf` (20 PDF pages, main paper plus supplement) and
local `../../VRM/mdistiller/distillers/VRM.py` / dataset transforms at VRM commit56d02ce.
Initial review below; the user subsequently selected the two-input full/CAM variant.
Implemented options and checks are documented in VRM_CAM_RUNNING.md (2026-11-09).

## Confirmed design

- Sections5.1-5.3 (PDF4-5): normalized cross-view relation vectors, inter-sample matrices
  [B,B,C] and inter-class matrices [C,C,B]. Work from class predictions rather than
  requiring feature/channel alignment. Same batch order/pairing must be retained.
- Section5.5/Equation7 (PDF6): CE on both views plus ISV and ICV losses. This supports
  preserving both supervised terms; ordinary KL KD is not an extra default objective.
- Supplement8.2 (PDF12): weak crop/flip; strong crop/flip plus14-operation RandAugment
  and gray Cutout (probability1). The paper's Weak-Weak ablation is also competitive
  (PDF7-8), so stronger augmentation should not be assumed universally better.
- Supplement8.4 (PDF14-15): CIFAR batch64, SGD240epochs, LR.01 forMobile/Shuffle students
  and.05 otherwise, momentum.9/weight decay.0005, temperature4, LRcuts150/180/210;
  reported results average3trials. The current EP-KD schedule differs and must be named explicitly.

## Important discrepancy to resolve

Section5.4 (PDF5) and Algorithm1 (PDF12) include unreliable-edge pruning using student
joint-entropy percentile masks. The local VRM.py has no entropy/percentile/mask operation:
it matches all cross-view edges using smooth_l1_loss. Its cross-view construction implements
the redundant-edge reduction structurally, but not the paper's explicit unreliable-edge mask.
Teacher no_grad is not an implementation of that student-based mask.

Loss conventions also require care: main text specifies Huber distance, Algorithm1 calls
its loss MSE with masked summation, and local code uses mean SmoothL1 with nested factors
(edge_weight50, then64 forISV and32 forICV). Do not translate paper coefficients directly
without accounting for reductions/normalization. This is not evidence of a simple coefficient typo.

Before claiming the full paper method, reconcile the authors' current published implementation
and paper; do not invent an entropy formula or pruning percentile. If retaining the exact supplied
code, name it explicitly as that code variant and record the pruning omission.

## Earlier four-input proposal (superseded by the user's two-input choice)

For the full-image branch, create a weak/strong full pair. For the selected CAM region,
create a weak/strong crop pair. Sum one VRM objective per pair; retain only the full pair
at32px. This preserves the official paired-view organization and makes four inputs per
teacher/student at restricted stages, versus two in the current DKD/SimKD/ReviewKD wrapper.
Crop-path BN running-stat suppression would apply to both crop forwards. Native small
crops plus strong augmentation may lose class evidence and require diagnostics.

The CAM/native-crop curriculum is our proposed extension, not a setting established by
the VRM paper. Replacing VRM's two views directly with full/crop would be another named
experiment. No generic extra KL term, quality gate or altered augmentation should be added
silently. First suggested pair: ResNet32x4->ResNet8x4; preserve batch64 initially.
