# Reference audit: DKD, SimKD, ReviewKD

2026-10-21. Source audit completed; numerical port checks are separate implementation tasks. PDFs and sibling repositories are development references only.

## DKD (task 01)

Reference: mdistiller revision `a08d46f10d6102bd6e3f258ca5ac880b020ea259`, `distillers/DKD.py`, `engine/cfg.py`, `configs/cifar100/dkd/`; supplied DKD PDF section on decoupling target/non-target knowledge.

TCKD is KL between two-bin target-versus-other probabilities. NCKD is KL after masking the target logit with -1000. Both use temperature-softened probabilities, class-sum KL, T squared, and batch division. Per-sample implementation must sum classes before gating and batch mean. Preserve alpha/beta independently; do not reintroduce classical KD's confidence coupling. CE uses ground truth. Loss ramp is `min(epoch/20,1)` with mdistiller's one-based epoch convention; zero warmup means 1.

Merged examples: resnet32x4 -> resnet8x4 inherits alpha=1,beta=8,T=4,warmup=20; resnet32x4 -> ShuffleV2 also inherits these, but its official YAML sets student LR .01 instead of .05. resnet56 -> resnet20 overrides beta=2; WRN -> WRN overrides beta=6; WRN -> ShuffleV1 has no override and inherits 8. TGSC optimizer/schedule are supplied separately from the existing pair profile.

Required TGSC changes: expose unreduced TCKD/NCKD; apply unified or component-specific gates; preserve CE and ramp. No new loss term.

## SimKD (task 02)

Reference: official clone revision `2b389c31ed7779aea31e7aaf0bb0f2d8b6ac2f01`, `models/util.py` SimKD class, `helper/loops.py`, `train_student.py`, `scripts/run_distill.sh`; supplied paper's feature regression/reused-classifier method and bottleneck discussion.

Official projector: 1x1(Cs,Ct/2), 3x3(Ct/2,Ct/2,pad=1), 1x1(Ct/2,Ct), no conv bias, each with BN/ReLU. Pool the larger feature grid to the smaller before regression. Loss is elementwise mean MSE; per-sample reduction must mean C/H/W then batch. The example command uses `-c 0 -d 0 -b 1`, factor 2: no CE/KL, feature coefficient 1. The unrelated Proj helper is not this projector.

Official `feat[-2]` is the final spatial feature before global pooling. In mdistiller, `feats[-1]` matches this for ResNet/VGG/ResNetV2. WRN's dictionary feature precedes its final BN/ReLU; ShuffleV2 and MobileNetV2 have a final conv after `feats[-1]`. Explicitly expose the actual prepool output as `final_feat` for these models rather than project a different layer. Student and teacher head dimensions must match the selected final spatial feature. Head attributes include fc/classifier/linear. The official SimKD clone does not include the WRN40-2 backbone; using the verified projector/objective with the supplied TGSC WRN architecture is a backbone extension, not a claim of reproducing a released WRN SimKD checkpoint.

Inference retains student backbone, projector, global pooling, and frozen teacher classifier. Its pooling target uses teacher spatial geometry in the reference; inference must reproduce that geometry without running teacher features (derive and store the supported-resolution target grids). This matters for heterogeneous pairs when the student final grid is larger. Teacher head parameters are frozen, not detached from the projected student computational graph. Student's unused classifier must not receive optimization.

## ReviewKD (task 03)

Reference: official clone revision `cede6ea6387ae9b6127de0e561507177bf19c11e`, `CIFAR-100/model/reviewkd.py`, `train.py`, `script/reviewKD.sh`; supplied paper sections 3.2/3.3 on residual review, ABF, and HCL. Compare local `distillers/ReviewKD.py`.

ABF uses 1x1/BN transform, two sigmoid attention maps for current/residual fusion, nearest interpolation, then 3x3/BN output. Native mid width is min(512, deepest student channels). HCL sums stage losses; each stage starts with full-grid MSE and adds valid adaptive pools [4,2,1] with successive half weights, normalizing by accumulated weights. Preserve this order per sample. Native objective is CE + coefficient * ramp * HCL; optional official KL flag is not active in the cited base scripts, so it is not added.

Homogeneous ResNet32x4 -> ResNet8x4: student postactivation feats[1:] plus pooled feature, teacher preactivation feats[1:] plus pooled feature. The current mdistiller defaults include an extra student stem but silently truncate it through zip; select the intended three stages explicitly instead. Reference channel vectors [64,128,256,256].

Heterogeneous ResNet32x4 -> ShuffleV2: three student stages [116,232,464] plus pooled 1024, teacher stages [64,128,256] plus pooled 256. At 32 input student grids [16,8,4,1], teacher [32,16,8,1]. Residual interpolation targets student grid before fusion; output interpolation targets teacher grid. Shapes are runtime values under TGSC.

Reference script weights: ResNet32x4 -> ResNet8x4=5; ->ShuffleV2=8; WRN40-2 -> WRN16-2/WRN40-1/ShuffleV1=5; ResNet56 -> ResNet20=.6. The single mdistiller default YAML only establishes weight 5 for its named pair. VGG/MobileNet combinations are described in factory branches but some imports are commented out in the released clone; shape compatibility can be implemented from those branches, but do not invent a verified pair weight.

Epoch discrepancy: official train.py loops from zero and uses epoch/20; mdistiller calls from one. Retain mdistiller one-based ramp for integration and document it. Official-script coefficient and original fusion/HCL are retained. No proof of bitwise official backbone parity is claimed for TGSC's modified internal ShuffleV1 pooling.

## Initial supported configs

Use the three existing TGSC profiles: ResNet32x4 -> ResNet8x4, WRN40-2 -> WRN40-1, VGG13 -> VGG8. VGG ReviewKD has no verified active script coefficient: do not issue a supposedly official VGG ReviewKD config until that coefficient is established. Add a clearly documented ResNet32x4 -> ShuffleV2 compatibility pilot using the ResNet32x4 teacher crop bank and official student LR .01; this is an explicit derived TGSC profile, not an existing JSON reproduction. New pairs must state profile provenance and verified method settings.

## Additional teacher audit from supplied PDFs (2026-10-22)

Re-read the actual local DKD.pdf, SimKD.pdf and ReviewKD.pdf using PDF text extraction. Page numbers below count physical PDF pages, including the first page; no new runtime configs or scientific defaults were introduced by this audit.

DKD Tables 6/7 (PDF pp. 6/7) and ReviewKD Tables 1/2 (PDF p. 5) establish these CIFAR-100 pairs:

| Teacher | Published students in those benchmark tables | DKD beta, appendix Table A.2 (PDF p. 12) |
|---|---|---|
| ResNet56 | ResNet20 | 2 |
| ResNet110 | ResNet32 | 2 |
| ResNet32x4 | ResNet8x4, ShuffleNetV1, ShuffleNetV2 | 8 |
| WRN40-2 | WRN16-2, WRN40-1, ShuffleNetV1 | 6 |
| VGG13 | VGG8, MobileNetV2 | 6 |
| ResNet50 | MobileNetV2 | 8 |

DKD appendix A.2 (PDF p. 11): 240 epochs, batch 64, SGD momentum .9, weight decay .0005, LR .05 for ordinary ResNet/VGG/WRN students and .01 for MobileNet/ShuffleNet, decay by .1 at 150/180/210; CE weight 1, alpha 1, temperature 4, linear distillation-loss warmup 20 epochs. The three additional DKD upstream YAMLs res56_res20.yaml, res110_res32.yaml and res50_mv2.yaml agree with these pair choices and beta values (ResNet50 inherits beta 8).

ReviewKD section 4.1 (PDF p. 6) instead uses 240 epochs, batch 128, LR .1 for ordinary students and .02 for MobileNet/ShuffleNet, decays .1 at 150/180/210. Released CIFAR-100/script/reviewKD.sh independently establishes HCL coefficients .6 for ResNet56 -> ResNet20 and 1.0 for ResNet110 -> ResNet32. It does not supply a ResNet50 -> MobileNetV2 command; do not guess that coefficient or claim the paper's benchmark table specifies it. Paper and method-specific optimization settings must be distinguished from the existing TGSC protocol retained by user instruction.

SimKD section 4.1 (PDF p. 4) uses 240 epochs, batch 64, initial LR .05 for ordinary students and .01 for MobileNet/ShuffleNet, decay at 150/180/210, weight decay .0005, and describes Nesterov momentum .9. This is a paper/code distinction: the inspected train_student.py optimizer supplies momentum without nesterov=True. Main CIFAR tables 1/2 (PDF p. 5) use different pairs, including ResNet110x2 -> ResNet110/ResNet116/ShuffleNetV2, and WRN40-2 -> MobileNetV2. ResNet110x2 weights are not the official archive's plain ResNet110 weights. Applying the implemented SimKD objective to the three additional DKD/ReviewKD pairs is a new pair evaluation, not a reproduction of those SimKD table entries.

The earlier phrase "not configured" refers to missing local JSONs, TGSC profile provenance and variable-crop integration checks, not missing published teacher-student choices. Only three supplied TGSC/configs JSONs exist (ResNet32x4 -> ResNet8x4, WRN40-2 -> WRN40-1, VGG13 -> VGG8). These distillation papers do not define TGSC crop scoring, multiplicities or curriculum profiles. Extending the same TGSC defaults to a new pair must be labeled as a derived profile; do not describe it as an existing teacher-specific TGSC setting. New feature-based pair alignment also still requires checks across curriculum resolutions.

## Provenance

Upstream code identities: https://github.com/megvii-research/mdistiller , https://github.com/DefangChen/SimKD , https://github.com/JIA-Lab-research/ReviewKD . Ported files retain attribution. Check source license terms before distributing copied modules; do not claim a license grant where a reference repository contains none.

## Additional pair delivery (2026-10-28)

Added ResNet56 -> ResNet20 and ResNet110 -> ResNet32 configs for SimKD and ReviewKD. ReviewKD coefficients .6 and 1.0 come directly from the inspected official CIFAR-100/script/reviewKD.sh. SimKD retains factor 2 and feature coefficient 1 with its feature-only objective; these are additional pair evaluations rather than claims of reproducing the distinct published SimKD benchmark pairs. TGSC/data/optimizer settings are copied unchanged from the already checked derived DKD profiles. Actual CPU forward/backward checks with official teachers passed at all five curriculum sizes (TGSC_EXTRA_METHOD_CHECKS.json). VGG13 ReviewKD remains unconfigured pending a verified pair coefficient; no fallback from another pair is used.
