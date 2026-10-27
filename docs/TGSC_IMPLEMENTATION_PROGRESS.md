# TGSC progress and handoff

Last updated: 2026-10-21 (Asia/Karachi).

## Read this first after compaction

1. Read this file for current state and accepted decisions.
2. Read [implementation tasks](../TGSC_IMPLEMENTATION_TASKS.md) for the next task's scope/checks.
3. Consult [integration plan](../TGSC_INTEGRATION_PLAN.md) for detailed contracts; FitNet notes are historical and deferred.
4. Check `git status --short` and relevant diffs in the mdistiller checkout before editing. Verify actual files/check results rather than treating this report as proof that code exists or still works.
5. Resume the next incomplete task. Do not repeat completed work unless new changes or evidence justify it.

Latest explicit user instructions take precedence over this report. When a decision changes, update the current snapshot and append the change to the history so older discussions are not mistaken for the current plan.

## Current snapshot

- **Phase:** implementation active, authorized by the user's latest request to begin work.
- **Completed:** reference audits (01-03), first implementation of self-contained TGSC runtime, model adaptations, separate oracle metrics, persistent crop banks, gates, three distillers, epoch resume/exports, runner, and four thin notebooks.
- **Next action:** task 15: real CIFAR/teacher-checkpoint GPU pilot and actual Kaggle execution. Start from `docs/TGSC_RUNNING.md` and the preparation notebook. This host has CPU-only torch, so do not report GPU acceptance as complete. Do not redo the completed audits/CPU checks without new changes.
- **Task status:** 01-14 implemented and checked at the source/CPU/isolated-notebook-structure level. Task 15 is partial: synthetic CPU end-to-end checks pass; real-data GPU and Kaggle acceptance remain pending.
- **Checks performed:** `python -m unittest discover -s tests -q`: 13 tests pass on CPU (all supplied method/pair profiles at 16/20/24/28/32, sample-weight gradients, crop save/replay, shared bank identity, all-method resume versus uninterrupted execution, and exports). `python tools/validate_tgsc_references.py --reference-root .. --output docs/TGSC_REFERENCE_CHECKS.json`: passes official SimKD/ReviewKD component comparisons and TGSC oracle crop parity on ResNet/VGG/WRN families.
- **Environment:** Windows PowerShell; Python 3.13, torch 2.6.0+cpu, torchvision 0.21.0+cpu. CUDA unavailable; YACS missing. New JSON-based runner deliberately does not require YACS. All six official teacher checkpoints have now passed CPU strict-load/forward checks; no real CIFAR training or Kaggle execution claimed.
- **Repository:** private https://github.com/ahmad903501/mdistiller-tgsc created and implementation pushed to main. Origin points there; Megvii preserved as upstream. Sibling references untouched. Core implementation commit c293f52 and notebook repository configuration commit 8bc0ec4 are uploaded.
- **Working tree:** checkpoint download/compatibility enhancement prepared for publication. Recheck live Git status/diffs and origin/main for the current published revision.
- **Known numerical difference:** TGSC's existing internal ShuffleV1 pooling adaptation changes 32x32 predictions (random-weight reference max absolute difference .01352 in saved check); other audited pooling changes match within FP32 tolerance.
- **Known tooling issue:** Python 3.13 tempfile(mode 0700) created inaccessible test folders in this sandbox. Tests now use a validated workspace scratch directory with compatible inherited permissions; initial test failures were filesystem errors, not algorithm failures.

## Accepted decisions

| Topic | Current instruction |
| --- | --- |
| Scope | Official DKD, SimKD, ReviewKD plus TGSC. FitNet deferred. Future methods should reuse the shared framework. |
| Implementation location | Change only mdistiller; sibling repos/PDFs are reference material. Runtime must work from one clone without siblings. |
| TGSC behavior | Preserve score definitions, ranking/clipping order, crop selection/multiplicity, gate, and selected pair settings. No silent redesign. |
| Augmentation | Keep TGSC defaults: `flip_prob=0.5`, `aug_padding=2`. Earlier proposal to disable them is superseded. Do not add duplicate mdistiller augmentation. |
| Misclassified crops | Retain existing top-two margin definition; do not reject a crop solely because teacher classification is wrong. |
| Losses/alignment | Preserve official losses and alignment mechanisms. No added CE/KL hybrids, new 1x1 FitNet adapters, or coefficient experiments. |
| Gating | Apply detached gates to per-sample distillation losses before batch mean. Preserve CE where native; do not add CE to feature-only SimKD. |
| DKD | Unified and requested decoupled gating; cache raw r1/r2/r3 alongside existing stored q. |
| Cache reuse | Prepare once per dataset/teacher checkpoint/preprocessing configuration and resolution bank. Reuse across methods/students. Different teachers generally require different banks. |
| Cache versus augmentation | Cache base crops and scores; training applies default TGSC augmentation. Scores still describe base crops: retain this existing approximation. Teacher/student see the same augmented tensor. |
| Pooling | Reuse existing TGSC input-size adaptations in mdistiller. Preserve official intermediate-feature alignment. |
| Baseline experiments | No newly trained runs without TGSC. Use available baselines with provenance/protocol differences disclosed. Unity-gate numerical checks remain allowed. |
| Delivery | Preparation notebook plus one notebook each for DKD/SimKD/ReviewKD; all call a shared Python runner. |
| GPU scope | One selected GPU per process initially; independent jobs on separate GPUs. DDP is deferred. |
| Recovery/export | Epoch-boundary resume and persistent Kaggle artifacts. SimKD export includes projector and frozen teacher head. |
| Publishing | User explicitly authorized creating a new GitHub repo and pushing in the latest turn. Private https://github.com/ahmad903501/mdistiller-tgsc created and pushed on main. Kaggle dataset uploads remain user-managed. Kaggle dataset uploads remain user-managed. |

## Verified facts worth carrying forward

- TGSC model copies already use adaptive final pooling in ResNet, WRN, MobileNetV2, and ShuffleNet; VGG/ResNetV2 also have adaptive final pooling. TGSC ShuffleNetV1 has an additional internal adaptive pooling replacement. Verify behavior when porting; not every change guarantees exact 32x32 parity.
- Existing TGSC caches save cropped tensors, labels, composite q, source indices, and selection flags; separate metric components are not persisted. Existing cache identity lacks teacher checkpoint content hashing.
- TGSC provides optional disk caching. ResNet32x4 and VGG JSONs have it off; WRN40-2 -> WRN40-1 has it on.
- All three inspected TGSC JSONs use batch 128 and five 48-epoch stages at 16/20/24/28/32. ResNet32x4/VGG milestones are `[48,144,192]`; WRN pair uses `[48,144,192,210]`. Do not flatten these differences into one invented default.
- DKD/ReviewKD's 20-epoch warmups apply to their distillation loss, not LR.
- Local ReviewKD resnet32x4 -> resnet8x4 YAML overrides review weight to 5, although global default is 1.
- Local DKD WRN40-2 -> ShuffleV1 YAML inherits beta 8; the user's initially pasted blanket WRN beta 6 does not describe that merged config.
- Official SimKD example uses feature-loss weight 1, CE/KL zero, factor 2. The reused teacher classifier and projector are needed at inference.
- mdistiller's original requirements pin torch 1.9/torchvision 0.10; don't install these blindly over Kaggle's compatible GPU stack.

These observations are from source/config inspection, not numerical validation. Recheck affected code during its assigned task.

## Reference locations and revisions

Relative to the mdistiller checkout:

- DKD: local `mdistiller/distillers/DKD.py`, `configs/cifar100/dkd/`, `../DKD.pdf`.
- SimKD: `../SimKD/models/util.py`, `helper/loops.py`, `train_student.py`, `scripts/run_distill.sh`, `../SimKD.pdf`.
- ReviewKD: `../ReviewKD/CIFAR-100/` and local `mdistiller/distillers/ReviewKD.py`, `../ReviewKD.pdf`.
- TGSC: `../TGSC/src/tgsc_kd/`, `../TGSC/configs/`, `../TGSC/model/python/`.
- FitNet: `../fitnet.pdf`; future work only. Do not spend current implementation effort recovering its historical epoch counts.

Previously inspected revisions, to be verified live if needed:

| Repository | Revision |
| --- | --- |
| mdistiller | `a08d46f10d6102bd6e3f258ca5ac880b020ea259` |
| SimKD | `2b389c31ed7779aea31e7aaf0bb0f2d8b6ac2f01` |
| ReviewKD | `cede6ea6387ae9b6127de0e561507177bf19c11e` |

## Task tracker

| ID | Task | Status |
| --- | --- | --- |
| 01 | DKD reference audit | Complete (CPU checks) |
| 02 | SimKD reference audit | Complete (CPU checks) |
| 03 | ReviewKD reference audit | Complete (CPU checks) |
| 04 | Self-contained TGSC config/package | Complete (offline checks) |
| 05 | Variable-size model compatibility | Complete (CPU checks) |
| 06 | Crop generation and separate metrics | Complete (CPU checks) |
| 07 | Persistent teacher-specific crop banks | Complete (CPU checks) |
| 08 | Shared per-sample gates | Complete (CPU checks) |
| 09 | DKD integration | Complete (CPU checks) |
| 10 | SimKD integration and inference | Complete (CPU checks) |
| 11 | ReviewKD integration | Complete (CPU checks) |
| 12 | Curriculum trainer/checkpoint recovery | Complete (CPU checks) |
| 13 | Python runner/GPU selection | Complete (offline checks) |
| 14 | Kaggle notebooks/instructions | Complete (offline checks) |
| 15 | GPU acceptance/pair expansion | Partial: CPU pilot passed; GPU/Kaggle pending |

## History and verification log

### 2026-10-21: planning consolidated; handoff initialized

- Created the integration plan and then the 15-task breakdown; no implementation changes.
- Superseded early proposals: disabling default augmentation, altering oracle clipping, imposing a universal 20 crop floor, changing crop multiplicity, adding loss hybrids, and training fresh ungated baselines.
- Investigated original FitNet's two-stage procedure and uncertain epoch budget; user subsequently deferred FitNet entirely.
- Confirmed mdistiller is the sole implementation/push target and references remain guidance only.
- Confirmed newly available `DKD.pdf`, `SimKD.pdf`, `ReviewKD.pdf` titles using first-page text extraction. Full formal audits remain pending.
- Added this progress/handoff record at the user's explicit request to survive context compaction.
- Verification during this handoff: inspected live Git status and task list; checked document links/15 task entries. No training result is claimed.

## Update discipline

Keep the snapshot and task tracker current; append substantive progress entries instead of rewriting historical outcomes. Update after each completed task, significant decision, failure that changes the approach, or interrupted partial task.

Each entry should include:

- Task ID/status and concrete outcome.
- Changed paths and why.
- Reference revision and paper section/equation actually consulted.
- Exact focused check commands, environment, concise results, and evidence/output paths when saved.
- For failed checks: primary error, what was tried, and the remaining issue; no large passing logs.
- Next action and any unresolved decision. For partial work, identify edited files and checks not yet run.

Record commit hashes only after commits actually exist. Record GPU/Kaggle verification only after execution. Before compaction or ending an implementation turn, ensure the next action is concrete enough to resume without reconstructing the chat.

### 2026-10-21: first implementation and CPU reference checks

- Authorized by the user's request to begin work; implementation proceeded through tasks without repeated approval gates.
- Added `docs/TGSC_REFERENCE_AUDIT.md` after comparing official DKD code/configs, official SimKD projector/loop, official ReviewKD ABF/HCL/factory/scripts, and relevant supplied PDF method sections.
- Ported TGSC config/utilities/oracle into `mdistiller/tgsc/`; retained formulas/selection and default augmentation while adding raw components/boxes/source metadata.
- Added persistent schema-2 crop bank with teacher checkpoint + dataset + normalized source identity, atomic writes, separate read/write roots and writer lock. No metric redesign.
- Ported inspected TGSC pooling changes into CIFAR models; exposed final prepool features where needed for SimKD.
- Added shared gate helpers, per-sample DKD unified/decoupled paths, official SimKD projector/inference graph, dynamic-size ReviewKD with native HCL, and explicit method configs.
- Added runtime trainer/resume/inference exports and GPU-selecting Python runner. Added preparation + three method notebooks and running guide; final isolated validation is next.
- Commands: `python -m unittest discover -s tests -q` (10 tests, pass); `python tools/validate_tgsc_references.py --reference-root .. --output docs/TGSC_REFERENCE_CHECKS.json` (pass). Subprocess capture was used to filter repetitive output; original legacy DKD emits a deprecation warning.
- Resume checks compare full final state against uninterrupted CPU execution for DKD/SimKD/ReviewKD across 16 -> 20 crop transition. Synthetic pixels/random weights only; not accuracy evidence.
- Official source discrepancy retained/documented: ReviewKD released script starts warmup epoch at zero; mdistiller starts at one. Pair-specific weights are explicit. VGG ReviewKD config omitted pending verified pair coefficient; no guessed default.
- No CUDA device here. Task 15 cannot be called complete until real-data/GPU and Kaggle execution occur.

### 2026-10-21: isolated delivery validation completed

- Tasks 04/13/14 completed for offline delivery: self-contained package/configs, shared Python CLI, GPU selection/error handling, four thin notebooks and running guide. README links to the workflow.
- Exact delivery check: `python tools/validate_tgsc_delivery.py`. It copies only runtime/config/tools/tests into a workspace scratch tree, clears PYTHONPATH, verifies imported runtime comes from that tree, runs all 11 self-contained tests, parses all four notebooks and checks CLI input errors. Result: pass. Evidence: `docs/TGSC_VALIDATION.json`.
- Checks now include independently zeroing DKD target/non-target gates while CE stays unchanged, all 11 supplied configs at five resolutions, reference component gradients, crop/cache replay, and per-method CPU resume versus uninterrupted final state.
- Corrected resume portability: checkpoint carries cloned best inference weights so `inference_best.pt` is available in a fresh output directory even when no later epoch improves. Run metadata includes launch arguments, dependency versions and parameter counts.
- Normalized source hashing across Windows/Linux line endings. Added crop-box/q consistency checks, retained only existing TGSC transforms, and documented derived heterogeneous profile provenance.
- `git diff --check`: passed. The two inaccessible workspace scratch folders from the initial tempfile failure were removed successfully using approved native PowerShell cleanup; no remaining cleanup issue is known.
- No reference repository changes, Git commit/push, dataset download, full training, or external publication. No CUDA/Kaggle execution claimed. Real checkpoint loading/accuracy, GPU memory/performance, and actual Kaggle artifact/session behavior still require task 15.
- Supported initial profiles: four pairs each for DKD/SimKD (three existing TGSC pair profiles plus explicit derived ResNet32x4 -> ShuffleV2), three ReviewKD pairs. VGG ReviewKD is omitted until a pair coefficient is verified; other pairs require explicit audited configs.
- FLOPs and long-run accuracy are not measured. CPU timings/synthetic test accuracy are correctness diagnostics only.

### 2026-10-21: publication preparation and Kaggle T4 guidance

- User authorized push and then clarified that a new repository must be created. Current origin is still Megvii upstream; do not push there. Planned new private repository name: `mdistiller-tgsc`, preserving Megvii as upstream. Owner account selection is pending; multiple saved GitHub accounts exist.
- Added private-repo clone support to all four notebooks using Kaggle Secret `GITHUB_TOKEN` and temporary Git askpass code. No token is embedded in notebook source, Git URLs, or generated files.
- Expanded running guide with one-T4-per-job recommendation, two independent GPU 0/1 subprocess launches within one session, cache-output-to-Dataset steps, and one-epoch smoke/resume workflow.
- `python tools/validate_tgsc_delivery.py`: all 11 tests/isolated checks and four notebook parses pass after these updates. Embedded askpass Python also parses; credential-pattern scan found no embedded secrets.
- CUDA/Kaggle throughput is still unmeasured. DDP is not implemented; selecting T4 x2 alone does not make a single method use both GPUs.

### 2026-10-21: GitHub repository created

- User selected `ahmad903501` as owner. Created private `ahmad903501/mdistiller-tgsc` using that account's saved Git credentials, without printing/storing tokens.
- Local implementation commit: `c293f52` (Integrate TGSC with DKD, SimKD, and ReviewKD). Initial push is in progress; do not treat repository creation as proof of upload.
- Notebook clone URLs now target the new repository on main; private cloning uses Kaggle Secrets. Added the T4 workflow to the running guide.
- Cleaned reference-inherited trailing whitespace in three Python files. No numerical behavior changes; source-based cache keys will reflect the final committed sources.

### 2026-10-21: initial push succeeded

- Created private https://github.com/ahmad903501/mdistiller-tgsc under the user-selected account.
- Preserved original Megvii remote as `upstream`, configured the new repository as `origin`, renamed the local branch to `main`, and pushed with tracking enabled. Git confirmed new remote branch main and exit status 0.
- Uploaded commits: c293f52 (TGSC implementation) and 8bc0ec4 (actual private-repo URL and whitespace cleanup). This handoff update is a subsequent documentation commit.
- User-facing Kaggle recommendation: prepare once per teacher on GPU 0, preserve/upload the bank as a Dataset, validate one epoch, then use independent GPU 0/1 jobs for experiment throughput if desired. One run across two GPUs remains unsupported/unbenchmarked.
- Private notebooks use read-only GitHub access via Kaggle Secret GITHUB_TOKEN. Use the same code revision and torch/torchvision versions for cache preparation/training. Attach original CIFAR, teacher weights and cache to method notebooks.
- GPU/Kaggle acceptance is still pending; pushing code does not establish GPU performance or accuracy.

### 2026-10-21: official teacher archive inspected; notebook download added

- User requested an automated teacher download in the preparation notebook and questioned whether the naming had been inspected. Earlier path advice was based on the model registry, not archive inspection; this distinction was stated explicitly.
- Downloaded the actual official GitHub `cifar_teachers.tar` asset: 364,390,400 bytes. Inspected all six `.pth` member paths; filenames match the registry convention, including case-sensitive `ResNet50`. Exact members, archive/checkpoint hashes, environment, and verification results are saved in `docs/TGSC_TEACHER_ARCHIVE_CHECKS.json`.
- Extracted and strictly loaded resnet32x4, wrn_40_2, vgg13, resnet56, resnet110 and ResNet50 into their declared architectures; all produced finite [2,100] logits on a CPU 32x32 probe. These are real official weights, not synthetic checkpoint fixtures; no dataset accuracy claim is made.
- Real inspection exposed legacy NumPy scalar accuracy metadata rejected by torch 2.6 weights-only loading. Added a scoped scalar/dtype constructor allowance (NumPy 1.x/2.x paths) while retaining `weights_only=True`; no unrestricted pickle fallback.
- Added `tools/download_teachers.py`: official URL only, actual archive-entry matching, controlled single-member extraction, checkpoint SHA comparison against archive contents, strict state-dict/forward verification. Downloaded weights/archive stay under ignored download_ckpts; no binary payload is committed.
- Added preparation-notebook `DOWNLOAD_TEACHER=True` and writable TEACHER_DOWNLOAD_ROOT defaults, plus a download/verification cell after clone/setup. CONFIG_NAME chooses teacher; the model registry supplies destination naming. Attached checkpoints remain supported when the flag is false.
- Added regression checks for NumPy metadata and case-sensitive member matching. `python tools/validate_tgsc_delivery.py`: all 13 tests, isolated runtime, notebook parses and CLI checks pass. Actual official-file checks used the new helper with --archive; the network archive was downloaded via curl.
- Update older pinned notebook revisions before using the new loader/download cell. GPU/Kaggle acceptance remains pending.

### 2026-10-21: selectable Kaggle/download sources and single-teacher upload folder

- User requested both downloading the selected teacher and reusing an attached Kaggle Dataset, plus simple CIFAR Dataset loading. All four notebooks now expose TEACHER_SOURCE and CIFAR_SOURCE (download/kaggle), validate source values/layout, and use writable roots for downloads. Preparation defaults to downloading both; method notebooks default to attached artifacts.
- Official release exposes only a combined teacher archive; choosing a teacher extracts only that checkpoint but still requires the archive download. No individual official URL was invented. The reference TGSC uses torchvision.datasets.CIFAR100; retained that loader and original sample order instead of adding a different dataset package/format.
- Downloader now writes teacher_manifest.json with the teacher key, archive member, SHA-256 and portable relative checkpoint path. Generated an ignored local upload-ready folder at download_ckpts/kaggle/resnet32x4/ from the already inspected archive: one 59,554,438-byte checkpoint plus manifest, no tar. Actual checkpoint strict-load/finite-forward verification passed; hash independently checked against the manifest.
- Running guide includes both notebook modes, local per-model extraction command, exact example Kaggle mount path, and separate teacher/CIFAR/crop-bank Dataset preservation. No Kaggle Dataset was uploaded or published automatically.
- Checks: python tools/validate_tgsc_delivery.py passed all 13 isolated tests, four notebook parses and CLI checks. A focused execution check covered all 16 notebook source combinations using mocked downloads, invalid source rejection, and upload folder contents/hash. git diff --check passed. GPU/Kaggle execution is still pending.
- Source-options implementation committed as ec4677b and pushed successfully to origin/main. Next action: user can upload contents of download_ckpts/kaggle/resnet32x4/ as tgsc-teacher-resnet32x4, attach it with original CIFAR and the crop bank, and perform the documented one-epoch T4 smoke/resume run.

### 2026-10-22: public GitHub page access diagnosis

- User made origin public and reported a page-loading error. Unauthenticated GitHub API returned HTTP 200 with private=false, visibility=public and default_branch=main. Direct GitHub repository HTML returned HTTP 500; raw preparation notebook returned HTTP 200 and parsed as valid notebook JSON. Repository visibility and raw file access are working; the page request is failing server-side, with root cause not yet established.
- Existing notebooks still default PRIVATE_REPO=True. For public clone access set it False in the settings cell; token setup is then unnecessary. No notebook defaults changed pending clarification of the failing page/cell.
- Asked user for failing URL/error/location. Next action: use raw/local notebook upload to proceed with Kaggle, and diagnose the exact reported error if different from the reproduced repository HTTP 500. No training or GPU checks performed. Diagnostic downloaded files are ignored under .test-artifacts/.

### 2026-10-22: replacement public repository

- User confirmed the failing page is the GitHub repository itself and authorized trying a new repository. Created public ahmad903501/tgsc-mdistiller via authenticated GitHub API under the previously selected account. Existing ahmad903501/mdistiller-tgsc is preserved; no repository deletion or history rewriting.
- Updated all four notebooks to the new clone URL and PRIVATE_REPO=False; GitHub token setup is optional for private forks only. Updated running guide. Local main will be pushed with its existing history, so previous revision pins remain valid.
- Publication checks: all four notebook code cells parse and public URL/flag assertions pass; git diff --check passes. Commit 9745b98 pushed successfully to new origin/main with tracking. Previous origin preserved as previous-origin; Megvii remains upstream. Raw new preparation notebook returns HTTP 200 and its published public settings were verified. New GitHub HTML-page check could not connect (HTTP 000 after timeout), so successful browser rendering is not yet confirmed. Next action: user opens the new repository URL; raw/local notebook import is available meanwhile. No numerical code changed; GPU/Kaggle training acceptance remains pending.

### 2026-10-22: stage method notebooks during Kaggle preparation

- User wants the three training notebooks staged in Kaggle while data preparation runs. No Kaggle CLI, local credential file or credential environment is configured here, so no account-side notebook creation/upload was claimed. Asked whether they will attach saved preparation output or separate Datasets, and for its URL/mounted paths.
- Pinned the three method notebooks to existing runtime revision 3d350da, the revision previously recommended for preparation; adjust to the actual preparation revision if different. Existing separate configs/output directories and public no-token settings retained. No runtime/scientific source changed, so this notebook-only edit does not alter cache identity.
- Added running-guide staging instructions, including paths relative to the whole saved preparation output, fresh-run defaults, and sequential preflight/train/evaluate flow. Actual attachment paths remain user-dependent; do not invent a Kaggle mount path or assert readiness before attachment/preflight.
- Checks passed: all three edited notebooks parse; pinned revision, public clone settings and distinct config/output assertions pass; git diff --check passes. Publication is the next operation; verify origin/main against HEAD if resuming. Next user action: import/save three drafts, attach completed inputs, run DKD one-epoch smoke/resume first and then SimKD/ReviewKD. Kaggle/GPU execution remains unverified.

### 2026-10-22: PDF check for remaining teachers

- User asked to check supplied PDFs after discussing teachers without current TGSC configs. Re-read DKD.pdf, ReviewKD.pdf and SimKD.pdf (local references only), with physical PDF page/table evidence recorded in TGSC_REFERENCE_AUDIT.md.
- DKD/ReviewKD publish ResNet56 -> ResNet20, ResNet110 -> ResNet32, ResNet50 -> MobileNetV2. DKD appendix beta values are 2/2/8, with alpha 1, T4 and 20-epoch warmup. Official DKD YAMLs agree. ReviewKD official commands establish coefficients .6 and 1 for first two; ResNet50 pair coefficient remains unverified by the checked sources.
- SimKD paper's CIFAR benchmark pairs differ, notably ResNet110x2 rather than archive ResNet110. Its paper says Nesterov momentum but inspected optimizer does not enable it; recorded the distinction without changing the implementation. Do not call applying SimKD to additional DKD pairs an exact paper-pair reproduction.
- Papers give pair choices and baseline training settings, but no TGSC crop profiles. Existing TGSC source still supplies only three pair JSONs. New crop profiles must be derived explicitly and feature alignment checked; no configs/runtime changes were made in this read/audit task. Current Kaggle preparation remains unaffected.
- Check: git diff --check; numerical tests unnecessary for documentation-only audit. Next action: use the verified pair choices when extending configs, preserving user-approved TGSC settings and recording derivation; resolve remaining ReviewKD coefficient before an official-default claim. No GPU/Kaggle execution claimed.

### 2026-10-22: additional teacher preparation configs

- User requested usable CONFIG_NAME/settings for ResNet56, ResNet110 and ResNet50 so their Dataset can be created. Added dkd_resnet56_to_resnet20.json, dkd_resnet110_to_resnet32.json and dkd_ResNet50_to_MobileNetV2.json, with explicit derived-profile provenance.
- Preserved existing ResNet32x4 TGSC data/crop schedule, oracle, scoring, augmentation and selection settings exactly. Pair-specific official DKD beta=2/2/8; MobileNetV2 LR=.01, other new students LR=.05. No SimKD/ReviewKD configs were silently added, no core runtime/source changes, and existing banks remain usable.
- Actual checks: loaded official pretrained weights for all three; prepared and reloaded cache records for two synthetic source images at 16/20/24/28/32; DKD loss and backward finite at every size. Evidence: TGSC_ADDITIONAL_TEACHER_CHECKS.json. The first PowerShell stderr-redirection wrapper returned shell failure on native warning/progress text despite completed checks; reran with captured subprocess output and confirmed Python exit 0 for all three. Full isolated delivery validator passes all 13 tests/notebook/CLI checks. git diff --check passes.
- Running guide now lists exact prep config names/output suggestions and the need to update old pinned checkouts after active jobs finish. Next action: publish these config/docs changes; user refreshes completed/idle preparation checkout, runs three new configs sequentially, and retains all banks/weights in final Dataset. Full-data Kaggle/GPU preparation and training remain unverified here.

### 2026-10-28: minimal upload packages from real Kaggle preparation archives

- User supplied ../datasets_cache and requested retaining important inputs because uploading CIFAR is slow/problematic. Inspected all four ZIP central directories and real cache manifests: ResNet32x4, VGG13, ResNet56, ResNet110, each 50,000 source images and five crop files. WRN40-2 and ResNet50 are absent from this folder. Originals total 9.77 GiB and include repeated CIFAR train/test/archive, teacher tar, repository/.git, logs and reference tensors.
- Implemented tools/package_tgsc_upload.py and executed it on all four originals. Generated ../datasets_cache/kaggle_upload/tgsc_{teacher}.zip (6.13 GiB total), plus upload_manifest.json and README_UPLOAD.txt. Exactly seven required members per ZIP: official teacher checkpoint, bank manifest, five res*.pt files. Original files preserved; excluded CIFAR, teacher tar, repository, logs and reference.pt (preparation-only). Info-ZIP compressed-copy mode avoids extraction/recompression; checkpoint hash, manifest-key identity, exact members/size/CRC, and SHA-256 of every compressed payload verified. Portable Python streaming fallback is covered by tests.
- Current runner still requires original CIFAR train data for identity/label validation and test data for evaluation. Chose existing supported CIFAR downloading instead of weakening cache checks: three method notebooks now default CIFAR_SOURCE=download, writable /kaggle/working/data, TEACHER_SOURCE=kaggle and runtime revision 499a60f (includes extra teacher configs). No scientific runtime/source changes and no bypass of original data validation. requirements-tgsc.txt needs numpy/tqdm plus compatible platform torch/torchvision; legacy requirements.txt torch1.9 pins are not appropriate.
- Real bank metadata records torch2.11.0+cu128/torchvision0.26.0+cu128. Source hash initially differed on Windows; confirmed archived source text is identical, and Linux lexicographic mixed-case filename ordering exactly reproduces stored source_sha256 3298408f1a6c5fab18d384bb4b7dce0757eeb4ebde4ed5c4108bd88020ea6d77. Do not rewrite cache keys or claim cross-version compatibility. Packaging does not modify payloads.
- Checks: 16 isolated tests pass (including new missing-file/wrong-teacher rejection, source preservation, exact selected contents, no-overwrite, compressed-copy and fallback contracts), all four notebook parses and CLI checks pass via validate_tgsc_delivery.py. Real artifact packaging checks passed for all four. Full crop CRC decompression/accuracy was not measured; compressed output payloads are proven identical to supplied inputs. No Kaggle upload or GPU run performed here.
- Next action: publish packaging/notebook/docs updates; user uploads only four generated tgsc_*.zip files to their public Dataset, copies actual mounted teacher/cache paths into notebooks, uses writable CIFAR download root, and runs preflight in the same dependency versions used during preparation.

### 2026-10-28: new ResNet50 ZIP is incomplete

- User added ../datasets_cache/resnet50cache.zip and requested matching minimal packaging. Inspected all 243 file members: teacher checkpoint and teacher manifest exist, but no crop bank manifest, reference.pt or res16/20/24/28/32 payloads. tgsc_cache contains only prepare.log.
- Log confirms CUDA out of memory during teacher-union preparation at batch 109/196 on a 14.56 GiB GPU; no bank was saved. Do not label it a complete fifth cache or fabricate missing manifests. No root-cause memory fix or successful retry has been established.
- Generated ../datasets_cache/kaggle_upload/teacher_only_ResNet50.zip (191.33 MiB) with only checkpoint and supplied teacher manifest; verified checkpoint SHA-256 9f546083e2e4e5aaa733b328f34c354acb93b168952684f7742f556ba1253c3d, exact members, CRC/size and compressed payload SHA-256 against original. Original and four prior complete packages untouched. Added ResNet50_status.json and appended explicit warning to README_UPLOAD.txt; aggregate complete-cache upload_manifest.json remains four records.
- Next action: successfully rerun ResNet50 preparation (investigate GPU memory/use a smaller preparation batch as appropriate), obtain all five res*.pt files plus bank manifest, then package the complete output. Existing four caches remain ready for upload. No runtime/code change, Kaggle upload or GPU execution performed in this turn.

### 2026-10-28: run matrix for uploaded four teachers

- User reports four complete caches uploaded and requests a tracker for all methods. Added docs/TGSC_RUN_TRACKER.md: 12 primary teacher/student/method runs, seed27/240epochs, DKD unified first. Seven config paths verified present, five explicitly marked pending setup. No training success inferred from uploaded data; all run statuses Not started.
- Recorded optional ResNet32x4 -> ShuffleV2 three-run extension, existing-cache reuse, additional-pair SimKD provenance, unresolved VGG ReviewKD coefficient, and output/accuracy columns. Dataset URL/version still pending. Next action: first ResNet32x4 DKD one-epoch GPU preflight/smoke; implement/test missing configs before claiming all twelve are runnable. No runtime changes or GPU checks performed.

### 2026-10-28: VGG13 Kaggle mount settings and ReviewKD blocker

- User is setting up ReviewKD + TGSC with uploaded VGG13 Dataset at /kaggle/input/datasets/ahmadjawwad903/tgsc-vgg13-try1. Given mounted root contains tgsc_cache/ and teacher_weights/cifar_teachers/vgg13_vanilla/; use the latter plus ckpt_epoch_240.pth, and the former as cache root. Actual remote mount not accessed here; paths are user-provided.
- Rechecked current config files: reviewkd_vgg13_to_vgg8.json still does not exist. ReviewKD VGG coefficient and feature correspondence remain unverified; do not leave ResNet CONFIG_NAME while changing only the VGG teacher checkpoint, or supply a nonexistent config as a ready-to-run setup.
- Shared input settings can be supplied now: teacher source kaggle, CIFAR source download into /kaggle/working/data, GPU0 seed27, distinct reviewkd_vgg13_vgg8 output path, fresh RESUME=None. Next action: establish official VGG ReviewKD coefficient/alignment and add/test that config before training. No GPU run or configuration implementation claimed.

### 2026-10-28: per-run Kaggle notebooks, method config prerequisites

- User supplied four public Dataset URLs and requested separate notebooks cloned/edited from existing method notebooks for all runs. Mapping: resnet56=toxica001/tgsc-resnet56-try1, vgg13=ahmadjawwad903/tgsc-vgg13-try1, resnet32x4=aj903501/tgsc-resnet32x4-try1, resnet110=ahmadjawwad903/tgsc-resnet110-try1.
- Added missing SimKD and ReviewKD configs for ResNet56 -> ResNet20 and ResNet110 -> ResNet32. Official ReviewKD .6/1.0; SimKD factor2/feature1 as additional pair evaluations. TGSC scientific settings unchanged. Official-weight CPU forward/backward checks pass at all five crop sizes; recorded in TGSC_EXTRA_METHOD_CHECKS.json. No runtime/model/oracle source changes, so supplied Linux crop-bank identities remain valid.
- VGG13 ReviewKD still lacks a verified pair weight; no invented fallback. Plan is 11 runnable notebooks plus 1 clearly blocked VGG13 ReviewKD notebook, with manifest/tracker distinguishing them. Each will use the supplied Dataset, exact teacher path, writable CIFAR download root, unique output and pinned runtime revision. Publication/runtime-pin commit is next, followed by generating and validating notebooks. Kaggle attachment and execution still require user account UI; no local credentials.

### 2026-10-28: all per-run notebooks generated and checked

- Runtime/config prerequisite commit exists: 5c01799. Generated notebooks/runs/ with 11 runnable primary experiments and tgsc_reviewkd_vgg13_to_vgg8_PENDING.ipynb, which raises an explicit blocker before cloning/training. Cloned the three existing method templates via tools/generate_tgsc_run_notebooks.py; every runnable notebook pins runtime 5c01799 and its own teacher/student/config/Dataset/output. Shared method templates are preserved.
- Exact Dataset mapping supplied by user is in notebooks/runs/run_manifest.json and README index. Input resolver accepts both /kaggle/input/datasets/<owner>/<slug> and /kaggle/input/<slug>, searches nested extracted layouts, requires one matching teacher and complete five-resolution bank, rejects missing/ambiguous attachments, and checks preparation torch/torchvision versions. It does not auto-attach Datasets or silently download crop banks. Internet needed for public clone/install and original CIFAR download; no token.
- Updated docs/TGSC_RUN_TRACKER.md to 11 configs available, one blocked; methods remain Not started because no training result supplied. New SimKD pairs are clearly additional evaluations, not published pair reproductions. VGG ReviewKD still needs verified coefficient/alignment, so it is not called ready.
- Created local ../kaggle_notebooks/TGSC_ALL_RUNS.zip (12 notebooks + index + manifest, 51.9 KiB), tested ZIP CRC and 12-notebook count. This is for notebook import, separate from teacher/cache upload folder.
- Checks passed: four new method configs with official teachers have finite CPU losses/backward at all five crop sizes; three notebook input/settings tests pass, and full isolated validate_tgsc_delivery.py passes 19 tests, parses all 16 notebooks, and CLI/import checks. Extended delivery validator to copy/validate run notebooks too. git diff --check passes. No GPU/Kaggle execution or Dataset version verification claimed.
- Next operation: commit/push generated delivery artifacts and validation report to origin/main, including ancestor runtime revision 5c01799. Verify live remote if resuming. Next user action: extract TGSC_ALL_RUNS.zip, import a runnable notebook into Kaggle, attach its linked Dataset, enable GPU+Internet, Run All (prefer DKD one-epoch first acceptance); preserve each result. No Kaggle account upload performed from this environment.
