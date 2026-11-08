"""Deliver the two requested full/crop VRM variants with official32px transforms."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from generate_epkd_notebooks import cell, INPUTS, RUN, GITHUB_BOOTSTRAP


def make_notebook(name, passes, revision=None):
    revision = revision or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    intro = f'''# VRM + CAM: ResNet32x4 to ResNet8x4 —28px {passes[3]} pass(es)

This is the **published-code VRM variant** (VISION-SJTU/VRM@56d02ce), wrapped in our
CAM curriculum. The code omits the paper's joint-entropy unreliable-edge mask; no
unverified pruning rule is added. Losses, reductions, logit normalization and nested
weights match that code exactly. No independent full/crop KD or extra KL loss is added.

| Size | Epochs | Stride | Passes | Two VRM inputs |
|---|---|---|---|---|
|16|1–48|8|3|Weak full image + weak CAM crop|
|20|49–96|8|3|Weak full image + weak CAM crop|
|24|97–144|8|3|Weak full image + weak CAM crop|
|28|145–192|2|{passes[3]}|Weak full image + weak CAM crop|
|32|193–240|—|1|**Original VRM weak + strong32px views**|

28px has nine grid positions; top3 are cached. One pass uses top1; three passes use
top1/top2/top3. Scores are mean ground-truth-class CAM, not teacher confidence.
At32px, original RandomCrop32/padding4/flip + customRandAugment(2,10)/Cutout and
original normalization are used, including original constant padding. Both32px student
forwards update BN as official VRM does. At smaller sizes crop forwards preserve BN
running statistics, while the full branch updates them (EP-KD policy).

Batch64, FP32, SGD LR.05/momentum.9/weight decay.0005,240epochs. This integration uses
the EP-KD LR cuts48/144/192/210, not the paper's baseline150/180/210 schedule.
Seeds0/1/2 run once each, sequentially on GPU. Change GPU to select another GPU.
Option2 has more optimizer steps at28px; this is not an equal-compute comparison.

**Kaggle:** enable Internet and a GPU, then Run All. Setup clones the published code
from GitHub at a pinned commit; no runtime Dataset or ZIP is needed.
Teacher/original CIFAR can be attached or downloaded.
Set SMOKE_TEST=True for a short5stage stop/resume/export test before full training.
Logging, plots and resumable checkpoints are saved under OUTPUT. Output paths distinguish
the two options and smoke runs. GPU throughput and full benchmark accuracy are unmeasured.
'''
    config = f'''from pathlib import Path
REPO = Path("/kaggle/working/mdistiller_vrm_github")
REPO_URL = "https://github.com/ahmad903501/tgsc-mdistiller.git"
REVISION = {revision!r}
REQUIRED_SOURCE_FILES = ["tools/run_epkd.py", "mdistiller/epkd/runtime.py", "mdistiller/distillers/VRM.py", "mdistiller/epkd/vrm_augment.py"]
DATA_ROOT = Path("/kaggle/working/data")
OUTPUT = Path("/kaggle/working/vrm_cam_28p{passes[3]}")
TEACHER = "resnet32x4"
TEACHER_CHECKPOINT = None  # Discover an attached teacher or download official weights.
CONFIG_NAME = {name!r}
GPU = 0
SEEDS = [0,1,2]
EVAL_EVERY = 5
TEST_DATA = None
SMOKE_TEST = False  # True:128images,1epoch/size,first seed only; retain actual pass counts.
'''
    setup = GITHUB_BOOTSTRAP + '''
subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(REPO/"requirements-tgsc.txt")],check=True)
import torch
assert torch.cuda.is_available(), "Enable GPU; this notebook does not silently train on CPU."
print("Runtime:",REPO,"GPU:",torch.cuda.get_device_name(GPU))
OUTPUT.mkdir(parents=True,exist_ok=True)
'''
    launcher = RUN.split("for name in CONFIGS:")[0] + '''config = REPO/"configs/epkd/cifar100"/CONFIG_NAME
if SMOKE_TEST:
    raw = json.loads(config.read_text())
    raw["data"]["stages"] = [[r,1] for r in (16,20,24,28,32)]
    raw["train"]["epochs"] = 5
    config = OUTPUT/(config.stem+"_SMOKE.json")
    config.write_text(json.dumps(raw,indent=2))
for seed in (SEEDS[:1] if SMOKE_TEST else SEEDS):
    output = OUTPUT/(config.stem+f"_seed{seed}")
    latest = output/"latest.pt"
    run("check",config,seed,output,["--limit","16"])
    if SMOKE_TEST:
        if not latest.exists():
            run("train",config,seed,output,["--limit","128","--stop-after-epoch","2"])
        run("train",config,seed,output,["--limit","128","--resume",str(latest)])
    else:
        run("train",config,seed,output,["--resume",str(latest)] if latest.exists() else [])
    export = output/("inference_latest.pt" if SMOKE_TEST else "inference_best.pt")
    run("evaluate",config,seed,output,["--export",str(export),*(["--limit","128"] if SMOKE_TEST else [])])
    if SMOKE_TEST:
        saved = torch.load(latest,map_location="cpu",weights_only=True)
        history = saved["metrics_history"]
        assert [r["resolution"] for r in history] == [16,20,24,28,32]
        assert history[3]["passes"] == json.loads(config.read_text())["epkd"]["passes"][3]
        assert history[-1]["second_view"] == "vrm_original_strong32"
        assert all(set(r["train"]) == {"pair.loss_ce","pair.loss_isv","pair.loss_icv","loss_total"} for r in history)
        print("PASS: all stages,paired loss,28px pass count,original32px augmentation,resume and export")
    from IPython.display import display,Image
    display(Image(filename=str(output/"training_curves.png")))
'''
    nb = dict(cells=[cell("markdown",intro,0),cell("code",config,1),cell("code",setup,2),
                    cell("code",INPUTS,3),cell("code",launcher,4)],nbformat=4,nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3",display_name="Python 3",language="python"),
                            language_info=dict(name="python",codemirror_mode=dict(name="ipython",version=3),
                                               pygments_lexer="ipython3",mimetype="text/x-python",file_extension=".py",
                                               nbconvert_exporter="python")))
    path = ROOT/"notebooks/runs"/(Path(name).stem+".ipynb")
    path.write_text(json.dumps(nb,indent=1),encoding="utf-8")
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",default=str(ROOT.parent/"kaggle_notebooks/VRM_CAM_RUNTIME.zip"))
    p.add_argument("--revision",help="Published implementation commit; defaults to current HEAD")
    args=p.parse_args()
    revision=args.revision or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    base=json.loads((ROOT/"configs/epkd/cifar100/dkd_resnet32x4_to_resnet8x4.json").read_text())
    base["method"]="VRM"
    base["method_options"]={}
    base["data"]["batch_size"]=64
    base["profile_source"]="VRM published code56d02ce + paired full/CAM curriculum; original weak/strong transforms at32px"
    for count in (1,3):
        cfg=json.loads(json.dumps(base))
        cfg["epkd"]["passes"]=[3,3,3,count,1]
        cfg["epkd"]["strides"]=[8,8,8,2,8]
        name=f"vrm_cam_resnet32x4_to_resnet8x4_28p{count}.json"
        cfg["train"]["run_name"]=Path(name).stem
        cfg["train"]["save_path"]="vrm_cam_runs/"+Path(name).stem+"/model.pth"
        (ROOT/"configs/epkd/cifar100"/name).write_text(json.dumps(cfg,indent=2))
        print(make_notebook(name,cfg["epkd"]["passes"],revision))
    bundle=Path(args.bundle)
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
        files=[*ROOT.glob("mdistiller/**/*.py"),*ROOT.glob("tools/*.py"),*ROOT.glob("configs/epkd/cifar100/*.json"),ROOT/"requirements-tgsc.txt",ROOT/"LICENSE"]
        for path in sorted(set(files)):
            if path.is_file():
                archive.write(path,path.relative_to(ROOT))
    print(bundle)


if __name__=="__main__":
    main()
