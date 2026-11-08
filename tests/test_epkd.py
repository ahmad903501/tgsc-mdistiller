"""Online CAM integration: official losses, BN behavior, resume, exports and delivery."""
import contextlib
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess
import sys
import zipfile


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from test_tgsc import test_directory
from mdistiller.tgsc.config import ExperimentConfig
from mdistiller.tgsc.experiment import load_run
from mdistiller.tgsc.runtime import build_model, build_distiller, load_teacher, seed_everything, load_export
from mdistiller.epkd.pipeline import teacher_cam, top_windows, stage_coordinates, ImageViews, crop_batchnorm
from mdistiller.epkd.runtime import preflight, method_step, train, settings

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class EPKDChecks(unittest.TestCase):
    def setup_run(self, method="DKD"):
        exp = ExperimentConfig(teacher_model="resnet8", student_model="resnet8")
        exp.data.stages = [(16,1), (20,1), (24,1), (28,1), (32,1)]
        exp.data.batch_size = 4
        exp.data.num_workers = 0
        exp.train.epochs = 5
        exp.optim.milestones = [2,4]
        cfg = settings(exp, {"epkd": {"passes": [1]*5, "strides": [8]*5, "proposal_batch_size": 2}})
        options = {"review_weight": 5} if method == "REVIEWKD" else {}
        seed_everything(27)
        teacher = build_model("resnet8").eval().requires_grad_(False)
        model = build_distiller(exp, method, options, teacher)
        raw = torch.randint(0, 256, (8,3,32,32), dtype=torch.uint8)
        labels = torch.arange(8).long()
        test = TensorDataset(raw.float()/255., labels)
        return exp, cfg, options, model, raw, labels, test

    def test_cam_equation_and_original_window_ranks(self):
        model = build_model("wrn_40_2").eval().requires_grad_(False)
        image = torch.randn(2,3,32,32)
        y = torch.tensor([0,1])
        logits, cam = teacher_cam(model, image, y)
        _, feats = model(image)
        feature = feats["final_feat"]
        head = model.fc
        wanted = F.relu(torch.einsum("bchw,bc->bhw", feature, head.weight[y])).unsqueeze(1)
        wanted = F.interpolate(wanted, (32,32), mode="bilinear", align_corners=False)
        wanted /= wanted.flatten(1).amax(1).clamp_min(1e-8).view(-1,1,1,1)
        torch.testing.assert_close(cam, wanted)
        scores = torch.arange(1024).float().reshape(1,1,32,32)
        coords = top_windows(scores, 16, 8)
        self.assertEqual(coords.tolist(), [[[16,16],[16,8],[16,0]]])
        self.assertEqual(top_windows(scores, 28, 8).tolist(), [[[0,0],[0,0],[0,0]]])
        self.assertEqual(top_windows(torch.zeros_like(scores), 16).shape, (1,3,2))

    def test_official_losses_are_reused_and_bn_scopes_cover_adapters(self):
        for method in ("DKD", "SIMKD", "REVIEWKD"):
            exp,cfg,options,model,raw,y,test = self.setup_run(method)
            views = ImageViews(exp.data.mean, exp.data.std, "cpu")
            x = views.normalize(raw[:4])
            crop = x[:,:,:16,:16]
            model.train()
            state = copy.deepcopy(model.state_dict())
            _, full_losses = model(image=x, target=y[:4], epoch=1)
            with crop_batchnorm(model):
                _, crop_losses = model(image=crop, target=y[:4], epoch=1)
            expected = sum(full_losses.values())+sum(crop_losses.values())
            model.load_state_dict(state)
            _, actual_losses, actual = method_step(model,x,crop,y[:4],1,cfg)
            torch.testing.assert_close(actual,expected)
            if method == "SIMKD":
                self.assertEqual(set(actual_losses), {"full.loss_kd", "crop.loss_kd"})
            buffers = {n:v.clone() for n,v in model.named_buffers() if "running_" in n}
            with crop_batchnorm(model):
                model(image=crop,target=y[:4],epoch=1)
            for name,value in model.named_buffers():
                if name in buffers:
                    torch.testing.assert_close(value,buffers[name],rtol=0,atol=0)

    def test_preflight_is_nonmutating_for_all_methods(self):
        for method in ("DKD", "SIMKD", "REVIEWKD"):
            exp,cfg,options,model,raw,y,test = self.setup_run(method)
            before = copy.deepcopy(model.state_dict())
            report = preflight(model,raw,y,exp,cfg,"cpu")
            self.assertEqual([r["resolution"] for r in report], [16,20,24,28,32])
            for name,value in model.state_dict().items():
                torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_resume_and_teacher_free_exports_all_methods(self):
        for method in ("DKD", "SIMKD", "REVIEWKD"):
            with test_directory() as root, contextlib.redirect_stdout(io.StringIO()):
                root = Path(root)
                exp,cfg,options,model,raw,y,test = self.setup_run(method)
                teacher_path = root/"teacher.pt"
                torch.save({"model":model.teacher.state_dict()},teacher_path)
                initial = copy.deepcopy(model.state_dict())
                seed_everything(103)
                train(exp,method,options,cfg,model,raw,y,test,"cpu",root/"full",teacher_path,eval_every=5)
                full = copy.deepcopy(model.state_dict())
                model.load_state_dict(initial)
                seed_everything(103)
                train(exp,method,options,cfg,model,raw,y,test,"cpu",root/"part",teacher_path,stop_after_epoch=2,eval_every=5)
                train(exp,method,options,cfg,model,raw,y,test,"cpu",root/"resumed",teacher_path,resume=root/"part/latest.pt",eval_every=5)
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,full[key],rtol=0,atol=0)
                payload = torch.load(root/"resumed/latest.pt",weights_only=True)
                self.assertEqual([r["resolution"] for r in payload["metrics_history"]],[16,20,24,28,32])
                self.assertEqual(payload["metrics_history"][-1]["source_draws"],8)
                self.assertTrue((root/"resumed/training_curves.png").exists())
                exported,_ = load_export(root/"resumed/inference_latest.pt")
                model.eval()
                probe = torch.randn(2,3,32,32)
                with torch.no_grad():
                    predicted = exported(probe)
                    if isinstance(predicted, tuple):
                        predicted = predicted[0]
                    torch.testing.assert_close(model(image=probe),predicted)
                with self.assertRaisesRegex(ValueError,"identity differs"):
                    train(exp,method,options,cfg,model,raw.flip(0),y,test,"cpu",root/"wrong",teacher_path,
                          resume=root/"part/latest.pt",eval_every=5)
                # A finished checkpoint must be safe to reopen, not reference an undefined last epoch.
                train(exp,method,options,cfg,model,raw,y,test,"cpu",root/"finished",teacher_path,
                      resume=root/"resumed/latest.pt",eval_every=5)

    def test_all_configured_pairs_with_existing_official_teacher_weights(self):
        # Local optional audit; standard tests still exercise every shape without network access.
        configs = sorted((ROOT/"configs/epkd/cifar100").glob("*.json"))
        self.assertGreaterEqual(len(configs),11)
        raw = torch.randint(0,256,(2,3,32,32),dtype=torch.uint8)
        y = torch.tensor([0,1])
        for path in configs:
            exp,method,options,source = load_run(path)
            teacher = build_model(exp.teacher_model)
            checkpoint = ROOT/"download_ckpts/cifar_teachers"/(exp.teacher_model+"_vanilla")/"ckpt_epoch_240.pth"
            if checkpoint.exists():
                load_teacher(teacher,checkpoint)
            else:
                teacher.eval().requires_grad_(False)
            model = build_distiller(exp,method,options,teacher)
            preflight(model,raw,y,exp,settings(exp,source),"cpu")

    def test_cli_and_portable_notebook_delivery(self):
        from tools.run_epkd import main
        with test_directory() as root, contextlib.redirect_stdout(io.StringIO()):
            root = Path(root)
            exp,cfg,options,model,raw,y,test = self.setup_run()
            checkpoint = root/"teacher.pt"
            torch.save({"model": model.teacher.state_dict()},checkpoint)
            config = root/"config.json"
            config.write_text(json.dumps({**asdict(exp), "method": "DKD", "epkd": cfg}))
            class FakeOriginals:
                data = raw.permute(0,2,3,1).numpy()
                targets = y.tolist()
            with patch("mdistiller.tgsc.cache.cifar_dataset", return_value=FakeOriginals()), \
                 patch("mdistiller.tgsc.cache.prepare_bank", side_effect=AssertionError("Old bank used")), \
                 patch("mdistiller.tgsc.evaluation_data.test_dataset", return_value=test):
                main(["check", "--cpu-check", "--config", str(config), "--teacher-checkpoint", str(checkpoint),
                      "--test-data", "fixture.npz"])
                main(["train", "--cpu-check", "--config", str(config), "--teacher-checkpoint", str(checkpoint),
                      "--output", str(root/"cli"), "--stop-after-epoch", "1", "--test-data", "fixture.npz"])
            self.assertTrue((root/"cli/latest.pt").is_file())
        manifest = json.loads((ROOT/"notebooks/runs/epkd_run_manifest.json").read_text())
        self.assertEqual(len(manifest),11)
        for path in (ROOT/"notebooks/runs").glob("epkd_*.ipynb"):
            nb = json.loads(path.read_text())
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile(cell["source"],str(path),"exec")
        bundle = ROOT.parent/"kaggle_notebooks/EPKD_RUNTIME.zip"
        with zipfile.ZipFile(bundle) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("mdistiller/epkd/runtime.py",archive.namelist())
            self.assertIn("tools/run_epkd.py",archive.namelist())
            for name in archive.namelist():
                self.assertFalse(name.endswith((".pth", ".pt", ".npz")))
            with test_directory() as root:
                archive.extractall(root)
                code = f"import sys,site; sys.path.append(site.getusersitepackages()); sys.path.insert(0,{str(root)!r}); from mdistiller.epkd.runtime import train; from mdistiller.tgsc.runtime import build_distiller"
                checked = subprocess.run([sys.executable,"-I","-c",code],cwd=root,capture_output=True,text=True)
                self.assertEqual(checked.returncode,0,checked.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()
