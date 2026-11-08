"""VRM code fidelity, original32px pixels, paired training and both requested pass budgets."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import zipfile
import subprocess
import uuid

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from test_tgsc import test_directory
from mdistiller.distillers._base import Distiller
from mdistiller.distillers.VRM import VRM, isv_loss, icv_loss
from mdistiller.epkd import vrm_augment
from mdistiller.epkd.pipeline import ImageViews, stage_coordinates
from mdistiller.epkd.runtime import settings, preflight, method_step, train, vrm_final_views
from mdistiller.tgsc.config import ExperimentConfig
from mdistiller.tgsc.experiment import load_run
from mdistiller.tgsc.runtime import build_model, build_distiller, seed_everything, load_teacher, load_export

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT.parent/'VRM'
torch.set_num_threads(2)


def reference_loss():
    source=(REFERENCE/'mdistiller/distillers/VRM.py').read_text()
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('isv_loss','icv_loss','VRM')]
    scope={'torch':torch,'nn':nn,'F':F,'np':np,'Distiller':Distiller}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official VRM','exec'),scope)
    return scope


def reference_transforms():
    spec=importlib.util.spec_from_file_location('vrm_original_cifar_transforms',REFERENCE/'mdistiller/dataset/cifar100.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VRMCAMChecks(unittest.TestCase):
    def test_exact_official_loss_values_and_gradients(self):
        reference=reference_loss()
        seed_everything(27)
        # Match the real CIFAR batch/class dimensions, not merely a tiny mock relation graph.
        sw=torch.randn(64,100,requires_grad=True)
        ss=torch.randn(64,100,requires_grad=True)
        tw=torch.randn(64,100)
        ts=torch.randn(64,100)
        y=torch.arange(64)%100
        for name,fn,weight in [('isv_loss',isv_loss,64.),('icv_loss',icv_loss,32.)]:
            expected=reference[name](sw,ss,tw,ts)*weight
            actual=fn(sw,ss,tw,ts)*weight
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            ga=torch.autograd.grad(actual,(sw,ss),retain_graph=True)
            ge=torch.autograd.grad(expected,(sw,ss),retain_graph=True)
            for a,e in zip(ga,ge):
                torch.testing.assert_close(a,e,rtol=0,atol=0)
        cfg=SimpleNamespace(KD=SimpleNamespace(TEMPERATURE=4,LOSS=SimpleNamespace(CE_WEIGHT=.1,KD_WEIGHT=.9)))
        teacher=build_model('resnet8').eval().requires_grad_(False)
        student=build_model('resnet8')
        ref=reference['VRM'](copy.deepcopy(student),copy.deepcopy(teacher),cfg).train()
        local=VRM(student,teacher,cfg).train()
        x=torch.randn(4,3,32,32)
        v=torch.randn(4,3,32,32)
        labels=torch.arange(4)
        a,losses=local(image_weak=x,image_strong=v,target=labels)
        e,expected=ref(image_weak=x,image_strong=v,target=labels)
        torch.testing.assert_close(a,e,rtol=0,atol=0)
        self.assertEqual(set(losses),{'loss_ce','loss_isv','loss_icv'})
        for key in losses:
            torch.testing.assert_close(losses[key],expected[key],rtol=0,atol=0)
        sum(losses.values()).backward();sum(expected.values()).backward()
        for a,e in zip(local.student.parameters(),ref.student.parameters()):
            torch.testing.assert_close(a.grad,e.grad,rtol=0,atol=0)
        for a,e in zip(local.student.buffers(),ref.student.buffers()):
            torch.testing.assert_close(a,e,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_original_32px_augmentation_pixel_for_pixel(self):
        reference=reference_transforms()
        self.assertEqual(len(vrm_augment.augment_list()),14)
        self.assertEqual(vrm_augment.get_cifar100_train_transform_strong().transforms[0].transforms[0].padding_mode,'constant')
        raw=np.random.default_rng(27).integers(0,256,(32,32,3),dtype=np.uint8)
        for seed in range(30):
            seed_everything(seed)
            expected=reference.get_cifar100_train_transform_strong()(Image.fromarray(raw))
            seed_everything(seed)
            actual=vrm_augment.get_cifar100_train_transform_strong()(Image.fromarray(raw))
            for a,e in zip(actual,expected):
                torch.testing.assert_close(a,e,rtol=0,atol=0)
        exp=ExperimentConfig()
        batch=torch.tensor(np.stack([raw,raw])).permute(0,3,1,2)
        seed_everything(91)
        transform=reference.get_cifar100_train_transform_strong()
        pairs=[transform(Image.fromarray(r)) for r in batch.permute(0,2,3,1).numpy()]
        expected=tuple(torch.stack([p[i] for p in pairs]) for i in (0,1))
        seed_everything(91)
        actual=vrm_final_views(exp,'cpu')(batch)
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a,e,rtol=0,atol=0)

    def test_real_config_preflight_and_three_distinct_28px_windows(self):
        for count in (1,3):
            path=ROOT/f'configs/epkd/cifar100/vrm_cam_resnet32x4_to_resnet8x4_28p{count}.json'
            exp,method,options,raw_cfg=load_run(path)
            teacher=build_model(exp.teacher_model)
            load_teacher(teacher,ROOT/'download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth')
            model=build_distiller(exp,method,options,teacher)
            raw=torch.randint(0,256,(2,3,32,32),dtype=torch.uint8)
            y=torch.tensor([0,1])
            cfg=settings(exp,raw_cfg)
            checks=preflight(model,raw,y,exp,cfg,'cpu')
            self.assertEqual(len(checks),5)
            self.assertEqual(set(checks[-1]['losses']),{'pair.loss_ce','pair.loss_isv','pair.loss_icv'})
            coords=stage_coordinates(teacher,raw,y,ImageViews(exp.data.mean,exp.data.std,'cpu'),28,2,2)
            self.assertTrue(all(len(set(map(tuple,c.tolist())))==3 for c in coords))

    def test_training_resume_and_export_both_options(self):
        for passes28 in (1,3):
            with test_directory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root=Path(directory)
                seed_everything(27)
                exp=ExperimentConfig(teacher_model='resnet8',student_model='resnet8')
                exp.data.stages=[(r,1) for r in (16,20,24,28,32)]
                exp.data.batch_size=4
                exp.data.num_workers=0
                exp.train.epochs=5
                exp.optim.milestones=[2,4]
                cfg=settings(exp,{'method':'VRM','epkd':{'passes':[3,3,3,passes28,1],'strides':[8,8,8,2,8],'proposal_batch_size':4}})
                teacher=build_model('resnet8').eval().requires_grad_(False)
                model=build_distiller(exp,'VRM',{},teacher)
                raw=torch.randint(0,256,(8,3,32,32),dtype=torch.uint8)
                y=torch.arange(8)
                test=TensorDataset(raw.float()/255.,y)
                checkpoint=root/'teacher.pt'
                torch.save({'model':teacher.state_dict()},checkpoint)
                initial=copy.deepcopy(model.state_dict())
                seed_everything(91)
                history=train(exp,'VRM',{},cfg,model,raw,y,test,'cpu',root/'full',checkpoint)
                expected=copy.deepcopy(model.state_dict())
                self.assertEqual(history[3]['optimizer_steps'],2*passes28)
                self.assertEqual(history[3]['source_draws'],8*passes28)
                self.assertEqual(history[-1]['second_view'],'vrm_original_strong32')
                self.assertIn('pair.loss_isv',history[-1]['train'])
                model.load_state_dict(initial)
                seed_everything(91)
                train(exp,'VRM',{},cfg,model,raw,y,test,'cpu',root/'part',checkpoint,stop_after_epoch=4)
                train(exp,'VRM',{},cfg,model,raw,y,test,'cpu',root/'resumed',checkpoint,resume=root/'part/latest.pt')
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
                exported,_=load_export(root/'resumed/inference_latest.pt')
                model.eval()
                probe=torch.randn(2,3,32,32)
                with torch.no_grad():
                    torch.testing.assert_close(model(image=probe),exported(probe)[0],rtol=0,atol=0)
                self.assertTrue((root/'resumed/training_curves.png').exists())

    def test_notebooks_and_new_runtime_bundle(self):
        configs=[]
        for passes in (1,3):
            stem=f'vrm_cam_resnet32x4_to_resnet8x4_28p{passes}'
            config=json.loads((ROOT/f'configs/epkd/cifar100/{stem}.json').read_text())
            configs.append(config)
            self.assertEqual(config['epkd']['passes'],[3,3,3,passes,1])
            self.assertEqual(config['epkd']['strides'],[8,8,8,2,8])
            nb=json.loads((ROOT/f'notebooks/runs/{stem}.ipynb').read_text())
            for cell in nb['cells']:
                if cell['cell_type']=='code':
                    compile(cell['source'],stem,'exec')
            self.assertIn('"git","clone"',nb['cells'][2]['source'])
            self.assertIn('REPO_URL = "https://github.com/ahmad903501/tgsc-mdistiller.git"',nb['cells'][1]['source'])
            self.assertIn('SEEDS = [0,1,2]',nb['cells'][1]['source'])
        a,b=copy.deepcopy(configs[0]),copy.deepcopy(configs[1])
        for cfg in (a,b):
            cfg['epkd']['passes'][3]=0
            cfg['train'].pop('run_name');cfg['train'].pop('save_path')
        self.assertEqual(a,b)
        with zipfile.ZipFile(ROOT.parent/'kaggle_notebooks/VRM_CAM_RUNTIME.zip') as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn('mdistiller/distillers/VRM.py',archive.namelist())
            self.assertIn('mdistiller/epkd/vrm_augment.py',archive.namelist())
            self.assertFalse(any(n.endswith(('.pt','.pth','.npz')) for n in archive.namelist()))
            # Exercise the exact Git bootstrap offline, including rerun and local-edit protection.
            nb=json.loads((ROOT/'notebooks/runs/vrm_cam_resnet32x4_to_resnet8x4_28p1.ipynb').read_text())
            bootstrap=nb['cells'][2]['source'].split('subprocess.run([sys.executable')[0]
            # Retain this tiny diagnostic Git fixture: Windows marks Git objects read-only,
            # while the generic test_directory cleanup assumes ordinary writable files.
            with contextlib.nullcontext(ROOT/'.test-artifacts'/('git_bootstrap_'+uuid.uuid4().hex)) as root:
                root.mkdir(parents=True)
                source=root/'source'
                source.mkdir()
                required=['tools/run_epkd.py','mdistiller/epkd/runtime.py','mdistiller/distillers/VRM.py','mdistiller/epkd/vrm_augment.py']
                for name in required:
                    file=source/name
                    file.parent.mkdir(parents=True,exist_ok=True)
                    file.write_text('# fixture\n')
                subprocess.run(['git','init','-q',str(source)],check=True,capture_output=True)
                subprocess.run(['git','-C',str(source),'add','.'],check=True,capture_output=True)
                subprocess.run(['git','-C',str(source),'-c','user.name=Notebook test','-c','user.email=notebook-test@example.invalid','commit','--no-gpg-sign','-qm','Fixture'],check=True,capture_output=True)
                revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
                scope=dict(Path=Path,REPO=root/'working',REPO_URL=source.as_posix(),REVISION=revision,REQUIRED_SOURCE_FILES=required)
                exec(bootstrap,scope)
                self.assertTrue(scope['complete'](scope['REPO']))
                exec(bootstrap,scope)
                tracked=scope['REPO']/'tools/run_epkd.py'
                tracked.write_text('# retain this local edit\n')
                with self.assertRaisesRegex(RuntimeError,'Tracked source changes'):
                    exec(bootstrap,scope)
                self.assertEqual(tracked.read_text(),'# retain this local edit\n')


if __name__=='__main__':
    unittest.main()
