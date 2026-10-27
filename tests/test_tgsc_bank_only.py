"""Cache-only training, test-only loading, evaluation cadence and resumable plots."""
from dataclasses import asdict
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_tgsc import MiniCifar
from test_tgsc_upload import scratch
from mdistiller.tgsc.cache import CropBank, manifest, prepare_bank
from mdistiller.tgsc.config import ExperimentConfig
from mdistiller.tgsc.runtime import build_model, build_distiller, seed_everything, train
from mdistiller.tgsc.evaluation_data import Cifar100Test, sha256
from tools.run_tgsc import main


torch.set_num_threads(2)


class BankOnlyContracts(unittest.TestCase):
    def setup_bank(self, root):
        exp = ExperimentConfig(teacher_model="resnet8", student_model="resnet8")
        exp.data.batch_size = 6
        exp.data.num_workers = 0
        exp.data.pin_memory = False
        exp.data.stages = ((16, 3), (32, 3))
        exp.train.epochs = 6
        exp.tgsc.precompute_batch_size = 3
        exp.tgsc.precompute_workers = 0
        dataset = MiniCifar()
        teacher = build_model("resnet8").eval().requires_grad_(False)
        checkpoint = root/"teacher.pt"
        torch.save({"model": teacher.state_dict()}, checkpoint)
        bank = CropBank(manifest(exp, checkpoint, dataset), root/"cache", root/"cache")
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_bank(exp, teacher, dataset, bank)
        return exp, dataset, teacher, checkpoint, bank

    def test_cli_check_never_loads_or_downloads_original_train_data(self):
        with scratch() as root:
            exp, dataset, teacher, checkpoint, bank = self.setup_bank(root)
            config = root/"run.json"
            config.write_text(json.dumps({**asdict(exp), "method": "DKD"}))
            with patch("mdistiller.tgsc.cache.cifar_dataset", side_effect=AssertionError("Original train data used")), \
                 patch("mdistiller.tgsc.evaluation_data.download_test", side_effect=AssertionError("Check downloaded evaluation data")), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(["check", "--cpu-check", "--config", str(config), "--teacher-checkpoint", str(checkpoint),
                      "--cache-root", str(bank.read_root), "--data-root", str(root/"no-cifar"), "--download"])
            self.assertFalse((root/"no-cifar").exists())
            restored = CropBank.from_prepared(exp, checkpoint, bank.read_root)
            labels = restored.establish_source_labels()
            self.assertEqual(labels.tolist(), dataset.targets)
            wrong_checkpoint = root/"different-teacher.pt"
            wrong_checkpoint.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "compatible"):
                CropBank.from_prepared(exp, wrong_checkpoint, bank.read_root)

    def test_crop_labels_must_match_full_image_source_labels(self):
        with scratch() as root:
            exp, dataset, teacher, checkpoint, bank = self.setup_bank(root)
            bank.establish_source_labels()
            payload = torch.load(bank.path(16), weights_only=True)
            payload["dataset"]["labels"][0] = (payload["dataset"]["labels"][0] + 1) % 100
            torch.save(payload, bank.path(16))
            with self.assertRaisesRegex(ValueError, "source labels"):
                bank.load(16, validation_samples=32, mmap=True)

    def test_test_only_npz_loader_and_hash_rejection(self):
        with scratch() as root:
            path = root/"test.npz"
            np.savez_compressed(path, images=np.zeros((10000,32,32,3), dtype=np.uint8),
                                labels=np.arange(10000, dtype=np.int64) % 100)
            with patch("mdistiller.tgsc.evaluation_data.TEST_SHA256", sha256(path)):
                dataset = Cifar100Test(path, (.5,.5,.5), (.5,.5,.5))
                self.assertEqual(len(dataset), 10000)
                image, label = dataset[17]
                torch.testing.assert_close(image, -torch.ones(3,32,32))
                self.assertEqual(label, 17)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Cifar100Test(path, (.5,.5,.5), (.5,.5,.5))

    def test_eval_every_five_and_final_resume_history_and_curves(self):
        with scratch() as root:
            exp, dataset, teacher, checkpoint, bank = self.setup_bank(root)
            bank.establish_source_labels()
            seed_everything(27)
            full = build_distiller(exp, "DKD", {}, teacher)
            with contextlib.redirect_stdout(io.StringIO()):
                train(exp, "DKD", {}, full, bank, dataset, torch.device("cpu"), root/"full", eval_every=5)
            rows = [json.loads(line) for line in (root/"full/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in rows if r["test"] is not None], [5,6])
            self.assertTrue(all(r["test"] is None for r in rows[:4]))
            seed_everything(27)
            part = build_distiller(exp, "DKD", {}, teacher)
            with contextlib.redirect_stdout(io.StringIO()):
                train(exp, "DKD", {}, part, bank, dataset, torch.device("cpu"), root/"part", stop_after_epoch=3)
                resumed = build_distiller(exp, "DKD", {}, teacher)
                train(exp, "DKD", {}, resumed, bank, dataset, torch.device("cpu"), root/"resumed", resume=root/"part/latest.pt")
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], msg=key)
            resumed_rows = [json.loads(line) for line in (root/"resumed/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in resumed_rows], [1,2,3,4,5,6])
            for name in ("metrics.csv", "training_curves.png", "training_curves.svg", "inference_best.pt"):
                self.assertGreater((root/"resumed"/name).stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "evaluation interval"):
                train(exp, "DKD", {}, resumed, bank, dataset, torch.device("cpu"), root/"wrong",
                      resume=root/"part/latest.pt", eval_every=1)
