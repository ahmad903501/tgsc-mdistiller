"""Fast preflight samples pixel pages; training retains exhaustive validation."""
from pathlib import Path
import unittest

import torch

from mdistiller.tgsc.cache import CropBank, atomic_save
from test_tgsc_upload import scratch


class FastPreflightContracts(unittest.TestCase):
    def fixture(self, root):
        n, size = 64, 16
        bank = CropBank({"count": n, "schema": 2}, root, root)
        state = {"crops": torch.rand(n, 3, size, size),
                 "labels": torch.arange(n) % 100, "scores": torch.full((n,), .5),
                 "source_indices": torch.arange(n), "quality_flags": torch.ones(n),
                 "components": torch.full((n, 3), .5),
                 "boxes": torch.tensor([[0,0,size,size]]).repeat(n,1)}
        return bank, state

    def save(self, bank, state):
        atomic_save({"manifest": bank.identity, "resolution": 16, "dataset": state}, bank.path(16, write=True))

    def test_mapped_sample_returns_all_original_records(self):
        with scratch() as root:
            bank, state = self.fixture(root)
            self.save(bank, state)
            quick = bank.load(16, validation_samples=32, mmap=True)
            self.assertEqual(len(quick), 64)
            torch.testing.assert_close(quick.crops, state["crops"])
            del quick

    def test_full_stage_check_rejects_bad_pixels_outside_sample(self):
        with scratch() as root:
            bank, state = self.fixture(root)
            sampled = set(torch.linspace(0,63,32,dtype=torch.long).tolist())
            unsampled = next(i for i in range(64) if i not in sampled)
            state["crops"][unsampled,0,0,0] = float("nan")
            self.save(bank, state)
            quick = bank.load(16, validation_samples=32, mmap=True)
            self.assertEqual(len(quick), 64)
            del quick
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                bank.load(16)

    def test_metadata_outside_pixel_sample_is_still_checked(self):
        with scratch() as root:
            bank, state = self.fixture(root)
            state["components"][1,0] = .1
            self.save(bank, state)
            with self.assertRaisesRegex(ValueError, "Stored q"):
                bank.load(16, validation_samples=32, mmap=True)

    def test_missing_file_and_identity_are_still_checked(self):
        with scratch() as root:
            bank, state = self.fixture(root)
            with self.assertRaises(FileNotFoundError):
                bank.load(16, validation_samples=32, mmap=True)
            self.save(bank, state)
            wrong = dict(bank.identity, count=65)
            atomic_save({"manifest": wrong, "resolution": 16, "dataset": state}, bank.path(16, write=True))
            with self.assertRaisesRegex(ValueError, "identity/schema"):
                bank.load(16, validation_samples=32, mmap=True)
