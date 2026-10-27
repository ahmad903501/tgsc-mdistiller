"""Verified CIFAR-100 evaluation data without downloading the training split."""
import hashlib
import os
from pathlib import Path
import urllib.request

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

TEST_URL = "https://github.com/ahmad903501/tgsc-mdistiller/releases/download/tgsc-data-v1/cifar100-test.npz"
TEST_SHA256 = "cc2cf570b7933bd8005279fed5e60f256d02ea03b93d0f157cdbc63595d87e2a"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_test(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    request = urllib.request.Request(TEST_URL, headers={"User-Agent": "TGSC-test-split"})
    print("Downloading only the CIFAR-100 test split (28 MB); no training images...", flush=True)
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as output:
        total = 0
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
            total += len(chunk)
            if total % (4 * 1024**2) == 0:
                print(f"Test split: {total / 1024**2:.0f} MiB downloaded", flush=True)
    if sha256(temporary) != TEST_SHA256:
        raise ValueError("Test download SHA-256 mismatch; partial file was not installed")
    os.replace(temporary, path)


class Cifar100Test(Dataset):
    def __init__(self, path, mean, std):
        path = Path(path)
        if sha256(path) != TEST_SHA256:
            raise ValueError(f"CIFAR-100 test artifact hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            self.data = payload["images"].copy()
            labels = payload["labels"].copy()
        if self.data.shape != (10000, 32, 32, 3) or self.data.dtype != np.uint8:
            raise ValueError("Invalid CIFAR-100 test image array")
        if labels.shape != (10000,) or labels.dtype != np.int64 or (labels < 0).any() or (labels >= 100).any():
            raise ValueError("Invalid CIFAR-100 test labels")
        self.targets = labels.tolist()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.targets[index]


def test_dataset(exp, download=False, limit=None, path=None):
    path = Path(path) if path else Path(exp.data.root) / "cifar100-test.npz"
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing test-only data: {path}; use --download or --test-data")
        download_test(path)
    dataset = Cifar100Test(path, exp.data.mean, exp.data.std)
    if limit is not None:
        if limit < 2:
            raise ValueError("Evaluation subset limit must be at least 2")
        dataset.data, dataset.targets = dataset.data[:limit], dataset.targets[:limit]
    return dataset
