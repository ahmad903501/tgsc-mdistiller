"""Packaging must retain every training input and leave source archives unchanged."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid
import zipfile

from tools.package_tgsc_upload import package, select_files


@contextmanager
def scratch():
    parent = (Path.cwd() / ".test-artifacts").resolve()
    root = parent / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.resolve().is_relative_to(parent):
            shutil.rmtree(root)


def fixture(path, missing=False, wrong_teacher=False):
    checkpoint = b"teacher fixture"
    identity = {"schema": 2, "split": "train", "teacher": "resnet56",
                "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("teacher_weights/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth",
                         b"wrong teacher" if wrong_teacher else checkpoint)
        archive.writestr(f"tgsc_cache/{key}/manifest.json", json.dumps(identity))
        for resolution in (16, 20, 24, 28, 32):
            if not (missing and resolution == 32):
                archive.writestr(f"tgsc_cache/{key}/res{resolution}.pt", bytes(range(256)) * 8)
        for name in ("data/cifar-100-python/train", "teacher_weights/cifar_teachers.tar",
                     f"tgsc_cache/{key}/reference.pt", "mdistiller/.git/config"):
            archive.writestr(name, b"not needed in upload")


class UploadContracts(unittest.TestCase):
    def test_selected_payloads_and_original_unchanged(self):
        with scratch() as root:
            source = root / "original.zip"
            fixture(source)
            before = source.read_bytes()
            # Cover both portable recompression and compressed-copy mode when installed.
            modes = [None] + ([shutil.which("zip")] if shutil.which("zip") else [])
            for index, tool in enumerate(modes):
                output = root / str(index)
                output.mkdir()
                report = package(source, output, tool)
                with zipfile.ZipFile(source) as original, zipfile.ZipFile(output / report["output"]) as result:
                    self.assertEqual(len(result.namelist()), 7)
                    self.assertEqual(result.testzip(), None)
                    for name in result.namelist():
                        self.assertEqual(original.read(name), result.read(name))
                self.assertEqual(source.read_bytes(), before)
                with self.assertRaises(FileExistsError):
                    package(source, output, tool)

    def test_missing_resolution_rejected(self):
        with scratch() as root:
            source = root / "missing.zip"
            fixture(source, missing=True)
            with zipfile.ZipFile(source) as archive, self.assertRaisesRegex(ValueError, "Missing"):
                select_files(archive)

    def test_wrong_checkpoint_rejected(self):
        with scratch() as root:
            source = root / "wrong.zip"
            fixture(source, wrong_teacher=True)
            with zipfile.ZipFile(source) as archive, self.assertRaisesRegex(ValueError, "Teacher bytes"):
                select_files(archive)
