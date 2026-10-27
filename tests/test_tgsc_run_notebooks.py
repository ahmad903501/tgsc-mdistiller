"""Dataset notebooks must select their own pair and reject incomplete attachments."""
import ast
import json
from pathlib import Path
import unittest

from test_tgsc_upload import scratch
from tools.generate_tgsc_run_notebooks import DATASETS, RESOLVE_CODE


class RunNotebookContracts(unittest.TestCase):
    def resolver(self):
        tree = ast.parse(RESOLVE_CODE)
        definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        namespace = dict(Path=Path, json=json, TEACHER_NAME="resnet56", DATASET_URL="fixture")
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "resolver", "exec"), namespace)
        return namespace["resolve_inputs"]

    def fixture(self, root, suffix=""):
        checkpoint = root / suffix / "teacher_weights/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture")
        manifest = root / suffix / "tgsc_cache/bank/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"schema": 2, "teacher": "resnet56"}))
        for size in (16,20,24,28,32):
            (manifest.parent/f"res{size}.pt").write_bytes(b"fixture")
        return checkpoint.resolve(), manifest.resolve()

    def test_nested_attachment_and_duplicate_root_are_resolved(self):
        with scratch() as root:
            expected = self.fixture(root, "extracted_zip")
            self.assertEqual(self.resolver()([root, root]), expected)

    def test_incomplete_and_ambiguous_inputs_are_rejected(self):
        with scratch() as root:
            _, manifest = self.fixture(root, "one")
            (manifest.parent/"res32.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                self.resolver()([root])
            (manifest.parent/"res32.pt").write_bytes(b"fixture")
            self.fixture(root, "two")
            with self.assertRaisesRegex(RuntimeError, "found 4"):
                self.resolver()([root])

    def test_run_notebook_settings_and_blocked_run(self):
        # Notebooks are delivery artifacts; they are not copied into the isolated runtime stage.
        folder = Path(__file__).resolve().parents[1]/"notebooks/runs"
        if not folder.is_dir():
            self.skipTest("Per-run notebooks are checked in the delivery source tree")
        runs = json.loads((folder/"run_manifest.json").read_text())
        self.assertEqual(len(runs), 12)
        outputs = set()
        expected = {t:(s,o,slug) for t,s,o,slug in DATASETS}
        for run in runs:
            nb = json.loads((folder/run["notebook"]).read_text())
            if not run["runnable"]:
                first = next(c for c in nb["cells"] if c["cell_type"] == "code")
                with self.assertRaisesRegex(RuntimeError, "pending"):
                    exec("".join(first["source"]))
                continue
            settings = next(c for c in nb["cells"] if "CONFIG_NAME =" in "".join(c["source"]))
            env = {}
            exec("".join(settings["source"]), env)
            student,owner,slug = expected[run["teacher"]]
            self.assertEqual(run["student"], student)
            self.assertEqual(env["CONFIG_NAME"], run["config"])
            self.assertEqual(env["DATASET_URL"], f"https://www.kaggle.com/datasets/{owner}/{slug}")
            self.assertEqual(env["CIFAR_SOURCE"], "download")
            self.assertEqual(env["TEACHER_SOURCE"], "kaggle")
            self.assertEqual(env["EVAL_EVERY"], 5)
            self.assertIsNone(env["TEST_DATA"])
            self.assertFalse(env["PRIVATE_REPO"])
            self.assertEqual(env["REVISION"], run["runtime_revision"])
            self.assertNotIn(str(env["OUTPUT_DIR"]), outputs)
            outputs.add(str(env["OUTPUT_DIR"]))
        self.assertEqual(len(outputs), 11)
