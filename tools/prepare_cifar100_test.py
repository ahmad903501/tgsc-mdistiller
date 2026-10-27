"""Create a pickle-free CIFAR-100 test-only artifact from verified official bytes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

TEST_MD5 = "f0ef6b0ae62326f3e7ffdfab6717acfc"


def build_test_artifact(archive_path, output):
    with zipfile.ZipFile(archive_path) as archive:
        matches = [n for n in archive.namelist() if n.endswith("cifar-100-python/test")]
        if len(matches) != 1:
            raise ValueError("Expected one original CIFAR-100 test file")
        payload = archive.read(matches[0])
    # Only deserialize the exact official file, never an arbitrary uploaded pickle.
    if hashlib.md5(payload).hexdigest() != TEST_MD5:
        raise ValueError("Test file does not match torchvision's official CIFAR-100 MD5")
    data = pickle.load(io.BytesIO(payload), encoding="latin1")
    images = np.ascontiguousarray(data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
    labels = np.asarray(data["fine_labels"], dtype=np.int64)
    if images.shape != (10000, 32, 32, 3) or images.dtype != np.uint8 or labels.shape != (10000,):
        raise ValueError("Unexpected CIFAR-100 test arrays")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, images=images, labels=labels)
    report = {"source": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
              "source_test_md5": TEST_MD5, "test_samples": len(labels),
              "format": "npz; uint8 NHWC images and int64 fine labels; allow_pickle=False",
              "bytes": output.stat().st_size,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Saved preparation ZIP containing original test file")
    parser.add_argument("--output", required=True, help="Output .npz path")
    args = parser.parse_args()
    print(json.dumps(build_test_artifact(args.archive, args.output), indent=2))
