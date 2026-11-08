"""Fetch one official teacher, falling back to the mdistiller archive; no crop banks."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

HASHES = {
    "resnet56": "4143895c82105da0a73236230db11aee5be39b660fe21e1323037c774f994155",
    "resnet110": "75dc04faf468c078f1dfa010b51069fa3a8a4b91145379574e05c8e63e7441a9",
    "resnet32x4": "22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f",
    "vgg13": "ff25893af9e28fe12cfcc4fd0c699596754eb89bfafd521dfb1301cd281b47f8",
    "wrn_40_2": "41100a8e8478b326645eb9aab09c18fc8ed34c1ab4004d658cb652125a81ca18",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True, choices=sorted(HASHES))
    p.add_argument("--output-dir", default="download_ckpts")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from mdistiller.tgsc.runtime import load_teacher, build_model
    from mdistiller.tgsc.cache import file_hash
    path = Path(args.output_dir)/"cifar_teachers"/(args.teacher+"_vanilla")/"ckpt_epoch_240.pth"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        url = f"https://shape2prog.csail.mit.edu/repo/{args.teacher}_vanilla/ckpt_epoch_240.pth"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mdistiller-EPKD"})
            with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as stream:
                while block := response.read(1024*1024):
                    stream.write(block)
            if file_hash(part) != HASHES[args.teacher]:
                raise ValueError("RepDistiller checkpoint hash mismatch")
            part.replace(path)
        except Exception as error:
            part.unlink(missing_ok=True)
            print(f"Individual checkpoint unavailable ({type(error).__name__}); using official mdistiller archive", flush=True)
            subprocess.run([sys.executable, str(root/"tools/download_teachers.py"), "--teacher", args.teacher,
                            "--output-dir", args.output_dir], check=True)
    if file_hash(path) != HASHES[args.teacher]:
        raise ValueError("Existing checkpoint does not match the audited official teacher")
    load_teacher(build_model(args.teacher), path)
    print(path.resolve())


if __name__ == "__main__":
    main()
