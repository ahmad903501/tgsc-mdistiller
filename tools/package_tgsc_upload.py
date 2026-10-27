"""Build minimal Kaggle teacher/crop-bank ZIPs from saved preparation outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile


def select_files(archive):
    manifests = [n for n in archive.namelist()
                 if n.startswith("tgsc_cache/") and n.endswith("/manifest.json")]
    if len(manifests) != 1:
        raise ValueError("Expected one teacher bank per input ZIP")
    name = manifests[0]
    identity = json.loads(archive.read(name))
    if identity.get("schema") != 2 or identity.get("split") != "train":
        raise ValueError("Expected a schema-2 training crop bank")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    if name != f"tgsc_cache/{key}/manifest.json":
        raise ValueError("Bank directory does not match its manifest hash")
    teacher = identity["teacher"]
    checkpoint = f"teacher_weights/cifar_teachers/{teacher}_vanilla/ckpt_epoch_240.pth"
    selected = [checkpoint, name] + [f"tgsc_cache/{key}/res{r}.pt" for r in (16, 20, 24, 28, 32)]
    for item in selected:
        if archive.namelist().count(item) != 1:
            raise ValueError(f"Missing or duplicate required member: {item}")
    with archive.open(checkpoint) as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != identity["checkpoint_sha256"]:
            raise ValueError("Teacher bytes do not match the bank's checkpoint hash")
    return identity, selected


def compressed_hash(path, info):
    """Compare compressed payloads without inflating multi-GB crop tensors."""
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        name_size, extra_size = struct.unpack_from("<HH", header, 26)
        stream.seek(name_size + extra_size, 1)
        digest = hashlib.sha256()
        remaining = info.compress_size
        while remaining:
            block = stream.read(min(8 * 1024**2, remaining))
            if not block:
                raise ValueError("Truncated ZIP payload")
            digest.update(block)
            remaining -= len(block)
        return digest.hexdigest()


def package(source, output_dir, zip_tool):
    with zipfile.ZipFile(source) as original:
        identity, selected = select_files(original)
        destination = output_dir / f"tgsc_{identity['teacher']}.zip"
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}; choose an empty output directory")
        partial = destination.with_suffix(".partial.zip")
        if partial.exists():
            raise FileExistsError(partial)
        if zip_tool:
            # Info-ZIP copy mode preserves compressed bytes; no extraction/recompression.
            result = subprocess.run([zip_tool, "-q", "-U", str(source), "--out", str(partial),
                                     "-i", *selected], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
        else:
            with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as target:
                for name in selected:
                    with original.open(name) as src, target.open(name, "w", force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024**2)
        with zipfile.ZipFile(partial) as target:
            if set(target.namelist()) != set(selected):
                raise ValueError("Unexpected files in output ZIP")
            for name in selected:
                before, after = original.getinfo(name), target.getinfo(name)
                if (before.CRC, before.file_size) != (after.CRC, after.file_size):
                    raise ValueError(f"ZIP member differs: {name}")
                if zip_tool:
                    if compressed_hash(source, before) != compressed_hash(partial, after):
                        raise ValueError(f"Compressed member bytes differ: {name}")
                else:
                    # Reading checks the generated CRC, including in the portable fallback.
                    with target.open(name) as stream:
                        while stream.read(8 * 1024**2):
                            pass
        partial.rename(destination)
        return {"teacher": identity["teacher"], "input": source.name, "output": destination.name,
                "input_bytes": source.stat().st_size, "output_bytes": destination.stat().st_size,
                "members": selected, "bank_identity": identity,
                "verification": "teacher SHA-256, bank identity, exact members, CRC/size and " +
                                ("identical compressed payload SHA-256" if zip_tool else "output CRC read"),
                "copy_mode": bool(zip_tool)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source_dir, output_dir = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("Use a separate output directory; originals are preserved")
    sources = sorted(source_dir.glob("*.zip"))
    if not sources:
        raise FileNotFoundError("No input ZIPs found")
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = shutil.which("zip")
    reports = []
    for source in sources:
        print(f"Packaging {source.name} ({'compressed copy' if tool else 'recompression'})", flush=True)
        report = package(source, output_dir, tool)
        reports.append(report)
        print(f"Verified {report['output']}: {report['output_bytes'] / 2**30:.2f} GiB", flush=True)
    (output_dir / "upload_manifest.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print("Originals preserved. Upload the generated ZIPs; download CIFAR in each Kaggle runtime.")


if __name__ == "__main__":
    main()
