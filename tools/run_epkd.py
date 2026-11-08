"""Run original EP-KD CAM curriculum around official DKD, SimKD or ReviewKD."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "train", "evaluate"])
    p.add_argument("--config", required=True)
    p.add_argument("--teacher-checkpoint")
    p.add_argument("--data-root")
    p.add_argument("--output", default="epkd_runs/default")
    p.add_argument("--gpu", default="0", help="One visible GPU index; one independent run per process")
    p.add_argument("--cpu-check", action="store_true", help="Permit small local verification on CPU")
    p.add_argument("--seed", type=int)
    p.add_argument("--download", action="store_true")
    p.add_argument("--test-data")
    p.add_argument("--limit", type=int, help="Explicit smoke-test training/evaluation subset; not a full benchmark")
    p.add_argument("--resume")
    p.add_argument("--stop-after-epoch", type=int)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--export")
    args = p.parse_args(argv)
    if args.gpu not in ("0", "1", "2", "3", "4", "5", "6", "7"):
        p.error("--gpu selects one GPU index0..7")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from torch.utils.data import DataLoader
    from mdistiller.tgsc.experiment import load_run
    from mdistiller.tgsc.cache import cifar_dataset, file_hash, dataset_hash
    from mdistiller.tgsc.evaluation_data import test_dataset, TEST_SHA256
    from mdistiller.tgsc.runtime import (seed_everything, build_model, load_teacher,
                                        build_distiller, load_export, evaluate)
    from mdistiller.epkd.runtime import settings, preflight, train
    exp, method, options, raw_cfg = load_run(args.config)
    cfg = settings(exp, raw_cfg)
    if args.data_root:
        exp.data.root = args.data_root
    if args.seed is not None:
        exp.train.seed = args.seed
    if args.limit is not None and args.limit < exp.data.batch_size and args.command == "train":
        p.error("--limit must allow at least one complete training batch")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= exp.train.epochs:
        p.error("--stop-after-epoch must lie within the configured epoch range")
    seed_everything(exp.train.seed)
    if not torch.cuda.is_available() and not args.cpu_check:
        raise RuntimeError("CUDA unavailable; CPU verification requires explicit --cpu-check")
    device = torch.device("cpu" if args.cpu_check else "cuda:0")
    if args.cpu_check:
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    def evaluation_inputs():
        if args.test_data:
            ds = test_dataset(exp, download=args.download, path=args.test_data, limit=args.limit)
            identity = {"artifact_sha256": TEST_SHA256, "samples": len(ds)}
        else:
            # The original archive already includes test images; avoid a second28MB download.
            ds = cifar_dataset(exp, train=False, download=args.download, limit=args.limit)
            identity = {"original_test_sha256": dataset_hash(ds), "samples": len(ds)}
        return ds, identity

    if args.command == "evaluate":
        if not args.export:
            p.error("evaluate requires --export")
        model, meta = load_export(args.export)
        if meta["method"] != method or meta["student_model"] != exp.student_model:
            raise ValueError("Export differs from method/student config")
        exp.data.mean, exp.data.std = tuple(meta["mean"]), tuple(meta["std"])
        ds, _ = evaluation_inputs()
        print(json.dumps(evaluate(model.to(device), DataLoader(ds, batch_size=exp.data.batch_size), device)))
        return
    path = args.teacher_checkpoint or exp.teacher_checkpoint
    if not path or not Path(path).is_file():
        raise FileNotFoundError("Provide --teacher-checkpoint (no crop bank required)")
    teacher = load_teacher(build_model(exp.teacher_model, exp.num_classes), path)
    distiller = build_distiller(exp, method, options, teacher).to(device)
    # Raw originals are required for clean-image CAM and the full-image anchor.
    ds = cifar_dataset(exp, download=args.download, limit=args.limit)
    images = torch.from_numpy(ds.data.copy()).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(ds.targets, dtype=torch.long)
    print(json.dumps({"pipeline": "epkd", "method": method, "teacher": exp.teacher_model,
                      "student": exp.student_model, "training_images": len(images),
                      "teacher_sha256": file_hash(path), "epkd": cfg, "device": str(device)}), flush=True)
    if args.command == "check":
        print(json.dumps({"preflight": preflight(distiller, images, labels, exp, cfg, device)}))
        print("EP-KD preflight passed. No training or crop bank creation.")
        return
    test, test_id = evaluation_inputs()
    train(exp, method, options, cfg, distiller, images, labels, test, device, args.output, path,
          resume=args.resume, stop_after_epoch=args.stop_after_epoch, eval_every=args.eval_every, test_identity=test_id)


if __name__ == "__main__":
    main()
