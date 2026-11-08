"""Full-image anchor + CAM crop training with official method losses, recovery and exports."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from ..tgsc.cache import atomic_save, file_hash
from ..tgsc.runtime import (environment, evaluate, export_payload, rng_state, restore_rng,
                           resume_contract)
from ..tgsc.reporting import write_reports
from .pipeline import ImageViews, crop_batchnorm, stage_coordinates, teacher_cam


DEFAULTS = {"passes": [3, 3, 3, 1, 1], "strides": [8, 8, 8, 8, 8],
            "anchor_weight": 1., "view_weight": 1., "proposal_batch_size": 256}


def settings(exp, raw):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(raw.get("epkd", {}))
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown EP-KD setting")
    if len(exp.data.stages) != len(cfg["passes"]) or len(exp.data.stages) != len(cfg["strides"]):
        raise ValueError("Specify one stride and pass count per stage")
    if any(type(v) is not int or v < 1 for v in cfg["passes"] + cfg["strides"]):
        raise ValueError("Strides and passes must be positive integers")
    if cfg["proposal_batch_size"] < 1 or cfg["anchor_weight"] <= 0 or cfg["view_weight"] <= 0:
        raise ValueError("Positive proposal batch size and anchor/view weights required")
    if raw.get("method", "").upper() == "VRM" and (cfg["anchor_weight"] != 1 or cfg["view_weight"] != 1):
        raise ValueError("VRM has one joint paired objective; independent branch weights must remain1")
    return cfg


def tensor_identity(raw, labels):
    if raw.dtype != torch.uint8 or raw.ndim != 4 or tuple(raw.shape[1:]) != (3, 32, 32):
        raise ValueError("Expected original uint8 CIFAR images [N,3,32,32]")
    if labels.dtype != torch.long or labels.shape != (len(raw),) or len(raw) < 2:
        raise ValueError("Expected one int64 target per image and at least two images")
    h = hashlib.sha256(raw.cpu().contiguous().numpy().tobytes())
    h.update(labels.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_identity():
    root = Path(__file__).parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return h.hexdigest()


def method_step(distiller, full, crop, target, epoch, cfg):
    """Reuse official losses unchanged; neither branch receives TGSC gate metadata."""
    if getattr(distiller, "paired_views", False):
        if crop is None:
            raise ValueError("VRM requires two views, including at32px")
        logits, losses = distiller(image_weak=full, image_strong=crop, target=target,
                                   crop_second=crop.shape[-1] < full.shape[-1], epoch=epoch)
        return logits, {"pair."+k: v for k,v in losses.items()}, sum(losses.values())
    logits, full_losses = distiller(image=full, target=target, epoch=epoch)
    losses = {"full."+k: v for k, v in full_losses.items()}
    total = cfg["anchor_weight"]*sum(full_losses.values()) if crop is not None else sum(full_losses.values())
    if crop is not None:
        with crop_batchnorm(distiller):
            _, crop_losses = distiller(image=crop, target=target, epoch=epoch)
        losses.update({"crop."+k: v for k, v in crop_losses.items()})
        total = total + cfg["view_weight"]*sum(crop_losses.values())
    return logits, losses, total


def preflight(distiller, raw, labels, exp, cfg, device):
    """Exercise real teacher CAM and method losses at every stage without altering training state."""
    state, rng = copy.deepcopy(distiller.state_dict()), rng_state()
    report = []
    views = ImageViews(exp.data.mean, exp.data.std, device)
    final_views = vrm_final_views(exp, device) if getattr(distiller, "paired_views", False) else None
    raw, labels = raw[:2].to(device), labels[:2].to(device)
    try:
        for stage, (size, _) in enumerate(exp.data.stages):
            distiller.train()
            coordinates = None
            if size < 32:
                coordinates = stage_coordinates(distiller.teacher, raw, labels, views, size, cfg["strides"][stage], 2)
                crop = views.view(raw, coordinates, size, 0)
            else:
                crop = None
            if size == 32 and final_views is not None:
                full, crop = final_views(raw)
            else:
                full = views.full(raw)
            _, losses, total = method_step(distiller, full, crop, labels, 1, cfg)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite preflight loss at{size}")
            total.backward()
            if any(p.grad is not None for p in distiller.teacher.parameters()):
                raise AssertionError("Teacher unexpectedly received gradients")
            if any(not torch.isfinite(p.grad).all() for p in distiller.get_learnable_parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite trainable gradients")
            report.append({"resolution": size, "losses": {k: float(v.detach()) for k, v in losses.items()},
                           "coordinate_shape": list(coordinates.shape) if coordinates is not None else None})
            distiller.zero_grad(set_to_none=True)
    finally:
        distiller.load_state_dict(state)
        distiller.zero_grad(set_to_none=True)
        restore_rng(rng)
    return report


def vrm_final_views(exp, device):
    """Use the original PIL transform pair, with original per-image RNG ordering at32px."""
    from PIL import Image
    from .vrm_augment import get_cifar100_train_transform_strong
    if tuple(exp.data.mean) != (.5071,.4867,.4408) or tuple(exp.data.std) != (.2675,.2565,.2761):
        raise ValueError("Official VRM32px transform expects its original CIFAR normalization")
    transform = get_cifar100_train_transform_strong()
    def apply(raw):
        pairs = [transform(Image.fromarray(image)) for image in raw.cpu().permute(0,2,3,1).numpy()]
        return tuple(torch.stack([p[i] for p in pairs]).to(device) for i in (0,1))
    return apply


def train(exp, method, options, cfg, distiller, raw, labels, test_dataset, device, output,
          teacher_path, resume=None, stop_after_epoch=None, eval_every=5, test_identity=None):
    if eval_every < 1 or len(raw) < exp.data.batch_size:
        raise ValueError("Positive evaluation interval and at least one complete training batch required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"latest.pt").exists() and not resume:
        raise FileExistsError("Use --resume or a fresh output directory")
    device = torch.device(device)
    data_sha = tensor_identity(raw, labels)
    raw, labels = raw.to(device), labels.to(device)
    distiller.to(device)
    params = [p for p in distiller.get_learnable_parameters() if p.requires_grad]
    if len(set(map(id, params))) != len(params):
        raise ValueError("Duplicate optimizer parameters")
    optimizer = torch.optim.SGD(params, lr=exp.optim.base_lr, momentum=exp.optim.momentum, weight_decay=exp.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(exp.optim.milestones), exp.optim.milestone_gamma)
    run_cfg = {"experiment": asdict(exp), "method": method, "options": options, "epkd": cfg}
    provenance = {"config": run_cfg, "pipeline": "epkd-v1", "source_sha256": source_identity(),
                  "teacher_sha256": file_hash(teacher_path), "train_sha256": data_sha,
                  "test_identity": test_identity, "evaluation": {"every": eval_every}, "environment": environment()}
    contract = {k:v for k,v in provenance.items() if k not in ("config", "environment")}
    contract["config"] = resume_contract(run_cfg)
    contract["torch"] = provenance["environment"]["torch"]
    contract["torchvision"] = provenance["environment"]["torchvision"]
    start, best, best_epoch, best_inference, history = 1, -1., None, None, []
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("contract") != contract:
            raise ValueError("Resume config/data/teacher/code/dependencies/evaluation identity differs")
        distiller.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best, best_epoch = saved["epoch"]+1, saved["best_top1"], saved["best_epoch"]
        history, best_inference = saved["metrics_history"], saved["best_inference"]
        restore_rng(saved["rng"])
    (output/"run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output/"metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in history), encoding="utf-8")
    if best_inference is not None:
        atomic_save(best_inference, output/"inference_best.pt")
    views = ImageViews(exp.data.mean, exp.data.std, device)
    final_views = vrm_final_views(exp, device) if getattr(distiller, "paired_views", False) else None
    test_loader = DataLoader(test_dataset, batch_size=exp.data.batch_size, num_workers=0)
    plan = [i for i, (_, duration) in enumerate(exp.data.stages) for _ in range(duration)]
    current, coordinates = None, None
    for epoch in range(start, exp.train.epochs+1):
        stage = plan[epoch-1]
        size = exp.data.stages[stage][0]
        active = size < 32
        if current != stage:
            before = rng_state()
            try:
                coordinates = stage_coordinates(distiller.teacher, raw, labels, views, size,
                                  cfg["strides"][stage], cfg["proposal_batch_size"]) if active else None
            finally:
                restore_rng(before)
            current = stage
            print(f"Stage{stage+1}: size{size}, passes{cfg['passes'][stage]}, CAM stride{cfg['strides'][stage]}, online coordinate cache", flush=True)
        distiller.train()
        begin = time.monotonic()
        totals, n, steps, full_correct = {}, 0, 0, 0
        teacher_correct, teacher_count = 0, 0
        teacher_output = []
        hook = distiller.teacher.register_forward_hook(lambda module, args, out: teacher_output.append(out[0].detach()))
        try:
            for pass_idx in range(cfg["passes"][stage]):
                order = torch.randperm(len(raw), device=device)
                for start_idx in range(0, len(raw)-exp.data.batch_size+1, exp.data.batch_size):
                    idx = order[start_idx:start_idx+exp.data.batch_size]
                    target = labels[idx]
                    if final_views is not None and not active:
                        full, crop = final_views(raw[idx])
                    else:
                        full = views.full(raw[idx])
                        crop = views.view(raw[idx], coordinates[idx], size, pass_idx % 3) if active else None
                    teacher_output.clear()
                    optimizer.zero_grad(set_to_none=True)
                    logits, losses, total = method_step(distiller, full, crop, target, epoch, cfg)
                    if not torch.isfinite(total):
                        raise FloatingPointError(f"Nonfinite loss at epoch{epoch}")
                    total.backward()
                    optimizer.step()
                    batch = len(idx)
                    full_correct += int((logits.argmax(1) == target).sum())
                    if active or final_views is not None:
                        teacher_correct += int((teacher_output[-1].argmax(1) == target).sum())
                        teacher_count += batch
                    for k, v in {**losses, "loss_total": total}.items():
                        totals[k] = totals.get(k, 0.) + float(v.detach())*batch
                    n += batch
                    steps += 1
        finally:
            hook.remove()
        stopping = stop_after_epoch is not None and epoch >= stop_after_epoch
        measured = epoch % eval_every == 0 or epoch == exp.train.epochs or stopping
        metrics = None
        before = rng_state()
        try:
            if measured:
                metrics = evaluate(distiller, test_loader, device)
        finally:
            restore_rng(before)
        improved = metrics is not None and metrics["top1"] > best
        if improved:
            best, best_epoch = metrics["top1"], epoch
        row = {"epoch": epoch, "pipeline": "epkd", "resolution": size, "passes": cfg["passes"][stage],
               "stride": cfg["strides"][stage] if active else None,
               "train": {k:v/n for k,v in totals.items()}, "train_full_top1": 100*full_correct/n,
               "teacher_crop_top1": 100*teacher_correct/teacher_count if active and teacher_count else None,
               "teacher_second_view_top1": 100*teacher_correct/teacher_count if final_views is not None and teacher_count else None,
               "second_view": ("vrm_original_strong32" if final_views is not None else None) if not active else "cam_crop",
               "source_draws": n, "optimizer_steps": steps, "lr": optimizer.param_groups[0]["lr"],
               "test": metrics, "test_evaluated": measured, "best_test_top1": best if best >= 0 else None,
               "best_epoch": best_epoch, "seconds": time.monotonic()-begin}
        scheduler.step()
        inference = export_payload(distiller, exp, method, options)
        if improved:
            best_inference = inference
        history.append(row)
        saved = {**provenance, "contract": contract, "epoch": epoch, "best_top1": best, "best_epoch": best_epoch,
                 "best_inference": best_inference, "metrics_history": history, "model": distiller.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state()}
        atomic_save(saved, output/"latest.pt")
        atomic_save(inference, output/"inference_latest.pt")
        if improved:
            atomic_save(saved, output/"best.pt")
            atomic_save(inference, output/"inference_best.pt")
        with (output/"metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row)+"\n")
        before = rng_state()
        try:
            write_reports(output, plots=measured)
        finally:
            restore_rng(before)
        print(json.dumps(row), flush=True)
        if stopping:
            break
    if start > exp.train.epochs:
        atomic_save(export_payload(distiller, exp, method, options), output/"inference_latest.pt")
        write_reports(output, plots=bool(history))
    return history
