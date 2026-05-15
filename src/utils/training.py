"""EMA, model unwrapping, backbone freezing, CUDA cleanup, resume/warm-start."""
import copy
import gc
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def update_ema(ema_model, model, decay):
    src = unwrap_model(model)
    ema_params = dict(ema_model.named_parameters())
    for name, p in src.named_parameters():
        ema_params[name].mul_(decay).add_(p.data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, b in src.named_buffers():
        ema_buffers[name].copy_(b)


def make_ema_copy(model, device):
    ema = copy.deepcopy(unwrap_model(model)).to(device).eval()
    for p in ema.parameters():
        p.requires_grad = False
    return ema


def freeze_all_backbone(model):
    """Freeze the DINOv2-like ViT trunk."""
    core = unwrap_model(model)
    for p in core.backbone.vit.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model, n_last_blocks=2):
    core = unwrap_model(model)
    freeze_all_backbone(core)
    if hasattr(core.backbone.vit, "blocks"):
        for blk in core.backbone.vit.blocks[-n_last_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    if hasattr(core.backbone.vit, "norm"):
        for p in core.backbone.vit.norm.parameters():
            p.requires_grad = True


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt


def load_warm_start(model, ckpt_path):
    """Load matching keys from a sibling-architecture checkpoint. Skip mismatches."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print("warm-start checkpoint not found, starting from random init:", ckpt_path)
        return {"loaded": 0, "skipped": 0, "missing": None, "unexpected": None}

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["ema"] if "ema" in ckpt else ckpt.get("model", ckpt)
    cur = model.state_dict()
    loadable, skipped = {}, []
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    print("loaded warm-start from", ckpt_path)
    print("  matched keys:", len(loadable))
    print("  skipped old keys:", len(skipped))
    print("  missing in new model:", len(missing))
    print("  unexpected during load:", len(unexpected))
    if len(skipped):
        print("  sample skipped:", skipped[:10])
    return {"loaded": len(loadable), "skipped": len(skipped),
            "missing": missing, "unexpected": unexpected}


def load_resume_state(core_model, ema_model, optimizer, scheduler, scaler,
                     run_dir, cfg):
    """Reload core/ema/opt/sched/scaler + log.csv to resume training mid-run."""
    run_dir = Path(run_dir)
    resume_ckpt = Path(cfg.get("resume_ckpt", ""))
    if not cfg.get("resume_training", False) or not resume_ckpt.exists():
        return {"enabled": False, "start_epoch": 0, "best_iou": -1.0,
                "best_ema_iou": -1.0, "log": [], "elapsed_minutes": 0.0}

    ckpt = torch.load(resume_ckpt, map_location="cpu")
    core_model.load_state_dict(ckpt["model"], strict=False)
    if "ema" in ckpt:
        ema_model.load_state_dict(ckpt["ema"], strict=False)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    log_path = run_dir / "log.csv"
    log_rows, elapsed_minutes = [], 0.0
    if log_path.exists():
        log_rows = pd.read_csv(log_path).to_dict("records")
        if len(log_rows):
            elapsed_minutes = float(log_rows[-1].get("minutes", 0.0) or 0.0)

    best_iou = float(ckpt.get("best_iou", -1.0))
    best_ema_iou = float(ckpt.get("best_ema_iou", -1.0))
    for src_path, key in [(run_dir / "best.pt", "best_iou"),
                          (run_dir / "ema_best.pt", "best_ema_iou")]:
        if src_path.exists():
            try:
                v = float(torch.load(src_path, map_location="cpu").get(key, -1.0))
                if key == "best_iou":
                    best_iou = max(best_iou, v)
                else:
                    best_ema_iou = max(best_ema_iou, v)
            except Exception:
                pass

    print("resumed from", resume_ckpt)
    print("  start_epoch:", start_epoch)
    print("  best_iou so far:", best_iou)
    print("  best_ema_iou so far:", best_ema_iou)
    print("  prior log rows:", len(log_rows))
    return {"enabled": True, "start_epoch": start_epoch,
            "best_iou": best_iou, "best_ema_iou": best_ema_iou,
            "log": log_rows, "elapsed_minutes": elapsed_minutes}
