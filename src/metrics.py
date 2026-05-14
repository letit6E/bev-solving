"""IoU on class 1 with 255 as ignore.

`streaming_threshold_sweep` keeps memory constant — fixes the OOM that
happens if you cache all val logits then sweep thresholds.
"""
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def iou_binary_batch(logits, gt, threshold=0.5, ignore_value=255):
    pred = (torch.sigmoid(logits) > threshold).long()
    valid = (gt != ignore_value).long()
    pred = pred * valid
    gt_b = (gt == 1).long() * valid
    inter = ((pred == 1) & (gt_b == 1)).sum().item()
    union = ((pred == 1) | (gt_b == 1)).sum().item()
    return inter, union


@torch.no_grad()
def streaming_threshold_sweep(model, loader, thresholds, device,
                              amp_enabled=True, ignore_value=255,
                              forward_fn=None):
    inter = torch.zeros(len(thresholds), dtype=torch.float64)
    union = torch.zeros(len(thresholds), dtype=torch.float64)
    model.eval()
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
    for batch in tqdm(loader, desc="threshold sweep"):
        imgs = batch["images"].to(device, non_blocking=True)
        intr = batch["intrinsics"].to(device, non_blocking=True)
        c2c = batch["car2cams"].to(device, non_blocking=True)
        gt = batch["gt"]
        with autocast(enabled=amp_enabled):
            logits = forward_fn(model, imgs, intr, c2c) if forward_fn else model(imgs, intr, c2c)
        probs = torch.sigmoid(logits).cpu()
        valid = (gt != ignore_value)
        gt_b = ((gt == 1) & valid).float()
        for ti, t in enumerate(thresholds):
            pred = ((probs > t) & valid).float()
            inter[ti] += (pred * gt_b).sum().item()
            union[ti] += (pred + gt_b).clamp(0, 1).sum().item()
    return {float(t): float(inter[ti] / max(union[ti].item(), 1.0))
            for ti, t in enumerate(thresholds)}
