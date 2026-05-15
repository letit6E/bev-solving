"""Validation-time IoU loop. Forward, threshold@0.5, accumulate inter/union.

Two batch-key conventions live in the notebooks ('gt' and 'gt_hard'); we read
either. Rover id is optional — older v1/v2/v3 models don't take it.
"""
import torch
from tqdm.auto import tqdm

from src.metrics import iou_binary_batch


@torch.inference_mode()
def evaluate_iou(model, loader, device, threshold=0.5, desc="val@0.5",
                 use_amp=True, forward_fn=None):
    model.eval()
    inter = 0
    union = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        intr = batch["intrinsics"].to(device, non_blocking=True)
        c2c = batch["car2cams"].to(device, non_blocking=True)
        rover_id = batch.get("rover_id")
        if rover_id is not None:
            rover_id = rover_id.to(device, non_blocking=True)
        gt = batch.get("gt", batch.get("gt_hard")).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and use_amp)):
            if forward_fn is not None:
                logits = forward_fn(model, images, intr, c2c, rover_id)
            elif rover_id is not None:
                logits = model(images, intr, c2c, rover_id)
            else:
                logits = model(images, intr, c2c)
        logits = logits.float()
        i, u = iou_binary_batch(logits, gt, threshold=threshold)
        inter += i
        union += u
        pbar.set_postfix(iou=f"{inter / max(union, 1):.4f}")
    return inter / max(union, 1)
