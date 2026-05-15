"""EMA, model unwrapping, backbone freezing, CUDA cleanup."""
import copy
import gc

import torch


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
