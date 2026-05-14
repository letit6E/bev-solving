"""v2: replace backbone with Simple-BEV's Encoder_res101 (nuScenes-pretrained).

Requires `external/simple_bev` (see scripts/setup_simplebev.sh).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.models.v1 import MultiCamBEV

SIMPLEBEV_REPO = Path("external/simple_bev")


def _import_simplebev():
    if not SIMPLEBEV_REPO.exists():
        raise FileNotFoundError(f"clone Simple-BEV into {SIMPLEBEV_REPO} first")
    p = str(SIMPLEBEV_REPO.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _find_ckpt():
    cands = list(SIMPLEBEV_REPO.rglob("model-*.pth")) + list(SIMPLEBEV_REPO.rglob("model-*.pt"))
    if not cands:
        raise FileNotFoundError(f"no checkpoint under {SIMPLEBEV_REPO}")
    return max(cands, key=lambda p: p.stat().st_size)


def load_pretrained_encoder(encoder, ckpt_path=None):
    ckpt = torch.load(Path(ckpt_path) if ckpt_path else _find_ckpt(), map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    enc_sd = {k.replace("encoder.", "", 1): v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    print(f"matched {len(enc_sd) - len(unexpected)}/{len(enc_sd)}; missing={len(missing)}")


class MultiCamBEVPretrainedEncoder(MultiCamBEV):
    def __init__(self, load_pretrained=True, freeze_encoder=True, ckpt=None):
        super().__init__(freeze_backbone=False)
        _import_simplebev()
        from nets.segnet import Encoder_res101

        del self.backbone
        self.backbone = Encoder_res101(C=128)
        if load_pretrained:
            load_pretrained_encoder(self.backbone, ckpt)
        self._frozen = freeze_encoder
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
