"""BCE + Dice + Lovasz hinge with ignore-mask support.

Lovasz hinge: direct surrogate for IoU on the positive class. Helped most for
the imbalanced occupancy task — see Berman et al. CVPR'18.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(loader, max_batches=200):
    pos = neg = 0
    for i, b in enumerate(loader):
        gt = b["gt"]
        valid = gt != 255
        pos += (gt[valid] == 1).sum().item()
        neg += (gt[valid] == 0).sum().item()
        if i >= max_batches:
            break
    return neg / max(pos, 1)


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    j = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        j[1:] = j[1:] - j[:-1]
    return j


def lovasz_hinge_flat(logits, labels):
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    grad = _lovasz_grad(labels[perm])
    return torch.dot(F.relu(errors_sorted), grad)


class CompoundLoss(nn.Module):
    """BCE(pos_weight) + lambda_dice * Dice (used in v1/v2)."""

    def __init__(self, pos_weight=5.0, lambda_dice=0.5, ignore_value=255):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.lambda_dice = lambda_dice
        self.ignore_value = ignore_value

    def forward(self, logits, gt):
        valid = (gt != self.ignore_value).float()
        gt_f = (gt == 1).float()
        bce = F.binary_cross_entropy_with_logits(
            logits, gt_f, pos_weight=self.pos_weight, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
        prob = torch.sigmoid(logits) * valid
        gt_d = gt_f * valid
        inter = (prob * gt_d).sum((1, 2, 3))
        denom = prob.sum((1, 2, 3)) + gt_d.sum((1, 2, 3))
        dice = (1 - (2 * inter + 1) / (denom + 1)).mean()
        return bce + self.lambda_dice * dice, {"bce": bce.item(), "dice": dice.item()}


class CompoundLossV2(nn.Module):
    """BCE + Dice + Lovasz (used from v3 onwards)."""

    def __init__(self, pos_weight=5.0, w_bce=0.5, w_dice=0.3, w_lovasz=0.2, ignore_value=255):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.w_bce, self.w_dice, self.w_lovasz = w_bce, w_dice, w_lovasz
        self.ignore_value = ignore_value

    def forward(self, logits, gt):
        valid = (gt != self.ignore_value)
        valid_f = valid.float()
        gt_f = (gt == 1).float()

        bce = F.binary_cross_entropy_with_logits(
            logits, gt_f, pos_weight=self.pos_weight, reduction="none")
        bce = (bce * valid_f).sum() / valid_f.sum().clamp(min=1.0)

        prob = torch.sigmoid(logits) * valid_f
        gt_d = gt_f * valid_f
        inter = (prob * gt_d).sum((1, 2, 3))
        denom = prob.sum((1, 2, 3)) + gt_d.sum((1, 2, 3))
        dice = (1 - (2 * inter + 1) / (denom + 1)).mean()

        lov_logits = logits[valid]
        lov_gt = gt_f[valid]
        lov = lovasz_hinge_flat(lov_logits, lov_gt) if lov_logits.numel() else logits.sum() * 0.0

        total = self.w_bce * bce + self.w_dice * dice + self.w_lovasz * lov
        return total, {"bce": bce.item(), "dice": dice.item(), "lovasz": lov.item()}
