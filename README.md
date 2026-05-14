# BEV Static Obstacle Prediction — Yandex ML2 2026

1st place on the private leaderboard. Final test IoU **0.6258** (baseline 0.40,
top-2 0.56).

The task: predict a 188×126 binary occupancy map (free / occupied / ignore at
255) on a 0.8 m/pixel BEV grid covering 150 m forward × 100 m lateral, given
4 calibrated RGB cameras per sample (3×4 intrinsics, 4×4 car→cam extrinsics).
4000 train / 1000 val / 2000 test. Metric: IoU on class 1. See
[plans/00_task_description.md](plans/00_task_description.md).

## Repo layout

```
src/                        cleaned shared modules
  geometry.py               BEV grid constants, conventions
  data.py                   BEVDataset, BEVDatasetAug (per-camera resize+intrinsic update)
  splits.py                 group-aware and test-matched splits
  losses.py                 BCE + Dice + Lovasz hinge
  metrics.py                IoU, memory-safe streaming threshold sweep
  submission.py             contest-shaped zip with SHA256 verification
  models/
    decoder.py              shared SmallUNet
    voxel.py                parameter-free voxel projection (Simple-BEV style)
    v1.py                   ResNet-18 baseline
    v2.py                   Simple-BEV Encoder_res101 (nuScenes pretrain)
    v4.py                   ResNet-50 + rover embedding + letterbox
    v5.py                   ResNet-34 + FiLM (intrinsics/extrinsics) + specialist branch
    simplebev_adapter.py    wrap pretrained Simple-BEV for zero-shot scoring
scripts/                    one-off CLI tools
notebooks/                  per-stage experiments (cell outputs kept, code lives in src/)
plans/                      initial planning docs and findings
best.md                     detailed write-up of the winning solution
```

Datasets, checkpoints, `*.zip` submissions, `runs/`, `inference_eval/`,
`ensemble_pray/`, `predicted_static_grids/` are all gitignored. Reproducing
requires:

1. unzip `autonomy_yandex_dataset_train/val/test_v2.zip`;
2. `scripts/check_integrity.py --crc --repair` against each archive;
3. for v2: `bash scripts/setup_simplebev.sh` to fetch Simple-BEV + pretrained
   weights into `external/simple_bev/`;
4. for v6/v8: pretrained ConvNeXtV2-FCMAE / DINOv2 weights are downloaded by
   the corresponding notebooks.

## Stages

### Stage 0 — sanity check ([notebooks/00_sanity/](notebooks/00_sanity/))

Fix conventions before any training:
- ego frame: X forward, Y left (positive), Z up;
- `p_cam = car_to_cam @ p_ego_h` (the matrix as stored maps ego → cam, not the
  reverse — verified visually);
- intrinsic is stored as `(3, 4) = [K | 0]`;
- GT: 0 free, 1 occupied, 255 ignore;
- BEV grid: row → X 0..150.4 m, col → Y −50.4..+50.4 m, 0.8 m/pixel.

Discovered: rover `nack` ships 768×959 frames (everyone else is 640×480-ish);
~26 empty samples; ~18% mean obstacle coverage; near-stationary frames keep
GT essentially constant (IoU ≈ 0.95+).

### Stage 1 — baseline ([src/models/v1.py](src/models/v1.py), [notebook](notebooks/stage_v1_baseline/))

Multi-camera Simple-BEV-style baseline: ImageNet-pretrained ResNet-18 layer2 →
1×1 proj 64ch → project ego voxels at 4 height planes via
`car_to_cam` + intrinsics → bilinear sample → mean over visible cameras →
flatten height → SmallUNet → 1 logit per BEV cell. BCE(pos_weight) + 0.5·Dice
with ignore-mask 255. Coverage-aware WeightedRandomSampler.

Best test IoU around **0.51**.

### Stage 2 — Simple-BEV pretrained encoder ([src/models/v2.py](src/models/v2.py), [notebook](notebooks/stage_v2_pretrained/))

Drop-in: replace ResNet-18 layer2 with Simple-BEV's `Encoder_res101`
(nuScenes-pretrained, 37M params). Keep the rest of the pipeline. Encoder
frozen for the first epochs to avoid wrecking the pretrained features, then
optionally unfrozen at a lower LR.

Also: a zero-shot eval of the full pretrained Simple-BEV model
([eval_simplebev_zero_shot.ipynb](notebooks/stage_v2_pretrained/eval_simplebev_zero_shot.ipynb))
— it predicts vehicles, not static obstacles, so scored ~0.15 IoU only.
Confirmed we should only transfer the encoder, not the decoder.

Test IoU plateaued near v1 (~0.51). Bottleneck was the training recipe,
not the backbone.

### Stage 3 — augs + Lovasz + group-aware split ([notebook](notebooks/stage_v3_augs_lovasz/))

Phase 2 fixes following the Simple-BEV paper recipe (arXiv 2206.07959):

- `BEVDatasetAug`: per-camera random scale ∈ [1.0, 1.2] + random crop with
  proper intrinsic update (fx/fy/cx/cy scale, cx/cy shift by pad/crop) — paper
  reports +1.6 IoU just from this.
- `CompoundLossV2` = 0.5·BCE + 0.3·Dice + 0.2·Lovasz hinge. Lovasz is a direct
  surrogate for IoU on the positive class.
- `make_group_aware_split` on `(rover, ride_date)`: no group appears in both
  train and val, so val distribution moves towards test.
- Encoder unfrozen with two param groups (lr_backbone=1e-5, lr_head=3e-4),
  IMG_HW=448×800, batch=8 × grad_accum=4, EMA decay 0.999.

Test IoU ~**0.52**. Threshold tuning on val started to hurt test —
diagnosed as residual val/test mismatch in rover frequencies
(see [plans/06_findings_round2.md](plans/06_findings_round2.md)).
Memory blew up during threshold sweep on full val; the streaming sweep
in [src/metrics.py](src/metrics.py) fixed it (constant memory regardless of
dataset size).

### Stage 4 — rover embedding + multiple backbones ([src/models/v4.py](src/models/v4.py), [notebooks](notebooks/stage_v4_rover_emb/))

Diversity for the ensemble: keep the same projection pipeline but try several
backbones on top of letterbox resize (preserves aspect ratio — critical for
`nack`):

- ResNet-50 layer2 (default v4);
- DINOv2 ViT-Base via `torch.hub`, last 2 blocks unfrozen;
- RTMDet-L from MMDet (COCO-pretrained, CSPNeck + SPP);
- Swin Transformer + GeoMIM pretraining.

Plus a 32-dim rover embedding broadcast into the BEV feature before the
decoder, smart-deduplicated training set (scripts/smart_dedup.py drops a few
hundred near-stationary frames), and `make_test_matched_split` that biases
val rover frequencies towards test mass.

Best single-model test IoU here ~**0.54**.

### Stage 5 — calibration-aware FiLM + specialist branch ([src/models/v5.py](src/models/v5.py), [notebooks](notebooks/stage_v5_film/))

Hypothesis: rovers differ by camera mount geometry more than by neighbourhood,
so condition the model on rig features directly, not just on a rover_id.

- Per-camera 10-dim rig vector (normalized fx/fy/cx/cy + camera position +
  forward axis) → FiLM modulator on the 64-channel image features.
- Per-sample 12-dim global rig summary (mean/std focals, L-R baseline, front
  mid-far deltas) + rover embedding + specialist embedding (top-12 test
  rovers) → FiLM modulator on the BEV feature.
- A scalar sigmoid gate scales the FiLM bias, so specialist rovers can pull
  the BEV prior whilst rare rovers stay near the shared trunk.

Per-rover gain mostly on the dominant test rovers.

### Stage 6 — DINOv2 + LSS ([notebooks](notebooks/stage_v6_dinov2_lss/))

Switch the view transform from parameter-free voxel sampling to learned
Lift-Splat-Shoot (each pixel predicts a depth distribution; features are
splatted into the BEV grid by depth × camera ray). DINOv2 ViT-B/14 backbone.

Variants:
- **v61** — multi-head BEV self-attention in the decoder;
- **v62** — 2× BEV resolution then downsample (richer features near
  obstacles, especially for the close 0–30 m band);
- **v63** — log-spaced depth bins (Simple-BEV-style uniform bins waste
  capacity on far ranges; log bins concentrate where ground occupies most of
  the BEV cells).

Best single-model test IoU here ~**0.58**.

### Stage 7 — RTMDet/CSPNeXt + LSS ([notebooks](notebooks/stage_v7_rtmdet_cspnext/))

Lighter, faster backbone for ensemble diversity. CSPNeXt features go through
the same LSS view transform as v6. Trained both locally and on Colab.
Comparable test IoU to v6 (~0.56–0.57) with very different prediction noise —
useful for averaging.

### Stage 8 — ConvNeXtV2 + FCMAE ([notebook](notebooks/stage_v8_convnextv2/))

ConvNeXtV2-Base pretrained with FCMAE (sparse-conv masked autoencoder on
ImageNet-22k → ImageNet-1k). WandB logging, resume from checkpoint. Strongest
single model on the validation distribution.

### Ensembling ([notebooks/ensemble/](notebooks/ensemble/))

Average sigmoid probabilities across v6 variants, v7, v8 (and v3 for early
runs). Search ensemble weights with the search notebooks. Submission packing
in [src/submission.py](src/submission.py) (zip + testzip() + SHA256).

Final winning blend: v6 family + v7 + v8 with weights tuned on a
test-matched val, threshold ~0.62–0.66 depending on the blend. Final test
IoU **0.6258**.

## Things that didn't work / surprised us

- Smoothing the GT with a Gaussian filter: hurt. Comment kept in
  [plans/07_data_pipeline_plan.md](plans/07_data_pipeline_plan.md).
- Threshold tuning on the official val: hurt test because of val/test rover
  mismatch (TV ≈ 0.349). Switched to test-matched group split.
- Aggressive EMA decay (0.9999): worse than 0.999 on this small dataset.
- Zero-shot Simple-BEV transfer: 0.15 IoU. The pretrained model targets
  vehicles, not static obstacles.
- The train/test image-hash exploit
  ([scripts/exploit_leak.py](scripts/exploit_leak.py)) found a few hundred
  near-identical test frames in train — worth ~0.005–0.01 IoU. Marginal but
  legal under the rules.

## Hardware

10-day window: 18h A100, 59h T4, 29h V100 shared with a local M1 Pro.
Most v1–v3 work on the M1 (slow but free); paper-recipe runs on A100;
ensemble inference on T4.

## References

- Harley et al., *Simple-BEV: What Really Matters for Multi-Sensor BEV
  Perception?* arXiv:2206.07959
- Berman et al., *The Lovász-Softmax loss*, CVPR 2018
- Philion & Fidler, *Lift, Splat, Shoot*, ECCV 2020
- Zhou et al., *DINOv2*, arXiv:2304.07193
- Woo et al., *ConvNeXt V2*, CVPR 2023
