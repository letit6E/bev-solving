# Winning solution write-up

Final private test IoU **0.6258** — 1st place. Public test IoU was higher
(~0.65); private dropped ~0.02 because of a threshold call I'd make
differently next time (see [Lessons learned](#lessons-learned)).

This document tells the full story of how the solution evolved, what worked,
what didn't, and what I'd do differently.

## 1. Data first

The starting split was train/val = 4000/1000, but I quickly noticed that
local validation IoU was a poor predictor of public leaderboard IoU — sometimes
by 2–3 points. Looking at rover frequencies, the official val skewed heavily
towards rovers that were rare on test, and one rover (`nack`) had a bunch of
almost-empty GT maps that inflated easy-to-predict cases.

So I merged train and val into a single pool of 5000 samples and resampled my
own validation:

- **Stratify by `(rover, ride_date)` group** so no group appears in both train
  and val. This removes the trivial leak where val is just frames from the
  same ride a few seconds before train frames.
- **Per-rover weights matched to test distribution.** I read off the per-rover
  count from test info.csv and pulled val samples in proportion. This brought
  the rover-frequency TV distance between val and test from ~0.35 down to
  ~0.18.

After two or three submissions I verified: **the new local val correlated with
public to within ~0.005 IoU**. This was the single most important investment
of the whole competition — it meant every subsequent experiment could be
trusted locally before spending a submission slot.

### Dataset cleanup

While I had the data merged anyway:

- **Empty GT maps**: dropped frames where the GT had 0 occupied pixels (~26
  samples). They added noise to BCE without teaching anything.
- **Near-stationary duplicates**: when the rover stops at a traffic light you
  get bursts of 10–30 frames that are visually identical. I grouped
  consecutive frames within each `(rover, ride_date)` where the frontal image
  hash had MAE < 0.02, and kept only the frame whose GT had the **most
  obstacle pixels** (richer supervision signal). Trimmed a few hundred
  redundant samples without losing any unique scene.

### Rover embeddings

The fleet has many rovers, and each rover carries cameras at different mount
positions, heights and focal lengths. Even with calibration as input, a
single shared model has trouble modelling the systematic ground-projection
bias per rover. I added a 32-dim rover embedding broadcast onto the BEV
feature map before the decoder, which let the model learn a per-rover prior.
Worth roughly +0.01 IoU on its own; more once combined with later
calibration-aware FiLM conditioning.

## 2. Splitting the pipeline: encoder vs post-image path

I separated the problem into two architectural questions:

1. **Image encoder** — extract per-pixel features that contain everything the
   downstream needs about what's in the image (semantic class, depth cues,
   surface orientation).
2. **Post-image path** — project those features from the 4 camera frames into
   the shared BEV grid and decode into an occupancy logit.

My intuition was that the **encoder was the bottleneck**, not the projection.
The post-image path is constrained by geometry; you can't make it lift better
than the input features allow. So I started by fixing a simple post-image
path — parameter-free voxel sampling (Simple-BEV style) + a small UNet — and
tried different encoders. Only later did I start playing with the post-image
path.

## 3. Encoder experiments

### Plain ResNet (didn't fly)

ResNet-18, -34, -50, -101 — all variants. ImageNet classification features
care about the global label of an image and are translation-invariant by
design. For BEV occupancy you need the *opposite*: features that are
spatially precise and depth-aware. The plateau was around 0.51–0.53 IoU
regardless of depth. Pretrained Simple-BEV's `Encoder_res101` (nuScenes-tuned)
gave only a marginal lift — the bottleneck was the representation, not the
weights.

### DINOv2 (the unlock)

DINOv2 is a vision transformer self-supervised on a huge web corpus. What
caught my eye in the DINOv2 paper: a *frozen* DINOv2 backbone plus a tiny
linear head predicts monocular depth maps competitively. That's exactly the
information BEV projection needs — for each image pixel, an estimate of where
it lives in 3D.

I plugged DINOv2 ViT-B/14 in as the image encoder (last two blocks
fine-tunable, the rest frozen) and trained ~20 epochs on the same post-image
path. Local val IoU jumped to **0.59+**, and the very first public submission
hit **0.60+**. This was the moment the problem opened up.

### Other backbones (saved for ensemble)

I also trained:

- **RTMDet-L** (COCO-pretrained, CSPNeXt backbone) — ~0.56 IoU. Slightly
  worse than DINOv2 alone but with very different failure modes.
- **ConvNeXtV2-Base** with FCMAE pretraining (ImageNet-22k → ImageNet-1k) —
  also ~0.57–0.58 IoU. WandB-tracked, resumable Colab training. Strongest
  CNN backbone of the bunch.
- **Swin Transformer + GeoMIM** pretraining — comparable but heavier to fine-
  tune at the batch sizes I had.

None of these caught DINOv2 solo, but they added diversity for the final
ensemble — they made different mistakes.

## 4. Post-image path experiments

Once DINOv2 was the encoder of choice I tried improving the projection:

- **Parameter-free voxel sampling vs Lift-Splat-Shoot.** Switching to LSS
  (learned per-pixel depth distribution that splats features into the BEV
  grid) helped a bit on average; my final v6 family is LSS-based.
- **BEV attention** (v61): self-attention layers in the BEV decoder so far-
  away cells could exchange information with near cells. Marginal effect
  solo.
- **2× BEV resolution** (v62): double the projection resolution, then
  bilinearly downsample. Idea was to give the decoder more spatial precision
  near obstacle boundaries. Solo: marginal.
- **Log-spaced depth bins** (v63): the default uniform depth bins in LSS
  waste capacity on far distances. Log spacing concentrates bins in the near
  range (0–30 m) where most obstacles sit. Solo: marginal.

**None of these moved the needle solo.** My read at the time, which I still
think is right: I'd hit the label noise ceiling. The GT itself is noisy —
near-stationary frames sometimes have IoU < 1.0 between consecutive frames
that should be identical. There's only so much signal to extract before
you're fitting label noise.

But the variations had **uncorrelated errors**, which is exactly what makes
an ensemble useful. I kept them all for the final blend.

## 5. Calibration-aware conditioning

In parallel I tried v5 — a model that doesn't just see rover_id but takes the
**rig geometry directly** as input:

- A 10-dim per-camera rig vector: normalised fx/fy/cx/cy + the camera's
  position and forward axis in ego frame.
- A 12-dim global rig summary: mean/std focals, left/right baseline,
  front mid-far deltas.
- FiLM modulators on both the image feature (per camera) and the BEV feature
  (per sample).
- A specialist embedding for the top-12 test rovers, gated by a sigmoid so
  rare rovers stay near the shared trunk.

The intuition: rovers differ by *mount geometry* more than by neighbourhood.
A rover_id embedding can only memorise this through correlation; explicit
rig features tell the model "this camera is mounted 30 cm lower and points
3° down", which is the geometric fact that actually matters.

This helped most on rovers that dominate the test set but are rare in train.

## 6. Final ensemble

The goal in the final week was **robustness**, not squeezing the last point.
The blend:

- Several DINOv2-LSS variants (v6, v61 BEV attention, v62 2× resolution,
  v63 log depth) — same backbone, different post-image paths.
- RTMDet-CSPNeXt + LSS — different backbone family.
- ConvNeXtV2 + LSS — strongest single model.
- (Some early v3-augs runs in earlier submissions.)

Predictions are sigmoid-averaged. Ensemble weights are searched on the
test-matched local val. The weight search notebooks are in
[notebooks/ensemble/](notebooks/ensemble/).

I also explored a small legal data-overlap exploit
([scripts/exploit_leak.py](scripts/exploit_leak.py)): a handful of test
frames have a near-identical (frontal MAE < 0.1) twin in train within the
same `(rover, ride_date)` group. Copying the matching train GT into those
test predictions is allowed under the rules. Worth ~0.005–0.01 IoU.

## 7. The threshold call (lessons learned)

When you average sigmoid probabilities and then threshold, the optimal
threshold sits somewhere between "what each individual model would pick" and
"what the average ensemble sees". With 6+ models averaged, the probability
distribution becomes much smoother — pixels that *any* model thinks are
suspicious push the average up.

My local test-matched val said optimal threshold ≈ **0.54**. Public
leaderboard sweep said **0.73**. That's a big gap. I trusted public.

**This was probably my one real mistake.** Private dropped ~0.02 from public,
which is large for a stable ensemble. I think public had a slightly different
balance of easy/hard frames than private + local val, and the high threshold
overfitted to the *public* test set's prior. My local val was, in retrospect,
a better approximation of *full* test (public + private). The right move
would have been to take the geometric mean of the two thresholds (~0.63) or
just trust local at 0.54.

Lessons for next time:

- **Validation set design is half the problem.** Build it carefully and then
  *trust* it, even when public says something else.
- **Treat the public leaderboard as a single noisy submission**, not as ground
  truth. With ~10–20 successful submissions, you have 10–20 measurements of
  noisy public score and 1 measurement of carefully-designed local val.
- When local and public disagree, **average the implied parameter**, don't
  pick the larger sample size.

## 8. What's in this repo

See [README.md](README.md) for the full stage-by-stage layout. The
ensemble-relevant pieces of code live in:

- [src/data.py](src/data.py), [src/splits.py](src/splits.py) — dataset and
  rover-matched val split.
- [src/models/v4.py](src/models/v4.py), [src/models/v5.py](src/models/v5.py)
  — rover embedding and FiLM models.
- [scripts/smart_dedup.py](scripts/smart_dedup.py) — near-stationary dedup.
- [scripts/exploit_leak.py](scripts/exploit_leak.py) — train/test overlap
  exploit.
- [notebooks/stage_v6_dinov2_lss/](notebooks/stage_v6_dinov2_lss/) — DINOv2 +
  LSS variants (the workhorses of the final blend).
- [notebooks/stage_v8_convnextv2/](notebooks/stage_v8_convnextv2/) —
  ConvNeXtV2 + FCMAE.
- [notebooks/ensemble/](notebooks/ensemble/) — blend weight search and
  submission packing.

## 9. References

- *DINOv2: Learning Robust Visual Features without Supervision*, Oquab et
  al., arXiv:2304.07193 — the unlocking paper, especially the linear-probe
  depth results.
- *Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?*, Harley
  et al., arXiv:2206.07959 — the post-image path baseline.
- *Lift, Splat, Shoot*, Philion & Fidler, ECCV 2020 — the learned depth
  view-transform used in v6.
- *ConvNeXt V2*, Woo et al., CVPR 2023 — the FCMAE pretraining used in v8.
- *The Lovász-Softmax loss*, Berman et al., CVPR 2018 — direct IoU surrogate.
