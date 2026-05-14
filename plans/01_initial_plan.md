# План атаки: BEV Static Occupancy Prediction (Yandex ML2 2026)

> **TL;DR.** Baseline даёт IoU≈0.4 потому что использует **только 1 камеру из 4** и игнорирует **всю калибровку**. Топ-1 (0.56) почти наверняка делает нормальный multi-camera lift в BEV. Реалистичный путь к 0.55+ — Simple-BEV или CVT с pretrained nuScenes весами + BCE+Dice+Lovasz + аугментации в BEV-space + ансамбль + TTA. Запасной мощный план — pseudo-LiDAR через Depth-Anything-V2 + UNet поверх грида.

---

## 1. Что есть, что не так в baseline

### 1.1 Постановка
- **Вход**: 4 RGB камеры
  - `/camera/inner/frontal/middle` — фронтальная (центр)
  - `/camera/inner/frontal/far` — фронтальная (телевик/дальняя)
  - `/side/left/forward`, `/side/right/forward` — боковые
- **Калибровка**: 4×(intrinsic 3×3 + car_to_cam 4×4)
- **Выход**: BEV grid `(1, 188, 126)` int32, `0=free`, `1=occupied`, `255/-1=ignore`. Один пиксель ≈ 0.8×0.8 м, область ≈ 150×100 м (188×0.8 ≈ 150 м, 126×0.8 ≈ 100 м)
- **Метрика**: IoU по классу `1`
- **Train/Val/Test**: 4000 / 500 / ?
- **Бюджет компьюта**: M1 Pro (для отладки), удалённо T4 59ч, V100 29ч, A100 18ч

### 1.2 Что делает baseline (и почему 0.4)

В [baseline_v4.ipynb](baseline_v4.ipynb):
```python
# train loop
images, _, _, gt = batch
image = images[0].to(device)  # ← БЕРЁТ ТОЛЬКО ПЕРВУЮ КАМЕРУ
gt = gt[0].to(device)
loss = MSELoss()(preds, gt)   # ← MSE для бинарного таргета
```

| Слабость | Почему критично | Что делать |
|---|---|---|
| Использует только 1 камеру из 4 | Боковые видят боковые препятствия (отбойники, стены), `far` нужен для дальней области карты | **Multi-camera fusion (главный буст!)** |
| Игнорирует intrinsic / car_to_cam | Без них модель не знает геометрию сцены — учится "среднему" представлению | **Geometric BEV lift (LSS / CVT / Simple-BEV)** |
| MSE loss | Бинарная задача, дисбаланс классов, метрика — IoU | **BCE+Dice+Lovasz** |
| lr=0.01 + Adam | Слишком высокий для свёрточной сетки | **lr=1e-4..3e-4 + cosine + warmup** |
| Encoder → AvgPool(1,1) → Linear → Upsample | Полностью теряет пространственную структуру | **U-Net / FPN декодер** |
| Без аугментаций | Переобучение на 4000 семплов | **Color/blur + BEV-space augs** |
| Resize до 256×512, без crop | Понижает разрешение, теряет мелкие препятствия | **512×1024 или 384×768 + правильный пересчёт intrinsic** |
| 5 эпох, без шедулера | Недообучение | **30-60 эпох + cosine LR + EMA** |
| Без ensemble / TTA | Оставляет 1-3 IoU на столе | **TTA (hflip), ensemble разных backbones** |

**Вывод**: только то, что baseline использует одну камеру и MSE — почти наверняка стоит ему 10-15 IoU. Аккуратный baseline на той же ResNet-50, но с правильной архитектурой и лоссом, должен давать 0.50+.

---

## 2. Литературный ландшафт: что использовать

### 2.1 Семейства подходов

| Подход | Идея | Сильные стороны | Слабые стороны |
|---|---|---|---|
| **LSS** (Lift-Splat-Shoot) | Каждый пиксель → распределение по depth → unproject в frustum → splat в BEV | Простая, легкая (~14M), MIT, есть pretrained | Дискретизация depth, чувствительна к калибровке |
| **CVT** (Cross-View Transformer) | BEV-queries учат attention к image features через camera-aware pos.embeddings | Очень лёгкая (~5M), single-GPU friendly, 36 IoU vehicle | Чуть ниже потолок |
| **Simple-BEV** | Билинейная проекция фичей в 3D voxel grid через extrinsic, без depth prediction | 47.6 IoU vehicle на nuScenes, идейно проста, есть HF | ResNet-101, тяжелее в трейне |
| **PointBeV** | Sparse BEV queries, в 1/10 точек тренируется | 47.58 IoU, минимум памяти, влезает в 1×A100 | Чуть сложнее в инфере |
| **BEVFormer** | Spatial+Temporal cross-attn | SOTA-класс с темпоралью | Тяжёлая (24 ep × 8 A100); на одном GPU только tiny |
| **TaDe** | BEV autoencoder + RGB→BEV alignment | **65.9 IoU vehicle** — SOTA single-frame | 80 GPU-hours, риск не сойтись |
| **Pseudo-LiDAR (Depth Anything V2)** | DA-V2 metric depth → unproject → BEV grid → small UNet | Не нужно тренировать BEV энкодер с нуля, легко интерпретируется | Зависит от качества depth, "вытягивание" вертикальных объектов |
| **IPM + UNet** (Cam2BEV) | Inverse perspective mapping каждой камеры → склейка → UNet | Тривиально | Flat-ground assumption, для obstacles **плохо** |

### 2.2 Кого точно стоит смотреть в первую очередь

**Топ-3 для нашего бюджета** (отсортировано по value/risk):

1. **CVT** — https://github.com/bradyz/cross_view_transformers
   - Лёгкая, single-GPU support документирован
   - На A100 18ч можно прогнать 30-50 эпох с нуля
   - 4 камеры → легко адаптируется (просто меняем число view embeddings)
   - **Почему первой**: самая быстрая итерация, лучшая отладка

2. **Simple-BEV** — https://github.com/aharley/simple_bev (+ [HF Qualcomm](https://huggingface.co/qualcomm/Simple-Bev))
   - 47.6 IoU vehicle на nuScenes single-frame — топ среди camera-only
   - Pretrained веса есть
   - На A100 fine-tune от nuScenes-checkpoint реально за 12-15ч
   - **Почему второй**: выше потолок, есть чекпоинт

3. **PointBeV** — https://github.com/valeoai/PointBeV
   - Pretrained EfficientNet-B4 чекпоинты доступны (38.7-47.6 IoU)
   - Sparse design = низкое потребление памяти
   - **Почему третий**: лучший memory profile + готовые веса

**Резерв на отдельный трек** (запускать параллельно):
4. **Pseudo-LiDAR через [Depth-Anything-V2-Metric-Outdoor-Base](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf)** — независимый baseline, ансамблируется с обучаемыми моделями. Можно довести до 0.40-0.50 IoU вообще без обучения BEV-сетки.

### 2.3 Что **не** брать (под наш бюджет)
- **BEVFormer-base / V2** — 24 ep × 8 A100, single-GPU не успеет
- **BEVFusion** — рассчитан на LiDAR, без него теряет качество
- **MapTR/MapTRv2** — таргет вектор, не raster
- **HDMapNet** — таргет lanes/dividers, не obstacles
- **UniAD / FB-BEV** — слишком тяжёлые
- **Marigold** — affine-invariant depth, не metric → не подходит для unproject

---

## 3. План по дням / ставкам

Я разделю на 3 уровня агрессивности. Можно идти по нарастающей или взять сразу средний путь.

### 3.1 День 1-2: Solid Multi-Camera Baseline (целевой IoU 0.45-0.50)

**Минимальные изменения относительно baseline для огромного скачка:**

1. **Использовать все 4 камеры**:
   - Энкодер пропускаем через **shared backbone** (ResNet-50 / EfficientNet-B0)
   - Получаем 4 feature maps; объединяем через простой concat в BEV-плоскости через билинейную проекцию (Simple-BEV style)
   
2. **Правильная архитектура**:
   - Backbone: `timm.create_model('tf_efficientnet_b0_ns', pretrained=True, features_only=True)` — лёгкий, проверенный, ImageNet pretrained
   - BEV Lift: Simple-BEV-style projection из (B, N=4, C, H, W) в (B, C, Z=1, X=188, Y=126) через сэмплинг по unprojected ego-frame voxels
   - Decoder: маленький FPN/UNet с Upsample+Conv → 1 logit channel

3. **Loss**: `BCEWithLogitsLoss(pos_weight=K) + 0.5*DiceLoss`, где `K = N_neg/N_pos` посчитанный на train. Маска `255` через явный multiplicative ignore-mask или `ignore_index` (BCE этого не имеет, делать руками).

4. **Гиперпараметры**:
   - `lr=3e-4`, AdamW, weight_decay=1e-4
   - Cosine schedule с warmup (5% steps)
   - `batch_size=8` (если хватит памяти на 4 камеры × 384x768)
   - 30 эпох, EMA decay 0.999

5. **Аугментации (image-space, БЕЗ ломания геометрии)**:
   - Color jitter, gaussian blur, random gamma
   - **НЕЛЬЗЯ**: random rotate/perspective/crop **per-camera** без пересчёта intrinsic
   - **МОЖНО**: одинаковый horizontal flip на ВСЕ камеры **+ свап left/right камер + flip BEV target по оси X**

6. **Вход**: ресайз до 384×768 (компромисс между качеством и памятью; пересчитать intrinsic пропорционально!)

7. **Ожидаемая метрика**: 0.45-0.50.

### 3.2 День 3-4: Pretrained + сильный backbone (целевой IoU 0.50-0.55)

**Добавляем поверх:**

8. **Pretrained веса nuScenes**:
   - Скачиваем Simple-BEV camera-only checkpoint: `bash get_rgb_model.sh`
   - Адаптируем: меняем число камер 6→4, меняем размер BEV grid 200×200→188×126
   - Загружаем веса backbone и BEV head (head — частично, по форме)
   - Fine-tune 5-10 эпох с малым LR (3e-5 на backbone, 3e-4 на голове)

9. **Замена backbone на DINOv2-S + LoRA**:
   - https://github.com/mrabiabrn/robustbev — есть готовая обвязка
   - DINOv2-Small (~22M) + LoRA rank=16 на attention
   - На 4000 семплов даёт устойчивость к domain shift и выигрыш на validation
   - Альтернатива: ConvNeXt-Tiny через timm

10. **Lovasz loss как добавочный**: `total = 0.5*BCE + 0.3*Dice + 0.2*Lovasz`. Lovasz прямо оптимизирует IoU — самый прямой путь для нашей метрики.

11. **BEV-space аугментации**:
    - После lift в BEV: random horizontal flip (+ swap left/right cams + flip target)
    - Random rotation вокруг ego на ±15° (применяется к BEV features ровно так же как к target)
    - Random scale 0.9-1.1
    - **Camera dropout**: с p=0.1 заменять одну случайную камеру нулями — учим устойчивости к отказам

12. **Class balancing через sampler**: WeightedRandomSampler, веса по доле occupied пикселей в семпле. Семплы где много препятствий — чаще.

13. **Ожидаемая метрика**: 0.50-0.55.

### 3.3 День 5-6: Финал, ансамбль, TTA (целевой IoU 0.55-0.58)

14. **Ансамбль из 3 моделей** разных backbones:
    - CVT с EfficientNet-B0
    - Simple-BEV с DINOv2-S
    - LSS с ResNet-50
    
    Усредняем sigmoid-логиты, потом threshold tuning.

15. **TTA**:
    - **Horizontal flip**: горизонтальный flip всех 4 камер + swap left↔right + flip BEV-output обратно по оси X. Усреднить с обычным предом.
    - **Camera resolution sweep**: 0.9× и 1.1× resize (с пересчётом intrinsic!) — но это рискованно если не проверять
    - **Multi-crop**: маленький crop в дальние камеры
    - Предупреждение: `ttach` сам не пересчитывает intrinsic, делать руками!

16. **Threshold optimization** на validation:
    ```python
    best_iou, best_t = 0, 0.5
    for t in np.linspace(0.2, 0.8, 31):
        pred = (logits.sigmoid() > t).astype(int)
        iou = compute_iou(pred, gt, ignore=255)
        if iou > best_iou: best_iou, best_t = iou, t
    ```
    Часто оптимальный threshold не 0.5, может быть 0.35-0.45 при дисбалансе. Может дать +1-3 IoU.

17. **EMA + SWA**:
    - Во время трейна держим EMA копию весов (decay 0.999), при инференсе используем её
    - Последние 20% эпох — SWA (Stochastic Weight Averaging)

18. **Pseudo-labels self-training** (если успеваем):
    - Обучить strong model → предсказать на test (если разрешено правилами; нужно проверить!) → добавить уверенные пиксели в train → re-train
    - Очень мощный буст обычно (+1-3 IoU)
    - **Внимание**: правила хакатона могут это запрещать. Проверь.

19. **Ожидаемая метрика**: 0.55-0.58.

---

## 4. Критические технические детали

### 4.1 Геометрический lift из image features в BEV

Это сердце всего подхода. У нас есть:
- `intrinsic[i]` — 3×3 матрица камеры `i`
- `car_to_cam[i]` — 4×4, преобразование точки из ego-frame в camera-frame для камеры `i`

**Цель**: для каждой ego-frame BEV-клетки `(x, y, z)` найти соответствие в каждой image-feature-map.

**Правильное направление преобразования**:
```python
# Для каждой BEV клетки p_ego = (X, Y, Z, 1) ∈ R^4
# Преобразуем в camera-frame:
p_cam = car_to_cam[i] @ p_ego  # ∈ R^4
# Проекция в image:
uv_homog = intrinsic[i] @ p_cam[:3]  # (u*z, v*z, z)
u, v = uv_homog[0]/uv_homog[2], uv_homog[1]/uv_homog[2]
# Дальше — bilinear sampling из image features
```

**Замечания**:
- BEV grid в ego-frame: `X ∈ [0, 150]`, `Y ∈ [-50, 50]`, `Z` фиксированный (например, 0..3 м поднятый по высоте над землёй)
- Один пиксель = 0.8 м, значит:
  - `X = 0..150` → 188 шагов
  - `Y = -50..50` → 126 шагов
  - **Внимание**: проверь ориентацию (вид сверху, инверсия осей в визуализации `extent=[-50, 50, 0, 150]` в baseline)
- В Simple-BEV они "сэмплят" фичи не в BEV-клетке, а в неявном **3D voxel столбике** (3-8 высот) и потом коллапсируют по Z
- Только **видимые** проекции (где `0 ≤ u < W`, `0 ≤ v < H`, `p_cam[2] > 0`) — остальные обнуляем/маскируем

### 4.2 Resize изображений и intrinsic

Если ресайзим (H, W) → (H', W') с фактором `s = H'/H`:
```python
intrinsic_resized = intrinsic.copy()
intrinsic_resized[0, :] *= W' / W  # fx, cx
intrinsic_resized[1, :] *= H' / H  # fy, cy
```
Если делаем **center crop** до (H', W') с offset (dy, dx):
```python
intrinsic_cropped = intrinsic.copy()
intrinsic_cropped[0, 2] -= dx  # cx
intrinsic_cropped[1, 2] -= dy  # cy
```
**Важно**: baseline ресайзит до 256×512 без пересчёта intrinsic. Это нормально для baseline (он intrinsic не использует), но **обязательно для нас**.

### 4.3 Маска ignore (255)

В нашем GT часть пикселей помечена как 255 (или -1). При обучении мы должны **полностью игнорировать** их в loss. Подходы:

```python
# Вариант 1: явная маска
mask = (gt != 255)  # bool tensor
loss = bce_loss(pred[mask], gt[mask].float())

# Вариант 2: через reduction='none' и mean по unmasked
loss_per_pixel = F.binary_cross_entropy_with_logits(pred, gt.float(), reduction='none')
loss = (loss_per_pixel * mask.float()).sum() / mask.sum().clamp(min=1)
```

**При расчёте Dice / Lovasz** тоже нужно **исключать** ignore-пиксели. Проще всего занулить и pred, и gt в маскированных местах перед loss.

### 4.4 Расчёт метрики IoU

```python
def iou_binary(pred, gt, ignore=255):
    valid = gt != ignore
    pred = pred[valid]
    gt = gt[valid]
    inter = ((pred == 1) & (gt == 1)).sum()
    union = ((pred == 1) | (gt == 1)).sum()
    return inter / max(union, 1)
```
Должно совпадать с метрикой системы (IoU только по классу 1).

### 4.5 `pos_weight` для BCE

Считаем долю occupied пикселей по train (ignore-маска **не считается**):
```python
pos = 0
neg = 0
for _, _, _, gt in train_dataset:
    g = gt[0]  # (1, 188, 126)
    valid = g != 255
    pos += (g[valid] == 1).sum()
    neg += (g[valid] == 0).sum()
pos_weight = neg / pos  # обычно 5-20
```
Подставляем в `BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))`. Это сразу +5-10 IoU над baseline.

---

## 5. Альтернативный трек: Pseudo-LiDAR через Depth-Anything-V2

Если основной трек завис или хочется параллельный strong baseline.

**Pipeline:**
```
Каждая RGB камера i:
  → DA-V2-Metric-Outdoor-Base (HF) → depth map (H, W) в метрах
  → unproject через intrinsic[i]^-1 → 3D points (cam frame)
  → car_to_cam[i]^-1 (в ego frame)
  → фильтр: 0 ≤ X ≤ 150, -50 ≤ Y ≤ 50, 0.3 ≤ Z ≤ 3
  → bin в BEV grid 188×126 → "raw occupancy" (counts/binary)

4 BEV-grid (по одной на камеру) → max/sum → "raw fused BEV"
  → 2D U-Net (EffNet-B0 backbone, ~5M params) → final binary mask
```

**Преимущества:**
- DA-V2 даёт чёткие edges → препятствия аккуратно локализованы
- Тренировка только маленькой 2D UNet поверх — 2-4 часа на A100
- Полностью независимый сигнал — отлично ансамблится с CVT/Simple-BEV
- Inference: можно прекомпьютить depth для всех train/val заранее на T4

**Реалистичная метрика:** 0.40-0.50 standalone, в ансамбле даёт +0.5-1.5 IoU поверх основной модели.

**Ссылки:**
- HF: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf
- Альтернатива: https://huggingface.co/zachL1/Metric3D (KITTI δ1 = 0.985, лучше для open-road)

---

## 6. Архитектурный набросок (стартовый шаблон)

Ниже — псевдокод того, что писать сразу для солидного multi-camera baseline. Этим уже можно бить 0.45-0.50.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import timm

class MultiCamBEV(nn.Module):
    def __init__(self, n_cameras=4, bev_h=188, bev_w=126,
                 x_range=(0., 150.), y_range=(-50., 50.),
                 z_levels=(0.3, 1.0, 2.0, 3.0)):
        super().__init__()
        # 1. Shared image encoder
        self.encoder = timm.create_model(
            'tf_efficientnet_b0_ns', pretrained=True, features_only=True,
            out_indices=(2, 3, 4)  # (1/8, 1/16, 1/32) feature maps
        )
        # 2. Reduce channels
        self.reduce = nn.ModuleList([
            nn.Conv2d(c, 64, 1) for c in self.encoder.feature_info.channels()
        ])
        # 3. BEV grid params
        self.bev_h, self.bev_w = bev_h, bev_w
        self.x_range, self.y_range = x_range, y_range
        self.z_levels = z_levels
        self.register_buffer("ego_voxels", self._make_ego_voxels())
        # 4. BEV decoder (small UNet)
        self.bev_decoder = SmallUNet(in_ch=64*len(z_levels), out_ch=1)

    def _make_ego_voxels(self):
        # Returns (Z, H, W, 4) in homogeneous ego coords
        xs = torch.linspace(*self.x_range, self.bev_h)
        ys = torch.linspace(*self.y_range, self.bev_w)
        zs = torch.tensor(self.z_levels)
        Z, X, Y = torch.meshgrid(zs, xs, ys, indexing='ij')
        ones = torch.ones_like(X)
        return torch.stack([X, Y, Z, ones], dim=-1)  # (Z, H, W, 4)

    def forward(self, images, intrinsics, car2cams):
        # images: (B, N=4, 3, Hi, Wi)
        # intrinsics: (B, N, 3, 3)
        # car2cams: (B, N, 4, 4)
        B, N = images.shape[:2]
        feats = self.encoder(images.flatten(0, 1))  # process all cams together
        feat = self.reduce[-1](feats[-1])           # use deepest scale
        _, C, Hf, Wf = feat.shape
        feat = feat.view(B, N, C, Hf, Wf)

        # 5. Lift: для каждого ego voxel найти соответствие в каждой камере
        bev_features = []
        for z_idx in range(len(self.z_levels)):
            voxels = self.ego_voxels[z_idx]  # (H, W, 4)
            voxels = voxels.view(-1, 4)      # (H*W, 4)
            # Project per camera
            cam_features = []
            for cam_idx in range(N):
                p_cam = car2cams[:, cam_idx] @ voxels.T.unsqueeze(0)  # (B, 4, H*W)
                p_cam = p_cam[:, :3]  # (B, 3, H*W)
                uv = intrinsics[:, cam_idx] @ p_cam  # (B, 3, H*W)
                uv_norm = uv[:, :2] / uv[:, 2:].clamp(min=1e-3)
                # Normalize to [-1, 1] для grid_sample
                uv_norm[:, 0] = 2 * uv_norm[:, 0] / (Wf * 8) - 1  # 8 = stride
                uv_norm[:, 1] = 2 * uv_norm[:, 1] / (Hf * 8) - 1
                # Mask invalid
                valid = (uv[:, 2] > 0) & (uv_norm.abs().max(1)[0] <= 1)
                grid = uv_norm.permute(0, 2, 1).view(B, 1, -1, 2)
                sampled = F.grid_sample(feat[:, cam_idx], grid,
                                       mode='bilinear', align_corners=False)
                # (B, C, 1, H*W)
                sampled = sampled.squeeze(2) * valid.unsqueeze(1).float()
                cam_features.append(sampled)
            # Aggregate over cameras (mean / max)
            agg = torch.stack(cam_features, dim=0).mean(0)
            agg = agg.view(B, C, self.bev_h, self.bev_w)
            bev_features.append(agg)
        bev = torch.cat(bev_features, dim=1)  # (B, C*Z, H_bev, W_bev)
        return self.bev_decoder(bev)


class SmallUNet(nn.Module):
    # ~3M params, 4 down-up blocks
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.enc1 = self._block(in_ch, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.bottleneck = self._block(256, 256)
        self.dec3 = self._block(256+256, 128)
        self.dec2 = self._block(128+128, 64)
        self.dec1 = self._block(64+64, 64)
        self.out = nn.Conv2d(64, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    @staticmethod
    def _block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1)
```

Это **рабочий каркас**, не идеальный. Берём — и доводим под ваши данные.

---

## 7. Loss и обучение (детально)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class CompoundLoss(nn.Module):
    def __init__(self, pos_weight=10.0, w_bce=0.5, w_dice=0.3, w_lovasz=0.2):
        super().__init__()
        self.pos_weight = pos_weight
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_lovasz = w_lovasz

    def forward(self, logits, gt, ignore_value=255):
        # logits: (B, 1, H, W), gt: (B, 1, H, W) int with 0/1/255
        valid = (gt != ignore_value)
        gt_f = gt.float() * valid.float()  # zero out ignore
        # BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, gt_f,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            reduction='none')
        bce = (bce * valid.float()).sum() / valid.float().sum().clamp(min=1)
        # Dice
        probs = logits.sigmoid() * valid.float()
        gt_f = gt_f * valid.float()
        inter = (probs * gt_f).sum()
        denom = probs.sum() + gt_f.sum()
        dice = 1 - (2 * inter + 1) / (denom + 1)
        # Lovasz (использовать готовую реализацию)
        lov = lovasz_hinge_flat(
            logits[valid].view(-1), gt_f[valid].view(-1).float()
        )
        return self.w_bce * bce + self.w_dice * dice + self.w_lovasz * lov
```

Для `lovasz_hinge_flat` — взять из https://github.com/Hsuxu/Loss_ToolBox-PyTorch или smp.

**Train loop checklist**:
- AdamW, lr=3e-4 backbone разморожен / 3e-5 если backbone frozen на старте
- Cosine annealing с warmup ~5% steps
- Gradient clipping `clip_grad_norm_(model.parameters(), 1.0)`
- AMP (`torch.cuda.amp.autocast` + `GradScaler`) — критично для скорости на A100
- EMA модели через `torch_ema` или ручной декей 0.999
- Validate каждые N эпох, сохранять best по IoU
- Сохранять обе версии: best (по IoU) и last (для возобновления)

---

## 8. Важные потенциальные подводные камни

| Камень | Почему опасен | Как проверить / обойти |
|---|---|---|
| Ориентация BEV grid | `extent=[-50, 50, 0, 150]` в baseline означает X — это лево/право, Y — вперёд/назад. **Не путать с моими ego coords.** | Визуализировать: преобразовать GT и убедиться что препятствия в BEV соответствуют тому, что видно в камерах |
| Inversion осей | `ax2.invert_xaxis()` в визуализации намекает что Y растёт вправо, но рисуется наоборот | Аккуратно протестировать на одном семпле: GT в координатах (row, col) это ... ? |
| Resolution intrinsic | Если ресайзишь — обязательно пересчитай fx/fy/cx/cy | Логи: на одном кадре для одной камеры визуализировать reprojection точек ego frame на изображение |
| Знак Y в car_to_cam | "вперёд, вправо, вверх" — стандарт автоматизаторов, но **проверить** | На фронтальной камере точка (X=10, Y=0, Z=0) должна быть около центра кадра |
| Формат car_to_cam (cam_to_car или car_to_cam?) | Имя в csv `car_to_cam` — но матрицы могут быть transposed/inverted | Проверить: `np.load(...)`, посмотреть translation вектор `T[:3, 3]` — это позиция камеры в car frame **или** позиция начала car frame в camera frame? |
| Маска 255 — может быть -1 | Условие `task_description.md` упоминает оба | Сделать `gt[gt < 0] = 255` или ignore = `(gt != 0) & (gt != 1)` |
| Размер выхода 188×126 vs 188×128 | Нужно **точно** соответствовать | Resize bilinear → threshold → cast int32 |
| Тип int32 | `int32` в submission | `preds.astype(np.int32)` |
| Shape (1, 188, 126) | Не (188, 126) | `preds.reshape(1, 188, 126)` |
| Submission zip | Только `info.csv` + `predicted_static_grids/` без других файлов | Перенести `images/`, `matrices/` наружу перед `make_archive` (как в baseline cell) |

**Перед первой посылкой обязательно прогнать**:
- Дамп одного предсказания → np.unique → должны быть только {0, 1}
- Shape (1, 188, 126), dtype int32
- Все имена файлов соответствуют `info.csv`
- В zip только `info.csv` и `predicted_static_grids/`

---

## 9. План использования железа

| GPU | Время | Что делать |
|---|---|---|
| **M1 Pro локально** | unlimited | Отладка кода, прогон на 50-100 семплах, проверка геометрии, визуализации, sanity-check sub-pipeline |
| **T4 (59ч)** | 59ч | (а) precompute Depth-Anything-V2 для всех картинок (1 раз, ~5-10ч); (б) лёгкие модели CVT-tiny / LSS-effb0 |
| **V100 (29ч)** | 29ч | Основной трек CVT или Simple-BEV (resnet-50 backbone), 1-2 эксперимента |
| **A100 (18ч)** | 18ч | Финальный трейн с DINOv2-S+LoRA или Simple-BEV-R50 на нормальном разрешении и большом batch |

**Стратегия**:
- Дни 1-2: всё на M1 Pro для отладки + 1 запуск на T4 для sanity check (LSS-effb0 от scratch на 4000)
- День 3: основной запуск Simple-BEV или CVT на V100 (29ч ≈ 25-30 эпох с ResNet-50)
- День 4: параллельно на T4 — precompute DA-V2 depth, второй backbone (CVT)
- День 5: финальный запуск с DINOv2 + LoRA на A100 (18ч)
- День 6: ансамбль + TTA + finalization

---

## 10. Прогноз и риски

**Реалистичные таргеты:**
| Уровень | IoU | Что нужно |
|---|---|---|
| Multi-cam baseline (вместо 1-cam baseline) | **0.45-0.50** | 1 день, очевидные фиксы |
| + правильный lift через LSS / Simple-BEV без pretrained | **0.48-0.53** | 2-3 дня |
| + pretrained nuScenes weights, fine-tune | **0.51-0.55** | 4 дня |
| + DINOv2 / ConvNeXt + аугментации + ensemble + TTA | **0.54-0.58** | 5-6 дней |
| + pseudo-labels self-training + Lovasz + threshold tuning | **0.56-0.60** | 6-7 дней + правила |

**Топ-1 0.56** в твоём хакатоне — это уровень solid Simple-BEV/CVT + ансамбль. С нашим бюджетом на A100 это достижимо.

**Главные риски**:
1. **Кривая интерпретация car_to_cam / intrinsic** → разрушает всю проекцию. **Митигейшн**: тщательный sanity-check на одном кадре с визуализацией reprojection point cloud в image plane.
2. **Memory issues на 4 камерах × 384x768 × backbone** → не влезет в V100. **Митигейшн**: gradient checkpointing, AMP, batch 4-6 вместо 8.
3. **Overfitting на 4000 семплов** → val хороший, test плохой. **Митигейшн**: CV (k-fold), сильные аугментации, EMA, early stopping.
4. **Domain gap** между nuScenes pretrained и нашим датасетом (другие камеры, другая страна, может, специфика "пункта оплаты" и "отбойников"). **Митигейшн**: full fine-tune (не только head), LoRA для устойчивости.
5. **Discretization** грид (0.8 м/пиксель) — мы можем "размазывать" тонкие препятствия. **Митигейшн**: предсказывать в higher resolution (376×252) и downsample.
6. **GT noise**: в реальных данных всегда есть шум разметки. **Митигейшн**: label smoothing, robust losses (focal).

---

## 11. Конкретные ссылки и ресурсы

### Code
- **Simple-BEV**: https://github.com/aharley/simple_bev (MIT)
- **CVT**: https://github.com/bradyz/cross_view_transformers (MIT)
- **LSS**: https://github.com/nv-tlabs/lift-splat-shoot (MIT)
- **PointBeV**: https://github.com/valeoai/PointBeV (MIT)
- **Cam2BEV (4-cam IPM)**: https://github.com/ika-rwth-aachen/Cam2BEV
- **TaDe (SOTA single-frame)**: https://github.com/happytianhao/TaDe
- **Robust DINOv2 BEV**: https://github.com/mrabiabrn/robustbev
- **BEVFormer (для tiny варианта)**: https://github.com/fundamentalvision/BEVFormer

### Веса
- LSS: Google Drive ссылка в README
- Simple-BEV: `bash get_rgb_model.sh` в репо
- HF: https://huggingface.co/qualcomm/Simple-Bev (Qualcomm export, оригинал найдёшь по ссылкам)
- PointBeV: чекпоинты в репо
- Depth-Anything-V2 metric: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf
- Metric3Dv2: https://huggingface.co/zachL1/Metric3D

### Утилиты
- **timm** (бэкбоны): https://github.com/huggingface/pytorch-image-models
- **smp** (loss + simple архитектуры): https://github.com/qubvel-org/segmentation_models.pytorch
- **Loss_Toolbox** (Lovasz, focal): https://github.com/Hsuxu/Loss_ToolBox-PyTorch
- **ttach** (TTA): https://github.com/qubvel/ttach
- **torch_ema**: https://github.com/fadel/pytorch_ema
- **albumentations** (image augs): https://albumentations.ai/

### Awesome-listings (для глубже копать)
- https://github.com/chaytonmin/Awesome-BEV-Perception-Multi-Cameras
- https://github.com/4DVLab/Vision-Centric-BEV-Perception
- https://github.com/Daniel-xsy/RoboBEV (33 модели в одном бенчмарке)

### Релевантные пейперы
- LSS: https://arxiv.org/abs/2008.05711
- CVT: https://arxiv.org/abs/2205.02833
- Simple-BEV: https://arxiv.org/abs/2206.07959
- BEVFormer: https://arxiv.org/abs/2203.17270
- PointBeV (CVPR 2024): https://arxiv.org/abs/2312.00703
- TaDe (CVPR 2024): https://arxiv.org/abs/2404.01925
- OccFeat (SSL pretraining): https://arxiv.org/abs/2404.14027
- Loss survey: https://arxiv.org/html/2312.05391v1
- Camera-view supervision boost: https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1431346/full

---

## 12. Ближайшие шаги, рекомендованный порядок

1. ✅ **Прочитал baseline** — понял слабые места.
2. **Прямо сейчас**: проверить геометрию данных. Загрузить 1 семпл, визуализировать GT BEV, спроецировать какие-нибудь точки ego frame через `car_to_cam` + `intrinsic` на каждую из 4 камер и убедиться что эти точки попадают в правдоподобные места.
3. **Потом**: написать `MultiCamBEVDataset` с правильным ресайзом + пересчётом intrinsic + ignore-mask.
4. **Потом**: написать `MultiCamBEV` модель (как в § 6) — простую, без претрейна.
5. **Потом**: запустить тренировку 2-3 эпохи на M1 Pro на 100 семплах, убедиться что лосс падает и предсказания вменяемые.
6. **Потом**: запустить полную тренировку на V100/A100 на 4000 семплах, 30 эпох. Это будет сильный baseline (0.45-0.50).
7. **Дальше** уже подключать pretrained, DINOv2, ансамбль, TTA — итеративно.

---

**Удачи! Если будут вопросы по конкретным шагам — пиши, поможем доводить до 0.6+.**
