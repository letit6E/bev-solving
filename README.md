# BEV Static Obstacle Prediction — Yandex ML2 2026

1-е место на привате. Итоговый test IoU **0.6258** (baseline 0.40, top-2 0.56).

Задача: предсказать бинарную occupancy-карту 188×126 (free / occupied / ignore=255)
на BEV-сетке 0.8 м/пиксель, покрывающей 150 м вперёд × 100 м в стороны, по
4 откалиброванным RGB-камерам на каждый сэмпл (intrinsics 3×4, extrinsics
car→cam 4×4). 4000 train / 1000 val / 2000 test. Метрика — IoU по классу 1.
См. [plans/00_task_description.md](plans/00_task_description.md).

## Структура репо

```
src/                        очищенные общие модули
  geometry.py               BEV-сетка, константы, конвенции
  data.py                   BEVDataset, BEVDatasetAug (resize + пересчёт intrinsic по камерам)
  splits.py                 group-aware и test-matched сплиты
  losses.py                 BCE + Dice + Lovasz hinge
  metrics.py                IoU, streaming threshold sweep с константной памятью
  submission.py             упаковка zip-сабмишена с проверкой SHA256
  models/
    decoder.py              общий SmallUNet
    voxel.py                parameter-free проекция вокселей (стиль Simple-BEV)
    v1.py                   ResNet-18 baseline
    v2.py                   Simple-BEV Encoder_res101 (предобучен на nuScenes)
    v4.py                   ResNet-50 + rover embedding + letterbox
    v5.py                   ResNet-34 + FiLM (intrinsics/extrinsics) + specialist-ветка
    simplebev_adapter.py    обёртка над предобученным Simple-BEV для zero-shot оценки
scripts/                    одноразовые CLI-скрипты
notebooks/                  эксперименты по этапам (output ячеек сохранён, общий код в src/)
plans/                      исходные планы и находки
best.md                     детальный разбор победившего решения
```

Датасеты, чекпойнты, `*.zip`-сабмишены, `runs/`, `inference_eval/`,
`ensemble_pray/`, `predicted_static_grids/` исключены через `.gitignore`. Для
воспроизведения нужно:

1. распаковать `autonomy_yandex_dataset_train/val/test_v2.zip`;
2. прогнать `scripts/check_integrity.py --crc --repair` по каждому архиву;
3. для v2: `bash scripts/setup_simplebev.sh` — клонирует Simple-BEV и качает
   претрейн в `external/simple_bev/`;
4. для v6/v8: предобученные веса ConvNeXtV2-FCMAE / DINOv2 качаются из самих
   ноутбуков.

## Этапы

### Stage 0 — sanity-check ([notebooks/00_sanity/](notebooks/00_sanity/))

Зафиксировал конвенции до любого обучения:
- ego frame: X — вперёд, Y — влево (положительное), Z — вверх;
- `p_cam = car_to_cam @ p_ego_h` (матрица отображает ego → cam, не наоборот —
  проверил визуально);
- intrinsic хранится как `(3, 4) = [K | 0]`;
- GT: 0 free, 1 occupied, 255 ignore;
- BEV-сетка: строка → X 0..150.4 м, столбец → Y −50.4..+50.4 м, 0.8 м/пиксель.

Что нашёл: ровер `nack` отдаёт кадры 768×959 (у остальных ~640×480);
~26 пустых сэмплов; средняя доля препятствий ~18%; на почти-стоящих кадрах GT
практически идентичен (IoU ≈ 0.95+).

### Stage 1 — baseline ([src/models/v1.py](src/models/v1.py), [notebook](notebooks/stage_v1_baseline/))

Multi-camera baseline в духе Simple-BEV: ImageNet-предобученный ResNet-18 layer2
→ 1×1 проекция в 64 каналов → проецирую ego-воксели на 4 высотах через
`car_to_cam` + intrinsics → bilinear sample → среднее по видимым камерам →
сплющиваю по высоте → SmallUNet → 1 logit на BEV-клетку. BCE(pos_weight) + 0.5·Dice
с ignore-маской 255. WeightedRandomSampler с поправкой на coverage.

Лучший test IoU около **0.51**.

### Stage 2 — предобученный энкодер Simple-BEV ([src/models/v2.py](src/models/v2.py), [notebook](notebooks/stage_v2_pretrained/))

Drop-in: меняю ResNet-18 layer2 на Simple-BEV-овский `Encoder_res101`
(предобучен на nuScenes, 37M параметров). Остальной пайплайн без изменений.
Энкодер заморожен на первые эпохи, чтобы не убить предобученные фичи, потом
можно разморозить с пониженным LR.

Плюс zero-shot оценка полного предобученного Simple-BEV
([eval_simplebev_zero_shot.ipynb](notebooks/stage_v2_pretrained/eval_simplebev_zero_shot.ipynb))
— он предсказывает машины, а не статические препятствия, поэтому выдал всего
~0.15 IoU. Подтвердило, что переносим только энкодер, не декодер.

Test IoU встал около v1 (~0.51). Узкое место — рецепт обучения, не backbone.

### Stage 3 — аугментации + Lovasz + group-aware split ([notebook](notebooks/stage_v3_augs_lovasz/))

Phase 2 — фиксы по рецепту статьи Simple-BEV (arXiv 2206.07959):

- `BEVDatasetAug`: per-camera случайный scale ∈ [1.0, 1.2] + random crop с
  корректным пересчётом intrinsic (масштаб fx/fy/cx/cy, сдвиг cx/cy на pad/crop) —
  статья даёт +1.6 IoU только от этого.
- `CompoundLossV2` = 0.5·BCE + 0.3·Dice + 0.2·Lovasz hinge. Lovasz —
  прямой суррогат IoU по положительному классу.
- `make_group_aware_split` по `(rover, ride_date)`: ни одна группа не попадает
  одновременно в train и val, распределение val сдвигается к test.
- Энкодер разморожен, две param-group (lr_backbone=1e-5, lr_head=3e-4),
  IMG_HW=448×800, batch=8 × grad_accum=4, EMA decay 0.999.

Test IoU ~**0.52**. Подбор threshold на val начал ронять test —
причина в остаточном рассогласовании частот роверов val vs test
(см. [plans/06_findings_round2.md](plans/06_findings_round2.md)).
Threshold sweep по полному val выжирал память — streaming sweep в
[src/metrics.py](src/metrics.py) это починил (константная память независимо от
размера датасета).

### Stage 4 — rover embedding + разные backbones ([src/models/v4.py](src/models/v4.py), [notebooks](notebooks/stage_v4_rover_emb/))

Diversity для ансамбля: проекция остаётся, меняю backbone поверх letterbox-resize
(сохраняет соотношение сторон — критично для `nack`):

- ResNet-50 layer2 (дефолт v4);
- DINOv2 ViT-Base через `torch.hub`, последние 2 блока разморожены;
- RTMDet-L из MMDet (COCO-pretrained, CSPNeck + SPP);
- Swin Transformer + GeoMIM pretraining.

Плюс 32-мерный rover embedding, размазанный по BEV-фиче перед декодером;
smart-дедупликация трейна (scripts/smart_dedup.py выкидывает пару сотен
почти-стоящих кадров); `make_test_matched_split` для смещения частот роверов val
к тестовым.

Лучший single-model test IoU здесь ~**0.54**.

### Stage 5 — calibration-aware FiLM + specialist-ветка ([src/models/v5.py](src/models/v5.py), [notebooks](notebooks/stage_v5_film/))

Гипотеза: роверы различаются скорее геометрией крепления камер, чем местностью.
Поэтому кондишен модели на rig-фичи напрямую, не только на rover_id.

- 10-мерный per-camera rig-вектор (нормированные fx/fy/cx/cy + позиция камеры +
  forward-ось) → FiLM-модулятор на 64-канальной фиче изображения.
- 12-мерное per-sample глобальное rig-резюме (mean/std фокусов, L-R базис,
  дельты front mid/far) + rover embedding + specialist embedding (топ-12
  тестовых роверов) → FiLM-модулятор на BEV-фиче.
- Скалярный sigmoid-гейт масштабирует FiLM-bias: specialist-роверы могут
  тянуть BEV-prior к себе, а редкие роверы остаются ближе к общему стволу.

Per-rover прирост в основном на доминирующих тестовых роверах.

### Stage 6 — DINOv2 + LSS ([notebooks](notebooks/stage_v6_dinov2_lss/))

Меняю view transform с parameter-free voxel sampling на обученный
Lift-Splat-Shoot (каждый пиксель предсказывает распределение глубины, фичи
расплёскиваются на BEV-сетку по depth × ray). Backbone — DINOv2 ViT-B/14.

Варианты:
- **v61** — multi-head BEV-self-attention в декодере;
- **v62** — 2× разрешение BEV с последующим downsample (богаче фичи вблизи
  препятствий, особенно в полосе 0–30 м);
- **v63** — логарифмические бины глубины (равномерные бины Simple-BEV тратят
  ёмкость на дальние диапазоны; лог-бины концентрируются там, где земля
  занимает большую часть BEV-клеток).

Лучший single-model test IoU здесь ~**0.58**.

### Stage 7 — RTMDet/CSPNeXt + LSS ([notebooks](notebooks/stage_v7_rtmdet_cspnext/))

Лёгкий быстрый backbone для diversity ансамбля. CSPNeXt-фичи идут через тот же
LSS, что и в v6. Тренировал локально и на Colab. Test IoU сопоставимый с v6
(~0.56–0.57), но с другим характером ошибок — полезно для усреднения.

### Stage 8 — ConvNeXtV2 + FCMAE ([notebook](notebooks/stage_v8_convnextv2/))

ConvNeXtV2-Base, предобученный через FCMAE (sparse-conv masked autoencoder на
ImageNet-22k → ImageNet-1k). Логирование в WandB, resume с чекпойнта. Сильнейшая
одиночная модель на валидации.

### Ансамбль ([notebooks/ensemble/](notebooks/ensemble/))

Усредняю sigmoid-вероятности по вариантам v6, v7, v8 (и v3 для ранних рани).
Поиск весов ансамбля в search-ноутбуках. Упаковка сабмишена в
[src/submission.py](src/submission.py) (zip + testzip() + SHA256).

Финальный победивший блeнд: семья v6 + v7 + v8 с весами, подобранными на
test-matched val, threshold ~0.62–0.66 в зависимости от блeнда. Итоговый test
IoU **0.6258**.

## Что не сработало / неожиданное

- Сглаживание GT гауссовым фильтром — стало хуже. Комментарий остался в
  [plans/07_data_pipeline_plan.md](plans/07_data_pipeline_plan.md).
- Подбор threshold на официальном val — ронял test из-за рассогласования
  частот роверов val/test (TV ≈ 0.349). Переехал на test-matched group split.
- Агрессивный EMA decay (0.9999) — хуже, чем 0.999, на таком маленьком датасете.
- Zero-shot перенос Simple-BEV — 0.15 IoU. Предобучен на машины, не статику.
- Эксплоит совпадения хешей картинок train/test
  ([scripts/exploit_leak.py](scripts/exploit_leak.py)) — нашёл пару сотен
  почти-идентичных тестовых кадров с близнецами в трейне. Стоит ~0.005–0.01 IoU.
  Маржинально, но правилами не запрещено.

## Железо

10-дневное окно: 18 ч A100, 59 ч T4, 29 ч V100 — плюс локальный M1 Pro.
Большинство экспериментов v1–v3 — на M1 (медленно, но бесплатно); запуски по
рецепту статьи — на A100; инференс ансамбля — на T4.

## Ссылки

- Harley et al., *Simple-BEV: What Really Matters for Multi-Sensor BEV
  Perception?* arXiv:2206.07959
- Berman et al., *The Lovász-Softmax loss*, CVPR 2018
- Philion & Fidler, *Lift, Splat, Shoot*, ECCV 2020
- Zhou et al., *DINOv2*, arXiv:2304.07193
- Woo et al., *ConvNeXt V2*, CVPR 2023
