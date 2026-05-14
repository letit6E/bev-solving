# Findings Round 2

## Главное

Сейчас главная проблема выглядит не как "архитектуры плохие", а как сочетание:

- сильного `val ↔ test` distribution shift;
- нескольких багов/несостыковок в evaluation pipeline;
- слишком слабого способа использовать rover/rig information.

Из-за этого новые архитектуры могут реально быть лучше, но локальная оценка этого не показывает.

---

## 1. Самые важные технические находки

### 1.1 `eval_after_train.ipynb` неправильно оценивает EMA

В [eval_after_train.ipynb](/Users/r-shangareev/PyProjects/shine-time/ML2_2026_Competition/eval_after_train.ipynb):

- при `WHICH_CKPT="ema"` грузится файл `ema.pt`;
- но state dict берётся из `ckpt["model"]`, а не из `ckpt["ema"]`.

То есть "EMA evaluation" фактически не использует EMA-веса.

Это затрагивает:

- основную загрузку чекпоинта;
- бонусное сравнение `best vs ema vs last`.

Следствие:

- все цифры про `ema` из этого ноутбука стоит считать ненадёжными;
- возможно, вы ещё не видели настоящую метрику EMA-модели.

### 1.2 `eval_after_train.ipynb` не поддерживает `v4`

В ноутбуке есть только:

- `MODEL_TYPE="v1"`
- `MODEL_TYPE="v2"`

Но [train_v4.ipynb](/Users/r-shangareev/PyProjects/shine-time/ML2_2026_Competition/train_v4.ipynb) в конце явно предлагает гонять `RUN_DIR='./runs/v4'` и `MODEL_TYPE='v4'`.

Сейчас это не реализовано.

Следствие:

- часть оценки `v4` могла идти не тем пайплайном или вообще не воспроизводиться корректно;
- сравнение `v3` и `v4` может быть невалидным.

### 1.3 Официальный `val` очень плохо совпадает с `test` по rover distribution

По rover frequencies:

- `TV(train, test) ≈ 0.187`
- `TV(val, test) ≈ 0.349`
- `TV(train, val) ≈ 0.384`

То есть официальный `val` распределён по роверам **значительно хуже**, чем сам `train`.

Особенно сильные расхождения:

- `orvy`: `61` в `val` против `365` в `test`
- `shelly`: `85` в `val` против `282` в `test`
- `greben`: `31` в `val` против `130` в `test`
- `soan`: `35` в `val` против `121` в `test`
- `benzon`: `5` в `val` против `58` в `test`

Следствие:

- подбор threshold по официальному `val` почти обречён;
- архитектурные сравнения по official val тоже очень шумные.

### 1.4 В official `val` есть патологический rover `nack`

Для `nack`:

- `train`: `40` samples, mean coverage `0.165`, mean positive fraction `0.392`
- `val`: `17` samples, mean coverage `0.0`
- `test`: `16` samples

То есть **все 17 val-сэмплов `nack` вообще без размеченной области**.

Следствие:

- любые выводы по `nack`, сделанные из official val, практически бесполезны;
- попытка занулить `nack` на test могла опираться на ложный локальный сигнал;
- падение с `0.531` до `0.529` после zeroing `nack` логично: train показывает, что `nack` не "пустой rover", а нормальный по статистике.

### 1.5 `rover_id` как фича полезен, но в текущем виде слишком грубый

В [bev_v4.py](/Users/r-shangareev/PyProjects/shine-time/ML2_2026_Competition/bev_v4.py):

- rover используется как embedding;
- embedding просто broadcast'ится в BEV перед decoder.

Проблема:

- у `31` rover'а меняется хотя бы `fx` между split'ами;
- есть реальные variation'ы rig geometry даже внутри одного rover name.

Следствие:

- `rover_id` помогает как coarse prior;
- но он не заменяет conditioning по реальной калибровке;
- модель может переучиться на label "rover", не научившись использовать actual rig parameters.

---

## 2. Что говорят данные про rover-specific подход

### 2.1 Очень сильный факт: в `test` нет новых rover'ов относительно `train`

Статистика:

- `train-test`: common rovers = `58`
- `only_test` = `0`

Это очень хороший сигнал.

Практический вывод:

- **per-rover specialization здесь имеет смысл**;
- test не требует generalization на unseen rover names.

### 2.2 Но rover name != rig identity

Есть отдельная тонкость:

- у многих rover'ов между split'ами плавают `fx` и другие calibration fingerprints;
- например, `31` rover имеет разные `fx` по split'ам.

Практический вывод:

- лучше думать не только про `rover-specific`, но и про `rig-specific`;
- можно строить фичу не из имени, а из calibration summary.

### 2.3 Official val содержит rover'ы, нехарактерные для test

Например:

- `kalem` есть в `val`, но его нет в `train` и `test`;
- `alvaro`, `angela` есть в `train/val`, но нет в `test`.

Практический вывод:

- они загрязняют локальную оценку;
- для model selection их влияние стоит уменьшать.

---

## 3. Почему архитектуры могут "не расти"

Моё текущее объяснение:

1. Вы меряете улучшения на валидации, которая плохо совпадает с leaderboard.
2. Часть eval-пайплайна для EMA/v4 сейчас сломана или неполна.
3. Rover signal пока внедрён слишком поздно и слишком грубо.
4. Архитектуры становятся немного лучше, но шум в оценке больше этого улучшения.

То есть plateau на `0.51-0.53` пока ещё **не доказательство**, что задача уткнулась в потолок текущего класса моделей.

---

## 4. Что я бы делал дальше

### 4.1 Первое: починить evaluation, прежде чем обучать ещё 3 модели

Минимум:

- корректно грузить `ckpt["ema"]` для EMA;
- добавить поддержку `MODEL_TYPE='v4'`;
- отдельно логировать score на:
  - official val
  - group val
  - rover-matched pseudo-test val

### 4.2 Построить новый validation split, имитирующий test

Это, по-моему, сейчас самый высокий ROI.

Идея:

- брать только rover'ы, присутствующие в test;
- веса rover'ов в val делать пропорционально test distribution;
- внутри rover'а holdout делать по `ride_date` или по timestamp blocks.

То есть нужен не просто `group split`, а **test-matched group split**.

### 4.3 Не делать отдельную модель на каждый rover сразу

Полный per-rover model zoo кажется рискованным:

- у многих rover'ов мало train samples;
- можно быстро переобучиться;
- трудно честно валидировать.

Гораздо лучше сначала один из этих вариантов:

1. shared trunk + rover/rig-conditioned FiLM
2. shared trunk + per-rover lightweight head/bias
3. global model + specialist fine-tune только для крупных rover'ов

### 4.4 Самый практичный rover hack

Я бы пробовал такой стек:

1. Обучить глобальную модель.
2. Выбрать крупные rover'ы из test mass:
   - `orvy`
   - `shelly`
   - `lerita`
   - `greben`
   - `soan`
   - `natelio`
   - `benzon`
   - `lucky`
   - `ward`
   - `greton`
3. Для rover'ов с достаточным train count сделать short fine-tune от глобальной модели.
4. На inference выбирать:
   - specialist, если rover покрыт и train count большой;
   - иначе global.

Это лучше, чем "одна полностью отдельная модель на каждого".

### 4.5 Ещё лучше: conditioning по calibration vector, а не только по имени

Предлагаемый rig vector:

- `fx, fy, cx, cy` для каждой камеры
- camera position offsets из `car_to_cam`
- baseline left-right
- высота/смещение middle/far камер

Дальше:

- прогнать через MLP;
- использовать как FiLM/gating в backbone или BEV fusion;
- можно совместить с rover embedding.

Это должно быть стабильнее, чем просто one-hot по rover.

---

## 5. Что говорит внешний ресерч

### 5.1 Viewpoint robustness для BEV действительно критична

Очень релевантная работа:

- *Towards Viewpoint Robustness in Bird's Eye View Segmentation* (ICCV 2023)
- NVIDIA page: https://research.nvidia.com/labs/toronto-ai/publication/2023_iccv_viewpoint_robustness/
- arXiv record: https://dblp.org/rec/journals/corr/abs-2309-05192.html

Главный вывод работы:

- даже небольшие изменения pitch/yaw/height/depth камеры заметно роняют BEV quality;
- rig/viewpoint shift сам по себе способен съесть большую часть IoU;
- авторы восстанавливают часть потерь за счёт viewpoint adaptation.

Это очень хорошо согласуется с тем, что вы наблюдаете по rover'ам.

### 5.2 Foundation backbones реально повышают robustness

Релевантная работа:

- *Robust Bird's Eye View Segmentation by Adapting DINOv2* (arXiv 2409.10228)
- summary/source trail: https://papers.cool/arxiv/2409.10228
- ResearchGate mirror with abstract: https://www.researchgate.net/publication/384074740_Robust_Bird%27s_Eye_View_Segmentation_by_Adapting_DINOv2

Главная идея:

- адаптировать `DINOv2` к BEV через `LoRA`;
- получать более устойчивые признаки и лучшее поведение под corruptions.

Для вашей задачи это особенно интересно, потому что rig shift частично выглядит как domain shift.

### 5.3 Conditioning on cameras matters

Релевантная недавняя работа:

- *Cameras as Relative Positional Encoding* (arXiv 2507.10496)
- summary/source trail: https://papers.cool/arxiv/2507.10496

Главный смысл:

- multi-view трансформеры выигрывают, когда камера condition'ится не грубо, а через explicit geometric representation of intrinsics/extrinsics.

Для вас это прямой аргумент в пользу:

- не ограничиваться `rover_id`;
- conditionить модель на реальную калибровку.

### 5.4 Calibration-free BEV — как запасной путь

Релевантная работа:

- *Multi-Camera Calibration Free BEV Representation for 3D Object Detection* (arXiv 2210.17252)
- source trail: https://dblp.org/rec/journals/corr/abs-2210-17252.html

Это не мой первый выбор для хакатона, но идея полезная:

- если модель очень чувствительна к calibration noise, geometry-free / weak-geometry branch может дать ансамблевую diversity.

---

## 6. Мой новый приоритет экспериментов

### Tier 1

1. Починить eval pipeline.
2. Сделать rover-matched validation.
3. Перемерить `v1/v3/v4/ensemble` честно.

### Tier 2

1. Calibration-vector conditioning.
2. Global + specialist heads for biggest test rovers.
3. Weight search для ансамбля на rover-matched val.

### Tier 3

1. DINOv2/LoRA backbone.
2. Geometry-aware + weak-geometry ensemble.
3. Rig clustering вместо raw rover id.

---

## 7. Короткий вердикт

Самый сильный новый вывод после просмотра кода и данных:

- **официальный val сейчас нельзя считать надёжным proxy для leaderboard**;
- **EMA-оценка у вас, вероятно, была некорректной**;
- **test полностью лежит внутри train по rover names**, значит rover/rig-specific specialization — это реальный рычаг;
- **но делать это лучше через calibration-aware conditioning или shared+specialist scheme, а не через грубый rover embedding в самом конце decoder'а**.
