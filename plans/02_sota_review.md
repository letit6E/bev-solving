# План по задаче предсказания карты статических препятствий из 4 камер

## Короткий вывод

Текущий бейзлайн застревает около `0.4` не потому, что задача упирается в потолок, а потому что он почти не использует реальную структуру задачи:

- учится только по одной камере из четырех;
- вообще не использует геометрию;
- теряет пространственную структуру через `flatten + linear`;
- использует неподходящий `MSELoss` вместо сегментационных лоссов;
- ресайзит изображения без корректировки `intrinsics`.

По моему мнению, у этой задачи хороший запас для роста. Реалистичный путь к `0.53-0.57` при ваших ресурсах выглядит так:

1. Быстро поднять сильный геометрический baseline: `4 cameras + geometry-aware warp/IPM + BEV decoder`.
2. Затем собрать одну из двух основных моделей:
   - `CVT-light / MatrixVT-light`;
   - `Simple-BEV / Lift-Splat` с хорошим 2D backbone.
3. Добавить 2-3 сильные фишки:
   - корректная работа с маской `255`;
   - sampling по количеству размеченных ячеек;
   - depth prior или distillation от `Depth Anything V2`;
   - ансамбль 2 разных view-transformer'ов;
   - аккуратный TTA и подбор порога по валидации.

Если цель именно "побиться за топ", мой главный кандидат на победный стек здесь:

`CVT-light или MatrixVT-light + сильный 2D encoder + masked BCE/Dice/Lovasz + coverage-aware sampling + optional depth prior + ensemble`.

---

## 1. Что это за задача по сути

Это не просто "сегментация картинки", а **camera-only BEV occupancy / static obstacle map prediction**:

- вход: 4 изображения с камер вокруг машины;
- выход: бинарная карта сверху `1 x 188 x 126`;
- `0` = свободно;
- `1` = занято статическим препятствием;
- `255` или `-1` = зона без разметки.

Практически это очень близко к:

- map-view semantic segmentation;
- BEV perception;
- camera-only occupancy / static map prediction.

Важно: задача именно про **статику**, а не про полную моментальную occupancy всего мира. Это означает, что некоторые сильные 3D occupancy-модели из литературы полезны как источник идей, но не всегда идеально совпадают с таргетом.

---

## 2. Что я увидел в ваших локальных данных

Ниже не общие слова, а наблюдения прямо по датасету.

### 2.1 Размер и структура

- `train`: `4000` сэмплов
- `val`: `1000`
- `test`: `2000`
- камер: `4`
- ширина изображений у всех камер `1024`, высота различается:
  - frontal middle: `546`
  - frontal far: `568`
  - side left: `540`
  - side right: `540`

### 2.2 Калибровки

- `intrinsics` имеют форму `3 x 4`, а не классическую `3 x 3`;
- `car_to_cam` имеют форму `4 x 4`;
- у фронтальной дальней камеры фокусное расстояние сильно больше, чем у остальных;
- у боковых камер параметры тоже заметно отличаются.

Отсюда два критичных вывода:

1. Нельзя считать, что все 4 камеры можно одинаково ресайзить и одинаково обрабатывать геометрически.
2. Нужно очень аккуратно проверить направление матрицы `car_to_cam`.

Важно: описание задачи текстом говорит, что эти матрицы переводят координаты камер в машинную систему, хотя имя `car_to_cam` намекает на обратное. Это потенциальная ловушка. Я бы обязательно провалидировал направление матрицы на 2-3 вручную выбранных точках через проекцию.

### 2.3 Разметка очень разреженная

По train:

- средняя доля размеченных BEV-ячеек: `0.1859`
- медиана: `0.1693`
- минимум: `0.0`
- максимум: `0.4370`
- сэмплов вообще без размеченных ячеек: `25`

Распределение доли размеченной области:

- `16.2%` сэмплов имеют `<= 5%` размеченных ячеек;
- `28.2%` имеют `<= 10%`;
- `59.8%` имеют `<= 20%`.

Это очень важный момент. Если просто случайно мешать данные и учить обычным способом, модель будет тратить много шагов на сэмплы со слабым обучающим сигналом.

### 2.4 Разметка в среднем сосредоточена впереди и близко к центру

Средняя покрытость маской максимальна:

- в ближней и средней зоне по ходу движения;
- около центра по боковой оси;
- сильно падает на дальних рядах и по краям.

Это означает:

- далеко не все части итоговой карты одинаково информативны;
- loss и sampling стоит делать aware к coverage;
- можно добавить auxiliary-задачу предсказания visibility / known-mask.

### 2.5 Пересечения между split'ами

Я проверил пересечения по `(rover, ride_date)`:

- `train-val`: `50` общих комбинаций
- `train-test`: `192`
- `val-test`: `26`

Это не значит утечка впрямую, но означает, что случайный сплит по train может легко оказаться слишком оптимистичным. Для внутренней валидации я бы делал хотя бы один групповой holdout по `(rover, ride_date)`.

---

## 3. Почему текущий baseline закономерно упирается в ~0.4

### 3.1 Используется только одна камера

В ноутбуке обучение идет по `images[0]`, то есть по одной фронтальной камере. Остальные 3 камеры загружаются, но игнорируются.

Для BEV-карты вокруг машины это почти гарантированный потолок.

### 3.2 Геометрия загружается, но не используется

`intrinsics` и `car_to_cam` читаются, но модель ими не пользуется.

Для такого класса задач это главная потеря качества. Без геометрии модель должна "угадать" связь пикселей с BEV чисто статистически.

### 3.3 Архитектура уничтожает пространственную структуру

В бейзлайне после encoder идет:

- `AdaptiveAvgPool2d((1, 1))`
- затем `Linear`
- затем reshape обратно в псевдо-BEV

Это выбрасывает почти всю полезную структуру изображения. Для map prediction это очень слабый inductive bias.

### 3.4 Неподходящий loss

`MSELoss` для бинарной occupancy-сегментации обычно заметно хуже, чем:

- `BCEWithLogitsLoss`;
- `DiceLoss`;
- `Lovasz`;
- `Focal` или `Tversky`.

### 3.5 Ресайз без корректировки intrinsics

Это одна из самых важных ошибок. Если изображение ресайзится, надо масштабировать:

- `fx`
- `fy`
- `cx`
- `cy`

Причем отдельно для каждой камеры, потому что высоты отличаются.

### 3.6 Нет учета неравномерности маски

При такой разреженной разметке полезно:

- отбрасывать или понижать вес почти пустых сэмплов;
- использовать sampler по coverage;
- считать метрику и loss строго по валидной маске.

---

## 4. Какие подходы из литературы здесь реально релевантны

Ниже я разделил методы на:

- **наиболее подходящие под вашу задачу и бюджет**;
- **полезные как источник идей, но тяжелые/избыточные**.

### 4.1 Lift-Splat-Shoot (LSS)

Источник:

- NVIDIA project page: https://research.nvidia.com/labs/toronto-ai/lift-splat-shoot/
- code: https://github.com/nv-tlabs/lift-splat-shoot

Идея:

- из каждой камеры извлекаются 2D признаки;
- модель предсказывает глубину / распределение глубины;
- признаки "поднимаются" в 3D frustum;
- затем "сплэтятся" в BEV.

Почему релевантно:

- это один из базовых и очень влиятельных camera-to-BEV подходов;
- он хорошо совпадает с постановкой "несколько камер -> карта сверху".

Плюсы:

- сильный геометрический inductive bias;
- понятная реализация;
- естественно работает с `intrinsics/extrinsics`.

Минусы:

- depth head может быть нестабильным на маленьком датасете;
- view transform заметно тяжелее, чем у самых легких альтернатив;
- при плохой работе глубины может проиграть attention-based CVT.

Вердикт для вас:

- **хороший путь**, особенно если добавить внешний depth prior;
- но я бы не делал его единственной ставкой.

### 4.2 Cross-View Transformers (CVT)

Источник:

- paper page: https://huggingface.co/papers/2205.02833
- official code: https://github.com/bradyz/cross_view_transformers
- HF checkpoint: https://huggingface.co/qualcomm/CVT

Идея:

- есть набор camera features;
- есть BEV queries;
- cross-attention учится перетягивать информацию из изображений в карту сверху;
- геометрия входит через camera-aware positional encoding.

Почему это очень подходит именно сюда:

- задача по смыслу почти совпадает с map-view semantic segmentation;
- CVT как раз был сделан под map-view segmentation;
- это ближе к вашему таргету, чем многие 3D detection-пайплайны.

Что важно:

- в official repo написано, что `50k` training iterations занимают около `8 часов`, и модель можно учить и на single GPU;
- в демо они показывают real-time map-view segmentation;
- Qualcomm держит очень маленький deployable checkpoint `qualcomm/CVT` c input `1x6x3x224x480` и размером около `1.33M` параметров.

Плюсы:

- отличный fit под задачу;
- легче, чем BEVFormer;
- attention-based fusion без обязательной явной depth supervision;
- может быть очень хорош по speed/quality.

Минусы:

- официальный код заточен под nuScenes и 6 камер;
- нужно аккуратно адаптировать под 4 камеры и ваш grid;
- HF-модель от Qualcomm больше ориентирована на inference/deployment, чем на удобный fine-tune.

Вердикт:

- **один из лучших кандидатов на основной победный пайплайн**.

### 4.3 Simple-BEV

Источник:

- paper page: https://huggingface.co/papers/2206.07959
- official code: https://github.com/aharley/simple_bev
- HF checkpoint: https://huggingface.co/qualcomm/Simple-Bev

Идея:

- простой geometry-aware lift в объем / ground plane;
- аккуратный, но не слишком сложный BEV decoder;
- сильный упор на практичность.

Что важно:

- в official repo есть pretrained camera-only модель;
- при `res_scale=2` (`448x800`) она дает `47.6` trainval mean IoU;
- камера+радар версия дает `55.8`, но радаров у вас нет.

Плюсы:

- очень практичный baseline;
- понятный код;
- хорошие pretrained веса;
- сильнее "с нуля", чем самописный IPM-UNet.

Минусы:

- код под другой датасет и 6 камер;
- camera-only вес обучен на nuScenes таргете, не на вашей бинарной статике;
- HF checkpoint тоже больше про inference/demo и мобайл-деплой.

Вердикт:

- **очень хороший второй основной кандидат**;
- если CVT будет капризным, Simple-BEV может оказаться самым выгодным по времени/качеству.

### 4.4 MatrixVT

Источник:

- paper page: https://huggingface.co/papers/2211.10593
- реализация идет внутри репозитория BEVDepth: https://github.com/Megvii-BaseDetection/BEVDepth

Идея:

- ускорить multi-camera -> BEV transform;
- вместо тяжелого splat использовать более efficient transporting matrix.

Почему это интересно:

- для ваших ограничений по GPU это очень логичное направление;
- paper claim'ит speed/memory improvement при близком качестве.

Плюсы:

- часто лучший компромисс speed / quality / memory;
- хороший fit для T4/V100.

Минусы:

- встроен в более большой стек;
- потребуется больше инженерной сборки, чем для "чистого" CVT.

Вердикт:

- **если делать не просто baseline, а реальный боевой стек - MatrixVT-light выглядит очень разумно**.

### 4.5 BEVDepth / BEVStereo

Источник:

- official repo: https://github.com/Megvii-BaseDetection/BEVDepth
- paper: https://arxiv.org/abs/2206.10092

Идея:

- научиться reliable depth estimation для camera-to-BEV;
- затем использовать depth-aware lift.

Плюсы:

- depth-моделирование действительно помогает camera-only BEV;
- из их стека можно заимствовать полезные идеи для geometry-aware depth branch.

Минусы:

- полный стек тяжеловат;
- это больше про 3D detection, чем про вашу бинарную карту;
- не лучший first bet под маленький локальный датасет.

Вердикт:

- **не как основной end-to-end репозиторий, а как источник идей для depth-aware ветки**.

### 4.6 BEVFormer

Источник:

- paper page: https://huggingface.co/papers/2203.17270
- official repo: https://github.com/zhiqi-li/BEVFormer
- HF model: https://huggingface.co/AXERA-TECH/bevformer

Идея:

- spatiotemporal transformer с BEV queries;
- spatial cross-attention + temporal self-attention.

Плюсы:

- очень сильная идея;
- отлично работает в больших camera-only BEV задачах.

Минусы:

- тяжело;
- избыточно для стартовой фазы;
- риск потратить все доступные GPU-часы на инфраструктуру и стабилизацию.

Вердикт:

- **как источник идей да, как основной хакатонный стек под ваши ресурсы - скорее нет**.

### 4.7 SurroundOcc / TPVFormer / OccFormer / GaussianFormer

Источники:

- SurroundOcc: https://huggingface.co/papers/2303.09551
- TPVFormer: https://huggingface.co/papers/2302.07817
- OccFormer: https://huggingface.co/papers/2304.05316
- GaussianFormer: https://github.com/huang-yh/GaussianFormer

Это уже скорее мир **3D semantic occupancy**, а не просто BEV static map.

Почему все равно стоит знать:

- они показывают, куда в целом идет SOTA;
- у них можно заимствовать идеи по:
  - sparse representations;
  - multi-scale supervision;
  - class imbalance;
  - volume/BEV decoders;
  - temporal and geometry priors.

Но для вашей задачи и бюджета они в основном **слишком тяжелые**.

Мой вывод:

- читать как вдохновение полезно;
- собирать под этот хакатон с нуля почти точно невыгодно.

---

## 5. Что из Hugging Face реально можно использовать

Здесь важно быть честным: на Hugging Face **есть полезные checkpoint'ы**, но именно готовых plug-and-play моделей под ваш точный формат "4 камеры -> бинарная static occupancy карта 188x126" почти нет.

### 5.1 Наиболее близкие готовые BEV-модели

#### `qualcomm/CVT`

Ссылка:

- https://huggingface.co/qualcomm/CVT

Что это:

- deployable версия Cross-View Transformer;
- в model card указаны:
  - checkpoint `vehicles_50k.pt`;
  - input `1x6x3x224x480`;
  - около `1.33M` параметров.

Когда полезно:

- если хочется быстро стартовать от уже существующей реализации;
- если удастся вытащить структуру модели / веса и адаптировать под 4 камеры.

Риски:

- этот репозиторий больше про AI Hub и deployment;
- fine-tune workflow может оказаться неудобнее, чем старт от official GitHub repo.

Мой вердикт:

- **полезен как reference checkpoint и для понимания минималистичной CVT-модели**.

#### `qualcomm/Simple-Bev`

Ссылка:

- https://huggingface.co/qualcomm/Simple-Bev

Что это:

- deployable Simple-BEV;
- ResNet-101 backbone;
- checkpoint `model-000025000.pth`;
- input `448x800`;
- около `49.7M` параметров.

Когда полезно:

- если хотите стартовать от уже обученной camera-only BEV модели.

Риски:

- та же история: это скорее inference/deployment package;
- придется смотреть, насколько удобно оттуда извлекать trainable PyTorch модель.

Мой вердикт:

- **полезно, но не так удобно, как просто взять official GitHub codebase**.

### 5.2 Очень полезные HF-модули не как финальная модель, а как строительные блоки

#### `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`

Ссылка:

- https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf

Почему это важно:

- это готовая outdoor metric depth модель;
- `24.8M` параметров;
- модель построена на `DINOv2` backbone;
- по model card обучалась на `~600K` synthetic labeled и `~62M` real unlabeled images.

Как использовать:

- как frozen teacher для pseudo-depth;
- как auxiliary depth branch;
- как источник geometric prior для LSS / Simple-BEV-подобной модели.

Мой вердикт:

- **один из самых полезных HF-ресурсов для вашей задачи**, даже если вы не будете его fine-tune'ить end-to-end.

#### `facebook/dinov2-base`

Ссылка:

- https://huggingface.co/facebook/dinov2-base

Почему это важно:

- сильный self-supervised vision encoder;
- `86.6M` параметров;
- очень хорош как feature extractor, если нужна более богатая 2D визуальная репрезентация.

Как использовать:

- как encoder для per-camera feature extraction;
- как teacher для distillation в более легкий backbone.

Риск:

- ViT-бэкбон может быть тяжелее в обучении и памяти, чем хороший ConvNet / efficient hybrid.

Мой вердикт:

- **очень хорош для эксперимента "сильный encoder + легкий view-transform"**, особенно на V100/A100.

#### `nvidia/segformer-b0-finetuned-cityscapes-1024-1024`

Ссылки:

- https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024
- https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024

Почему это может помочь:

- у вас задача про статические препятствия;
- 2D semantic prior по классам типа road / curb / wall / building / sidewalk часто помогает.

Как использовать:

- не как финальный predictor;
- а как auxiliary teacher, который подсказывает "что за пиксели" до projection в BEV.

Мой вердикт:

- **не must-have, но интересный буст через distillation или pseudo-label features**.

### 5.3 Итог по Hugging Face

Если коротко:

- **готовые BEV-модели на HF есть, но они в основном demo/deployment-oriented**;
- **для реального обучения удобнее official GitHub repos**;
- **на HF особенно ценны backbones и priors**:
  - Depth Anything V2;
  - DINOv2;
  - SegFormer.

---

## 6. Какой стек я бы выбрал на практике

### Вариант A: основной кандидат на победу

`CVT-light / MatrixVT-light`

#### Архитектура

1. 4 независимых image encoder'а с общими весами.
2. Camera-aware positional encoding на основе `intrinsics + extrinsics`.
3. View transform в BEV:
   - либо cross-view attention;
   - либо MatrixVT-style efficient transport.
4. Нормальный BEV decoder:
   - UNet / FPN / lightweight SegFormer-style head.
5. Финальный binary occupancy head на `188 x 126`.

#### Бэкбон

Лучшие практичные кандидаты:

- `resnet34`
- `resnet50`
- `convnext_tiny`
- `efficientnet-b0/b2`

Если хочется сильнее:

- `DINOv2-base` как encoder или teacher

#### Почему я ставлю на этот стек

- очень хороший fit под map-view segmentation;
- легче, чем BEVFormer;
- обычно лучше масштабируется по качеству, чем совсем простой IPM baseline;
- можно адаптировать под 4 камеры без чудовищной инфраструктуры.

### Вариант B: самый практичный и надежный

`Simple-BEV + хороший 2D backbone + optional depth prior`

#### Почему это сильно

- код проще;
- легче дебажить;
- geometry-aware lift уже встроен в саму идею;
- pretrained camera-only Simple-BEV дает понятную стартовую точку.

#### Как усилить

- заменить или усилить encoder;
- добавить auxiliary depth prior от Depth Anything;
- добавить better decoder и masked losses;
- обучить вторую версию с чуть другой геометрической сеткой и усреднить.

### Вариант C: быстрая промежуточная модель

`Hard/Soft IPM + BEV UNet`

Идея:

- из каждой камеры извлекаются признаки;
- через `grid_sample` они проектируются на ground plane;
- далее фьюзятся и проходят через BEV decoder.

Плюсы:

- можно быстро поднять;
- отличный дебаг-инструмент;
- часто уже сильно лучше "однокамерного flat baseline".

Минус:

- ceiling обычно ниже, чем у хорошего CVT / MatrixVT / Simple-BEV.

Мой совет:

- **обязательно сделать такой baseline первым**, даже если финально хотите идти в CVT.

---

## 7. Мой рекомендованный roadmap экспериментов

### Этап 0. Привести базу в порядок

Сделать обязательно:

1. Использовать все 4 камеры.
2. Корректно масштабировать `intrinsics` после resize.
3. Проверить направление `car_to_cam`.
4. Перейти на `BCEWithLogits + Dice` или `BCEWithLogits + Lovasz`.
5. Считать loss только по `gt != 255`.
6. Подбирать threshold по валидации, а не фиксировать `0.5` по умолчанию.

Даже это уже должно дать заметный рост относительно текущего ноутбука.

### Этап 1. Сильный geometry-aware baseline

Цель:

- быстро получить рабочую multi-camera модель.

Я бы сделал:

- per-camera encoder;
- projection через IPM / homography / grid_sample;
- BEV UNet head;
- masked segmentation loss.

Ожидание:

- хороший шанс быстро уйти заметно выше текущего baseline.

### Этап 2. Основная боeвая модель

Запустить в работу одну из двух линий:

- `CVT-light / MatrixVT-light`
- `Simple-BEV + depth prior`

Не надо строить сразу 5 архитектур. Лучше 2 линии:

1. одна attention-based;
2. одна geometry-lift based.

Это потом еще и даст хороший ансамбль.

### Этап 3. Усиления

Добавить по очереди:

- coverage-aware sampler;
- camera dropout;
- auxiliary depth branch;
- auxiliary visible-mask head;
- EMA / checkpoint averaging;
- TTA;
- ensemble.

### Этап 4. Финальная сборка

- выбрать 2-3 лучших чекпоинта;
- обучить финальные версии на `train + val`;
- сделать ансамбль;
- подобрать финальный threshold.

---

## 8. Какие фишки реально могут зарешать именно здесь

Ниже перечисляю не "косметику", а вещи, которые часто реально дают leaderboard gain.

### 8.1 Coverage-aware sampling

Так как разметка очень разреженная, стоит сэмплировать батчи так, чтобы чаще видеть примеры с хорошим coverage.

Идеи:

- вес сэмпла пропорционален доле `gt != 255`;
- либо смешанный sampler: часть батча из "сильных" примеров, часть из обычных.

Это особенно важно, потому что почти `16%` train-сэмплов имеют `<= 5%` размеченной области.

### 8.2 Auxiliary head на visibility / known-mask

Поскольку карта размечена не везде, модель может учиться не только occupancy, но и:

- "где вообще есть надежный наблюдаемый участок".

Это хорошо работает как регуляризатор BEV-представления.

### 8.3 Depth prior без полноценного depth-supervision

Даже если у вас нет GT depth:

- можно прогнать `Depth Anything V2`;
- получить pseudo-depth;
- использовать ее как:
  - auxiliary supervision;
  - soft prior для lift;
  - дополнительный вход в fusion block.

Это одна из самых практичных "чужих" идей, которые можно быстро перенести на ваш датасет.

### 8.4 Camera dropout

Во время обучения случайно убирать одну камеру из 4.

Зачем:

- модель меньше переучивается на одну фронтальную доминирующую камеру;
- повышается устойчивость к неполной информации;
- fusion становится осмысленнее.

### 8.5 Distillation от 2D teacher'ов

Сильный и недооцененный ход:

- teacher depth: `Depth Anything V2`
- teacher semantics: `SegFormer Cityscapes`

Даже если финальная модель маленькая, distillation может дать ощутимый буст в малоданных условиях.

### 8.6 Подбор порога по валидации

Для IoU бинарной occupancy порог очень часто важен. Не фиксируйте `0.5` по привычке.

Надо прогнать sweep, например:

- `0.30`
- `0.35`
- `0.40`
- ...
- `0.70`

И выбрать лучший threshold по masked IoU.

### 8.7 EMA и checkpoint averaging

Для segmentation/BEV задач это нередко дает "бесплатный" бонус.

### 8.8 Две разные школы в ансамбле

Самый логичный ансамбль:

- attention-based модель (`CVT-light`)
- geometry-lift модель (`Simple-BEV / LSS-like`)

Они делают разные ошибки, а значит усреднение логитов часто дает хороший gain.

### 8.9 Проверка transform direction вручную

Это не "косметика", а потенциальный killer bug.

Я бы до обучения сделал маленький debug-ноутбук:

1. взять несколько точек на земле в car frame;
2. спроецировать в каждую камеру;
3. убедиться, что точки попадают в ожидаемые места изображения.

Если перепутать направление `car_to_cam` / `cam_to_car`, модель может учиться неделями в неверной геометрии.

### 8.10 Групповая валидация

Так как есть заметное пересечение по `(rover, ride_date)`, я бы держал две валидации:

- официальную `val`;
- внутреннюю group-holdout по `(rover, ride_date)`.

Это поможет не переоценить локальные улучшения.

---

## 9. Какие лоссы и метрики я бы использовал

### Основной loss

Мой фаворит:

- `0.5 * BCEWithLogitsLoss`
- `0.3 * DiceLoss`
- `0.2 * LovaszHinge` или `Focal`

Все строго по маске `gt != 255`.

### Почему не только BCE

- BCE хорошо калибрует пиксельно;
- Dice/Lovasz лучше оптимизируют overlap-качество, близкое к IoU.

### Что еще важно

- `pos_weight` или focal-параметры нужно подбирать по валидации;
- можно сделать sample-wise normalization, чтобы сэмплы с маленькой маской не терялись.

### Локальная метрика

Нужно реализовать локально тот же masked IoU, что и в соревновании:

- ignore `255`;
- считать IoU по бинарной occupancy;
- проверять, что на submission уходят только `0/1`.

---

## 10. Аугментации, которые здесь уместны

### Image-level

- `ColorJitter`
- `RandomBrightnessContrast`
- `Gamma`
- небольшой `GaussianNoise`

### Осторожно и только с обновлением геометрии

- resize / crop
- horizontal flip
- BEV rotation / translation

Если делать такие аугментации, надо:

- обновлять `intrinsics`;
- при необходимости менять `extrinsics`;
- для горизонтального flip учитывать перестановку левой и правой камеры.

### Что я бы не делал на старте

- агрессивные перспективные деформации;
- слишком сильные кропы;
- тяжелые photometric tricks без абляции.

---

## 11. Как я бы распланировал вычисления

У вас:

- `59h` на `T4`
- или `29h` на `V100`
- или `18h` на `A100`

### Что делать на M1 Pro

Только:

- дебаг датасета;
- sanity-check projection;
- tiny overfit на 32-128 примерах;
- проверка loss / metric / infer pipeline.

Не тратить M1 на серьезное обучение.

### Если выбирать один главный ресурс

Мой выбор:

- использовать `T4` для первых серий недорогих экспериментов;
- если есть возможность переключиться, оставить `A100` на финальные 1-2 боевые прогона и ансамбль.

### Практичный план

#### План A

- `T4`: 4-6 быстрых экспериментов по 6-8 часов
- `A100`: 1-2 финальных прогона сильной модели + ensemble inference

#### План B

- `V100`: 2-3 серьезных прогона средней тяжести

### Что именно я бы запускал

1. IPM baseline
2. CVT-light
3. Simple-BEV-light
4. Лучшая из двух + depth prior
5. Ensemble

---

## 12. Оценка по ROI: что делать в каком порядке

### Самый высокий ROI

1. Использовать все 4 камеры
2. Починить геометрию и resize/intrinsics
3. Перейти на masked BCE/Dice/Lovasz
4. Сделать geometry-aware projection
5. Sampling по coverage

### Следующий слой

1. CVT-light или MatrixVT-light
2. Simple-BEV branch
3. Depth Anything prior

### Финальные проценты leaderboard

1. threshold tuning
2. EMA
3. TTA
4. ensemble

---

## 13. Какой результат я считаю реалистичным

Это оценка, а не гарантия.

### Консервативно

- хороший multi-camera geometry-aware baseline может заметно улучшить текущую точку.

### Реалистично

- сильный CVT-light / Simple-BEV + нормальный train pipeline выглядит как путь в район `0.5+`.

### Агрессивная цель

- `0.56+` выглядит достижимой, если:
  - основная модель будет действительно geometry-aware;
  - будет хороший encoder;
  - train pipeline будет чистым;
  - будет хотя бы небольшой ансамбль / финетюнинг.

---

## 14. Что бы я делал прямо следующим шагом

Если идти максимально прагматично, я бы делал так:

### Шаг 1

Переписать бейзлайн в нормальный multi-camera geometry-aware baseline:

- shared encoder for 4 cameras
- projection в BEV
- BEV UNet head
- masked BCE + Dice

### Шаг 2

Параллельно подготовить вторую ветку:

- CVT-light или MatrixVT-light

### Шаг 3

Подключить `Depth Anything V2` как prior / teacher.

### Шаг 4

Сделать финальный ensemble.

Если бы нужно было выбрать **одну** стартовую архитектуру, я бы начал с:

`Simple-BEV-style model, но с более аккуратным train pipeline и с возможностью потом добавить depth prior`.

Если бы нужно было выбрать **одну** ставку на лучший потолок, я бы выбрал:

`CVT-light / MatrixVT-light`.

---

## 15. Список источников

Основные источники, на которые я опирался:

- Lift-Splat-Shoot project page: https://research.nvidia.com/labs/toronto-ai/lift-splat-shoot/
- Lift-Splat-Shoot code: https://github.com/nv-tlabs/lift-splat-shoot
- Cross-View Transformers paper page: https://huggingface.co/papers/2205.02833
- Cross-View Transformers code: https://github.com/bradyz/cross_view_transformers
- Qualcomm CVT model card: https://huggingface.co/qualcomm/CVT
- Simple-BEV paper page: https://huggingface.co/papers/2206.07959
- Simple-BEV code: https://github.com/aharley/simple_bev
- Qualcomm Simple-Bev model card: https://huggingface.co/qualcomm/Simple-Bev
- BEVDepth code: https://github.com/Megvii-BaseDetection/BEVDepth
- BEVFormer paper page: https://huggingface.co/papers/2203.17270
- SurroundOcc paper page: https://huggingface.co/papers/2303.09551
- TPVFormer paper page: https://huggingface.co/papers/2302.07817
- OccFormer paper page: https://huggingface.co/papers/2304.05316
- GaussianFormer code: https://github.com/huang-yh/GaussianFormer
- Depth Anything V2 metric outdoor small: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
- DINOv2 base: https://huggingface.co/facebook/dinov2-base
- SegFormer Cityscapes B0: https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024
- SegFormer Cityscapes B3: https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024

---

## 16. Финальный вердикт

Если отбросить лишнее, то мой совет такой:

- не пытаться сразу строить "тяжелый SOTA-танк";
- быстро сделать сильный геометрический baseline;
- затем инвестировать время в `CVT-light` или `MatrixVT-light`;
- использовать HF не столько как финальную готовую модель, сколько как источник:
  - depth priors;
  - сильных image encoders;
  - reference checkpoint'ов.

Самая опасная ошибка здесь не "выбрать не ту SOTA-архитектуру", а:

- неверно обработать геометрию;
- проигнорировать маску;
- получить слишком оптимистичную локальную валидацию;
- потратить все GPU-часы на тяжелую модель до того, как появится сильный baseline.

Если захотите, следующим сообщением я могу уже перейти от ресерча к практике и предложить **конкретный технический план переписывания `baseline_v4.ipynb` в сильный geometry-aware training pipeline**.
