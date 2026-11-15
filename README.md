# AnimalSight

AnimalSight is a small UFPR Computer Vision final project investigating:

> Can classical region-of-interest selection reduce Vision Transformer inference cost while preserving image classification accuracy?

It combines a frozen `facebook/dinov2-small` feature extractor with a supervised SVM. A classical Sobel/ Gaussian ROI selector is evaluated as preprocessing for lower-resolution inputs. There is no detector, fine-tuning, confidence cascade, or feature concatenation in the current implementation.

## Dataset and protocol

Use the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), restricted to **beagle, boxer, chihuahua, pug, and samoyed**. Its official annotation lists define 500 `trainval` images and 499 `test` images for these breeds. No bounding boxes, trimaps, labels, or test information are used to select a crop.

1. Keep the official `trainval` / `test` partition and annotation order.
2. Split `trainval` into 400 training and 100 validation examples using stratification, a 20% validation fraction, and seed 42.
3. Evaluate each representation on that validation set. Fit each scaler only on its 400 training embeddings.
4. With settings fixed, refit a separate scaler/SVM on all 500 `trainval` examples for each representation.
5. Evaluate once on the same 499 official test images. Never choose resolution, ROI scale, or SVM parameters from test scores.

All classifiers use `StandardScaler` followed by `SVC(kernel="rbf", C=10, gamma="scale")`, retaining the original classifier configuration. These parameters and the ROI scale of 0.8 are fixed defaults, not test-tuned hyperparameters. A numerical gamma is estimated separately from each training representation by scikit-learn's `scale` rule. Training manifests, validation indices, labels and SHA-256 image hashes are saved.

## Representations

| Configuration | Source image | Shortest-edge resize | Actual transformer input | Patch tokens + CLS |
|---|---|---:|---:|---:|
| Baseline | Original | 256 | 224 × 224 | 256 + 1 |
| Reduced | Original | 128 | 112 × 112 | 64 + 1 |
| ROI Reduced | Classical square crop | 128 | 112 × 112 | 64 + 1 |

The original image processor's bicubic resize, square center crop and ImageNet normalization policy is preserved. Both resize and center crop are scaled together, so Reduced and Baseline retain the same field of view. **Original** means no classical ROI; the model's standard center crop still applies. A square ROI also undergoes this same resize/center crop policy, retaining its central 87.5% along each dimension.

Images are decoded as OpenCV BGR uint8 arrays, optionally cropped, converted to RGB/PIL, and passed to `AutoImageProcessor`. The last layer's 384-dimensional CLS embedding is used. DINOv2 stays in evaluation/inference mode with all parameters frozen. One loaded runtime is reused across representations, dataset splits and predictions.

The [Hugging Face DINOv2 implementation](https://huggingface.co/docs/transformers/model_doc/dinov2) interpolates positional embeddings internally. The installed implementation was inspected and tests run with a real small DINOv2 instance. Both `pixel_values` shape and output token count are checked at runtime. Smaller crops processed at 224 would not save transformer computation; the reduced configurations actually pass 112 × 112 tensors.

The checkpoint is pinned to revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`.

## Installation

Python 3.10+ and a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The tested versions are pinned in `requirements.txt`. Install a suitable PyTorch CPU or CUDA wheel for your machine. The experiment automatically uses CUDA when available; `--device cpu` forces CPU. `--threads 4` controls PyTorch and OpenCV CPU threads.

Place the official files at:

```text
data/oxford-pet/
├── annotations/
│   ├── trainval.txt
│   └── test.txt
└── images/
    ├── beagle_1.jpg
    └── ...
```

The official download links are available on the dataset page. For a new dataset directory:

```bash
mkdir -p data/oxford-pet
curl -L https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz -o data/oxford-pet/images.tar.gz
curl -L https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz -o data/oxford-pet/annotations.tar.gz
tar -xzf data/oxford-pet/images.tar.gz -C data/oxford-pet
tar -xzf data/oxford-pet/annotations.tar.gz -C data/oxford-pet
```

The first experiment downloads DINOv2 from Hugging Face. Add `--offline` when the pinned weights are already cached. A missing dataset or missing offline weights causes an explicit error; no synthetic metrics are substituted.

## Training and evaluation

Run commands from the repository root with the virtual environment activated:

```bash
# Validate within trainval, then refit and save three separate classifiers.
python -m src.experiments train

# Evaluate the saved classifiers on the official test split.
python -m src.experiments evaluate

# Or train, evaluate and benchmark with one shared model load.
python -m src.experiments all --offline --device cpu

# Generate another seeded validation preview without loading DINOv2.
python -m src.roi --scale 0.8 --seed 42 --count 15
```

Validation previews sample 15 validation images uniformly without replacement, using seed 42, rather than selecting one favorable example per breed. The selected paths are recorded in `roi_samples.json`.

Models are `models/baseline_224.joblib`, `models/reduced_112.joblib` and `models/roi_reduced_112_s0p8.joblib`. Each contains the scaler, SVM and representation/training metadata. These local joblib artifacts must be trusted files.

Feature caches under `models/cache/` have separate content-derived names. Their identity includes model/revision, processor/backend, software versions, preprocessing/ROI algorithm version, resolution, ROI scale, and the ordered paths and content hashes. Old `dinov2_*_features.npz` files and legacy classifiers are preserved but never loaded by the new workflow. Single-image inference never writes a feature cache.

For an additional 168 experiment, decide on it using validation, before inspecting its test results:

```bash
python -m src.experiments train --reduced-size 168 --models-dir models/roi_168
python -m src.experiments evaluate --reduced-size 168 --models-dir models/roi_168
python -m src.benchmark --reduced-size 168 --models-dir models/roi_168
```

This uses resize 192, input 168 × 168, and 145 tokens. The baseline remains 224. The separate model directory keeps the primary run's artifacts intact. Use matching `--roi-scale`, `--dataset`, `--models-dir`, and `--results-dir` across commands when changing defaults. The app uses the primary 112/0.8 configuration.

## Computational benchmark

```bash
python -m src.benchmark --offline --device cpu --threads 4 --repeats 3 --warmup 5
```

The benchmark loads the shared DINOv2 and all classifiers once. It uses **batch size one**, every official test image, three passes, and five warm-up predictions per configuration per pass. Image order and configuration block order are seeded and shuffled across passes; the same image order is used by each configuration in a pass. CUDA is synchronized before and after each measured prediction.

Timing starts from an already decoded OpenCV image and includes BGR/RGB/PIL conversion, ROI extraction when enabled, the image processor, device transfers, DINOv2, CLS extraction, feature scaling and SVM prediction. ROI preview generation is outside timing, but ROI selection is recomputed inside the measured path. Disk reading/JPEG decoding, weight downloads, model loading, training, embedding cache access and warm-up are excluded. No cached embeddings are used for timing. This is in-memory image inference latency, not upload-to-screen latency or cold-start latency.

Mean, median, standard deviation, p95 and raw per-image/per-repeat latency are saved. Speedup is baseline mean latency divided by the configuration's mean latency. Accuracy and macro F1 are computed from the same benchmark predictions; repeated predictions are checked for consistency.

Storage distinguishes one shared safetensors checkpoint plus model/processor configuration files from each SVM/scaler/provenance artifact. Total storage for a standalone configuration is their sum; combined storage counts the DINOv2 files once. Tensor payload bytes are also reported separately. Dataset, feature caches and Python/CUDA dependencies are excluded. Reducing input resolution changes activation/computation cost, **not the frozen model's weight size**. Parameter bytes are not peak runtime memory.

## Results

New experiment outputs are under `results/roi_112/`:

- `training.json`, `validation.csv`: protocol, file hashes, split indices and validation metrics.
- `evaluation.json`, `evaluation.csv`, `predictions.csv`: full official-test metrics, confusion matrices and predictions.
- `benchmark.json`, `comparison.csv`, `latencies.csv`: environment, storage, accuracy, F1, latency, speedup and raw timings.
- [Accuracy versus inference time](results/roi_112/accuracy_vs_latency.png): mean latency versus official-test accuracy.
- [ROI validation examples](results/roi_112/roi_validation.png), `roi_samples.json`: multiple unselected validation examples and their identities.
- `environment.txt`, `verification.txt`: installed package versions and executed checks.

Measured on 2026-11-14 with the fixed protocol above:

| Configuration | Validation accuracy (100) | Test accuracy (499) | Test macro F1 |
|---|---:|---:|---:|
| baseline_224 | 99.00% | 99.80% | 0.997990 |
| reduced_112 | 95.00% | 98.80% | 0.987989 |
| roi_reduced_112_s0p8 | 96.00% | 96.99% | 0.969987 |

Baseline classified 498/499 images correctly; Reduced 493/499; ROI Reduced 484/499. **ROI Reduced did not outperform Reduced:** it lost 1.80 percentage points (nine additional errors) at the same 112 × 112 transformer input. The validation trend (96% ROI versus 95% Reduced) did not carry over to the official test. No settings were changed after seeing these results.

The CPU benchmark used two complete passes (998 measurements per configuration, 2,994 total), batch size 1, five warm-ups per configuration/pass, and four PyTorch/OpenCV threads. Hardware: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz. No CUDA device was available.

| Configuration | Mean (ms/image) | Median (ms/image) | Speedup | SVM artifact (MiB) | Total model storage (MiB) |
|---|---:|---:|---:|---:|---:|
| baseline_224 | 202.41 | 228.00 | 1.00× | 0.805 | 84.967 |
| reduced_112 | 69.87 | 76.69 | 2.90× | 1.026 | 85.189 |
| roi_reduced_112_s0p8 | 67.08 | 63.12 | 3.02× | 1.032 | 85.195 |

The shared DINOv2 checkpoint and configuration files occupy 84.163 MiB. All three solutions together require 87.025 MiB, counting shared files once. Benchmark predictions exactly matched the cached-feature evaluation for all 2,994 timed predictions.

Both reduced inputs were faster than the baseline. ROI Reduced was slightly faster on average than Reduced in this run, but its lower accuracy provides no evidence that this ROI heuristic preserves information better at 112. The small latency gap should not be treated as a general advantage without independent repeated runs. Reduced alone lost 1.00 percentage point relative to baseline; ROI Reduced lost 2.81 points.

Reproduce this measurement with `python -m src.benchmark --offline --device cpu --threads 4 --repeats 2 --warmup 5`. The command default remains three passes. The 168 option and CUDA timings were not run.


Previous experiments, reported before this refactoring on the same 499-image official test split:

| Historical representation | Accuracy |
|---|---:|
| Gabor + LoG + SVM | 26.65% |
| Gabor + LoG + HOG + color histograms + SVM | 44.49% |
| DINOv2-small + SVM | 99.80% |
| Concatenated classical and DINOv2 features + SVM | 99.60% |

These historical figures are not new benchmark measurements. Their models, caches, reports and dataset files remain on disk. The obsolete classical, enhanced-classical, hybrid, confidence-analysis and dataset-exploration scripts were removed after checking dependencies. There was no `efficient_model.py` to remove.

## Local application

```bash
streamlit run app.py
```

Train the three primary classifiers first. Upload a photograph, inspect the original, ROI overlay, gradient map and crop, select a configuration, then classify. DINOv2 and classifiers are cached once across Streamlit reruns. The app displays predicted breed and measured processing time separately from initial model loading. It uses locally cached weights and the same decoding/preprocessing/prediction functions as the experiments. Uploaded images are not persisted.

Only the five selected breeds are recognized. Other dogs, cats, objects and mixed breeds will still be assigned one of those labels; the application does not implement rejection or calibrated confidence.

## Code and tests

```text
app.py                       Local Streamlit interface
src/pipeline.py              Official dataset loading, splits, scaler/SVM and metrics
src/roi.py                   Sobel/blur window selection and validation previews
src/transformer_model.py     Shared frozen model, preprocessing, feature cache and prediction
src/experiments.py           Train/validate/refit/evaluate CLI and reproducible outputs
src/benchmark.py             Warmed full-pipeline timing, storage and comparison plot
tests/                       ROI, dataset, embeddings, cache, experiment and app tests
```

```bash
python -m unittest discover -s tests -v
```

Tests use generated images and a small randomly initialized DINOv2 to verify real interpolation and CLS extraction without downloading weights. Their numerical scores are test fixtures, not experimental results. Separate real dataset runs produce the metrics above.

## Limitations

- Five breeds and one fixed validation split are a small academic experiment; conclusions do not generalize automatically to all 37 Oxford Pets classes or other datasets.
- Gradient concentration can favor background texture and can cut off the animal. It has no semantic or anatomical understanding.
- The official test was also used for historical experiments. This new protocol avoids new test-driven tuning but cannot undo that earlier exposure.
- Runtime depends on CPU/GPU, thread count, software versions, image dimensions and competing workloads. Repeated per-image measurements describe variation within one run, not confidence intervals over independent machines or training splits.
- Some original JPEGs produce OpenCV corruption warnings while remaining decodable; the workflow keeps them and does not silently change the official sample set.
- The 168 option is supported but is not part of the primary three-way comparison. No additional architecture or model training is introduced.
