# AnimalSight

UFPR Computer Vision project: can a classical region of interest (ROI) help DINOv2 classify dog breeds at smaller image resolutions?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.10+ and place the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) here:

```text
data/oxford-pet/images/*.jpg
data/oxford-pet/annotations/trainval.txt
data/oxford-pet/annotations/test.txt
```

The five breeds are **beagle, boxer, chihuahua, samoyed and Yorkshire terrier**. The first run downloads DINOv2; later runs reuse the local weights and feature caches.

## Run the experiment

```bash
python -m src.experiments
```

This single command:

1. Loads the official **500 training images** (`trainval`) and **499 test images** (`test`).
2. Extracts frozen DINOv2 CLS embeddings for each configuration.
3. Trains a separate `StandardScaler` + RBF SVM on all 500 training images.
4. Evaluates each classifier on the same 499 test images.
5. Saves the classifiers, accuracy/F1 tables, predictions and comparison plot.

The baseline fits the **entire photograph into 224 × 224**. The experiment compares two approaches at **112, 84, 56, 42 and 28** pixels:

- **Reduced:** resize the entire image while preserving its proportions, then center it on a square canvas (letterbox). Nothing is cropped.
- **ROI Reduced:** first select a square region using Sobel gradients and Gaussian blur, then fit that entire region into the same square input.

All configurations use bicubic interpolation and DINOv2 normalization, with its automatic resize and center crop disabled. The padding is RGB `(124, 116, 104)`, the rounded DINOv2 mean, so it is approximately zero after normalization. For example, a 600 × 400 photograph becomes 112 × 75 plus borders in Reduced 112. The baseline uses this same policy at 224 pixels. ROI is square and needs no padding.

The ROI scale is fixed at `0.8`. The SVM uses `C=10`, `kernel="rbf"`, `gamma="scale"`. DINOv2 is loaded once and remains frozen. Smaller inputs really produce fewer tokens; weights are unchanged.

To run only the original three-way comparison:

```bash
python -m src.experiments --resolutions 112
```

Add `--offline` to use cached weights without network access. Use `--device cpu` to force CPU. No separate validation split or automated test suite is required.

## Results

The current direct experiment writes to `results/experiment/`:

- [evaluation.csv](results/experiment/evaluation.csv): accuracy and macro F1 for all configurations, including the baseline.
- [comparison.csv](results/experiment/comparison.csv): Reduced versus ROI, including how many images ROI helps or harms.
- [accuracy_vs_resolution.png](results/experiment/accuracy_vs_resolution.png): accuracy curves.
- `predictions.csv` and `experiment.json`: predictions and the settings used.

Measured on the 499 official test images:

| Input | Reduced accuracy | ROI Reduced accuracy |
|---|---:|---:|
| 112 × 112 | 95.39% | 97.60% |
| 84 × 84 | 87.98% | 95.19% |
| 56 × 56 | 60.32% | 75.95% |
| 42 × 42 | 47.29% | 61.32% |
| 28 × 28 | 34.47% | 52.10% |

The 224 baseline reached **100.00%**. ROI improved total accuracy at every reduced resolution, although it still harmed some individual images. At 112 pixels, it corrected 14 errors and introduced 3; at 28 pixels, it corrected 151 and introduced 63. Preserving the full scene also makes the animal smaller within the input, so full-image reduction does not necessarily preserve the most useful details.

The previous experiment with center cropping, including its report and timings, is preserved in `results/archive_center_crop/`, with classifiers in `models/archive_center_crop/`. The older pug experiment is in `results/experiment_pug/` and `models/archive_pug/`; `results/roi_112/` and `results/resolution_sweep/` also refer to that older subset. The current experiment uses Yorkshire terrier and letterboxing. The comparison remains exploratory on this dataset.

## Optional: measure inference time

After running the experiment:

```bash
python -m src.benchmark --repeats 2
```

This compares the baseline and the two 112 configurations. Use `--reduced-size 56`, for example, to time a smaller input. Outputs go to `results/benchmark_112/` (or the selected size).

Timing includes preprocessing, ROI when enabled, DINOv2 and SVM. It excludes image decoding, model loading, training and feature caches. The benchmark uses warm-up, batch size one, repeated passes and CUDA synchronization when applicable. It also reports shared DINOv2 storage separately from SVM storage.

The current CPU run measured **222.34 ms** for Baseline, **81.52 ms** for Reduced 112 and **87.37 ms** for ROI Reduced 112: speedups of **2.73×** and **2.54×**, respectively. ROI improved accuracy by **2.20 percentage points** over Reduced 112 and took **5.85 ms** longer per image on average. These measurements use an Intel i7-1065G7, four threads, two passes and the Yorkshire subset. Both reduced inputs use 65 tokens; timing includes ROI selection when enabled. Smaller inputs were not timed.

## Optional: local interface

```bash
streamlit run app.py
```

Upload an image, inspect the gradient map/ROI and actual model input, select Baseline, Reduced or ROI Reduced, and classify. The app uses the saved baseline and 112 classifiers. Initial model loading is shown separately from inference time. Only the five selected breeds are supported.

To save ROI examples from training images: `python -m src.roi`.

## Report

[report.tex](report.tex) is the Portuguese IEEE report in two columns, with the current letterbox results. To compile it in Overleaf, upload it together with `results/report/`, preserving that folder path. The report figures show the resolution comparison and actual model inputs for an ROI success and failure.

## Reading the code

Start with `run_experiments()` in `src/experiments.py`; it contains the train-and-evaluate loop.

| File | Purpose |
|---|---|
| `src/pipeline.py` | Dataset loading, SVM training and metrics |
| `src/roi.py` | Classical crop selection |
| `src/transformer_model.py` | Frozen DINOv2, preprocessing, cache and prediction |
| `src/experiments.py` | Direct experiment, tables and resolution plot |
| `src/benchmark.py` | Optional timing and storage measurements |
| `app.py` | Optional Streamlit interface |

Model and feature caches live in `models/` and are ignored by Git. Their metadata prevents reuse with the wrong resolution, ROI settings or image order. Keep training and test images separate; the ROI itself uses neither labels nor a detector.
