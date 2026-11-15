"""Repeated batch-one inference, including preprocessing, ROI, transformer and SVM."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import sklearn
import torch

from src.experiments import add_common_arguments, evaluation_data, load_trained_artifacts, result_row, write_csv
from src.pipeline import ROOT, file_manifest, fingerprint, prediction_metrics, read_image, write_json
from src.transformer_model import config_id, configurations, load_runtime, predict_image


def synchronize(runtime):
    if runtime['device'].type == 'cuda':
        torch.cuda.synchronize(runtime['device'])


def timed_prediction(image, runtime, artifact):
    synchronize(runtime)
    start = time.perf_counter_ns()
    prediction = predict_image(image, runtime, artifact)
    synchronize(runtime)
    return prediction, (time.perf_counter_ns() - start) / 1_000_000


def measure(runtime, artifacts, images, labels, *, repeats=3, warmup=5, seed=42):
    if not images or len(images) != len(labels) or not artifacts or repeats < 2 or warmup < 1:
        raise ValueError('Provide matching nonempty images/labels, classifiers, >=2 repeats and >=1 warm-up.')
    rng = np.random.default_rng(seed)
    latencies = [[] for _ in artifacts]
    predictions = [[None] * len(images) for _ in artifacts]
    timings = []
    for repeat in range(repeats):
        image_order = rng.permutation(len(images))
        for artifact_index in rng.permutation(len(artifacts)):
            artifact = artifacts[artifact_index]
            name = config_id(artifact['representation']['configuration'])
            for i in range(warmup):
                predict_image(images[image_order[i % len(images)]], runtime, artifact)
            synchronize(runtime)
            for index in image_order:
                prediction, latency = timed_prediction(images[index], runtime, artifact)
                previous = predictions[artifact_index][index]
                if previous is not None and previous != prediction:
                    raise ValueError('Predictions changed across benchmark repetitions.')
                predictions[artifact_index][index] = prediction
                latencies[artifact_index].append(latency)
                timings.append({"configuration": name, "repeat": repeat + 1, "image_index": int(index),
                                "label": str(labels[index]), "prediction": prediction, "latency_ms": latency})
            print(f"Timed {name}, repeat {repeat + 1}/{repeats}: {len(images)} images", flush=True)
    baseline_index = next(i for i, a in enumerate(artifacts) if a['representation']['configuration']['name'] == 'baseline')
    baseline_mean = float(np.mean(latencies[baseline_index]))
    rows = []
    for artifact, predicted, times in zip(artifacts, predictions, latencies):
        row = result_row(artifact['representation']['configuration'], prediction_metrics(labels, predicted))
        row.update(mean_latency_ms=float(np.mean(times)), median_latency_ms=float(np.median(times)),
                   std_latency_ms=float(np.std(times)), p95_latency_ms=float(np.percentile(times, 95)),
                   measurements=len(times), speedup=baseline_mean / float(np.mean(times)))
        rows.append(row)
    return rows, timings


def storage_requirements(runtime, classifier_path):
    svm_bytes = classifier_path.stat().st_size
    return {"shared_dinov2_storage_bytes": runtime['shared_storage_bytes'],
            "dinov2_tensor_bytes": runtime['parameter_bytes'],
            "svm_storage_bytes": svm_bytes,
            "total_storage_bytes": runtime['shared_storage_bytes'] + svm_bytes}


def environment(runtime):
    cpu = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        cpu = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), cpu)
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            "device": str(runtime['device']),
            "gpu": torch.cuda.get_device_name(runtime['device']) if runtime['device'].type == 'cuda' else None,
            "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
            "sklearn": sklearn.__version__, "cuda_version": torch.version.cuda,
            "batch_size": 1, "runtime": runtime['identity']}


def save_comparison(rows, results_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fields = ['configuration', 'resolution', 'tokens', 'samples', 'accuracy', 'macro_f1',
              'mean_latency_ms', 'median_latency_ms', 'std_latency_ms', 'p95_latency_ms',
              'measurements', 'speedup', 'shared_dinov2_storage_bytes', 'dinov2_tensor_bytes',
              'svm_storage_bytes', 'total_storage_bytes']
    write_csv(results_dir / 'comparison.csv', rows, fields)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        ax.scatter(row['mean_latency_ms'], row['accuracy'] * 100, s=65,
                   label=f"{row['configuration']} ({row['tokens']} tokens)")
    ax.set(xlabel='Mean end-to-end inference time (ms/image)', ylabel='Official test accuracy (%)',
           title='AnimalSight: accuracy vs. inference time')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / 'accuracy_vs_latency.png', dpi=160)
    plt.close(fig)


def run_benchmark(runtime, configs, *, dataset, models_dir, results_dir, repeats=3, warmup=5, seed=42):
    artifacts = load_trained_artifacts(runtime, configs, models_dir)
    paths, labels = evaluation_data(artifacts, dataset)
    images = [read_image(path) for path in paths]
    rows, timings = measure(runtime, artifacts, images, labels, repeats=repeats, warmup=warmup, seed=seed)
    for row, config in zip(rows, configs):
        row.update(storage_requirements(runtime, models_dir / f'{config_id(config)}.joblib'))
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(runtime),
                "seed": seed, "repeats": repeats, "warmup_per_configuration_per_repeat": warmup,
                "test_files": file_manifest(paths), "training_fingerprint": fingerprint(artifacts[0]['training']),
                "timing_includes": ['BGR to RGB/PIL conversion', 'ROI if enabled', 'image processor',
                                    'device transfers', 'DINOv2 CLS extraction', 'scaler and SVM prediction'],
                "timing_excludes": ['disk image decoding', 'weights download', 'model/classifier loading',
                                    'training', 'feature caches', 'warm-up'],
                "storage_note": 'One shared safetensors checkpoint + model/processor configs; each joblib includes SVM, scaler and provenance. Excludes dataset, caches and Python dependencies.',
                "combined_storage_bytes": runtime['shared_storage_bytes'] + sum(r['svm_storage_bytes'] for r in rows),
                "results": rows}
    write_json(results_dir / 'benchmark.json', metadata)
    write_csv(results_dir / 'latencies.csv', timings, ['configuration', 'repeat', 'image_index', 'label', 'prediction', 'latency_ms'])
    save_comparison(rows, results_dir)
    for row in rows:
        print(f"{row['configuration']}: {row['accuracy']:.2%}, {row['mean_latency_ms']:.2f} ms, {row['speedup']:.2f}x")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    configs = configurations(args.reduced_size, args.roi_scale)
    runtime = load_runtime(args.device, local_files_only=args.offline, threads=args.threads)
    run_benchmark(runtime, configs, dataset=args.dataset, models_dir=args.models_dir,
                  results_dir=args.results_dir or ROOT / f'results/roi_{args.reduced_size}',
                  repeats=args.repeats, warmup=args.warmup, seed=args.seed)


if __name__ == '__main__':
    main()
