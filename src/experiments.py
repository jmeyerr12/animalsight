"""Validate on trainval, refit on trainval, and evaluate the untouched official test."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from src.pipeline import (BREEDS, DATASET, ROOT, SVM_PARAMETERS, evaluate_classifier,
                          file_manifest, fingerprint, load_dataset, train_classifier,
                          validation_indices, write_json)
from src.roi import save_roi_examples
from src.transformer_model import (config_id, configurations, extract_features, load_artifact,
                                   load_runtime, make_artifact)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def result_row(config, metrics):
    return {"configuration": config_id(config), "resolution": config['resolution'],
            "tokens": 1 + (config['resolution'] // 14) ** 2, **metrics}


def train_experiments(runtime, configs, *, dataset=DATASET, models_dir=ROOT / 'models',
                      results_dir=ROOT / 'results/roi_112', batch_size=4, seed=42,
                      validation_size=0.2):
    paths, labels = load_dataset('trainval', dataset)
    if set(labels) != BREEDS:
        raise ValueError('The trainval split must contain all five target breeds.')
    train, val = validation_indices(labels, validation_size, seed)
    training = {"files": file_manifest(paths), "labels": labels.tolist(),
                "train_indices": train.tolist(), "validation_indices": val.tolist(),
                "seed": seed, "validation_size": validation_size,
                "svm": SVM_PARAMETERS, "final_fit": "all_trainval"}
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in configs:
        features = extract_features(paths, runtime, config, cache_dir=models_dir / 'cache', batch_size=batch_size)
        validation_classifier = train_classifier(features[train], labels[train])
        metrics = evaluate_classifier(validation_classifier, features[val], labels[val])
        rows.append(result_row(config, metrics))
        final_classifier = train_classifier(features, labels)
        artifact = make_artifact(final_classifier, runtime, config, training)
        joblib.dump(artifact, models_dir / f'{config_id(config)}.joblib')
        print(f"Validation {config_id(config)}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}", flush=True)
    write_json(results_dir / 'training.json', {"runtime": runtime['identity'], "training": training,
                                               "configurations": configs, "validation": rows})
    write_csv(results_dir / 'validation.csv', rows, ['configuration', 'resolution', 'tokens', 'samples', 'accuracy', 'macro_f1'])
    scale = next(config['roi_scale'] for config in configs if config['roi_scale'] is not None)
    samples = save_roi_examples([paths[i] for i in val], labels[val], results_dir / 'roi_validation.png',
                               scale=scale, seed=seed)
    write_json(results_dir / 'roi_samples.json', {"seed": seed, "source": "trainval_validation", "paths": samples})
    return rows


def load_trained_artifacts(runtime, configs, models_dir):
    artifacts = [load_artifact(models_dir / f'{config_id(config)}.joblib', runtime) for config in configs]
    for artifact, config in zip(artifacts, configs):
        if artifact['representation']['configuration'] != config:
            raise ValueError('Saved classifier does not match the requested configuration.')
    if len({fingerprint(artifact['training']) for artifact in artifacts}) != 1:
        raise ValueError('Classifiers were not trained on identical splits and SVM settings.')
    return artifacts


def evaluation_data(artifacts, dataset):
    if not artifacts or len({fingerprint(a['training']) for a in artifacts}) != 1:
        raise ValueError('Classifiers must use the same training data and settings.')
    training = artifacts[0]['training']
    train_paths, train_labels = load_dataset('trainval', dataset)
    if training['files'] != file_manifest(train_paths) or training['labels'] != train_labels.tolist():
        raise ValueError('Dataset differs from classifier training metadata; retrain before evaluation.')
    paths, labels = load_dataset('test', dataset)
    if set(labels) != BREEDS or set(train_paths) & set(paths):
        raise ValueError('Invalid official test split: missing breeds or trainval overlap.')
    train_hashes = {entry['sha256'] for entry in training['files']}
    if train_hashes & {entry['sha256'] for entry in file_manifest(paths)}:
        raise ValueError('Identical image contents occur in trainval and test.')
    return paths, labels


def evaluate_experiments(runtime, configs, *, dataset=DATASET, models_dir=ROOT / 'models',
                         results_dir=ROOT / 'results/roi_112', batch_size=4):
    artifacts = load_trained_artifacts(runtime, configs, models_dir)
    paths, labels = evaluation_data(artifacts, dataset)
    rows, predictions = [], []
    for config, artifact in zip(configs, artifacts):
        features = extract_features(paths, runtime, config, cache_dir=models_dir / 'cache', batch_size=batch_size)
        metrics = evaluate_classifier(artifact['classifier'], features, labels)
        rows.append(result_row(config, metrics))
        predicted = artifact['classifier'].predict(features)
        predictions.extend({"configuration": config_id(config), "image": str(path), "label": label, "prediction": pred}
                           for path, label, pred in zip(paths, labels, predicted))
        print(f"Test {config_id(config)}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}", flush=True)
    write_json(results_dir / 'evaluation.json', {"runtime": runtime['identity'], "test_files": file_manifest(paths),
                                                "training_fingerprint": fingerprint(artifacts[0]['training']),
                                                "results": rows})
    write_csv(results_dir / 'evaluation.csv', rows, ['configuration', 'resolution', 'tokens', 'samples', 'accuracy', 'macro_f1'])
    write_csv(results_dir / 'predictions.csv', predictions, ['configuration', 'image', 'label', 'prediction'])
    return rows


def add_common_arguments(parser):
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--reduced-size', type=int, choices=[112, 168], default=112)
    parser.add_argument('--roi-scale', type=float, default=0.8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--offline', action='store_true', help='Use already downloaded Hugging Face weights.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'all'])
    add_common_arguments(parser)
    parser.add_argument('--batch-size', type=int, default=4, help='Embedding extraction batch size; timing always uses one.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-size', type=float, default=0.2)
    parser.add_argument('--repeats', type=int, default=3, help='Benchmark repeats for the all command.')
    parser.add_argument('--warmup', type=int, default=5, help='Benchmark warm-up for the all command.')
    args = parser.parse_args()
    configs = configurations(args.reduced_size, args.roi_scale)
    results_dir = args.results_dir or ROOT / f'results/roi_{args.reduced_size}'
    runtime = load_runtime(args.device, local_files_only=args.offline, threads=args.threads)
    common = dict(dataset=args.dataset, models_dir=args.models_dir, results_dir=results_dir, batch_size=args.batch_size)
    if args.command in ('train', 'all'):
        train_experiments(runtime, configs, seed=args.seed, validation_size=args.validation_size, **common)
    if args.command in ('evaluate', 'all'):
        evaluate_experiments(runtime, configs, **common)
    if args.command == 'all':
        from src.benchmark import run_benchmark
        run_benchmark(runtime, configs, dataset=args.dataset, models_dir=args.models_dir,
                      results_dir=results_dir, repeats=args.repeats, warmup=args.warmup, seed=args.seed)


if __name__ == '__main__':
    main()
