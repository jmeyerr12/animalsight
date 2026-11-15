"""Train on official trainval images and compare on official test images."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from src.pipeline import (BREEDS, DATASET, ROOT, SVM_PARAMETERS, file_manifest, fingerprint,
                          load_dataset, prediction_metrics, train_classifier, write_json)
from src.transformer_model import (REDUCED_SIZES, config_id, configurations, extract_features,
                                   load_artifact, load_runtime, make_artifact)

DEFAULT_RESOLUTIONS = (112, 84, 56, 42, 28)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def result_row(config, metrics):
    return {'configuration': config_id(config), 'resolution': config['resolution'],
            'tokens': 1 + (config['resolution'] // 14) ** 2, **metrics}


def experiment_configurations(resolutions, roi_scale):
    if not resolutions or len(set(resolutions)) != len(resolutions):
        raise ValueError('Provide distinct input resolutions.')
    configs = [configurations(resolutions[0], roi_scale)[0]]
    for size in resolutions:
        configs.extend(configurations(size, roi_scale)[1:])
    return configs


def load_trained_artifacts(runtime, configs, models_dir):
    artifacts = [load_artifact(models_dir / f'{config_id(config)}.joblib', runtime) for config in configs]
    for artifact, config in zip(artifacts, configs):
        if artifact['representation']['configuration'] != config:
            raise ValueError('Saved classifier does not match the requested configuration.')
    if len({fingerprint(artifact['training']) for artifact in artifacts}) != 1:
        raise ValueError('Classifiers must use the same training data and SVM settings.')
    return artifacts


def evaluation_data(artifacts, dataset):
    training = artifacts[0]['training']
    train_paths, train_labels = load_dataset('trainval', dataset)
    if training['files'] != file_manifest(train_paths) or training['labels'] != train_labels.tolist():
        raise ValueError('Dataset differs from the saved classifiers; run the experiment again.')
    paths, labels = load_dataset('test', dataset)
    if set(labels) != BREEDS or set(train_paths) & set(paths):
        raise ValueError('Invalid official test split.')
    return paths, labels


def compare_resolutions(resolutions, roi_scale, predictions, labels):
    rows = []
    for size in resolutions:
        reduced_config, roi_config = configurations(size, roi_scale)[1:]
        reduced = predictions[config_id(reduced_config)] == labels
        roi = predictions[config_id(roi_config)] == labels
        rows.append({'resolution': size, 'tokens': 1 + (size // 14) ** 2,
                     'reduced_accuracy': float(reduced.mean()), 'roi_accuracy': float(roi.mean()),
                     'delta_accuracy_pp': float(100 * (roi.mean() - reduced.mean())),
                     'roi_only_correct': int(np.sum(roi & ~reduced)),
                     'reduced_only_correct': int(np.sum(reduced & ~roi))})
    return rows


def save_resolution_plot(rows, baseline_accuracy, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: row['resolution'])
    sizes = [row['resolution'] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, [100 * row['reduced_accuracy'] for row in rows], 'o-', label='Reduced')
    ax.plot(sizes, [100 * row['roi_accuracy'] for row in rows], 's-', label='ROI Reduced')
    ax.axhline(100 * baseline_accuracy, color='gray', linestyle='--', label='Baseline 224')
    ax.set(xlabel='Transformer input side (pixels)', ylabel='Official test accuracy (%)',
           title='AnimalSight: accuracy vs. resolution', xticks=sizes, ylim=(0, 102))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_experiments(runtime, resolutions=DEFAULT_RESOLUTIONS, *, roi_scale=0.8,
                    dataset=DATASET, models_dir=ROOT / 'models',
                    results_dir=ROOT / 'results/experiment', batch_size=4):
    configs = experiment_configurations(resolutions, roi_scale)
    train_paths, train_labels = load_dataset('trainval', dataset)
    test_paths, test_labels = load_dataset('test', dataset)
    if set(train_labels) != BREEDS or set(test_labels) != BREEDS or set(train_paths) & set(test_paths):
        raise ValueError('Expected five breeds and separate official trainval/test splits.')
    training = {'files': file_manifest(train_paths), 'labels': train_labels.tolist(),
                'svm': SVM_PARAMETERS, 'final_fit': 'all_trainval'}
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = models_dir / 'cache'
    rows, predictions, prediction_rows = [], {}, []
    print(f'Training: {len(train_paths)} images | Test: {len(test_paths)} images', flush=True)

    for config in configs:
        name = config_id(config)
        train_features = extract_features(train_paths, runtime, config, cache_dir=cache_dir, batch_size=batch_size)
        classifier = train_classifier(train_features, train_labels)
        joblib.dump(make_artifact(classifier, runtime, config, training), models_dir / f'{name}.joblib')

        test_features = extract_features(test_paths, runtime, config, cache_dir=cache_dir, batch_size=batch_size)
        predicted = classifier.predict(test_features)
        predictions[name] = predicted
        metrics = prediction_metrics(test_labels, predicted)
        rows.append(result_row(config, metrics))
        prediction_rows.extend({'configuration': name, 'image': str(path), 'label': label, 'prediction': pred}
                               for path, label, pred in zip(test_paths, test_labels, predicted))
        print(f"{name}: accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.4f}", flush=True)

    comparison = compare_resolutions(resolutions, roi_scale, predictions, test_labels)
    write_csv(results_dir / 'evaluation.csv', rows,
              ['configuration', 'resolution', 'tokens', 'samples', 'accuracy', 'macro_f1'])
    write_csv(results_dir / 'comparison.csv', comparison, list(comparison[0]))
    write_csv(results_dir / 'predictions.csv', prediction_rows, ['configuration', 'image', 'label', 'prediction'])
    write_json(results_dir / 'experiment.json', {'runtime': runtime['identity'], 'training': training,
                                                'test_files': file_manifest(test_paths),
                                                'configurations': configs, 'results': rows})
    save_resolution_plot(comparison, rows[0]['accuracy'], results_dir / 'accuracy_vs_resolution.png')
    print(f'\nResults saved to {results_dir}', flush=True)
    return rows


def add_common_arguments(parser):
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--roi-scale', type=float, default=0.8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--offline', action='store_true', help='Use already downloaded weights.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument('--resolutions', type=int, nargs='+', choices=REDUCED_SIZES,
                        default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    experiment_configurations(args.resolutions, args.roi_scale)
    runtime = load_runtime(args.device, local_files_only=args.offline, threads=args.threads)
    run_experiments(runtime, args.resolutions, roi_scale=args.roi_scale, dataset=args.dataset,
                    models_dir=args.models_dir, results_dir=args.results_dir or ROOT / 'results/experiment',
                    batch_size=args.batch_size)


if __name__ == '__main__':
    main()
