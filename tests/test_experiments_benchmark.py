import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

from src import transformer_model as tm
from src import experiments, benchmark
from src.pipeline import BREEDS, read_image


class ExperimentTests(unittest.TestCase):
    def test_training_evaluation_and_timing_share_representations_and_splits(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        processor = BitImageProcessor(size={'shortest_edge': 256}, crop_size={'height': 224, 'width': 224})
        model = Dinov2Model(Dinov2Config(image_size=224, patch_size=14, hidden_size=12,
                                       num_hidden_layers=1, num_attention_heads=3, intermediate_size=24))
        runtime = tm.make_runtime(processor, model, 'cpu', revision='unit-test')
        runtime['shared_storage_bytes'] = 1000

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / 'data'
            (dataset / 'images').mkdir(parents=True)
            (dataset / 'annotations').mkdir()
            rng = np.random.default_rng(42)
            for split, numbers in [('trainval', range(5)), ('test', range(5, 7))]:
                names = [f'{breed}_{i}' for breed in sorted(BREEDS) for i in numbers]
                (dataset / 'annotations' / f'{split}.txt').write_text('\n'.join(names))
                for name in names:
                    cv2.imwrite(str(dataset / 'images' / f'{name}.jpg'),
                                rng.integers(0, 256, (28, 39, 3), dtype=np.uint8))
            configs = tm.configurations()
            models_dir, results_dir = root / 'models', root / 'results'
            training = experiments.train_experiments(runtime, configs, dataset=dataset,
                                                    models_dir=models_dir, results_dir=results_dir, batch_size=5)
            self.assertEqual([r['samples'] for r in training], [5, 5, 5])
            artifacts = experiments.load_trained_artifacts(runtime, configs, models_dir)
            for artifact in artifacts:
                self.assertEqual(artifact['classifier'][0].n_samples_seen_, 25)
            evaluation = experiments.evaluate_experiments(runtime, configs, dataset=dataset,
                                                          models_dir=models_dir, results_dir=results_dir, batch_size=5)
            self.assertEqual([r['samples'] for r in evaluation], [10, 10, 10])
            paths, labels = experiments.evaluation_data(artifacts, dataset)
            images = [read_image(path) for path in paths]
            rows, timings = benchmark.measure(runtime, artifacts, images, labels, repeats=2, warmup=1)
            self.assertEqual(len(timings), 60)
            self.assertEqual([r['measurements'] for r in rows], [20, 20, 20])
            self.assertTrue(all(r['mean_latency_ms'] > 0 for r in rows))
            self.assertEqual(rows[0]['speedup'], 1.)
            for measured, evaluated in zip(rows, evaluation):
                self.assertEqual(measured['accuracy'], evaluated['accuracy'])
            artifact = artifacts[0]
            prediction, latency = benchmark.timed_prediction(images[0], runtime, artifact)
            self.assertIn(prediction, BREEDS)
            self.assertGreater(latency, 0)
            storage = benchmark.storage_requirements(runtime, models_dir / f'{tm.config_id(configs[0])}.joblib')
            self.assertEqual(storage['total_storage_bytes'], 1000 + storage['svm_storage_bytes'])
            artifacts[1]['training'] = {'different_split': True}
            with self.assertRaises(ValueError):
                experiments.evaluation_data(artifacts, dataset)

    def test_benchmark_rejects_empty_data_or_no_warmup(self):
        for images, labels, warmup in [([], [], 1), ([None], ['a'], 0)]:
            with self.assertRaises(ValueError):
                benchmark.measure({}, [], images, labels, warmup=warmup)


if __name__ == '__main__':
    unittest.main()
