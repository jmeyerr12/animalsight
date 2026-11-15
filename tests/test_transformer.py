import tempfile
import unittest
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

from src import transformer_model as tm
from src.pipeline import train_classifier


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.image = np.random.default_rng(3).integers(0, 256, (81, 113, 3), dtype=np.uint8)
        cls.processor = BitImageProcessor(size={'shortest_edge': 256}, crop_size={'height': 224, 'width': 224})
        cls.model = Dinov2Model(Dinov2Config(image_size=224, patch_size=14, hidden_size=24,
                                          num_hidden_layers=1, num_attention_heads=3, intermediate_size=48))

    def runtime(self):
        return tm.make_runtime(self.processor, self.model, 'cpu', revision='test-random-weights')

    def test_real_transformer_shapes_and_positional_interpolation(self):
        runtime = self.runtime()
        for config in tm.configurations() + tm.configurations(168)[1:]:
            inputs = tm.prepare_inputs([self.image], runtime, config)
            size = config['resolution']
            self.assertEqual(tuple(inputs['pixel_values'].shape), (1, 3, size, size))
            with torch.inference_mode():
                expected = self.model(**inputs).last_hidden_state
            self.assertEqual(expected.shape[1], 1 + (size // 14) ** 2)
            features = tm.embed_images([self.image], runtime, config)
            self.assertEqual(features.shape, (1, 24))
            np.testing.assert_allclose(features, expected[:, 0].numpy(), atol=1e-6)
        self.assertFalse(self.model.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_cache_invalidates_order_content_and_representation(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / f'{i}.png' for i in range(2)]
            cv2.imwrite(str(paths[0]), self.image)
            cv2.imwrite(str(paths[1]), 255 - self.image)
            cache = root / 'cache'
            config = tm.configurations()[1]
            first = tm.extract_features(paths, runtime, config, cache_dir=cache)
            cache_files = list(cache.glob('*.npz'))
            self.assertEqual(len(cache_files), 1)
            timestamp = cache_files[0].stat().st_mtime_ns
            again = tm.extract_features(paths, runtime, config, cache_dir=cache)
            np.testing.assert_array_equal(first, again)
            self.assertEqual(cache_files[0].stat().st_mtime_ns, timestamp)
            reverse = tm.extract_features(paths[::-1], runtime, config, cache_dir=cache)
            np.testing.assert_allclose(reverse, first[::-1], atol=1e-6)
            cv2.imwrite(str(paths[0]), np.zeros_like(self.image))
            changed = tm.extract_features(paths, runtime, config, cache_dir=cache)
            self.assertFalse(np.allclose(first[0], changed[0]))
            tm.extract_features(paths, runtime, tm.configurations()[2], cache_dir=cache)
            tm.extract_features(paths, runtime, tm.configurations(roi_scale=0.7)[2], cache_dir=cache)
            self.assertEqual(len(list(cache.glob('*.npz'))), 5)

    def test_saved_prediction_matches_batch_features_and_rejects_wrong_runtime(self):
        runtime = self.runtime()
        config = tm.configurations()[2]
        images = [self.image, 255 - self.image, np.zeros_like(self.image), np.full_like(self.image, 255)]
        features = tm.embed_images(images, runtime, config)
        classifier = train_classifier(features, ['a', 'a', 'b', 'b'])
        artifact = tm.make_artifact(classifier, runtime, config, {'test_fixture': True})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(artifact, path)
            restored = tm.load_artifact(path, runtime)
            before = list(Path(folder).iterdir())
            prediction = tm.predict_image(self.image, runtime, restored)
            self.assertEqual(prediction, classifier.predict(features[:1])[0])
            self.assertEqual(before, list(Path(folder).iterdir()))
            wrong = {**runtime, 'identity': {**runtime['identity'], 'revision': 'other'}}
            with self.assertRaises(ValueError):
                tm.load_artifact(path, wrong)


if __name__ == '__main__':
    unittest.main()
