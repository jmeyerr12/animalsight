import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src import pipeline, roi


class RoiTests(unittest.TestCase):
    def test_crop_is_bounded_for_small_and_rectangular_images(self):
        for height, width in [(1, 1), (2, 7), (31, 60), (70, 19)]:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            for scale in (0.1, 0.8, 1.0):
                crop, box = roi.extract_roi(image, scale)
                x1, y1, x2, y2 = box
                self.assertTrue(0 <= x1 < x2 <= width)
                self.assertTrue(0 <= y1 < y2 <= height)
                self.assertEqual(crop.shape[:2], (max(1, int(min(height, width) * scale)),) * 2)
                np.testing.assert_array_equal(crop, image[y1:y2, x1:x2])

    def test_window_maximizes_gradient_sum(self):
        image = np.random.default_rng(42).integers(0, 256, (21, 31, 3), dtype=np.uint8)
        crop, (x, y, _, _), gradient = roi.extract_roi(image, 0.5, return_gradient=True)
        size = crop.shape[0]
        scores = np.array([[gradient[j:j+size, i:i+size].sum()
                            for i in range(32-size)] for j in range(22-size)])
        self.assertAlmostEqual(float(scores[y, x]), float(scores.max()), delta=0.01)

    def test_invalid_images_and_scales_are_rejected(self):
        for image in (None, np.zeros((0, 2, 3), dtype=np.uint8), np.zeros((3, 3)), np.zeros((3, 3, 3))):
            with self.assertRaises(ValueError):
                roi.extract_roi(image)
        for scale in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                roi.extract_roi(np.zeros((10, 10, 3), dtype=np.uint8), scale)


class PipelineTests(unittest.TestCase):
    def test_loader_uses_annotation_order_and_filters_breeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'annotations').mkdir()
            (root / 'images').mkdir()
            (root / 'annotations/trainval.txt').write_text('# header\nboxer_2 1\nAbyssinian_1 2\nbeagle_1 3\n')
            for name in ('boxer_2', 'beagle_1'):
                cv2.imwrite(str(root / 'images' / f'{name}.jpg'), np.zeros((4, 4, 3), dtype=np.uint8))
            paths, labels = pipeline.load_dataset('trainval', root)
            self.assertEqual([p.stem for p in paths], ['boxer_2', 'beagle_1'])
            self.assertEqual(labels.tolist(), ['boxer', 'beagle'])
            paths[0].unlink()
            with self.assertRaises(FileNotFoundError):
                pipeline.load_dataset('trainval', root)
            with self.assertRaises(ValueError):
                pipeline.load_dataset('../test', root)

    def test_validation_is_disjoint_stratified_and_repeatable(self):
        labels = np.repeat(sorted(pipeline.BREEDS), 20)
        train, val = pipeline.validation_indices(labels)
        train_again, val_again = pipeline.validation_indices(labels)
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(val, val_again)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(100)))
        self.assertEqual(np.unique(labels[val], return_counts=True)[1].tolist(), [4] * 5)

    def test_scaler_fits_training_only(self):
        classifier = pipeline.train_classifier(np.array([[0., 0.], [1., 1.], [10., 10.], [11., 11.]]),
                                               np.array(['a', 'a', 'b', 'b']))
        np.testing.assert_array_equal(classifier[0].mean_, [5.5, 5.5])
        metrics = pipeline.evaluate_classifier(classifier, np.array([[0., 0.], [11., 11.]]), np.array(['a', 'b']))
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertEqual(metrics['macro_f1'], 1.)
        np.testing.assert_array_equal(classifier[0].mean_, [5.5, 5.5])


if __name__ == '__main__':
    unittest.main()
