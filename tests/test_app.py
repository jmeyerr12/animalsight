import unittest

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from app import decode_upload
from src.pipeline import ROOT


class AppTests(unittest.TestCase):
    def test_upload_decoder_preserves_bgr_pixels(self):
        expected = np.zeros((20, 30, 3), dtype=np.uint8)
        expected[:, :, 0] = 255
        _, data = cv2.imencode('.png', expected)
        np.testing.assert_array_equal(decode_upload(data.tobytes()), expected)
        for contents in (b'', b'not an image'):
            with self.assertRaises(ValueError):
                decode_upload(contents)

    def test_app_starts_without_upload_or_loading_weights(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, 'AnimalSight')
        self.assertEqual(len(app.selectbox[0].options), 3)
        app.selectbox[0].select(2).run()
        self.assertEqual(len(app.exception), 0)

    def test_upload_shows_four_previews_and_invalid_upload_shows_error(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        _, data = cv2.imencode('.png', image)
        app.get('file_uploader')[0].upload('dog.png', data.tobytes(), 'image/png').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get('image')), 4)
        app.get('file_uploader')[0].clear().upload('broken.png', b'invalid', 'image/png').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)


if __name__ == '__main__':
    unittest.main()
