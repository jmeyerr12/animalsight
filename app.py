"""Local AnimalSight demo using exactly the experiment's inference path."""
from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st

from src.benchmark import timed_prediction
from src.experiments import load_trained_artifacts
from src.pipeline import BREEDS, ROOT
from src.roi import extract_roi
from src.transformer_model import config_id, configurations, load_runtime, prepare_image


def decode_upload(contents):
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR) if contents else None
    if image is None:
        raise ValueError('This file could not be decoded as an image.')
    return image


@st.cache_resource(show_spinner='Loading DINOv2 and classifiers…')
def load_resources(artifact_stamps):
    start = time.perf_counter()
    runtime = load_runtime(local_files_only=True)
    models_dir = Path(artifact_stamps[0][0]).parent
    artifacts = load_trained_artifacts(runtime, configurations(), models_dir)
    return runtime, artifacts, time.perf_counter() - start


def main():
    st.set_page_config(page_title='AnimalSight', layout='wide')
    st.title('AnimalSight')
    st.write('Classical ROI selection + frozen DINOv2 + SVM')
    breeds = ', '.join(breed.replace('_', ' ') for breed in sorted(BREEDS))
    st.caption(f'UFPR Computer Vision · Five breeds: {breeds}.')
    labels = ['Baseline · 224 × 224', 'Reduced · 112 × 112', 'ROI Reduced · 112 × 112']
    selected = st.selectbox('Configuration', range(3), format_func=lambda i: labels[i])
    upload = st.file_uploader('Upload a dog photograph', type=['jpg', 'jpeg', 'png', 'webp'])
    configs = configurations()
    model_paths = [ROOT / 'models' / f'{config_id(config)}.joblib' for config in configs]
    missing = [path.name for path in model_paths if not path.is_file()]
    if missing:
        st.info('Train the classifiers first: python -m src.experiments')
    if upload is None:
        st.caption('Predictions are restricted to these five breeds; other animals or objects are not detected.')
        return
    try:
        image = decode_upload(upload.getvalue())
        _, (x1, y1, x2, y2), gradient = extract_roi(image, return_gradient=True)
    except (ValueError, cv2.error) as error:
        st.error(str(error))
        return
    columns = st.columns(4)
    columns[0].image(image, channels='BGR', caption='Original photograph')
    preview = image.copy()
    cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    columns[1].image(preview, channels='BGR', caption='Classical ROI selection')
    normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    columns[2].image(normalized, caption='Gradient magnitude')
    columns[3].image(prepare_image(image, configs[selected]), caption='Actual model input (before normalization)')
    st.caption('Baseline and Reduced keep the entire photograph, preserve its proportions and fill the remaining borders. ROI Reduced keeps only the selected region. No center crop is applied.')
    if st.button('Classify', disabled=bool(missing), type='primary'):
        try:
            stamps = tuple((str(path), path.stat().st_mtime_ns) for path in model_paths)
            runtime, artifacts, load_seconds = load_resources(stamps)
            with st.spinner('Classifying…'):
                # Recompute ROI inside timed prediction so its cost is included.
                prediction, latency = timed_prediction(image, runtime, artifacts[selected])
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f'Unable to classify: {error}')
            st.info('Run python -m src.experiments with the current environment, then retry.')
            return
        st.success(f'Predicted breed: {prediction.replace("_", " ").title()}')
        st.metric('Inference time', f'{latency:.2f} ms')
        st.caption(f'Device: {runtime["device"]} · Batch size: 1 · Initial model load: {load_seconds:.2f} s (cached, excluded from inference).')
        st.caption('Inference includes preprocessing, ROI when selected, DINOv2 and SVM. Upload decoding and previews are excluded. A first prediction may be slower than the warmed benchmark.')


if __name__ == '__main__':
    main()
