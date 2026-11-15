"""One frozen DINOv2 runtime for extraction, benchmarking, and application inference."""
from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers.utils.hub import cached_file

from src.pipeline import ROOT, file_manifest, fingerprint, read_image
from src.roi import ROI_VERSION, extract_roi, validate_image

MODEL_NAME = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
SCHEMA_VERSION = 1
PREPROCESSING = "rgb-pil-bicubic-shortest-edge-256/224-center-crop-v1"


def configurations(reduced_size=112, roi_scale=0.8):
    if reduced_size not in (112, 168) or not 0 < roi_scale <= 1:
        raise ValueError("Reduced size must be 112 or 168; ROI scale must be in (0, 1].")
    return [{"name": "baseline", "resolution": 224, "roi_scale": None},
            {"name": "reduced", "resolution": reduced_size, "roi_scale": None},
            {"name": "roi_reduced", "resolution": reduced_size, "roi_scale": roi_scale}]


def config_id(config):
    name = f"{config['name']}_{config['resolution']}"
    if config['roi_scale'] is not None:
        name += "_s" + str(config['roi_scale']).replace('.', 'p')
    return name


def make_runtime(processor, model, device, *, revision=MODEL_REVISION):
    device = torch.device(device)
    model = model.to(device).eval()
    model.requires_grad_(False)
    identity = {"model": MODEL_NAME, "revision": revision,
                "hidden_size": model.config.hidden_size, "patch_size": model.config.patch_size,
                "processor": processor.to_dict(), "processor_class": type(processor).__name__,
                "preprocessing": PREPROCESSING, "roi_algorithm": ROI_VERSION,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "opencv": cv2.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
                "dtype": str(next(model.parameters()).dtype),
                "attention": model.config._attn_implementation}
    identity = json.loads(json.dumps(identity))
    weight_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return {"model": model, "processor": processor, "device": device,
            "identity": identity, "parameter_bytes": weight_bytes}


def load_runtime(device="auto", *, local_files_only=False, threads=4):
    if threads < 1:
        raise ValueError("Thread count must be positive.")
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"revision": MODEL_REVISION, "local_files_only": local_files_only}
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, backend="pil", **kwargs)
    model = AutoModel.from_pretrained(MODEL_NAME, **kwargs)
    runtime = make_runtime(processor, model, device)
    files = [cached_file(MODEL_NAME, name, **kwargs)
             for name in ("model.safetensors", "config.json", "preprocessor_config.json")]
    runtime["shared_storage_bytes"] = sum(Path(path).stat().st_size for path in files)
    return runtime


def prepare_inputs(images, runtime, config):
    rgb_images = []
    for image in images:
        validate_image(image)
        if config['roi_scale'] is not None:
            image, _ = extract_roi(image, config['roi_scale'])
        rgb_images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    size = config['resolution']
    inputs = runtime['processor'](images=rgb_images, return_tensors="pt",
                                  size={"shortest_edge": size * 256 // 224},
                                  crop_size={"height": size, "width": size},
                                  do_resize=True, do_center_crop=True)
    if tuple(inputs['pixel_values'].shape) != (len(images), 3, size, size):
        raise ValueError(f"Unexpected pixel_values shape: {inputs['pixel_values'].shape}")
    return inputs.to(runtime['device'])


def embed_images(images, runtime, config):
    inputs = prepare_inputs(images, runtime, config)
    with torch.inference_mode():
        # Hugging Face DINOv2 interpolates positional embeddings inside forward.
        hidden = runtime['model'](**inputs).last_hidden_state
        tokens = 1 + (config['resolution'] // runtime['model'].config.patch_size) ** 2
        expected = (len(images), tokens, runtime['model'].config.hidden_size)
        if tuple(hidden.shape) != expected:
            raise ValueError(f"Unexpected DINOv2 output shape: {tuple(hidden.shape)} != {expected}")
        features = hidden[:, 0, :].cpu().numpy()
    if not np.isfinite(features).all():
        raise ValueError("Non-finite embeddings.")
    return features


def representation(runtime, config):
    return {"schema": SCHEMA_VERSION, "runtime": runtime['identity'], "configuration": config}


def extract_features(paths, runtime, config, *, cache_dir=ROOT / 'models/cache', batch_size=4):
    if not paths or batch_size < 1:
        raise ValueError("Provide images and a positive batch size.")
    metadata = {**representation(runtime, config), "files": file_manifest(paths)}
    cache_path = Path(cache_dir) / f"{config_id(config)}_{fingerprint(metadata)}.npz"
    expected_shape = (len(paths), runtime['model'].config.hidden_size)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            features = cached['features']
            if (json.loads(str(cached['metadata'])) == metadata
                    and features.shape == expected_shape and np.isfinite(features).all()):
                print(f"Cache hit: {config_id(config)}, {len(paths)} images", flush=True)
                return features
    batches = []
    for start in range(0, len(paths), batch_size):
        images = [read_image(path) for path in paths[start:start + batch_size]]
        batches.append(embed_images(images, runtime, config))
        if start == 0 or (start // batch_size + 1) % 25 == 0:
            print(f"{config_id(config)}: {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    features = np.vstack(batches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, metadata=json.dumps(metadata, sort_keys=True))
    return features


def make_artifact(classifier, runtime, config, training):
    return {"representation": representation(runtime, config),
            "classifier": classifier, "training": training}


def load_artifact(path, runtime):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or 'representation' not in artifact:
        raise ValueError("Legacy classifier; retrain with python -m src.experiments train.")
    config = artifact['representation']['configuration']
    if artifact['representation'] != representation(runtime, config):
        raise ValueError("Classifier preprocessing/model version differs from this runtime. Retrain the classifier.")
    if artifact['classifier'].n_features_in_ != runtime['model'].config.hidden_size:
        raise ValueError("Classifier embedding dimension mismatch.")
    return artifact


def predict_image(image, runtime, artifact):
    config = artifact['representation']['configuration']
    return str(artifact['classifier'].predict(embed_images([image], runtime, config))[0])
