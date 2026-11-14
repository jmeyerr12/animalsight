
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from src.pipeline import ROOT, load_dataset, train_and_evaluate

MODEL_NAME = "facebook/dinov2-small"


def extract_features(paths, cache_name):
    cache_path = ROOT / "models" / f"{cache_name}_features.npz"
    file_paths = [str(path) for path in paths]

    if cache_path.exists():
        with np.load(cache_path) as cache:
            if np.array_equal(cache["paths"], file_paths):
                print("Loading cached features...")
                return cache["features"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device).eval()

    batch_size = 8 if device.type == "cuda" else 4
    embeddings = []

    for start in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[start:start + batch_size]
        images = []

        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        inputs = processor(
            images=images,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

        embeddings.append(features.cpu().numpy())

    embeddings = np.vstack(embeddings)

    cache_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=embeddings,
        paths=file_paths
    )

    return embeddings



def main():
    train_paths, train_labels = load_dataset("trainval")
    test_paths, test_labels = load_dataset("test")

    print("Extracting transformer features...")

    train_features = extract_features(
        train_paths, "dinov2_train"
    )

    test_features = extract_features(
        test_paths, "dinov2_test"
    )

    train_and_evaluate(
        train_features,
        train_labels,
        test_features,
        test_labels,
        "transformer"
    )


if __name__ == "__main__":
    main()