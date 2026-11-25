"""Official dataset splits and the shared supervised classification pipeline."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "oxford-pet"
BREEDS = {
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
}
SVM_PARAMETERS = {"kernel": "rbf", "C": 10, "gamma": "scale"}


def load_dataset(split, dataset=DATASET):
    if split not in {"trainval", "test"}:
        raise ValueError("Use the official trainval or test split.")
    dataset = Path(dataset)
    annotation_file = dataset / "annotations" / f"{split}.txt"
    paths, labels = [], []
    for line in annotation_file.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        image_id = line.split()[0]
        breed = image_id.rsplit("_", 1)[0].lower()
        if breed not in BREEDS:
            continue
        path = dataset / "images" / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        labels.append(breed)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"Empty or duplicate dataset entries: {annotation_file}")
    return paths, np.array(labels)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def file_manifest(paths):
    return [{"path": str(Path(path).resolve()),
             "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in paths]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def train_classifier(features, labels):
    classifier = make_pipeline(StandardScaler(), SVC(**SVM_PARAMETERS))
    classifier.fit(features, labels)
    return classifier


def prediction_metrics(labels, predictions):
    classes = sorted(set(labels) | set(predictions))
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "samples": len(labels), "classes": classes,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
            "report": classification_report(labels, predictions, labels=classes,
                                             output_dict=True, zero_division=0)}
