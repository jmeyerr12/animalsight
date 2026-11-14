
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "oxford-pet"

BREEDS = {
    "beagle",
    "boxer",
    "chihuahua",
    "pug",
    "samoyed",
}


def load_dataset(split):
    annotation_file = DATASET / "annotations" / f"{split}.txt"

    paths = []
    labels = []

    for line in annotation_file.read_text().splitlines():
        if not line or line.startswith("#"):
            continue

        image_id = line.split()[0]
        breed = image_id.rsplit("_", 1)[0].lower()

        if breed not in BREEDS:
            continue

        path = DATASET / "images" / f"{image_id}.jpg"

        if not path.is_file():
            raise FileNotFoundError(path)

        paths.append(path)
        labels.append(breed)

    return paths, np.array(labels)


def train_and_evaluate(
    train_features,
    train_labels,
    test_features,
    test_labels,
    name
):
    classifier = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10)
    )

    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="macro")
    report = classification_report(test_labels, predictions)

    output = (
        f"Accuracy: {accuracy:.4f}\n"
        f"F1 macro: {f1:.4f}\n\n"
        f"{report}"
    )

    print(output)

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    joblib.dump(classifier, ROOT / "models" / f"{name}.joblib")
    (ROOT / "results" / f"{name}_report.txt").write_text(output)
