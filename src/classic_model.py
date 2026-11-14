
import cv2
import numpy as np

from src.pipeline import load_dataset, train_and_evaluate


def extract_features(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    image = image.astype(np.float32) / 255.0
    features = []

    # Gabor
    for angle in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            (15, 15), 3.5, angle, 8, 0.5, 0,
            ktype=cv2.CV_32F
        )

        kernel /= np.sum(np.abs(kernel)) + 1e-8
        response = cv2.filter2D(image, cv2.CV_32F, kernel)

        features.append(np.abs(response).mean())
        features.append(response.std())

    # Laplacian
    blurred = cv2.GaussianBlur(image, (5, 5), 1.2)
    laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

    features.append(np.abs(laplacian).mean())
    features.append(laplacian.std())

    return np.array(features, dtype=np.float32)



def main():
    train_paths, train_labels = load_dataset("trainval")
    test_paths, test_labels = load_dataset("test")

    print("Extracting classic features...")

    train_features = np.array([
        extract_features(path) for path in train_paths
    ])

    test_features = np.array([
        extract_features(path) for path in test_paths
    ])

    train_and_evaluate(
        train_features,
        train_labels,
        test_features,
        test_labels,
        "classic"
    )


if __name__ == "__main__":
    main()
