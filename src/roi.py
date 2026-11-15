"""Classical gradient concentration ROI, without labels or learned detectors."""
import cv2
import numpy as np

ROI_VERSION = "sobel-gaussian-window-v1"


def validate_image(image):
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.size == 0 or image.dtype != np.uint8):
        raise ValueError("Expected a nonempty uint8 OpenCV BGR image with three channels.")


def extract_roi(image, scale=0.8, *, return_gradient=False):
    """Return (crop, (x1, y1, x2, y2)), optionally followed by the gradient map."""
    validate_image(image)
    if not 0 < scale <= 1:
        raise ValueError("ROI scale must be in (0, 1].")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.GaussianBlur(cv2.magnitude(gx, gy), (15, 15), 0)
    size = max(1, int(min(height, width) * scale))
    scores = cv2.boxFilter(gradient, cv2.CV_32F, (size, size), normalize=False)
    half = size // 2
    valid = scores[half:half + height - size + 1, half:half + width - size + 1]
    _, _, _, (x, y) = cv2.minMaxLoc(valid)
    box = (x, y, x + size, y + size)
    crop = image[y:y + size, x:x + size]
    return (crop, box, gradient) if return_gradient else (crop, box)


def save_roi_examples(paths, labels, output, *, scale=0.8, seed=42, count=15):
    """Draw randomly sampled images, without selecting by visual quality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.pipeline import read_image

    indices = np.random.default_rng(seed).choice(len(paths), min(count, len(paths)), replace=False)
    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 2.6 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        image = read_image(paths[index])
        crop, (x1, y1, x2, y2), gradient = extract_roi(image, scale, return_gradient=True)
        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        axes[row, 0].imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        axes[row, 1].imshow(gradient, cmap="inferno")
        axes[row, 2].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"{paths[index].name} ({labels[index]})", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    axes[0, 1].set_title("Gradient magnitude")
    axes[0, 2].set_title("Selected ROI")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return [str(paths[index]) for index in indices]


def main():
    import argparse
    from src.pipeline import DATASET, ROOT, load_dataset
    parser = argparse.ArgumentParser(description="Preview ROIs from training images.")
    parser.add_argument("--dataset", type=type(DATASET), default=DATASET)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=15)
    args = parser.parse_args()
    paths, labels = load_dataset("trainval", args.dataset)
    save_roi_examples(paths, labels, ROOT / "results/roi_examples.png",
                      scale=args.scale, seed=args.seed, count=args.count)


if __name__ == "__main__":
    main()
