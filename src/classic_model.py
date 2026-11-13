
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent.parent


def extrair_caracteristicas(caminho):
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Não foi possível abrir: {caminho}")

    img = img.astype(np.float32) / 255.0
    features = []

    # Gabor: 4 orientations
    for angulo in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            (15, 15), 3.5, angulo, 8, 0.5, 0,
            ktype=cv2.CV_32F
        )

        kernel /= np.sum(np.abs(kernel)) + 1e-8

        resposta = cv2.filter2D(img, cv2.CV_32F, kernel)

        features.append(np.abs(resposta).mean())
        features.append(resposta.std())

    # LoG: Gaussian + Laplacian
    suavizada = cv2.GaussianBlur(img, (5, 5), 1.2)
    log = cv2.Laplacian(suavizada, cv2.CV_32F, ksize=3)

    features.append(np.abs(log).mean())
    features.append(log.std())

    return np.array(features, dtype=np.float32)


def main():
    extensoes = {".jpg", ".jpeg", ".png", ".bmp"}

    arquivos = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensoes
        and not any(
            parte in {".venv", ".git", "results"}
            for parte in p.relative_to(ROOT).parts
        )
    )

    y = np.array([p.parent.name for p in arquivos])

    print("Extraindo características...")
    X = np.array([extrair_caracteristicas(p) for p in arquivos])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10)
    )

    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    print("\nAcurácia:", accuracy_score(y_test, previsoes))
    print("F1 macro:", f1_score(y_test, previsoes, average="macro"))
    print("\n", classification_report(y_test, previsoes))

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(modelo, ROOT / "models" / "classico.joblib")

    print("Modelo salvo em models/classico.joblib")


if __name__ == "__main__":
    main()
