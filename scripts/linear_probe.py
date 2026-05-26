from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


def evaluate_features(x: np.ndarray, y: np.ndarray, name: str) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test, preds))


def main() -> None:
    meta_path = Path("data/sample_metadata.csv")
    resnet_path = Path("data/resnet_features_demo.npy")
    vit_path = Path("data/vit_features_demo.npy")

    for path in [meta_path, resnet_path, vit_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    meta = pd.read_csv(meta_path)
    y = meta["movement"].to_numpy()

    resnet = np.load(resnet_path)
    vit = np.load(vit_path)

    evaluate_features(resnet, y, "ResNet-50")
    evaluate_features(vit, y, "Vision Transformer")


if __name__ == "__main__":
    main()