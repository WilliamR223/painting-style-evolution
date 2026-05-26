from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def get_scores(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    return acc, macro_f1


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

    resnet_acc, resnet_f1 = get_scores(resnet, y)
    vit_acc, vit_f1 = get_scores(vit, y)

    comparison = pd.DataFrame(
        {
            "Model": ["ResNet-50", "Vision Transformer"],
            "Accuracy": [resnet_acc, vit_acc],
            "Macro-F1": [resnet_f1, vit_f1],
        }
    )

    print(comparison)
    comparison.to_csv("data/model_comparison.csv", index=False)
    print("Saved comparison to data/model_comparison.csv")


if __name__ == "__main__":
    main()