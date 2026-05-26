from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.models as models


def load_sample_metadata(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def create_demo_resnet_features(df: pd.DataFrame) -> np.ndarray:
    models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    rng = np.random.default_rng(42)
    return rng.normal(size=(len(df), 2048))


def main() -> None:
    data_path = Path("data/sample_metadata.csv")
    output_path = Path("data/resnet_features_demo.npy")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}")

    df = load_sample_metadata(str(data_path))
    features = create_demo_resnet_features(df)
    np.save(output_path, features)

    print(f"Saved demo ResNet features to {output_path}")
    print(f"Shape: {features.shape}")


if __name__ == "__main__":
    main()