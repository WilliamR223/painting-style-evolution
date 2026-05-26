from pathlib import Path

import numpy as np
import pandas as pd
import timm


def load_sample_metadata(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def create_demo_vit_features(df: pd.DataFrame) -> np.ndarray:
    timm.create_model("vit_base_patch16_224", pretrained=True)
    rng = np.random.default_rng(123)
    return rng.normal(size=(len(df), 768))


def main() -> None:
    data_path = Path("data/sample_metadata.csv")
    output_path = Path("data/vit_features_demo.npy")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}")

    df = load_sample_metadata(str(data_path))
    features = create_demo_vit_features(df)
    np.save(output_path, features)

    print(f"Saved demo ViT features to {output_path}")
    print(f"Shape: {features.shape}")


if __name__ == "__main__":
    main()