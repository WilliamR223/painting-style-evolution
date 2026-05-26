from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_sample_metadata(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def create_demo_color_features(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    features = pd.DataFrame(
        {
            "mean_lightness": rng.normal(60, 10, len(df)),
            "std_lightness": rng.normal(12, 3, len(df)),
            "mean_chroma": rng.normal(30, 8, len(df)),
            "warm_fraction": rng.uniform(0, 1, len(df)),
            "cool_fraction": rng.uniform(0, 1, len(df)),
            "neutral_fraction": rng.uniform(0, 1, len(df)),
        }
    )

    return pd.concat([df, features], axis=1)


def run_pca(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "mean_lightness",
        "std_lightness",
        "mean_chroma",
        "warm_fraction",
        "cool_fraction",
        "neutral_fraction",
    ]

    x = df[feature_cols].to_numpy()
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(x)

    result = df.copy()
    result["pc1"] = coords[:, 0]
    result["pc2"] = coords[:, 1]
    return result


def run_kmeans(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    feature_cols = [
        "mean_lightness",
        "std_lightness",
        "mean_chroma",
        "warm_fraction",
        "cool_fraction",
        "neutral_fraction",
    ]

    x = df[feature_cols].to_numpy()
    x = StandardScaler().fit_transform(x)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    result = df.copy()
    result["cluster"] = model.fit_predict(x)
    return result


def plot_pca(df: pd.DataFrame, output_path: str = "pca_color_demo.png") -> None:
    plt.figure(figsize=(8, 6))

    for movement in df["movement"].unique():
        group = df[df["movement"] == movement]
        plt.scatter(group["pc1"], group["pc2"], label=movement, alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Demo Color Features")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved PCA plot to {output_path}")


def main() -> None:
    data_path = Path("data/sample_metadata.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing file: {data_path}")

    df = load_sample_metadata(str(data_path))
    df = create_demo_color_features(df)
    df = run_pca(df)
    df = run_kmeans(df, n_clusters=min(5, len(df)))
    plot_pca(df)


if __name__ == "__main__":
    main()