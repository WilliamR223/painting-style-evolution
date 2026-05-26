from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def compute_centroids(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        mask = labels == label
        centroids.append(features[mask].mean(axis=0))

    return unique_labels, np.vstack(centroids)


def main() -> None:
    meta_path = Path("data/sample_metadata.csv")
    resnet_path = Path("data/resnet_features_demo.npy")
    vit_path = Path("data/vit_features_demo.npy")

    for path in [meta_path, resnet_path, vit_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    meta = pd.read_csv(meta_path)
    labels = meta["movement"].to_numpy()

    for model_name, feature_path in [
        ("ResNet", resnet_path),
        ("ViT", vit_path),
    ]:
        features = np.load(feature_path)
        movement_labels, centroids = compute_centroids(features, labels)
        dist_matrix = pairwise_distances(centroids, metric="euclidean")

        dist_df = pd.DataFrame(dist_matrix, index=movement_labels, columns=movement_labels)
        output_path = Path(f"data/{model_name.lower()}_centroid_distances.csv")
        dist_df.to_csv(output_path)

        print(f"\n{model_name} centroid distance matrix")
        print(dist_df)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()