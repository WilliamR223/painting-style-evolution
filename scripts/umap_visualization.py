from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


def plot_umap(features: np.ndarray, labels: np.ndarray, title: str, output_path: str) -> None:
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for movement in np.unique(labels):
        mask = labels == movement
        plt.scatter(coords[mask, 0], coords[mask, 1], label=movement, alpha=0.7)

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved UMAP plot to {output_path}")


def main() -> None:
    meta = pd.read_csv("data/sample_metadata.csv")
    labels = meta["movement"].to_numpy()

    resnet = np.load("data/resnet_features_demo.npy")
    vit = np.load("data/vit_features_demo.npy")

    plot_umap(resnet, labels, "ResNet UMAP", "resnet_umap_demo.png")
    plot_umap(vit, labels, "ViT UMAP", "vit_umap_demo.png")


if __name__ == "__main__":
    main()