# Painting Style Evolution

This repository contains a simplified code version of my final year project on computational analysis of painting style evolution.

The project compares two main approaches:
- color-based analysis
- deep visual representations using ResNet-50 and Vision Transformer

The goal is to study how different feature spaces represent paintings across major art movements.

## Project Summary

This project examines painting evolution across 12 art movements, from Ancient to Contemporary.

Main components:
- dataset preparation
- dominant palette extraction
- color feature analysis
- PCA and clustering
- ResNet-50 feature extraction
- Vision Transformer feature extraction
- linear probe evaluation
- UMAP visualization
- centroid distance analysis
- model comparison

## Repository Structure

```text
painting-style-evolution/
│── README.md
│── requirements.txt
│── .gitignore
│── scripts/
│   ├── prepare_dataset.py
│   ├── color_analysis.py
│   ├── extract_resnet_features.py
│   ├── extract_vit_features.py
│   ├── linear_probe.py
│   ├── umap_visualization.py
│   ├── centroid_analysis.py
│   ├── compare_models.py
│── data/
│   └── sample_metadata.csv