from pathlib import Path

import pandas as pd


def assign_movement(year: int) -> str:
    if year < 500:
        return "Ancient"
    if year < 1400:
        return "Medieval"
    if year < 1600:
        return "Renaissance"
    if year < 1720:
        return "Baroque"
    if year < 1780:
        return "Rococo"
    if year < 1830:
        return "Neoclassicism"
    if year < 1850:
        return "Romanticism"
    if year < 1880:
        return "Realism"
    if year < 1890:
        return "Impressionism"
    if year < 1910:
        return "Post-Impressionism"
    if year < 1970:
        return "Modernism"
    return "Contemporary"


def main() -> None:
    input_path = Path("data/sample_metadata.csv")
    output_path = Path("data/sample_metadata_prepared.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_csv(input_path)
    df = df.dropna(subset=["rep_year"]).copy()
    df["movement_from_year"] = df["rep_year"].astype(int).apply(assign_movement)
    df.to_csv(output_path, index=False)

    print(f"Saved prepared metadata to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()