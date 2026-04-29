from __future__ import annotations

import argparse
import polars as pl
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset statistics for output.csv")
    parser.add_argument("input_csv", type=str, help="Path to output.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pl.read_csv(args.input_csv)

    print("===== DATASET OVERVIEW =====")
    print(f"# of players (rows): {df.height}")
    print(f"# of features (columns): {df.width}")

    # --- Missing values ---
    print("\n===== MISSING VALUES =====")
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    print(f"total missing values: {total_nulls}")

    if total_nulls > 0:
        print(null_counts)

    # --- Numeric columns only ---
    numeric_df = df.select(pl.selectors.numeric())

    # --- Feature stats ---
    print("\n===== FEATURE SUMMARY =====")
    stats = numeric_df.select([
    pl.all().mean().name.suffix("_mean"),
    pl.all().std().name.suffix("_std"),
    pl.all().min().name.suffix("_min"),
    pl.all().max().name.suffix("_max"),
    ])
    print(stats)

    # --- Variance ranking ---
    print("\n===== TOP FEATURES BY VARIANCE =====")
    variances = numeric_df.select(pl.all().var()).to_dicts()[0]

    sorted_vars = sorted(variances.items(), key=lambda x: -x[1])
    for name, var in sorted_vars[:15]:
        print(f"{name}: {var:.4f}")

    # --- Correlation matrix (sampled if large) ---
    print("\n===== CORRELATION (TOP PAIRS) =====")

    pdf = numeric_df.to_pandas()
    corr = pdf.corr()

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    pairs.sort(key=lambda x: -abs(x[2]))

    for a, b, c in pairs[:15]:
        print(f"{a} vs {b}: {c:.3f}")

    # --- AWPer sanity check ---
    print("\n===== AWP SIGNAL CHECK =====")
    awp_cols = [c for c in df.columns if "awp" in c.lower()]
    if awp_cols:
        for col in awp_cols:
            print(f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    else:
        print("No AWP-related columns found.")

    # --- Outlier detection (simple) ---
    print("\n===== EXTREME VALUES (Z > 3) =====")
    for col in numeric_df.columns:
        series = numeric_df[col].to_numpy()
        if len(series) < 2:
            continue

        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            continue

        z = (series - mean) / std
        outliers = np.sum(np.abs(z) > 3)

        if outliers > 0:
            print(f"{col}: {outliers} outliers")


if __name__ == "__main__":
    main()