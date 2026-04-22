from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

import hdbscan


# ---------- SETTINGS ----------
INPUT_PATH = "output.csv"
OUTPUT_DIR = Path("outputs")

MIN_ROUNDS_PLAYED = 10
PCA_VARIANCE = 0.90
K = 5


def load_grouped_players(path: str) -> pd.DataFrame:
    df = pl.read_csv(path).fill_null(0)

    meta_cols = ["player_name", "map_name"]
    numeric_cols = [c for c in df.columns if c not in meta_cols]

    grouped = (
        df.group_by("player_name")
        .agg([pl.col(c).mean().alias(c) for c in numeric_cols])
        .sort("player_name")
    )

    pdf = grouped.to_pandas()

    round_cols = [c for c in ["rounds_played_T", "rounds_played_CT"] if c in pdf.columns]
    if len(round_cols) == 2:
        pdf["rounds_played_total"] = pdf["rounds_played_T"] + pdf["rounds_played_CT"]
    elif len(round_cols) == 1:
        pdf["rounds_played_total"] = pdf[round_cols[0]]
    else:
        raise ValueError("No rounds_played columns found in output.csv")

    pdf = pdf[pdf["rounds_played_total"] > MIN_ROUNDS_PLAYED].copy().reset_index(drop=True)
    return pdf


def valid_cluster_stats(labels: np.ndarray) -> dict:
    counts = pd.Series(labels).value_counts().sort_index()

    non_noise = labels[labels != -1]
    n_clusters = len(set(non_noise)) if len(non_noise) > 0 else 0

    non_noise_counts = counts.loc[counts.index != -1] if any(counts.index != -1) else pd.Series(dtype=int)
    min_cluster_size = int(non_noise_counts.min()) if len(non_noise_counts) > 0 else 0

    n_noise = int((labels == -1).sum())

    return {
        "n_clusters": int(n_clusters),
        "min_cluster_size": min_cluster_size,
        "n_noise": n_noise,
        "counts": counts.to_dict(),
    }


def score_clustering(X: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    non_noise_mask = labels != -1
    X_scored = X[non_noise_mask]
    labels_scored = labels[non_noise_mask]

    if len(np.unique(labels_scored)) < 2:
        return None, None

    sil = silhouette_score(X_scored, labels_scored)
    dbi = davies_bouldin_score(X_scored, labels_scored)
    return sil, dbi


def run_methods(X_pca: np.ndarray) -> tuple[pd.DataFrame, dict]:
    methods = {}

    methods["kmeans"] = KMeans(
        n_clusters=K,
        random_state=42,
        n_init=10,
    ).fit_predict(X_pca)

    methods["gmm"] = GaussianMixture(
        n_components=K,
        random_state=42,
    ).fit_predict(X_pca)

    methods["hdbscan"] = hdbscan.HDBSCAN(
        min_cluster_size=2,
    ).fit_predict(X_pca)

    rows = []
    for name, labels in methods.items():
        stats = valid_cluster_stats(labels)
        sil, dbi = score_clustering(X_pca, labels)

        rows.append(
            {
                "method": name,
                "n_clusters": stats["n_clusters"],
                "min_cluster_size": stats["min_cluster_size"],
                "n_noise": stats["n_noise"],
                "silhouette": sil,
                "davies_bouldin": dbi,
                "cluster_sizes": str(stats["counts"]),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=["silhouette", "davies_bouldin"],
        ascending=[False, True],
        na_position="last",
    )

    return results_df, methods


def save_assignments(pdf: pd.DataFrame, methods: dict, X_scaled: np.ndarray) -> None:
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    base = pdf.copy()
    base["pc1"] = X_2d[:, 0]
    base["pc2"] = X_2d[:, 1]

    for method_name, labels in methods.items():
        out = base.copy()
        out["cluster"] = labels
        out.to_csv(OUTPUT_DIR / f"{method_name}_players.csv", index=False)

        summary = (
            out.loc[out["cluster"] != -1]
            .groupby("cluster")
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("cluster")
        )
        summary.to_csv(OUTPUT_DIR / f"{method_name}_summary.csv", index=False)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    pdf = load_grouped_players(INPUT_PATH)

    feature_cols = [c for c in pdf.columns if c not in ["player_name", "rounds_played_total"]]
    X = pdf[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA components kept: {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    results_df, methods = run_methods(X_pca)

    print("\nMethod comparison:")
    print(results_df)

    results_df.to_csv(OUTPUT_DIR / "clustering_method_comparison.csv", index=False)
    save_assignments(pdf, methods, X_scaled)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'clustering_method_comparison.csv'}")
    for method_name in methods:
        print(f"- {OUTPUT_DIR / f'{method_name}_players.csv'}")
        print(f"- {OUTPUT_DIR / f'{method_name}_summary.csv'}")


if __name__ == "__main__":
    main()