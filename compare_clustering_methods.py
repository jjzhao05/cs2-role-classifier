from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

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
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
RANDOM_STATE = 42


# ---------- HELPERS ----------
def ensure_output_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_DIR,
        "assignments": OUTPUT_DIR / "assignments",
        "summaries": OUTPUT_DIR / "summaries",
        "diagnostics": OUTPUT_DIR / "diagnostics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_grouped_players(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path.resolve()}")

    df = pl.read_csv(csv_path).fill_null(0)

    required_cols = {"player_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    meta_cols = [c for c in ["player_name", "map_name"] if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in meta_cols]

    if not numeric_cols:
        raise ValueError("No numeric feature columns found in input CSV.")

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
        raise ValueError("No rounds_played columns found. Expected rounds_played_T and/or rounds_played_CT.")

    pdf = pdf[pdf["rounds_played_total"] > MIN_ROUNDS_PLAYED].copy().reset_index(drop=True)

    if pdf.empty:
        raise ValueError(
            f"No players remain after filtering with MIN_ROUNDS_PLAYED={MIN_ROUNDS_PLAYED}."
        )

    return pdf


def prepare_feature_matrix(pdf: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    excluded_cols = {"player_name", "rounds_played_total"}
    feature_cols = [c for c in pdf.columns if c not in excluded_cols]

    if not feature_cols:
        raise ValueError("No feature columns available after excluding metadata.")

    X = pdf[feature_cols].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    zero_var_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]

    if X.shape[1] == 0:
        raise ValueError("All features were constant or invalid after cleaning.")

    return X, feature_cols


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

    if len(X_scored) < 2:
        return None, None

    unique_labels = np.unique(labels_scored)
    if len(unique_labels) < 2:
        return None, None

    smallest_cluster = pd.Series(labels_scored).value_counts().min()
    if smallest_cluster < 2:
        return None, None

    try:
        sil = float(silhouette_score(X_scored, labels_scored))
    except Exception:
        sil = None

    try:
        dbi = float(davies_bouldin_score(X_scored, labels_scored))
    except Exception:
        dbi = None

    return sil, dbi


def sanitize_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def run_methods(
    X_pca: np.ndarray,
    k_values: list[int],
) -> tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, str]]:
    n_samples = X_pca.shape[0]

    methods: Dict[str, np.ndarray] = {}
    labels_to_display_name: Dict[str, str] = {}
    rows = []

    valid_k_values = sorted({int(k) for k in k_values if isinstance(k, int) and k >= 2 and k < n_samples})

    for k in valid_k_values:
        # KMeans
        km_display = f"kmeans_k{k}"
        km_key = sanitize_name(km_display)
        km_labels = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        ).fit_predict(X_pca)

        methods[km_key] = km_labels
        labels_to_display_name[km_key] = km_display

        km_stats = valid_cluster_stats(km_labels)
        km_sil, km_dbi = score_clustering(X_pca, km_labels)
        rows.append(
            {
                "method": "kmeans",
                "k": k,
                "n_clusters": km_stats["n_clusters"],
                "min_cluster_size": km_stats["min_cluster_size"],
                "n_noise": km_stats["n_noise"],
                "silhouette": km_sil,
                "davies_bouldin": km_dbi,
                "cluster_sizes": str(km_stats["counts"]),
                "output_name": km_key,
            }
        )

        # GMM
        gmm_display = f"gmm_k{k}"
        gmm_key = sanitize_name(gmm_display)
        gmm_labels = GaussianMixture(
            n_components=k,
            random_state=RANDOM_STATE,
            covariance_type="full",
        ).fit(X_pca).predict(X_pca)

        methods[gmm_key] = gmm_labels
        labels_to_display_name[gmm_key] = gmm_display

        gmm_stats = valid_cluster_stats(gmm_labels)
        gmm_sil, gmm_dbi = score_clustering(X_pca, gmm_labels)
        rows.append(
            {
                "method": "gmm",
                "k": k,
                "n_clusters": gmm_stats["n_clusters"],
                "min_cluster_size": gmm_stats["min_cluster_size"],
                "n_noise": gmm_stats["n_noise"],
                "silhouette": gmm_sil,
                "davies_bouldin": gmm_dbi,
                "cluster_sizes": str(gmm_stats["counts"]),
                "output_name": gmm_key,
            }
        )

    # HDBSCAN once
    min_cluster_size = max(2, min(5, n_samples // 4 if n_samples >= 8 else 2))
    hdb_display = f"hdbscan_mcs{min_cluster_size}"
    hdb_key = sanitize_name(hdb_display)

    try:
        hdb_labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
        ).fit_predict(X_pca)
    except Exception:
        hdb_labels = np.full(n_samples, -1)

    methods[hdb_key] = hdb_labels
    labels_to_display_name[hdb_key] = hdb_display

    hdb_stats = valid_cluster_stats(hdb_labels)
    hdb_sil, hdb_dbi = score_clustering(X_pca, hdb_labels)
    rows.append(
        {
            "method": "hdbscan",
            "k": None,
            "n_clusters": hdb_stats["n_clusters"],
            "min_cluster_size": hdb_stats["min_cluster_size"],
            "n_noise": hdb_stats["n_noise"],
            "silhouette": hdb_sil,
            "davies_bouldin": hdb_dbi,
            "cluster_sizes": str(hdb_stats["counts"]),
            "output_name": hdb_key,
        }
    )

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(
            by=["silhouette", "davies_bouldin"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    return results_df, methods, labels_to_display_name


def save_pca_diagnostics(
    feature_cols: list[str],
    pca: PCA,
    X_pca: np.ndarray,
    dirs: dict[str, Path],
) -> None:
    explained = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(dirs["diagnostics"] / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    ).reset_index(names="feature")
    loadings.to_csv(dirs["diagnostics"] / "pca_loadings.csv", index=False)

    pca_scores = pd.DataFrame(
        X_pca,
        columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])],
    )
    pca_scores.to_csv(dirs["diagnostics"] / "pca_transformed_features.csv", index=False)


def save_assignments(
    pdf: pd.DataFrame,
    methods: Dict[str, np.ndarray],
    X_scaled: np.ndarray,
    dirs: dict[str, Path],
) -> None:
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_scaled)

    base = pdf.copy()
    base["pc1"] = X_2d[:, 0]
    base["pc2"] = X_2d[:, 1]

    for method_name, labels in methods.items():
        out = base.copy()
        out["cluster"] = labels
        out.to_csv(dirs["assignments"] / f"{method_name}_players.csv", index=False)

        summary = (
            out.loc[out["cluster"] != -1]
            .groupby("cluster")
            .agg(
                player_count=("player_name", "count"),
                **{
                    c: (c, "mean")
                    for c in out.columns
                    if c not in {"player_name", "cluster"}
                },
            )
            .reset_index()
            .sort_values("cluster")
        )
        summary.to_csv(dirs["summaries"] / f"{method_name}_summary.csv", index=False)


def main() -> None:
    dirs = ensure_output_dirs()

    pdf = load_grouped_players(INPUT_PATH)
    X, feature_cols = prepare_feature_matrix(pdf)

    feature_matrix_out = pd.concat([pdf[["player_name", "rounds_played_total"]], X], axis=1)
    feature_matrix_out.to_csv(dirs["diagnostics"] / "model_input_features.csv", index=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Players used: {len(pdf)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"PCA components kept: {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"K values tested: {K_VALUES}")

    save_pca_diagnostics(feature_cols, pca, X_pca, dirs)

    results_df, methods, labels_to_display_name = run_methods(X_pca, K_VALUES)

    print("\nMethod comparison:")
    if not results_df.empty:
        display_cols = [
            "method",
            "k",
            "n_clusters",
            "min_cluster_size",
            "n_noise",
            "silhouette",
            "davies_bouldin",
            "cluster_sizes",
        ]
        print(results_df[display_cols])
    else:
        print("No clustering methods produced valid output.")

    results_df.to_csv(dirs["root"] / "clustering_method_comparison.csv", index=False)
    save_assignments(pdf, methods, X_scaled, dirs)

    print("\nSaved:")
    print(f"- {dirs['root'] / 'clustering_method_comparison.csv'}")
    print(f"- {dirs['diagnostics'] / 'model_input_features.csv'}")
    print(f"- {dirs['diagnostics'] / 'pca_explained_variance.csv'}")
    print(f"- {dirs['diagnostics'] / 'pca_loadings.csv'}")
    print(f"- {dirs['diagnostics'] / 'pca_transformed_features.csv'}")
    for method_key in methods:
        display_name = labels_to_display_name.get(method_key, method_key)
        print(f"- {dirs['assignments'] / f'{method_key}_players.csv'} ({display_name})")
        print(f"- {dirs['summaries'] / f'{method_key}_summary.csv'} ({display_name})")


if __name__ == "__main__":
    main()