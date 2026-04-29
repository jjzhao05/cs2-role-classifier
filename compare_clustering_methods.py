from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan


# ---------- SETTINGS ----------
INPUT_PATH = "output.csv"
OUTPUT_DIR = Path("outputs")

MIN_ROUNDS_PLAYED = 10
PCA_VARIANCE = 0.99
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
RANDOM_STATE = 42


# ---------- IO ----------
def ensure_dirs():
    (OUTPUT_DIR / "assignments").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summaries").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "diagnostics").mkdir(parents=True, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).fillna(0)

    required = {"player_name", "rounds_played_ct", "rounds_played_t"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["rounds_played_total"] = df["rounds_played_ct"] + df["rounds_played_t"]
    df = df[df["rounds_played_total"] > MIN_ROUNDS_PLAYED].copy()

    if df.empty:
        raise ValueError(f"No players remain after MIN_ROUNDS_PLAYED={MIN_ROUNDS_PLAYED}")

    return df.reset_index(drop=True)


# ---------- FEATURES ----------
def prepare_features(df: pd.DataFrame):
    excluded = {
        "player_name",
        "rounds_played_ct",
        "rounds_played_t",
        "rounds_played_total",
        "awpy_rounds_ct",
        "awpy_rounds_t",
    }

    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    zero_var_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    return X, feature_cols


# ---------- SCORING ----------
def clustering_scores(X: np.ndarray, labels: np.ndarray):
    if len(set(labels)) < 2:
        return None, None

    if -1 in labels:
        mask = labels != -1
        X = X[mask]
        labels = labels[mask]

    if len(X) < 2 or len(set(labels)) < 2:
        return None, None

    cluster_sizes = pd.Series(labels).value_counts()
    if cluster_sizes.min() < 2:
        return None, None

    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = None

    try:
        dbi = float(davies_bouldin_score(X, labels))
    except Exception:
        dbi = None

    return sil, dbi


def cluster_stats(labels: np.ndarray):
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    non_noise = labels[labels != -1]
    n_clusters = len(set(non_noise)) if len(non_noise) else 0
    n_noise = int((labels == -1).sum())
    min_cluster_size = min(
        [v for k, v in counts.items() if k != -1],
        default=0,
    )
    return n_clusters, min_cluster_size, n_noise, counts


# ---------- CLUSTERING ----------
def run_methods(X_pca: np.ndarray):
    methods = {}
    rows = []
    n_samples = len(X_pca)

    valid_k = [k for k in K_VALUES if 2 <= k < n_samples]

    for k in valid_k:
        km_name = f"kmeans_k{k}"
        km_labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X_pca)
        methods[km_name] = km_labels

        n_clusters, min_cluster_size, n_noise, counts = cluster_stats(km_labels)
        sil, dbi = clustering_scores(X_pca, km_labels)
        rows.append({
            "method": "kmeans",
            "k": k,
            "n_clusters": n_clusters,
            "min_cluster_size": min_cluster_size,
            "n_noise": n_noise,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "cluster_sizes": str(counts),
            "output_name": km_name,
        })

        gmm_name = f"gmm_k{k}"
        gmm_labels = GaussianMixture(
            n_components=k,
            random_state=RANDOM_STATE,
            covariance_type="full",
        ).fit(X_pca).predict(X_pca)
        methods[gmm_name] = gmm_labels

        n_clusters, min_cluster_size, n_noise, counts = cluster_stats(gmm_labels)
        sil, dbi = clustering_scores(X_pca, gmm_labels)
        rows.append({
            "method": "gmm",
            "k": k,
            "n_clusters": n_clusters,
            "min_cluster_size": min_cluster_size,
            "n_noise": n_noise,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "cluster_sizes": str(counts),
            "output_name": gmm_name,
        })

    hdb_name = "hdbscan"
    try:
        hdb_labels = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1).fit_predict(X_pca)
    except Exception:
        hdb_labels = np.full(n_samples, -1)

    methods[hdb_name] = hdb_labels
    n_clusters, min_cluster_size, n_noise, counts = cluster_stats(hdb_labels)
    sil, dbi = clustering_scores(X_pca, hdb_labels)
    rows.append({
        "method": "hdbscan",
        "k": None,
        "n_clusters": n_clusters,
        "min_cluster_size": min_cluster_size,
        "n_noise": n_noise,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "cluster_sizes": str(counts),
        "output_name": hdb_name,
    })

    results = pd.DataFrame(rows).sort_values(
        by=["silhouette", "davies_bouldin"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    return results, methods


# ---------- SAVING ----------
def save_diagnostics(df: pd.DataFrame, X: pd.DataFrame, feature_cols: list[str], pca: PCA, X_pca: np.ndarray):
    pd.concat(
        [df[["player_name", "rounds_played_total"]], X],
        axis=1,
    ).to_csv(OUTPUT_DIR / "diagnostics" / "model_input_features.csv", index=False)

    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(OUTPUT_DIR / "diagnostics" / "pca_explained_variance.csv", index=False)

    pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    ).reset_index(names="feature").to_csv(
        OUTPUT_DIR / "diagnostics" / "pca_loadings.csv",
        index=False,
    )

    pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])],
    ).to_csv(OUTPUT_DIR / "diagnostics" / "pca_transformed_features.csv", index=False)


def save_assignments(df: pd.DataFrame, X_scaled: np.ndarray, methods: dict):
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    base = df.copy()
    base["pc1"] = coords[:, 0]
    base["pc2"] = coords[:, 1]

    for name, labels in methods.items():
        out = base.copy()
        out["cluster"] = labels
        out.to_csv(OUTPUT_DIR / "assignments" / f"{name}_players.csv", index=False)

        non_noise = out[out["cluster"] != -1].copy()
        if non_noise.empty:
            summary = pd.DataFrame(columns=["cluster", "player_count"])
        else:
            numeric_cols = [c for c in non_noise.columns if c not in {"player_name", "cluster"}]
            summary = (
                non_noise.groupby("cluster")[numeric_cols]
                .mean()
                .reset_index()
            )
            counts = (
                non_noise.groupby("cluster")
                .size()
                .reset_index(name="player_count")
            )
            summary = counts.merge(summary, on="cluster", how="left")

        summary.to_csv(OUTPUT_DIR / "summaries" / f"{name}_summary.csv", index=False)


# ---------- MAIN ----------
def main():
    ensure_dirs()

    df = load_data(INPUT_PATH)
    X, feature_cols = prepare_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Players used: {len(df)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"PCA components kept: {X_pca.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"K values tested: {K_VALUES}")

    save_diagnostics(df, X, feature_cols, pca, X_pca)

    results, methods = run_methods(X_pca)

    print("\nMethod comparison:")
    print(results[[
        "method",
        "k",
        "n_clusters",
        "min_cluster_size",
        "n_noise",
        "silhouette",
        "davies_bouldin",
        "cluster_sizes",
    ]])

    results.to_csv(OUTPUT_DIR / "clustering_method_comparison.csv", index=False)
    save_assignments(df, X_scaled, methods)

    print("\nSaved outputs to:")
    print(f"- {OUTPUT_DIR}")


if __name__ == "__main__":
    main()