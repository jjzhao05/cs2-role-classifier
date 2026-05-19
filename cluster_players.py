from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan

INPUT_PATH = "output.csv"
OUTPUT_DIR = Path("outputs")
MIN_ROUNDS_PLAYED = 26
K_VALUES = [3, 4, 5, 6, 7, 8]
RANDOM_STATE = 69420

# Features excluded from clustering but kept in the output CSV for plotting.
CLUSTER_EXCLUDE_FEATURES = {"adr", "kpr"}

# HDBSCAN hyperparameter grid
HDBSCAN_MIN_CLUSTER_SIZES = [3, 5, 10, 15]
HDBSCAN_MIN_SAMPLES = [1, 3, 5]

# Stability analysis config
STABILITY_N_BOOTS = 50       # bootstrap iterations per model
STABILITY_RESAMPLE_FRAC = 0.80  # fraction of players sampled per bootstrap
# Composite score weights (all positive = higher is better after sign-flipping DB)
STABILITY_WEIGHT = 0.40
SILHOUETTE_WEIGHT = 0.40
DB_WEIGHT = 0.20             # applied to (1 / davies_bouldin), so lower DB → higher score


def load_data():
    df = pd.read_csv(INPUT_PATH).fillna(0)
    required = {"player_name", "side", "rounds_played"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["side"] = df["side"].str.lower()
    df = df[df["side"].isin(["ct", "t"])]
    df = df[df["rounds_played"] > MIN_ROUNDS_PLAYED].copy()
    if df.empty:
        raise ValueError("No rows left after filtering.")
    return df


def get_features(df):
    exclude = {"player_name", "side", "rounds_played"} | CLUSTER_EXCLUDE_FEATURES
    X = df.drop(columns=[c for c in exclude if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]
    if X.empty:
        raise ValueError("No usable feature columns.")
    return X


def run_hdbscan(X_scaled, min_cluster_size, min_samples):
    """
    Fit HDBSCAN and return labels. Returns (None, None) if the result is
    degenerate (all noise, or only one cluster after ignoring noise label -1).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_scaled)
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        return None, None
    return labels, clusterer


# ---------------------------------------------------------------------------
# Stability helpers
# ---------------------------------------------------------------------------

def _fit_labels(X_scaled, method_name: str, k: int | None,
                mcs: int | None, ms: int | None,
                random_state: int) -> np.ndarray | None:
    """Re-fit a named method and return labels (or None if degenerate)."""
    if method_name == "kmeans":
        return KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        ).fit_predict(X_scaled)

    if method_name == "gmm":
        return (
            GaussianMixture(
                n_components=k,
                random_state=random_state,
                covariance_type="full",
            )
            .fit(X_scaled)
            .predict(X_scaled)
        )

    if method_name == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X_scaled)
        if len(set(labels) - {-1}) < 2:
            return None
        return labels

    raise ValueError(f"Unknown method: {method_name}")


def compute_stability(
    X_scaled: np.ndarray,
    full_labels: np.ndarray,
    method_name: str,
    k: int | None = None,
    mcs: int | None = None,
    ms: int | None = None,
    n_boots: int = STABILITY_N_BOOTS,
    resample_frac: float = STABILITY_RESAMPLE_FRAC,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """
    Bootstrap stability analysis via Adjusted Rand Index.

    For each bootstrap iteration:
      1. Sample `resample_frac` of the rows *with replacement*.
      2. Refit the model on the subsample.
      3. Compute ARI between the subsample's new labels and the corresponding
         full-data labels (same row indices).

    Returns (mean_ari, std_ari). Skipped / degenerate boots are excluded from
    the mean; if fewer than 5 valid boots succeed, returns (nan, nan).
    """
    rng = np.random.RandomState(random_state)
    n = X_scaled.shape[0]
    sample_size = max(3, int(n * resample_frac))

    ari_scores: list[float] = []

    for boot_i in range(n_boots):
        idx = rng.choice(n, size=sample_size, replace=True)
        X_boot = X_scaled[idx]

        boot_labels = _fit_labels(
            X_boot,
            method_name,
            k=k,
            mcs=mcs,
            ms=ms,
            random_state=random_state + boot_i,
        )

        if boot_labels is None:
            continue  # degenerate — skip this iteration

        # Compare against the full-data labels for the same indices.
        ref_labels = full_labels[idx]

        # If reference labels are all noise (-1) or trivial, ARI is undefined.
        ref_valid = ref_labels != -1
        if ref_valid.sum() < 2 or len(set(ref_labels[ref_valid])) < 2:
            continue

        boot_valid = boot_labels != -1
        # Intersect valid masks so we score on the same points.
        both_valid = ref_valid & boot_valid
        if both_valid.sum() < 2 or len(set(ref_labels[both_valid])) < 2:
            continue

        ari = adjusted_rand_score(ref_labels[both_valid], boot_labels[both_valid])
        ari_scores.append(ari)

    if len(ari_scores) < 5:
        return float("nan"), float("nan")

    return float(np.mean(ari_scores)), float(np.std(ari_scores))


def composite_score(silhouette: float, davies_bouldin: float,
                    stability_mean: float, stability_std: float) -> float:
    """
    Weighted composite (higher = better).
      - silhouette:      higher is better  [already in [-1, 1]]
      - davies_bouldin:  lower is better   → convert to 1/(1+db)
      - stability:       1-sigma lower confidence bound max(0, mean - std),
                         so high variance is penalised even if the mean is good.

    Any NaN component contributes 0 rather than propagating NaN, so models
    without stability data still rank on silhouette + DB alone.
    """
    sil_term = SILHOUETTE_WEIGHT * (silhouette if not np.isnan(silhouette) else 0.0)
    db_term = DB_WEIGHT * (1.0 / (1.0 + davies_bouldin) if not np.isnan(davies_bouldin) else 0.0)

    if np.isnan(stability_mean) or np.isnan(stability_std):
        stab_score = 0.0
    else:
        # Lower-confidence bound: reward consistent stability, punish high variance.
        stab_score = max(0.0, stability_mean - stability_std)
    stab_term = STABILITY_WEIGHT * stab_score

    return sil_term + db_term + stab_term


# ---------------------------------------------------------------------------
# Main clustering routine
# ---------------------------------------------------------------------------

def cluster_side(df, side):
    side_df = df[df["side"] == side].reset_index(drop=True)
    if len(side_df) < 3:
        print(f"Skipping {side}: not enough players.")
        return

    X = get_features(side_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    methods = {}
    results = []

    # --- KMeans + GMM ---
    for k in K_VALUES:
        if k >= len(side_df):
            continue

        # KMeans
        kmeans_name = f"kmeans_k{k}"
        kmeans_labels = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        ).fit_predict(X_scaled)
        methods[kmeans_name] = kmeans_labels

        sil = silhouette_score(X_scaled, kmeans_labels)
        db = davies_bouldin_score(X_scaled, kmeans_labels)
        stab_mean, stab_std = compute_stability(
            X_scaled, kmeans_labels, "kmeans", k=k
        )
        results.append({
            "side": side,
            "method": "kmeans",
            "name": kmeans_name,
            "k": k,
            "noise_pct": 0.0,
            "silhouette": sil,
            "davies_bouldin": db,
            "stability_mean": stab_mean,
            "stability_std": stab_std,
            "composite_score": composite_score(sil, db, stab_mean, stab_std),
        })

        # GMM
        gmm_name = f"gmm_k{k}"
        gmm_labels = GaussianMixture(
            n_components=k,
            random_state=RANDOM_STATE,
            covariance_type="full",
        ).fit(X_scaled).predict(X_scaled)
        methods[gmm_name] = gmm_labels

        sil = silhouette_score(X_scaled, gmm_labels)
        db = davies_bouldin_score(X_scaled, gmm_labels)
        stab_mean, stab_std = compute_stability(
            X_scaled, gmm_labels, "gmm", k=k
        )
        results.append({
            "side": side,
            "method": "gmm",
            "name": gmm_name,
            "k": k,
            "noise_pct": 0.0,
            "silhouette": sil,
            "davies_bouldin": db,
            "stability_mean": stab_mean,
            "stability_std": stab_std,
            "composite_score": composite_score(sil, db, stab_mean, stab_std),
        })

    # --- HDBSCAN grid search ---
    for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
        for ms in HDBSCAN_MIN_SAMPLES:
            if mcs > len(side_df) // 2:
                continue
            labels, _ = run_hdbscan(X_scaled, mcs, ms)
            if labels is None:
                continue

            name = f"hdbscan_mcs{mcs}_ms{ms}"
            methods[name] = labels

            noise_mask = labels != -1
            noise_pct = (~noise_mask).mean()

            if noise_mask.sum() < 2 or len(set(labels[noise_mask])) < 2:
                continue

            sil = silhouette_score(X_scaled[noise_mask], labels[noise_mask])
            db = davies_bouldin_score(X_scaled[noise_mask], labels[noise_mask])
            k_found = len(set(labels) - {-1})

            stab_mean, stab_std = compute_stability(
                X_scaled, labels, "hdbscan", mcs=mcs, ms=ms
            )
            results.append({
                "side": side,
                "method": "hdbscan",
                "name": name,
                "k": k_found,
                "noise_pct": round(noise_pct, 4),
                "silhouette": sil,
                "davies_bouldin": db,
                "stability_mean": stab_mean,
                "stability_std": stab_std,
                "composite_score": composite_score(sil, db, stab_mean, stab_std),
            })

    if not results:
        print(f"No valid clustering results for {side}.")
        return

    results_df = pd.DataFrame(results).sort_values(
        "composite_score",
        ascending=False,
    )
    # Disqualify models whose mean ARI is below the minimum threshold.
    # Models with NaN stability (too few valid boots) are kept so they can
    # still rank on silhouette + DB alone; they appear with a warning in tiers.
    MIN_STABILITY = 0.50
    eligible = results_df[
        results_df["stability_mean"].isna()
        | (results_df["stability_mean"] >= MIN_STABILITY)
    ]
    if eligible.empty:
        print(f"[warn] all models below stability threshold ({MIN_STABILITY:.2f}); "
              "falling back to top-2 by composite score.")
        eligible = results_df
    best_models = eligible.head(2)

    side_dir = OUTPUT_DIR / side
    side_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(side_dir / "model_scores.csv", index=False)

    # Stability-only summary (convenient for downstream analysis / plotting)
    stability_cols = ["side", "method", "name", "k", "noise_pct",
                      "silhouette", "davies_bouldin",
                      "stability_mean", "stability_std", "composite_score"]
    results_df[stability_cols].to_csv(side_dir / "stability_scores.csv", index=False)

    # Player-level cluster assignments — write top 2 per algorithm so the
    # plotter can select top-N per algorithm without missing CSVs.
    TOP_N_PER_ALGO = 2
    for method_name, group in results_df.groupby("method"):
        top = group.head(TOP_N_PER_ALGO)
        for _, row in top.iterrows():
            name = row["name"]
            if name not in methods:
                continue
            labels = methods[name]
            output = side_df.copy()
            output["cluster"] = labels
            output["pc1"] = coords[:, 0]
            output["pc2"] = coords[:, 1]
            output.to_csv(side_dir / f"{name}_player_clusters.csv", index=False)

    # --- Console summary ---
    print(f"\n{side.upper()} analysis complete")
    print(f"Players used: {len(side_df)}")
    print(f"Features used: {X.shape[1]}")
    print(f"Bootstrap iterations per model: {STABILITY_N_BOOTS} "
          f"(resample fraction: {STABILITY_RESAMPLE_FRAC:.0%})")
    print("\nBest 2 models (ranked by composite score):")
    display_cols = ["name", "k", "silhouette", "davies_bouldin",
                    "stability_mean", "stability_std", "composite_score"]
    print(best_models[display_cols].to_string(index=False))

    _print_stability_interpretation(results_df)

    # --- Feature importance for each best model ---
    for _, row in best_models.iterrows():
        name = row["name"]
        labels = methods[name]

        valid = labels != -1
        if len(set(labels[valid])) < 2 or valid.sum() < 10:
            continue

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X.values[valid], labels[valid])

        importance = (
            pd.Series(rf.feature_importances_, index=X.columns)
            .sort_values(ascending=False)
        )

        print(f"\nFeature importance — {side.upper()} {name}:")
        print(importance.to_string())


def _print_stability_interpretation(results_df: pd.DataFrame) -> None:
    """Print a human-readable stability tier summary to the console."""
    stab = (
        results_df[["name", "stability_mean", "stability_std"]]
        .dropna()
        .sort_values("stability_mean", ascending=False)
    )
    if stab.empty:
        return

    def tier(mean_ari):
        if mean_ari >= 0.80:
            return "HIGH  ✓"
        if mean_ari >= 0.50:
            return "MEDIUM ~"
        return "LOW   ✗"

    print("\nStability tiers (ARI vs bootstrap resamples):")
    for _, row in stab.iterrows():
        t = tier(row["stability_mean"])
        print(f"  {row['name']:<30}  mean={row['stability_mean']:.3f}  "
              f"std={row['stability_std']:.3f}  [{t}]")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    cluster_side(df, "ct")
    cluster_side(df, "t")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()