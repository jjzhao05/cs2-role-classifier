from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "output.csv"
OUTPUT_DIR = Path("outputs")

MIN_ROUNDS_PLAYED = 26
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
RANDOM_STATE = 69420


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
    exclude = {
        "player_name",
        "side",
        "rounds_played",
    }

    X = df.drop(columns=[c for c in exclude if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Remove useless constant columns
    X = X.loc[:, X.nunique() > 1]

    if X.empty:
        raise ValueError("No usable feature columns.")

    return X


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

    for k in K_VALUES:
        if k >= len(side_df):
            continue

        kmeans_name = f"kmeans_k{k}"
        kmeans_labels = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        ).fit_predict(X_scaled)

        methods[kmeans_name] = kmeans_labels

        results.append({
            "side": side,
            "method": "kmeans",
            "name": kmeans_name,
            "k": k,
            "silhouette": silhouette_score(X_scaled, kmeans_labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, kmeans_labels),
        })

        gmm_name = f"gmm_k{k}"
        gmm_labels = GaussianMixture(
            n_components=k,
            random_state=RANDOM_STATE,
            covariance_type="full",
        ).fit(X_scaled).predict(X_scaled)

        methods[gmm_name] = gmm_labels

        results.append({
            "side": side,
            "method": "gmm",
            "name": gmm_name,
            "k": k,
            "silhouette": silhouette_score(X_scaled, gmm_labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, gmm_labels),
        })

    results_df = pd.DataFrame(results).sort_values(
        ["silhouette", "davies_bouldin"],
        ascending=[False, True],
    )

    best_models = results_df.head(2)

    side_dir = OUTPUT_DIR / side
    side_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(side_dir / "model_scores.csv", index=False)

    for _, row in best_models.iterrows():
        name = row["name"]
        labels = methods[name]

        output = side_df.copy()
        output["cluster"] = labels
        output["pc1"] = coords[:, 0]
        output["pc2"] = coords[:, 1]

        output.to_csv(side_dir / f"{name}_player_clusters.csv", index=False)

    best_models.to_csv(side_dir / "best_models.csv", index=False)

    print(f"\n{side.upper()} analysis complete")
    print(f"Players used: {len(side_df)}")
    print(f"Features used: {X.shape[1]}")
    print("\nBest 2 models:")
    print(best_models)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()

    cluster_side(df, "ct")
    cluster_side(df, "t")

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()