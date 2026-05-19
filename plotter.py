from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("outputs")
ASSIGNMENTS_DIR = INPUT_DIR / "assignments"
SUMMARIES_DIR = INPUT_DIR / "summaries"
PLOTS_DIR = Path("plots")

RADAR_FEATURES = [
    "opening_kill_rate",
    "opening_duel_success",
    "trade_kill_rate",
    "death_traded_rate",
    "flash_assists_per_round",
    "util_damage_per_round",
    "awp_kill_share",
    "rifle_kill_share",
    "multi_kill_rate",
    "grenades_per_round",
]


def get_top_methods(results_path: Path, top_n: int = 3) -> list[str]:
    if not results_path.exists():
        print(f"[skip] missing {results_path}")
        return []

    df = pd.read_csv(results_path)
    if df.empty:
        print("[skip] clustering results file is empty")
        return []

    top_methods = []
    # Group by side AND method so each side gets equal representation.
    groups = (
        df[["side", "method"]].dropna().drop_duplicates()
        .sort_values(["side", "method"])
        .itertuples(index=False)
    )
    for side, method_name in groups:
        sub = df[(df["side"] == side) & (df["method"] == method_name)].copy()
        sub = sub.sort_values(
            by=["silhouette", "davies_bouldin"],
            ascending=[False, True],
            na_position="last",
        ).head(top_n)

        names = sub["output_name"].dropna().tolist()
        top_methods.extend(names)

        print(f"\nTop {top_n} for {side.upper()} {method_name}:")
        cols = ["output_name", "k", "silhouette", "davies_bouldin", "cluster_sizes"]
        print(sub[cols].to_string(index=False))

    return top_methods


def get_combined_side_metric(df: pd.DataFrame, base_name: str) -> tuple[pd.Series | None, str | None]:
    if base_name in df.columns:
        s = pd.to_numeric(df[base_name], errors="coerce")
        return s, base_name

    t_col = f"{base_name}_t"
    ct_col = f"{base_name}_ct"

    has_t = t_col in df.columns
    has_ct = ct_col in df.columns

    if has_t and has_ct:
        s = (
            pd.to_numeric(df[t_col], errors="coerce").fillna(0) +
            pd.to_numeric(df[ct_col], errors="coerce").fillna(0)
        ) / 2.0
        return s, f"{base_name}_avg"

    if has_t:
        s = pd.to_numeric(df[t_col], errors="coerce")
        return s, t_col

    if has_ct:
        s = pd.to_numeric(df[ct_col], errors="coerce")
        return s, ct_col

    return None, None


def plot_pca_scatter(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    if players.empty:
        print(f"[skip] {method_name} scatter players file is empty")
        return

    required = {"player_name", "pc1", "pc2", "cluster"}
    missing = required - set(players.columns)
    if missing:
        print(f"[skip] {method_name} scatter missing columns: {sorted(missing)}")
        return

    plt.figure(figsize=(14, 10))

    non_noise = players.loc[players["cluster"] != -1]
    noise = players.loc[players["cluster"] == -1]

    if len(non_noise) > 0:
        plt.scatter(
            non_noise["pc1"],
            non_noise["pc2"],
            c=non_noise["cluster"],
            s=80,
            label="Clustered players",
        )

    if len(noise) > 0:
        plt.scatter(
            noise["pc1"],
            noise["pc2"],
            s=140,
            marker="x",
            linewidths=2,
            label="Noise / outliers",
        )

    for _, row in players.iterrows():
        plt.text(row["pc1"], row["pc2"], row["player_name"], fontsize=8)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Scatter: {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{method_name}_pca.png", dpi=200)
    plt.close()


def plot_rating_box(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    if players.empty:
        print(f"[skip] {method_name} rating box players file is empty")
        return

    if "cluster" not in players.columns:
        print(f"[skip] {method_name} missing cluster column")
        return

    rating_vals, rating_label = get_combined_side_metric(players, "rating")
    if rating_vals is None:
        print(f"[skip] {method_name} no rating column found")
        return

    plot_df = pd.DataFrame(
        {
            "cluster": players["cluster"],
            "rating": rating_vals,
        }
    ).dropna()

    if plot_df.empty:
        print(f"[skip] {method_name} no usable rating data")
        return

    non_noise = plot_df.loc[plot_df["cluster"] != -1].copy()
    noise = plot_df.loc[plot_df["cluster"] == -1].copy()

    cluster_order = sorted(non_noise["cluster"].unique()) if not non_noise.empty else []
    labels = [str(c) for c in cluster_order]
    data = [non_noise.loc[non_noise["cluster"] == c, "rating"].tolist() for c in cluster_order]

    if not noise.empty:
        labels.append("noise")
        data.append(noise["rating"].tolist())

    if not data:
        print(f"[skip] {method_name} no cluster data for rating box")
        return

    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, tick_labels=labels, patch_artist=True)

    for patch in box["boxes"]:
        patch.set_alpha(0.5)

    plt.xlabel("Cluster")
    plt.ylabel(rating_label)
    plt.title(f"Rating Distribution by Cluster: {method_name}")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / f"{method_name}_rating_box.png", dpi=200)
    plt.close()


def make_radar(ax, values, categories, title):
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals = values.tolist()

    vals += vals[:1]
    angles += angles[:1]

    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(title, y=1.1)


def plot_radar(summary: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    if summary.empty:
        print(f"[skip] {method_name} radar summary is empty")
        return

    if "cluster" not in summary.columns:
        print(f"[skip] {method_name} radar missing 'cluster'")
        return

    radar_features = [f for f in RADAR_FEATURES if f in summary.columns]
    if len(radar_features) < 3:
        print(f"[skip] {method_name} radar not enough matching features")
        return

    radar_df = summary.copy()
    for c in radar_features:
        radar_df[c] = pd.to_numeric(radar_df[c], errors="coerce")

    mins = radar_df[radar_features].min()
    maxs = radar_df[radar_features].max()
    denom = (maxs - mins).replace(0, 1)

    radar_scaled = radar_df.copy()
    radar_scaled[radar_features] = (radar_scaled[radar_features] - mins) / denom

    num_clusters = len(radar_scaled)
    if num_clusters == 0:
        print(f"[skip] {method_name} radar has 0 rows after scaling")
        return

    rows = int(np.ceil(num_clusters / 2))
    cols = 2 if num_clusters > 1 else 1

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(14, 5 * rows),
        subplot_kw=dict(polar=True),
    )

    axes = np.array(axes).reshape(-1)

    last_i = -1
    for i, (_, row) in enumerate(radar_scaled.iterrows()):
        last_i = i
        make_radar(
            axes[i],
            row[radar_features],
            radar_features,
            f"{method_name} cluster {int(row['cluster'])}",
        )

    for j in range(last_i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_dir / f"{method_name}_radar.png", dpi=200)
    plt.close()


def plot_k_metrics(results_path: Path, output_dir: Path) -> None:
    if not results_path.exists():
        print(f"[skip] missing {results_path}")
        return

    df = pd.read_csv(results_path)
    if df.empty:
        print("[skip] clustering results file is empty")
        return

    k_df = df[df["k"].notna()].copy()
    if not k_df.empty:
        k_df["k"] = k_df["k"].astype(int)

    methods = sorted(k_df["method"].dropna().unique()) if not k_df.empty else []
    sil_df = k_df.dropna(subset=["silhouette"]).copy() if not k_df.empty else pd.DataFrame()

    hdbscan_df = df[
        (df["method"] == "hdbscan") &
        (df["silhouette"].notna())
    ].copy()

    if sil_df.empty and hdbscan_df.empty:
        print("[skip] no valid silhouette values found")
        return

    plt.figure(figsize=(10, 6))

    for method in methods:
        sub = sil_df[sil_df["method"] == method].sort_values("k")
        if sub.empty:
            continue
        plt.plot(sub["k"], sub["silhouette"], marker="o", label=method)

    if not hdbscan_df.empty:
        hdb_sil = float(hdbscan_df.iloc[0]["silhouette"])
        plt.axhline(
            y=hdb_sil,
            linestyle="--",
            linewidth=2,
            label=f"hdbscan ({hdb_sil:.3f})",
        )

    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_vs_k.png", dpi=200)
    plt.close()

    print("[ok] saved silhouette_vs_k.png")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    results_path = INPUT_DIR / "clustering_method_comparison.csv"
    top_method_names = get_top_methods(results_path, top_n=3)

    if not top_method_names:
        print("[skip] no top methods found")
        return

    for method_name in top_method_names:
        players_path = ASSIGNMENTS_DIR / f"{method_name}_players.csv"
        summary_path = SUMMARIES_DIR / f"{method_name}_summary.csv"

        if not players_path.exists():
            print(f"[skip] missing {players_path}")
            continue

        players = pd.read_csv(players_path)

        plot_pca_scatter(players, method_name, PLOTS_DIR)

        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            plot_radar(summary, method_name, PLOTS_DIR)
        else:
            print(f"[skip] missing {summary_path}")

    plot_k_metrics(results_path, PLOTS_DIR)

    print(f"\nSaved plots in ./{PLOTS_DIR}")


if __name__ == "__main__":
    main()