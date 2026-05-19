from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = Path("outputs")
PLOTS_DIR = Path("plots")
SIDES = ["ct", "t"]
TOP_N_PER_ALGORITHM = 2  # best N models per (side, algorithm) to plot

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


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def get_top_models(side: str, top_n: int = TOP_N_PER_ALGORITHM) -> list[str]:
    """
    Read outputs/{side}/model_scores.csv and return the names of the top N
    models per algorithm, ranked by composite_score descending.
    """
    scores_path = INPUT_DIR / side / "model_scores.csv"
    if not scores_path.exists():
        print(f"[skip] missing {scores_path}")
        return []

    df = pd.read_csv(scores_path)
    if df.empty:
        print(f"[skip] {scores_path} is empty")
        return []

    selected = []
    for method, group in df.groupby("method"):
        top = (
            group.dropna(subset=["composite_score"])
            .sort_values("composite_score", ascending=False)
            .head(top_n)
        )
        names = top["name"].tolist()
        selected.extend(names)
        print(f"  [{side.upper()}] {method}: {names}")

    return selected


# ---------------------------------------------------------------------------
# Per-cluster summary (replaces the old summaries/ CSV dir)
# ---------------------------------------------------------------------------

def build_cluster_summary(players: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster mean of all numeric columns (excluding noise cluster -1).
    Returns one row per cluster with a 'cluster' column.
    """
    numeric_cols = players.select_dtypes(include="number").columns.tolist()
    exclude = {"pc1", "pc2", "cluster"}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    non_noise = players[players["cluster"] != -1].copy()
    if non_noise.empty or "cluster" not in non_noise.columns:
        return pd.DataFrame()

    return non_noise.groupby("cluster")[feature_cols].mean().reset_index()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_pca_scatter(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    required = {"player_name", "pc1", "pc2", "cluster"}
    missing = required - set(players.columns)
    if missing:
        print(f"[skip] {method_name} scatter — missing columns: {sorted(missing)}")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    non_noise = players[players["cluster"] != -1]
    noise = players[players["cluster"] == -1]

    if not non_noise.empty:
        scatter = ax.scatter(
            non_noise["pc1"],
            non_noise["pc2"],
            c=non_noise["cluster"],
            cmap="tab10",
            s=80,
            zorder=3,
        )
        # Cluster legend
        handles, labels = scatter.legend_elements(prop="colors")
        cluster_ids = sorted(non_noise["cluster"].unique())
        ax.legend(handles, [f"Cluster {c}" for c in cluster_ids],
                  title="Cluster", loc="upper right")

    if not noise.empty:
        ax.scatter(
            noise["pc1"],
            noise["pc2"],
            s=140,
            marker="x",
            linewidths=2,
            color="grey",
            label="Noise / outliers",
            zorder=4,
        )

    for _, row in players.iterrows():
        ax.annotate(
            row["player_name"],
            (row["pc1"], row["pc2"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA Scatter — {method_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"{method_name}_pca.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


def plot_rating_box(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    if "cluster" not in players.columns:
        print(f"[skip] {method_name} rating box — missing cluster column")
        return

    # rating column dropped from final CSV; fall back to adr as proxy
    rating_col = next(
        (c for c in ["adr", "damage_per_round"] if c in players.columns),
        None,
    )
    if rating_col is None:
        print(f"[skip] {method_name} rating box — no rating/adr column found")
        return

    plot_df = players[["cluster", rating_col]].copy()
    plot_df[rating_col] = pd.to_numeric(plot_df[rating_col], errors="coerce")
    plot_df = plot_df.dropna()

    non_noise = plot_df[plot_df["cluster"] != -1]
    noise = plot_df[plot_df["cluster"] == -1]

    cluster_order = sorted(non_noise["cluster"].unique())
    tick_labels = [f"Cluster {c}" for c in cluster_order]
    data = [non_noise.loc[non_noise["cluster"] == c, rating_col].tolist() for c in cluster_order]

    if not noise.empty:
        tick_labels.append("Noise")
        data.append(noise[rating_col].tolist())

    if not data:
        print(f"[skip] {method_name} rating box — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_alpha(0.5)

    ax.set_xlabel("Cluster")
    ax.set_ylabel(rating_col)
    ax.set_title(f"ADR by Cluster — {method_name}")
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    out = output_dir / f"{method_name}_rating_box.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


def _make_radar(ax, values, categories, title):
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals = values.tolist() + values.tolist()[:1]
    angles = angles + angles[:1]

    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in categories],
        fontsize=8,
    )
    ax.set_yticklabels([])
    ax.set_title(title, y=1.12, fontsize=10)


def plot_radar(summary: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    if summary.empty or "cluster" not in summary.columns:
        print(f"[skip] {method_name} radar — empty or missing cluster column")
        return

    radar_features = [f for f in RADAR_FEATURES if f in summary.columns]
    if len(radar_features) < 3:
        print(f"[skip] {method_name} radar — fewer than 3 matching features")
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
        return

    ncols = 2 if num_clusters > 1 else 1
    nrows = int(np.ceil(num_clusters / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
        subplot_kw=dict(polar=True),
    )
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(radar_scaled.iterrows()):
        _make_radar(
            axes[i],
            row[radar_features],
            radar_features,
            f"Cluster {int(row['cluster'])}",
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Cluster Profiles — {method_name}", fontsize=12, y=1.01)
    plt.tight_layout()
    out = output_dir / f"{method_name}_radar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def plot_stability_bar(side: str, output_dir: Path) -> None:
    """
    Horizontal bar chart of stability_mean ± stability_std for all models
    on a given side, coloured by algorithm.
    """
    scores_path = INPUT_DIR / side / "model_scores.csv"
    if not scores_path.exists():
        return

    df = pd.read_csv(scores_path).dropna(subset=["stability_mean"])
    if df.empty:
        return

    df = df.sort_values("stability_mean", ascending=True)

    method_colors = {"kmeans": "#4C72B0", "gmm": "#DD8452", "hdbscan": "#55A868"}
    colors = [method_colors.get(m, "#888888") for m in df["method"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    ax.barh(df["name"], df["stability_mean"], xerr=df["stability_std"],
            color=colors, alpha=0.8, capsize=3)
    ax.axvline(x=0.50, linestyle="--", color="grey", linewidth=1, label="Min threshold (0.50)")
    ax.axvline(x=0.80, linestyle=":", color="green", linewidth=1, label="High tier (0.80)")
    ax.set_xlabel("Mean ARI (± 1 std)")
    ax.set_title(f"Bootstrap Stability — {side.upper()}")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)

    # Legend for algorithm colours
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=m) for m, c in method_colors.items()]
    ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0][-2:],
              fontsize=8, loc="lower right")

    plt.tight_layout()
    out = output_dir / f"{side}_stability_bar.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


def plot_silhouette_vs_k(side: str, output_dir: Path) -> None:
    scores_path = INPUT_DIR / side / "model_scores.csv"
    if not scores_path.exists():
        return

    df = pd.read_csv(scores_path).dropna(subset=["silhouette", "k"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for method, group in df.groupby("method"):
        if method == "hdbscan":
            continue  # k is emergent for HDBSCAN — not meaningful to plot vs fixed k
        sub = group.sort_values("k")
        ax.plot(sub["k"], sub["silhouette"], marker="o", label=method)

    hdb = df[df["method"] == "hdbscan"]
    if not hdb.empty:
        best_hdb_sil = hdb["silhouette"].max()
        ax.axhline(y=best_hdb_sil, linestyle="--", linewidth=1.5,
                   label=f"HDBSCAN best ({best_hdb_sil:.3f})")

    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Silhouette vs k — {side.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"{side}_silhouette_vs_k.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Cluster dominance: per-cluster mean table + z-score heatmap
# ---------------------------------------------------------------------------

# Features to exclude from dominance analysis (positional/meta, not role-defining)
_DOMINANCE_EXCLUDE = {"pc1", "pc2", "cluster", "rounds_played"}
# Std columns add noise to the mean table; keep them for the heatmap only
_STD_SUFFIX = "_std"

TOP_FEATURES_PER_CLUSTER = 8  # rows shown in the console mean table


def print_cluster_means(summary: pd.DataFrame, method_name: str) -> None:
    """
    Print a side-by-side table of per-cluster means for the most
    discriminating features (highest variance across clusters).
    Shows TOP_FEATURES_PER_CLUSTER features ranked by inter-cluster std.
    """
    if summary.empty or "cluster" not in summary.columns:
        return

    feat_cols = [
        c for c in summary.columns
        if c not in _DOMINANCE_EXCLUDE and not c.endswith(_STD_SUFFIX)
    ]
    if not feat_cols:
        return

    sub = summary.set_index("cluster")[feat_cols]

    # Rank features by how much they vary across clusters
    inter_std = sub.std(axis=0).sort_values(ascending=False)
    top_feats = inter_std.head(TOP_FEATURES_PER_CLUSTER).index.tolist()

    table = sub[top_feats].T
    table.index.name = "feature"

    print(f"\nCluster means — {method_name} (top {TOP_FEATURES_PER_CLUSTER} discriminating features):")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))


def plot_zscore_heatmap(
    summary: pd.DataFrame,
    players: pd.DataFrame,
    method_name: str,
    output_dir: Path,
) -> None:
    """
    Z-score heatmap: each cell is how many global std-devs above/below the
    population mean that cluster sits for each feature.

    - Rows = features (sorted by absolute max z-score so the most
      cluster-defining features float to the top)
    - Columns = clusters
    - Colour: diverging (red = above average, blue = below average)
    """
    if summary.empty or "cluster" not in summary.columns:
        print(f"[skip] {method_name} heatmap — empty summary")
        return

    feat_cols = [
        c for c in summary.columns
        if c not in _DOMINANCE_EXCLUDE
    ]
    if not feat_cols:
        return

    # Global stats from the full player population (non-noise only)
    non_noise = players[players["cluster"] != -1]
    available = [c for c in feat_cols if c in non_noise.columns]
    if not available:
        return

    global_mean = non_noise[available].mean()
    global_std = non_noise[available].std().replace(0, np.nan)

    cluster_means = summary.set_index("cluster")[available]
    z = (cluster_means - global_mean) / global_std
    z = z.dropna(axis=1, how="all")

    if z.empty:
        return

    # Sort features by max absolute z across clusters (most distinctive first)
    z = z.loc[:, z.abs().max(axis=0).sort_values(ascending=False).index]

    # Cap columns to avoid an unreadably wide plot
    MAX_FEATURES = 40
    z = z.iloc[:, :MAX_FEATURES]

    z_plot = z.T  # features as rows, clusters as columns

    fig_h = max(6, len(z_plot) * 0.35)
    fig_w = max(5, len(z_plot.columns) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = min(3.0, z_plot.abs().max().max())
    im = ax.imshow(z_plot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(z_plot.columns)))
    ax.set_xticklabels([f"Cluster {c}" for c in z_plot.columns], fontsize=10)
    ax.set_yticks(range(len(z_plot.index)))
    ax.set_yticklabels(z_plot.index, fontsize=8)

    # Annotate each cell with the z value
    for row_i, feat in enumerate(z_plot.index):
        for col_i, clust in enumerate(z_plot.columns):
            val = z_plot.loc[feat, clust]
            if pd.notna(val):
                ax.text(col_i, row_i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 1.5 else "white")

    plt.colorbar(im, ax=ax, label="Z-score (std devs from population mean)")
    ax.set_title(f"Feature Z-scores by Cluster — {method_name}\n"
                 f"(sorted by discriminating power, top {MAX_FEATURES} features)")
    plt.tight_layout()
    out = output_dir / f"{method_name}_zscore_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    for side in SIDES:
        side_plot_dir = PLOTS_DIR / side
        side_plot_dir.mkdir(exist_ok=True)

        print(f"\n=== {side.upper()} ===")

        model_names = get_top_models(side, top_n=TOP_N_PER_ALGORITHM)
        if not model_names:
            print(f"[skip] no models found for {side}")
            continue

        for name in model_names:
            players_path = INPUT_DIR / side / f"{name}_player_clusters.csv"
            if not players_path.exists():
                print(f"[skip] missing {players_path}")
                continue

            players = pd.read_csv(players_path)
            if players.empty:
                print(f"[skip] {players_path} is empty")
                continue

            summary = build_cluster_summary(players)

            print_cluster_means(summary, name)
            plot_pca_scatter(players, name, side_plot_dir)
            plot_rating_box(players, name, side_plot_dir)
            plot_radar(summary, name, side_plot_dir)
            plot_zscore_heatmap(summary, players, name, side_plot_dir)

        plot_stability_bar(side, side_plot_dir)
        plot_silhouette_vs_k(side, side_plot_dir)

    print(f"\nSaved plots in ./{PLOTS_DIR}")


if __name__ == "__main__":
    main()