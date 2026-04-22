from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("outputs")
ASSIGNMENTS_DIR = INPUT_DIR / "assignments"
SUMMARIES_DIR = INPUT_DIR / "summaries"
PLOTS_DIR = Path("plots")

RADAR_FEATURES = [
    "opening_attempts_per_round_T",
    "opening_success_rate_T",
    "trade_success_rate_T",
    "flash_assists_per_round_T",
    "awp_kill_share_T",
    "rifle_kill_share_T",
    "avg_distance_from_team_T",
    "isolation_rate_T",
]


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

    for i, (_, row) in enumerate(radar_scaled.iterrows()):
        make_radar(
            axes[i],
            row[radar_features],
            radar_features,
            f"{method_name} cluster {int(row['cluster'])}",
        )

    for j in range(i + 1, len(axes)):
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

    # Plot k-based methods
    k_df = df[df["k"].notna()].copy()
    if not k_df.empty:
        k_df["k"] = k_df["k"].astype(int)

    methods = sorted(k_df["method"].dropna().unique()) if not k_df.empty else []

    sil_df = k_df.dropna(subset=["silhouette"]).copy() if not k_df.empty else pd.DataFrame()

    # HDBSCAN reference line
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

    if not ASSIGNMENTS_DIR.exists():
        print(f"[skip] missing assignments directory: {ASSIGNMENTS_DIR}")
        return

    player_files = sorted(ASSIGNMENTS_DIR.glob("*_players.csv"))
    if not player_files:
        print(f"[skip] no player assignment files found in {ASSIGNMENTS_DIR}")
        return

    for players_path in player_files:
        method_name = players_path.stem.replace("_players", "")
        summary_path = SUMMARIES_DIR / f"{method_name}_summary.csv"

        players = pd.read_csv(players_path)
        plot_pca_scatter(players, method_name, PLOTS_DIR)

        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            plot_radar(summary, method_name, PLOTS_DIR)
        else:
            print(f"[skip] missing {summary_path}")

    plot_k_metrics(
        INPUT_DIR / "clustering_method_comparison.csv",
        PLOTS_DIR,
    )

    print(f"\nSaved plots in ./{PLOTS_DIR}")


if __name__ == "__main__":
    main()