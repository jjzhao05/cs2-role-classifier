from awpy import Demo
import polars as pl
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print grenades grouped by entity_id")
    parser.add_argument("demo_path", type=Path, help="Path to .dem file")
    args = parser.parse_args()

    demo = Demo(str(args.demo_path))
    demo.parse()

    grenades = demo.grenades

    if "entity_id" not in grenades.columns:
        raise RuntimeError(f"'entity_id' not found. Available columns: {grenades.columns}")

    print("\n=== Grenades grouped by entity_id ===")

    grouped = (
        grenades
        .group_by("entity_id")
        .agg([
            pl.len().alias("rows"),
            pl.col("thrower").first().alias("thrower"),
            pl.col("grenade_type").first().alias("grenade_type"),
            pl.col("round_num").first().alias("round_num"),
            pl.col("tick").min().alias("start_tick"),
            pl.col("tick").max().alias("end_tick"),
        ])
        .sort("entity_id")
    )

    print(grouped)

    print("\n=== Summary ===")
    print(f"Unique grenades (entity_id count): {grouped.height}")


if __name__ == "__main__":
    main()
    