from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from awpy import Demo
from awpy.stats import adr, calculate_trades, kast, rating


TEAM_ALIASES = {
    "t": "t",
    "terrorist": "t",
    "terrorists": "t",
    "ct": "ct",
    "counterterrorist": "ct",
    "counterterrorists": "ct",
    "counter-terrorist": "ct",
    "counter-terrorists": "ct",
}

RIFLES = {"ak47", "m4a1", "m4a1_silencer", "aug", "sg556", "galilar", "famas"}
AWP = {"awp"}
SMGS = {"mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"}
PISTOLS = {
    "glock",
    "usp",
    "usp_silencer",
    "p2000",
    "hkp2000",
    "p250",
    "deagle",
    "elite",
    "fiveseven",
    "cz75a",
    "tec9",
    "revolver",
}
SHOTGUNS = {"xm1014", "mag7", "nova", "sawedoff"}
SCOUT = {"ssg08"}
UTIL_KILL_WEAPONS = {"hegrenade", "inferno", "molotov", "incgrenade"}
UTIL_DAMAGE_WEAPONS = {"hegrenade", "inferno", "molotov", "incgrenade"}


def pick_col(df: pl.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns exist: {list(candidates)}. Available: {df.columns}")
    return None


def empty_feature_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def normalize_side_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .replace_strict(TEAM_ALIASES, default=None)
    )


def resolve_input_path(input_path: str | None, test_mode: bool) -> Path:
    if test_mode:
        return Path("demos_test")
    if input_path is None:
        raise ValueError("You must provide an input demo path unless using --test.")
    return Path(input_path)


def iter_demo_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".dem":
            raise ValueError(f"Input file is not a .dem file: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    return sorted(path.rglob("*.dem"))


def safe_get_demo_table(demo: Demo, attr_names: Sequence[str]) -> pl.DataFrame:
    for name in attr_names:
        if hasattr(demo, name):
            value = getattr(demo, name)
            if value is None:
                continue
            if isinstance(value, pl.DataFrame):
                return value
            try:
                return pl.DataFrame(value)
            except Exception:
                continue
    return pl.DataFrame()


def build_awpy_builtin_stats(demo: Demo) -> pl.DataFrame:
    adr_df = adr(demo)
    kast_df = kast(demo, trade_length_in_seconds=5.0)
    rating_df = rating(demo)

    pieces: list[pl.DataFrame] = []

    if not adr_df.is_empty():
        pieces.append(
            adr_df.select(
                [
                    pl.col("name").cast(pl.Utf8).alias("player_name"),
                    normalize_side_expr("side").alias("side"),
                    pl.col("n_rounds").cast(pl.Int64).alias("awpy_rounds"),
                    pl.col("dmg").cast(pl.Float64).alias("awpy_damage"),
                    pl.col("adr").cast(pl.Float64).alias("adr"),
                ]
            )
        )

    if not kast_df.is_empty():
        pieces.append(
            kast_df.select(
                [
                    pl.col("name").cast(pl.Utf8).alias("player_name"),
                    normalize_side_expr("side").alias("side"),
                    pl.col("kast_rounds").cast(pl.Int64).alias("kast_rounds"),
                    pl.col("kast").cast(pl.Float64).alias("kast"),
                ]
            )
        )

    if not rating_df.is_empty():
        pieces.append(
            rating_df.select(
                [
                    pl.col("name").cast(pl.Utf8).alias("player_name"),
                    normalize_side_expr("side").alias("side"),
                    pl.col("impact").cast(pl.Float64).alias("impact"),
                    pl.col("rating").cast(pl.Float64).alias("rating"),
                ]
            )
        )

    if not pieces:
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "awpy_rounds": pl.Int64,
                "awpy_damage": pl.Float64,
                "adr": pl.Float64,
                "kast_rounds": pl.Int64,
                "kast": pl.Float64,
                "impact": pl.Float64,
                "rating": pl.Float64,
            }
        )

    out = (
        pl.concat(pieces, how="diagonal_relaxed")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
        .group_by(["player_name", "side"])
        .agg(
            [
                pl.max("awpy_rounds").alias("awpy_rounds"),
                pl.max("awpy_damage").alias("awpy_damage"),
                pl.max("adr").alias("adr"),
                pl.max("kast_rounds").alias("kast_rounds"),
                pl.max("kast").alias("kast"),
                pl.max("impact").alias("impact"),
                pl.max("rating").alias("rating"),
            ]
        )
    )

    return out


def build_rounds_played_from_builtin_stats(builtin_stats: pl.DataFrame) -> pl.DataFrame:
    if builtin_stats.is_empty():
        return empty_feature_frame(
            {"player_name": pl.Utf8, "side": pl.Utf8, "rounds_played": pl.Int64}
        )

    n_rounds_col = pick_col(builtin_stats, ["awpy_rounds", "n_rounds"], required=False)
    if n_rounds_col is None:
        return empty_feature_frame(
            {"player_name": pl.Utf8, "side": pl.Utf8, "rounds_played": pl.Int64}
        )

    return (
        builtin_stats.select(
            [
                pl.col("player_name").cast(pl.Utf8).alias("player_name"),
                pl.col("side").cast(pl.Utf8).alias("side"),
                pl.col(n_rounds_col).cast(pl.Int64).alias("rounds_played"),
            ]
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
        .group_by(["player_name", "side"])
        .agg(pl.max("rounds_played").alias("rounds_played"))
    )


def build_kill_features(kills_df: pl.DataFrame) -> pl.DataFrame:
    if kills_df.is_empty():
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "kills": pl.Int64,
                "rifle_kills": pl.Int64,
                "awp_kills": pl.Int64,
                "smg_kills": pl.Int64,
                "pistol_kills": pl.Int64,
                "shotgun_kills": pl.Int64,
                "scout_kills": pl.Int64,
                "util_kills": pl.Int64,
            }
        )

    attacker_name_col = pick_col(kills_df, ["attacker_name", "killer_name"])
    attacker_side_col = pick_col(kills_df, ["attacker_side", "killer_side"])
    weapon_col = pick_col(kills_df, ["weapon", "weapon_name", "weapon_class"], required=False)

    df = (
        kills_df.select(
            [
                pl.col(attacker_name_col).cast(pl.Utf8).alias("player_name"),
                normalize_side_expr(attacker_side_col).alias("side"),
                (
                    pl.col(weapon_col).cast(pl.Utf8).str.to_lowercase()
                    if weapon_col is not None
                    else pl.lit(None, dtype=pl.Utf8)
                ).alias("weapon"),
            ]
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
    )

    return df.group_by(["player_name", "side"]).agg(
        [
            pl.len().alias("kills"),
            pl.col("weapon").is_in(list(RIFLES)).sum().alias("rifle_kills"),
            pl.col("weapon").is_in(list(AWP)).sum().alias("awp_kills"),
            pl.col("weapon").is_in(list(SMGS)).sum().alias("smg_kills"),
            pl.col("weapon").is_in(list(PISTOLS)).sum().alias("pistol_kills"),
            pl.col("weapon").is_in(list(SHOTGUNS)).sum().alias("shotgun_kills"),
            pl.col("weapon").is_in(list(SCOUT)).sum().alias("scout_kills"),
            pl.col("weapon").is_in(list(UTIL_KILL_WEAPONS)).sum().alias("util_kills"),
        ]
    )


def build_death_features(kills_df: pl.DataFrame) -> pl.DataFrame:
    if kills_df.is_empty():
        return empty_feature_frame(
            {"player_name": pl.Utf8, "side": pl.Utf8, "deaths": pl.Int64}
        )

    victim_name_col = pick_col(kills_df, ["victim_name", "player_name"])
    victim_side_col = pick_col(kills_df, ["victim_side", "player_side"])

    return (
        kills_df.select(
            [
                pl.col(victim_name_col).cast(pl.Utf8).alias("player_name"),
                normalize_side_expr(victim_side_col).alias("side"),
            ]
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        .group_by(["player_name", "side"])
        .agg(pl.len().alias("deaths"))
    )


def build_assist_features(kills_df: pl.DataFrame) -> pl.DataFrame:
    if kills_df.is_empty():
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "assists": pl.Int64,
                "flash_assists": pl.Int64,
            }
        )

    assister_name_col = pick_col(kills_df, ["assister_name", "assister"], required=False)
    assister_side_col = pick_col(kills_df, ["assister_side"], required=False)
    assistedflash_col = pick_col(kills_df, ["assistedflash"], required=False)

    if assister_name_col is None or assister_side_col is None:
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "assists": pl.Int64,
                "flash_assists": pl.Int64,
            }
        )

    return (
        kills_df.select(
            [
                pl.col(assister_name_col).cast(pl.Utf8).alias("player_name"),
                normalize_side_expr(assister_side_col).alias("side"),
                (
                    pl.col(assistedflash_col).fill_null(False).cast(pl.Boolean())
                    if assistedflash_col is not None
                    else pl.lit(False)
                ).alias("assistedflash"),
            ]
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        .group_by(["player_name", "side"])
        .agg(
            [
                pl.len().alias("assists"),
                pl.col("assistedflash").sum().alias("flash_assists"),
            ]
        )
    )


def build_opening_features(kills_df: pl.DataFrame) -> pl.DataFrame:
    if kills_df.is_empty():
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "opening_kills": pl.Int64,
                "opening_deaths": pl.Int64,
            }
        )

    round_col = pick_col(kills_df, ["round_num", "round_number", "round", "round_index"])
    tick_col = pick_col(kills_df, ["tick", "game_tick", "event_tick"], required=False)
    attacker_name_col = pick_col(kills_df, ["attacker_name", "killer_name"])
    attacker_side_col = pick_col(kills_df, ["attacker_side", "killer_side"])
    victim_name_col = pick_col(kills_df, ["victim_name", "player_name"])
    victim_side_col = pick_col(kills_df, ["victim_side", "player_side"])

    if tick_col is None:
        first_kills = kills_df.unique(subset=[round_col], keep="first")
    else:
        first_kills = kills_df.sort([round_col, tick_col]).unique(subset=[round_col], keep="first")

    attackers = first_kills.select(
        [
            pl.col(attacker_name_col).cast(pl.Utf8).alias("player_name"),
            normalize_side_expr(attacker_side_col).alias("side"),
            pl.lit(1).alias("opening_kills"),
            pl.lit(0).alias("opening_deaths"),
        ]
    )

    victims = first_kills.select(
        [
            pl.col(victim_name_col).cast(pl.Utf8).alias("player_name"),
            normalize_side_expr(victim_side_col).alias("side"),
            pl.lit(0).alias("opening_kills"),
            pl.lit(1).alias("opening_deaths"),
        ]
    )

    return (
        pl.concat([attackers, victims], how="vertical")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        .group_by(["player_name", "side"])
        .agg(
            [
                pl.sum("opening_kills").alias("opening_kills"),
                pl.sum("opening_deaths").alias("opening_deaths"),
            ]
        )
    )


def build_multikill_features(kills_df: pl.DataFrame) -> pl.DataFrame:
    if kills_df.is_empty():
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "rounds_with_2k": pl.Int64,
                "rounds_with_3k": pl.Int64,
                "rounds_with_4k": pl.Int64,
                "rounds_with_5k": pl.Int64,
                "multi_kill_rounds": pl.Int64,
            }
        )

    round_col = pick_col(kills_df, ["round_num", "round_number", "round", "round_index"])
    attacker_name_col = pick_col(kills_df, ["attacker_name", "killer_name"])
    attacker_side_col = pick_col(kills_df, ["attacker_side", "killer_side"])

    per_round_kills = (
        kills_df.select(
            [
                pl.col(attacker_name_col).cast(pl.Utf8).alias("player_name"),
                normalize_side_expr(attacker_side_col).alias("side"),
                pl.col(round_col).alias("round_id"),
            ]
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        .group_by(["player_name", "side", "round_id"])
        .agg(pl.len().alias("kills_in_round"))
    )

    return per_round_kills.group_by(["player_name", "side"]).agg(
        [
            (pl.col("kills_in_round") >= 2).sum().alias("rounds_with_2k"),
            (pl.col("kills_in_round") >= 3).sum().alias("rounds_with_3k"),
            (pl.col("kills_in_round") >= 4).sum().alias("rounds_with_4k"),
            (pl.col("kills_in_round") >= 5).sum().alias("rounds_with_5k"),
            (pl.col("kills_in_round") >= 2).sum().alias("multi_kill_rounds"),
        ]
    )


def build_player_round_side_map(kills_df: pl.DataFrame) -> pl.DataFrame:
    if kills_df.is_empty():
        return empty_feature_frame(
            {"player_name": pl.Utf8, "round_num": pl.Int64, "side": pl.Utf8}
        )

    round_col = pick_col(kills_df, ["round_num", "round_number", "round", "round_index"])
    attacker_name_col = pick_col(kills_df, ["attacker_name", "killer_name"], required=False)
    attacker_side_col = pick_col(kills_df, ["attacker_side", "killer_side"], required=False)
    victim_name_col = pick_col(kills_df, ["victim_name", "player_name"], required=False)
    victim_side_col = pick_col(kills_df, ["victim_side", "player_side"], required=False)

    pieces: list[pl.DataFrame] = []

    if attacker_name_col and attacker_side_col:
        pieces.append(
            kills_df.select(
                [
                    pl.col(attacker_name_col).cast(pl.Utf8).alias("player_name"),
                    pl.col(round_col).cast(pl.Int64).alias("round_num"),
                    normalize_side_expr(attacker_side_col).alias("side"),
                ]
            )
        )

    if victim_name_col and victim_side_col:
        pieces.append(
            kills_df.select(
                [
                    pl.col(victim_name_col).cast(pl.Utf8).alias("player_name"),
                    pl.col(round_col).cast(pl.Int64).alias("round_num"),
                    normalize_side_expr(victim_side_col).alias("side"),
                ]
            )
        )

    if not pieces:
        return empty_feature_frame(
            {"player_name": pl.Utf8, "round_num": pl.Int64, "side": pl.Utf8}
        )

    return (
        pl.concat(pieces, how="vertical")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        .unique()
    )


def build_damage_features(damages_df: pl.DataFrame) -> pl.DataFrame:
    if damages_df.is_empty():
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "damage": pl.Float64,
                "damage_taken": pl.Float64,
                "util_damage": pl.Float64,
            }
        )

    attacker_name_col = pick_col(damages_df, ["attacker_name", "attacker"], required=False)
    attacker_side_col = pick_col(damages_df, ["attacker_side"], required=False)
    victim_name_col = pick_col(damages_df, ["victim_name", "victim"], required=False)
    victim_side_col = pick_col(damages_df, ["victim_side"], required=False)
    dmg_col = pick_col(
        damages_df,
        ["dmg_health", "hp_damage", "health_damage", "damage", "damage_health"],
        required=False,
    )
    weapon_col = pick_col(damages_df, ["weapon", "weapon_name"], required=False)

    pieces: list[pl.DataFrame] = []

    if attacker_name_col and attacker_side_col and dmg_col:
        attacker_piece = (
            damages_df.select(
                [
                    pl.col(attacker_name_col).cast(pl.Utf8).alias("player_name"),
                    normalize_side_expr(attacker_side_col).alias("side"),
                    pl.col(dmg_col).cast(pl.Float64).fill_null(0.0).alias("damage"),
                    (
                        pl.col(weapon_col).cast(pl.Utf8).str.to_lowercase()
                        if weapon_col is not None
                        else pl.lit(None, dtype=pl.Utf8)
                    ).alias("weapon"),
                ]
            )
            .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
            .with_columns(
                pl.when(pl.col("weapon").is_in(list(UTIL_DAMAGE_WEAPONS)))
                .then(pl.col("damage"))
                .otherwise(0.0)
                .alias("util_damage"),
                pl.lit(0.0).alias("damage_taken"),
            )
            .select(["player_name", "side", "damage", "damage_taken", "util_damage"])
        )
        pieces.append(attacker_piece)

    if victim_name_col and victim_side_col and dmg_col:
        victim_piece = (
            damages_df.select(
                [
                    pl.col(victim_name_col).cast(pl.Utf8).alias("player_name"),
                    normalize_side_expr(victim_side_col).alias("side"),
                    pl.lit(0.0).alias("damage"),
                    pl.col(dmg_col).cast(pl.Float64).fill_null(0.0).alias("damage_taken"),
                    pl.lit(0.0).alias("util_damage"),
                ]
            )
            .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        )
        pieces.append(victim_piece)

    if not pieces:
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "damage": pl.Float64,
                "damage_taken": pl.Float64,
                "util_damage": pl.Float64,
            }
        )

    return pl.concat(pieces, how="vertical").group_by(["player_name", "side"]).agg(
        [
            pl.sum("damage").alias("damage"),
            pl.sum("damage_taken").alias("damage_taken"),
            pl.sum("util_damage").alias("util_damage"),
        ]
    )


def build_grenade_features(shots_df: pl.DataFrame) -> pl.DataFrame:
    if shots_df.is_empty():
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "grenades_thrown": pl.Int64,
                "he_grenades_thrown": pl.Int64,
                "flashbangs_thrown": pl.Int64,
                "smokes_thrown": pl.Int64,
                "fire_nades_thrown": pl.Int64,
                "decoys_thrown": pl.Int64,
            }
        )

    player_col = pick_col(shots_df, ["player_name", "player", "name"], required=False)
    side_col = pick_col(shots_df, ["player_side", "side"], required=False)
    weapon_col = pick_col(shots_df, ["weapon"], required=False)

    if player_col is None or side_col is None or weapon_col is None:
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "grenades_thrown": pl.Int64,
                "he_grenades_thrown": pl.Int64,
                "flashbangs_thrown": pl.Int64,
                "smokes_thrown": pl.Int64,
                "fire_nades_thrown": pl.Int64,
                "decoys_thrown": pl.Int64,
            }
        )

    df = (
        shots_df.select(
            [
                pl.col(player_col).cast(pl.Utf8).alias("player_name"),
                normalize_side_expr(side_col).alias("side"),
                pl.col(weapon_col).cast(pl.Utf8).str.to_lowercase().alias("weapon"),
            ]
        )
        .filter(
            pl.col("player_name").is_not_null()
            & pl.col("side").is_in(["ct", "t"])
        )
        .filter(
            pl.col("weapon").is_in(
                [
                    "weapon_hegrenade",
                    "weapon_flashbang",
                    "weapon_smokegrenade",
                    "weapon_molotov",
                    "weapon_incgrenade",
                    "weapon_decoy",
                ]
            )
        )
    )

    return df.group_by(["player_name", "side"]).agg(
        [
            pl.len().alias("grenades_thrown"),
            (pl.col("weapon") == "weapon_hegrenade").sum().alias("he_grenades_thrown"),
            (pl.col("weapon") == "weapon_flashbang").sum().alias("flashbangs_thrown"),
            (pl.col("weapon") == "weapon_smokegrenade").sum().alias("smokes_thrown"),
            (
                (pl.col("weapon") == "weapon_molotov")
                | (pl.col("weapon") == "weapon_incgrenade")
            ).sum().alias("fire_nades_thrown"),
            (pl.col("weapon") == "weapon_decoy").sum().alias("decoys_thrown"),
        ]
    )


def build_trade_features(kills_df: pl.DataFrame) -> pl.DataFrame:
    if kills_df.is_empty():
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "trade_kills": pl.Int64,
                "traded_deaths": pl.Int64,
            }
        )

    required_cols = {"round_num", "tick", "attacker_name", "attacker_side", "victim_name", "victim_side", "was_traded"}
    missing = required_cols - set(kills_df.columns)
    if missing:
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "trade_kills": pl.Int64,
                "traded_deaths": pl.Int64,
            }
        )

    df = (
        kills_df.select(
            [
                pl.col("round_num").cast(pl.Int64).alias("round_num"),
                pl.col("tick").cast(pl.Int64).alias("tick"),
                pl.col("attacker_name").cast(pl.Utf8).alias("attacker_name"),
                normalize_side_expr("attacker_side").alias("attacker_side"),
                pl.col("victim_name").cast(pl.Utf8).alias("victim_name"),
                normalize_side_expr("victim_side").alias("victim_side"),
                pl.col("was_traded").fill_null(False).cast(pl.Boolean()).alias("was_traded"),
            ]
        )
        .with_row_index("kill_id")
        .filter(
            pl.col("attacker_name").is_not_null()
            & pl.col("victim_name").is_not_null()
            & pl.col("attacker_side").is_not_null()
            & pl.col("victim_side").is_not_null()
        )
    )

    traded_deaths = (
        df.filter(pl.col("was_traded"))
        .select(
            [
                pl.col("victim_name").alias("player_name"),
                pl.col("victim_side").alias("side"),
                pl.lit(0).alias("trade_kills"),
                pl.lit(1).alias("traded_deaths"),
            ]
        )
    )

    traded_orig = (
        df.filter(pl.col("was_traded"))
        .select(
            [
                pl.col("kill_id").alias("orig_kill_id"),
                pl.col("round_num"),
                pl.col("tick").alias("orig_tick"),
                pl.col("attacker_name").alias("orig_attacker_name"),
                pl.col("attacker_side").alias("orig_attacker_side"),
                pl.col("victim_name").alias("orig_victim_name"),
                pl.col("victim_side").alias("orig_victim_side"),
            ]
        )
    )

    later_kills = df.select(
        [
            pl.col("kill_id"),
            pl.col("round_num"),
            pl.col("tick").alias("trade_tick"),
            pl.col("attacker_name").alias("trade_attacker_name"),
            pl.col("attacker_side").alias("trade_attacker_side"),
            pl.col("victim_name").alias("trade_victim_name"),
            pl.col("victim_side").alias("trade_victim_side"),
        ]
    )

    trade_matches = (
        traded_orig.join(later_kills, on="round_num", how="inner")
        .filter(pl.col("trade_tick") > pl.col("orig_tick"))
        .filter(pl.col("trade_victim_name") == pl.col("orig_attacker_name"))
        .filter(pl.col("trade_attacker_side") == pl.col("orig_victim_side"))
        .filter(pl.col("trade_victim_side") == pl.col("orig_attacker_side"))
        .sort(["orig_kill_id", "trade_tick", "kill_id"])
        .unique(subset=["orig_kill_id"], keep="first")
    )

    trade_kills = trade_matches.select(
        [
            pl.col("trade_attacker_name").alias("player_name"),
            pl.col("trade_attacker_side").alias("side"),
            pl.lit(1).alias("trade_kills"),
            pl.lit(0).alias("traded_deaths"),
        ]
    )

    pieces: list[pl.DataFrame] = []
    if traded_deaths.height > 0:
        pieces.append(traded_deaths)
    if trade_kills.height > 0:
        pieces.append(trade_kills)

    if not pieces:
        return empty_feature_frame(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "trade_kills": pl.Int64,
                "traded_deaths": pl.Int64,
            }
        )

    return (
        pl.concat(pieces, how="vertical")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        .group_by(["player_name", "side"])
        .agg(
            [
                pl.sum("trade_kills").alias("trade_kills"),
                pl.sum("traded_deaths").alias("traded_deaths"),
            ]
        )
    )


def build_base_player_side_table(
    kills_df: pl.DataFrame,
    grenades_df: pl.DataFrame,
    damages_df: pl.DataFrame,
    builtin_stats: pl.DataFrame,
) -> pl.DataFrame:
    dfs: list[pl.DataFrame] = []

    if not builtin_stats.is_empty():
        dfs.append(
            builtin_stats.select(
                [
                    pl.col("player_name").cast(pl.Utf8).alias("player_name"),
                    pl.col("side").cast(pl.Utf8).alias("side"),
                ]
            )
        )

    if not kills_df.is_empty():
        attacker_name_col = pick_col(kills_df, ["attacker_name", "killer_name"], required=False)
        attacker_side_col = pick_col(kills_df, ["attacker_side", "killer_side"], required=False)
        victim_name_col = pick_col(kills_df, ["victim_name", "player_name"], required=False)
        victim_side_col = pick_col(kills_df, ["victim_side", "player_side"], required=False)
        assister_name_col = pick_col(kills_df, ["assister_name", "assister"], required=False)
        assister_side_col = pick_col(kills_df, ["assister_side"], required=False)

        if attacker_name_col and attacker_side_col:
            dfs.append(
                kills_df.select(
                    [
                        pl.col(attacker_name_col).cast(pl.Utf8).alias("player_name"),
                        normalize_side_expr(attacker_side_col).alias("side"),
                    ]
                )
            )

        if victim_name_col and victim_side_col:
            dfs.append(
                kills_df.select(
                    [
                        pl.col(victim_name_col).cast(pl.Utf8).alias("player_name"),
                        normalize_side_expr(victim_side_col).alias("side"),
                    ]
                )
            )

        if assister_name_col and assister_side_col:
            dfs.append(
                kills_df.select(
                    [
                        pl.col(assister_name_col).cast(pl.Utf8).alias("player_name"),
                        normalize_side_expr(assister_side_col).alias("side"),
                    ]
                )
            )

    if not grenades_df.is_empty():
        player_col = pick_col(grenades_df, ["thrower_name", "thrower", "player_name", "owner_name"], required=False)
        if player_col:
            round_col = pick_col(grenades_df, ["round_num", "round_number", "round", "round_index"], required=False)
            side_col = pick_col(grenades_df, ["thrower_side", "player_side", "owner_side"], required=False)

            if side_col:
                dfs.append(
                    grenades_df.select(
                        [
                            pl.col(player_col).cast(pl.Utf8).alias("player_name"),
                            normalize_side_expr(side_col).alias("side"),
                        ]
                    )
                )
            elif round_col and not kills_df.is_empty():
                round_side_map = build_player_round_side_map(kills_df)
                inferred = (
                    grenades_df.select(
                        [
                            pl.col(player_col).cast(pl.Utf8).alias("player_name"),
                            pl.col(round_col).cast(pl.Int64).alias("round_num"),
                        ]
                    )
                    .join(round_side_map, on=["player_name", "round_num"], how="left")
                    .select(["player_name", "side"])
                )
                dfs.append(inferred)

    if not damages_df.is_empty():
        attacker_name_col = pick_col(damages_df, ["attacker_name", "attacker"], required=False)
        attacker_side_col = pick_col(damages_df, ["attacker_side"], required=False)
        victim_name_col = pick_col(damages_df, ["victim_name", "victim"], required=False)
        victim_side_col = pick_col(damages_df, ["victim_side"], required=False)

        if attacker_name_col and attacker_side_col:
            dfs.append(
                damages_df.select(
                    [
                        pl.col(attacker_name_col).cast(pl.Utf8).alias("player_name"),
                        normalize_side_expr(attacker_side_col).alias("side"),
                    ]
                )
            )

        if victim_name_col and victim_side_col:
            dfs.append(
                damages_df.select(
                    [
                        pl.col(victim_name_col).cast(pl.Utf8).alias("player_name"),
                        normalize_side_expr(victim_side_col).alias("side"),
                    ]
                )
            )

    if not dfs:
        return empty_feature_frame({"player_name": pl.Utf8, "side": pl.Utf8})

    return (
        pl.concat(dfs, how="vertical")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_not_null())
        .unique()
    )


def add_derived_side_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([pl.col(c).fill_null(0) for c in df.columns if c not in {"player_name", "side"}])

    def rate(num: str, den: str, out: str) -> pl.Expr:
        return (
            pl.when(pl.col(den) > 0)
            .then(pl.col(num) / pl.col(den))
            .otherwise(0.0)
            .alias(out)
        )

    exprs: list[pl.Expr] = []

    if {"trade_kills", "kills"} <= set(df.columns):
        exprs.append(rate("trade_kills", "kills", "trade_kill_rate"))

    if {"traded_deaths", "deaths"} <= set(df.columns):
        exprs.append(rate("traded_deaths", "deaths", "death_traded_rate"))

    if {"opening_kills", "kills"} <= set(df.columns):
        exprs.append(rate("opening_kills", "kills", "opening_kill_rate"))

    if {"opening_deaths", "deaths"} <= set(df.columns):
        exprs.append(rate("opening_deaths", "deaths", "opening_death_rate"))

    if {"grenades_thrown", "rounds_played"} <= set(df.columns):
        exprs.append(rate("grenades_thrown", "rounds_played", "grenades_per_round"))

    if {"he_grenades_thrown", "rounds_played"} <= set(df.columns):
        exprs.append(rate("he_grenades_thrown", "rounds_played", "he_grenades_per_round"))

    if {"flashbangs_thrown", "rounds_played"} <= set(df.columns):
        exprs.append(rate("flashbangs_thrown", "rounds_played", "flashbangs_per_round"))

    if {"smokes_thrown", "rounds_played"} <= set(df.columns):
        exprs.append(rate("smokes_thrown", "rounds_played", "smokes_per_round"))

    if {"fire_nades_thrown", "rounds_played"} <= set(df.columns):
        exprs.append(rate("fire_nades_thrown", "rounds_played", "fire_nades_per_round"))

    if {"decoys_thrown", "rounds_played"} <= set(df.columns):
        exprs.append(rate("decoys_thrown", "rounds_played", "decoys_per_round"))

    if {"kills", "rounds_played"} <= set(df.columns):
        exprs.append(rate("kills", "rounds_played", "kpr"))

    if {"deaths", "rounds_played"} <= set(df.columns):
        exprs.append(rate("deaths", "rounds_played", "dpr"))

    if {"kills", "deaths"} <= set(df.columns):
        exprs.append(rate("kills", "deaths", "kdr"))

    if {"rounds_played", "deaths"} <= set(df.columns):
        exprs.append(
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("rounds_played") - pl.col("deaths")) / pl.col("rounds_played"))
            .otherwise(0.0)
            .clip(lower_bound=0.0)
            .alias("survival_rate")
        )

    if {"opening_kills", "opening_deaths"} <= set(df.columns):
        exprs.append((pl.col("opening_kills") + pl.col("opening_deaths")).alias("opening_duel_attempts"))
        exprs.append(
            pl.when((pl.col("opening_kills") + pl.col("opening_deaths")) > 0)
            .then(pl.col("opening_kills") / (pl.col("opening_kills") + pl.col("opening_deaths")))
            .otherwise(0.0)
            .alias("opening_duel_success")
        )

    if {"trade_kills", "traded_deaths", "rounds_played"} <= set(df.columns):
        exprs.append(
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("trade_kills") + pl.col("traded_deaths")) / pl.col("rounds_played"))
            .otherwise(0.0)
            .alias("trade_participation")
        )

    if {"awp_kills", "kills"} <= set(df.columns):
        exprs.append(rate("awp_kills", "kills", "awp_kill_share"))

    if {"rifle_kills", "kills"} <= set(df.columns):
        exprs.append(rate("rifle_kills", "kills", "rifle_kill_share"))

    if {"multi_kill_rounds", "rounds_played"} <= set(df.columns):
        exprs.append(rate("multi_kill_rounds", "rounds_played", "multi_kill_rate"))

    if {"assists", "rounds_played"} <= set(df.columns):
        exprs.append(rate("assists", "rounds_played", "assists_per_round"))

    if {"flash_assists", "rounds_played"} <= set(df.columns):
        exprs.append(rate("flash_assists", "rounds_played", "flash_assists_per_round"))

    if {"damage", "rounds_played"} <= set(df.columns):
        exprs.append(rate("damage", "rounds_played", "damage_per_round"))

    if {"damage_taken", "rounds_played"} <= set(df.columns):
        exprs.append(rate("damage_taken", "rounds_played", "damage_taken_per_round"))

    if {"damage", "damage_taken", "rounds_played"} <= set(df.columns):
        exprs.append(
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("damage") - pl.col("damage_taken")) / pl.col("rounds_played"))
            .otherwise(0.0)
            .alias("damage_diff_per_round")
        )

    if {"util_damage", "rounds_played"} <= set(df.columns):
        exprs.append(rate("util_damage", "rounds_played", "util_damage_per_round"))

    if exprs:
        df = df.with_columns(exprs)

    return df.fill_null(0).fill_nan(0)


def parse_single_demo(demo_path: Path) -> pl.DataFrame:
    demo = Demo(str(demo_path))
    demo.parse()

    kills_df = calculate_trades(demo)
    grenades_df = safe_get_demo_table(demo, ["grenades", "grenades_df"])
    shots_df = safe_get_demo_table(demo, ["shots", "shots_df"])
    damages_df = safe_get_demo_table(demo, ["damages", "damages_df"])
    builtin_stats = build_awpy_builtin_stats(demo)

    if (
        kills_df.is_empty()
        and grenades_df.is_empty()
        and shots_df.is_empty()
        and damages_df.is_empty()
        and builtin_stats.is_empty()
    ):
        raise RuntimeError("No usable event tables found in demo.")

    base = build_base_player_side_table(kills_df, grenades_df, damages_df, builtin_stats)
    rounds_played = build_rounds_played_from_builtin_stats(builtin_stats)
    kill_features = build_kill_features(kills_df)
    death_features = build_death_features(kills_df)
    assist_features = build_assist_features(kills_df)
    trade_features = build_trade_features(kills_df)
    opening_features = build_opening_features(kills_df)
    multikill_features = build_multikill_features(kills_df)
    damage_features = build_damage_features(damages_df)
    grenade_features = build_grenade_features(shots_df)

    out = (
        base.join(rounds_played, on=["player_name", "side"], how="left")
        .join(kill_features, on=["player_name", "side"], how="left")
        .join(death_features, on=["player_name", "side"], how="left")
        .join(assist_features, on=["player_name", "side"], how="left")
        .join(trade_features, on=["player_name", "side"], how="left")
        .join(opening_features, on=["player_name", "side"], how="left")
        .join(multikill_features, on=["player_name", "side"], how="left")
        .join(damage_features, on=["player_name", "side"], how="left")
        .join(grenade_features, on=["player_name", "side"], how="left")
        .join(builtin_stats, on=["player_name", "side"], how="left")
    )

    return add_derived_side_features(out)


def split_sides_wide(df: pl.DataFrame) -> pl.DataFrame:
    value_cols = [c for c in df.columns if c not in {"player_name", "side"}]

    wide = df.pivot(
        index="player_name",
        on="side",
        values=value_cols,
        aggregate_function="first",
    )

    rename_map: dict[str, str] = {}
    for col in wide.columns:
        if col == "player_name":
            continue

        if col.endswith("_ct") or col.endswith("_t"):
            rename_map[col] = col
            continue

        if col.startswith("ct_"):
            rename_map[col] = f"{col[3:]}_ct"
            continue
        if col.startswith("t_"):
            rename_map[col] = f"{col[2:]}_t"
            continue

        parts = col.split("_")
        if len(parts) >= 2 and parts[0] in {"ct", "t"}:
            rename_map[col] = f'{"_".join(parts[1:])}_{parts[0]}'

    if rename_map:
        wide = wide.rename(rename_map)

    return wide.fill_null(0).fill_nan(0)


def sort_output_columns(df: pl.DataFrame) -> pl.DataFrame:
    preferred = [
        "player_name",

        "rounds_played_ct", "rounds_played_t",
        "awpy_rounds_ct", "awpy_rounds_t",

        "kills_ct", "kills_t",
        "deaths_ct", "deaths_t",
        "kpr_ct", "kpr_t",
        "dpr_ct", "dpr_t",
        "kdr_ct", "kdr_t",
        "survival_rate_ct", "survival_rate_t",

        "damage_ct", "damage_t",
        "awpy_damage_ct", "awpy_damage_t",
        "damage_taken_ct", "damage_taken_t",
        "damage_per_round_ct", "damage_per_round_t",
        "damage_taken_per_round_ct", "damage_taken_per_round_t",
        "damage_diff_per_round_ct", "damage_diff_per_round_t",
        "util_damage_ct", "util_damage_t",
        "util_damage_per_round_ct", "util_damage_per_round_t",
        "adr_ct", "adr_t",
        "kast_rounds_ct", "kast_rounds_t",
        "kast_ct", "kast_t",
        "impact_ct", "impact_t",
        "rating_ct", "rating_t",

        "assists_ct", "assists_t",
        "assists_per_round_ct", "assists_per_round_t",
        "flash_assists_ct", "flash_assists_t",
        "flash_assists_per_round_ct", "flash_assists_per_round_t",

        "trade_kills_ct", "trade_kills_t",
        "traded_deaths_ct", "traded_deaths_t",
        "trade_kill_rate_ct", "trade_kill_rate_t",
        "death_traded_rate_ct", "death_traded_rate_t",
        "trade_participation_ct", "trade_participation_t",

        "opening_kills_ct", "opening_kills_t",
        "opening_deaths_ct", "opening_deaths_t",
        "opening_kill_rate_ct", "opening_kill_rate_t",
        "opening_death_rate_ct", "opening_death_rate_t",
        "opening_duel_attempts_ct", "opening_duel_attempts_t",
        "opening_duel_success_ct", "opening_duel_success_t",

        "rounds_with_2k_ct", "rounds_with_2k_t",
        "rounds_with_3k_ct", "rounds_with_3k_t",
        "rounds_with_4k_ct", "rounds_with_4k_t",
        "rounds_with_5k_ct", "rounds_with_5k_t",
        "multi_kill_rounds_ct", "multi_kill_rounds_t",
        "multi_kill_rate_ct", "multi_kill_rate_t",

        "rifle_kills_ct", "rifle_kills_t",
        "awp_kills_ct", "awp_kills_t",
        "smg_kills_ct", "smg_kills_t",
        "pistol_kills_ct", "pistol_kills_t",
        "shotgun_kills_ct", "shotgun_kills_t",
        "scout_kills_ct", "scout_kills_t",
        "util_kills_ct", "util_kills_t",
        "rifle_kill_share_ct", "rifle_kill_share_t",
        "awp_kill_share_ct", "awp_kill_share_t",

        "grenades_thrown_ct", "grenades_thrown_t",
        "he_grenades_thrown_ct", "he_grenades_thrown_t",
        "flashbangs_thrown_ct", "flashbangs_thrown_t",
        "smokes_thrown_ct", "smokes_thrown_t",
        "fire_nades_thrown_ct", "fire_nades_thrown_t",
        "decoys_thrown_ct", "decoys_thrown_t",
        "grenades_per_round_ct", "grenades_per_round_t",
        "he_grenades_per_round_ct", "he_grenades_per_round_t",
        "flashbangs_per_round_ct", "flashbangs_per_round_t",
        "smokes_per_round_ct", "smokes_per_round_t",
        "fire_nades_per_round_ct", "fire_nades_per_round_t",
        "decoys_per_round_ct", "decoys_per_round_t",
    ]

    existing = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df.select(existing + remaining).sort("player_name")

def combine_demo_results(per_demo_frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    frames = list(per_demo_frames)
    if not frames:
        raise RuntimeError("No demos parsed successfully.")

    df = pl.concat(frames, how="diagonal_relaxed")

    key_cols = {"player_name", "side"}

    raw_sum_cols = [
        c
        for c in df.columns
        if c not in key_cols
        and c not in {
            "adr",
            "kast",
            "impact",
            "rating",
            "kpr",
            "dpr",
            "kdr",
            "survival_rate",
            "trade_kill_rate",
            "death_traded_rate",
            "opening_kill_rate",
            "opening_death_rate",
            "opening_duel_attempts",
            "opening_duel_success",
            "trade_participation",
            "awp_kill_share",
            "rifle_kill_share",
            "multi_kill_rate",
            "assists_per_round",
            "flash_assists_per_round",
            "damage_per_round",
            "damage_taken_per_round",
            "damage_diff_per_round",
            "util_damage_per_round",
            "grenades_per_round",
            "he_grenades_per_round",
            "flashbangs_per_round",
            "smokes_per_round",
            "fire_nades_per_round",
            "decoys_per_round",
        }
    ]

    work = df

    if "kast" in work.columns and "awpy_rounds" in work.columns:
        work = work.with_columns(
            (pl.col("kast").fill_null(0.0) * pl.col("awpy_rounds").fill_null(0)).alias("_kast_weighted")
        )

    if "impact" in work.columns and "awpy_rounds" in work.columns:
        work = work.with_columns(
            (pl.col("impact").fill_null(0.0) * pl.col("awpy_rounds").fill_null(0)).alias("_impact_weighted")
        )

    if "rating" in work.columns and "awpy_rounds" in work.columns:
        work = work.with_columns(
            (pl.col("rating").fill_null(0.0) * pl.col("awpy_rounds").fill_null(0)).alias("_rating_weighted")
        )

    agg_exprs = [pl.sum(c).alias(c) for c in raw_sum_cols]

    if "_kast_weighted" in work.columns:
        agg_exprs.append(pl.sum("_kast_weighted").alias("_kast_weighted"))
    if "_impact_weighted" in work.columns:
        agg_exprs.append(pl.sum("_impact_weighted").alias("_impact_weighted"))
    if "_rating_weighted" in work.columns:
        agg_exprs.append(pl.sum("_rating_weighted").alias("_rating_weighted"))

    combined = work.group_by(["player_name", "side"]).agg(agg_exprs)

    extra_exprs: list[pl.Expr] = []

    if {"awpy_damage", "awpy_rounds"} <= set(combined.columns):
        extra_exprs.append(
            pl.when(pl.col("awpy_rounds") > 0)
            .then(pl.col("awpy_damage") / pl.col("awpy_rounds"))
            .otherwise(0.0)
            .alias("adr")
        )

    if {"_kast_weighted", "awpy_rounds"} <= set(combined.columns):
        extra_exprs.append(
            pl.when(pl.col("awpy_rounds") > 0)
            .then(pl.col("_kast_weighted") / pl.col("awpy_rounds"))
            .otherwise(0.0)
            .alias("kast")
        )

    if {"_impact_weighted", "awpy_rounds"} <= set(combined.columns):
        extra_exprs.append(
            pl.when(pl.col("awpy_rounds") > 0)
            .then(pl.col("_impact_weighted") / pl.col("awpy_rounds"))
            .otherwise(0.0)
            .alias("impact")
        )

    if {"_rating_weighted", "awpy_rounds"} <= set(combined.columns):
        extra_exprs.append(
            pl.when(pl.col("awpy_rounds") > 0)
            .then(pl.col("_rating_weighted") / pl.col("awpy_rounds"))
            .otherwise(0.0)
            .alias("rating")
        )

    if extra_exprs:
        combined = combined.with_columns(extra_exprs)

    drop_cols = [c for c in ["_kast_weighted", "_impact_weighted", "_rating_weighted"] if c in combined.columns]
    if drop_cols:
        combined = combined.drop(drop_cols)

    return add_derived_side_features(combined)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", nargs="?", default=None)
    parser.add_argument("output_csv", nargs="?", default=None)
    parser.add_argument("--test", dest="test_output_csv", default=None)
    args = parser.parse_args()

    test_mode = args.test_output_csv is not None
    output_csv = args.test_output_csv if test_mode else args.output_csv

    if output_csv is None:
        raise ValueError("You must provide an output CSV path, or use --test <output_csv>.")

    input_path = resolve_input_path(args.input_path, test_mode)
    demo_files = iter_demo_files(input_path)

    print(f"[info] input path: {input_path.resolve()}")
    print(f"[info] found {len(demo_files)} demos to process")

    parsed_frames: list[pl.DataFrame] = []
    success_count = 0

    for i, demo_path in enumerate(demo_files, start=1):
        print(f"\n[{i}/{len(demo_files)}] parsing: {demo_path}")
        try:
            df = parse_single_demo(demo_path)
            parsed_frames.append(df)
            success_count += 1
            print(f"[ok {i}/{len(demo_files)}] parsed successfully")
        except Exception as e:
            print(f"[skip {i}/{len(demo_files)}] {demo_path}: {e}")

    if not parsed_frames:
        raise RuntimeError("No demos parsed successfully.")

    final_df = combine_demo_results(parsed_frames)
    final_df = split_sides_wide(final_df)
    final_df = sort_output_columns(final_df)
    final_df = final_df.fill_null(0).fill_nan(0)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_csv(output_path)

    print(f"\n[done] parsed {success_count}/{len(demo_files)} demos successfully")
    print(f"[done] output written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()