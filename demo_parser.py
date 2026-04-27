from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from awpy import Demo
from awpy.stats import adr, kast, rating, calculate_trades

TEAM_ALIASES = {
    "t": "t",
    "terrorist": "t",
    "terrorists": "t",
    "ct": "ct",
    "counterterrorist": "ct",
    "counterterrorists": "ct",
    "counter-terrorist": "ct",
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
    "tec9",
    "cz75",
    "cz75a",
    "revolver",
}
SHOTGUNS = {"nova", "xm1014", "mag7", "sawedoff"}
SCOUT = {"ssg08"}
UTILITY_KILL_WEAPONS = {"hegrenade", "molotov", "incendiary", "inferno"}


def pick_col(df: pl.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing columns: {list(candidates)}. Available: {df.columns}")
    return None


def require_columns(df: pl.DataFrame, mapping: dict[str, Sequence[str]]) -> dict[str, str]:
    return {alias: pick_col(df, cols, required=True) for alias, cols in mapping.items()}


def normalize_team(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return TEAM_ALIASES.get(s, str(value).strip().lower())


def normalize_weapon(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def categorize_weapon(weapon: str | None) -> str | None:
    if weapon is None:
        return None
    if weapon in RIFLES:
        return "rifle"
    if weapon in AWP:
        return "awp"
    if weapon in SMGS:
        return "smg"
    if weapon in PISTOLS:
        return "pistol"
    if weapon in SHOTGUNS:
        return "shotgun"
    if weapon in SCOUT:
        return "scout"
    if "knife" in weapon:
        return "knife"
    return None


def map_name_from_demo(demo: Demo, demo_path: Path) -> str:
    header = getattr(demo, "header", None)
    if isinstance(header, dict):
        for key in ("map_name", "map", "mapName"):
            if key in header and header[key]:
                return str(header[key])
    return demo_path.stem


def get_player_round_totals_by_side(demo: Demo) -> pl.DataFrame:
    prt = demo.player_round_totals
    cols = require_columns(
        prt,
        {
            "player_name": ["name", "player_name"],
            "side": ["side", "team_side"],
            "n_rounds": ["n_rounds", "rounds", "rounds_played"],
        },
    )

    return (
        prt.select(
            [
                pl.col(cols["player_name"]).cast(pl.Utf8).alias("player_name"),
                pl.col(cols["side"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("side"),
                pl.col(cols["n_rounds"]).cast(pl.Int64).alias("rounds_played"),
            ]
        )
        .filter(pl.col("side").is_in(["t", "ct"]))
        .group_by(["player_name", "side"])
        .agg(pl.col("rounds_played").sum().alias("rounds_played"))
        .sort(["player_name", "side"])
    )


def get_ticks_for_side_lookup(demo: Demo) -> pl.DataFrame:
    ticks = demo.ticks
    cols = require_columns(
        ticks,
        {
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "name": ["name", "player_name"],
            "team": ["team_name", "side", "player_side", "team"],
        },
    )

    return (
        ticks.select(
            [
                pl.col(cols["round_num"]).cast(pl.Int64).alias("round_num"),
                pl.col(cols["tick"]).cast(pl.Int64).alias("tick"),
                pl.col(cols["name"]).cast(pl.Utf8).alias("player_name"),
                pl.col(cols["team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("team"),
            ]
        )
        .filter(pl.col("player_name").is_not_null())
        .filter(pl.col("team").is_in(["t", "ct"]))
        .unique(subset=["round_num", "tick", "player_name"])
    )


def get_grenade_events(demo: Demo) -> pl.DataFrame:
    gren = demo.grenades
    cols = require_columns(
        gren,
        {
            "entity_id": ["entity_id"],
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "thrower": ["thrower_name", "player_name", "thrower", "player"],
            "grenade_type": ["grenade_type", "weapon", "grenade"],
        },
    )

    return (
        gren.select(
            [
                pl.col(cols["entity_id"]).alias("entity_id"),
                pl.col(cols["round_num"]).cast(pl.Int64).alias("round_num"),
                pl.col(cols["tick"]).cast(pl.Int64).alias("tick"),
                pl.col(cols["thrower"]).cast(pl.Utf8).alias("thrower"),
                pl.col(cols["grenade_type"]).cast(pl.Utf8).str.to_lowercase().alias("grenade_type_raw"),
            ]
        )
        .filter(pl.col("thrower").is_not_null())
        .filter(pl.col("grenade_type_raw").is_not_null())
        .group_by("entity_id")
        .agg(
            [
                pl.col("round_num").first().alias("round_num"),
                pl.col("thrower").first().alias("thrower"),
                pl.col("grenade_type_raw").first().alias("grenade_type_raw"),
                pl.col("tick").min().alias("start_tick"),
            ]
        )
        .sort(["round_num", "start_tick"])
    )


def normalize_grenade_types(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            [
                pl.when(pl.col("grenade_type_raw").str.contains("flash"))
                .then(pl.lit("flashbang"))
                .when(pl.col("grenade_type_raw").str.contains("smoke"))
                .then(pl.lit("smoke"))
                .when(pl.col("grenade_type_raw").str.contains(r"(^|[^a-z])he([^a-z]|$)|hegrenade"))
                .then(pl.lit("hegrenade"))
                .when(pl.col("grenade_type_raw").str.contains("molotov"))
                .then(pl.lit("molotov"))
                .when(pl.col("grenade_type_raw").str.contains("incendiary"))
                .then(pl.lit("incendiary"))
                .when(pl.col("grenade_type_raw").str.contains("decoy"))
                .then(pl.lit("decoy"))
                .otherwise(pl.lit(None))
                .alias("grenade_type")
            ]
        )
        .filter(pl.col("grenade_type").is_not_null())
    )


def attach_grenade_side(grenades: pl.DataFrame, ticks_df: pl.DataFrame) -> pl.DataFrame:
    tick_lookup = (
        ticks_df.select(
            [
                "round_num",
                pl.col("tick").alias("start_tick"),
                pl.col("player_name").alias("thrower"),
                pl.col("team").alias("side"),
            ]
        )
        .unique()
    )

    return (
        grenades.join(
            tick_lookup,
            on=["round_num", "start_tick", "thrower"],
            how="left",
        )
        .filter(pl.col("side").is_in(["t", "ct"]))
    )


def get_kills(demo: Demo) -> pl.DataFrame:
    kills = demo.kills
    cols = require_columns(
        kills,
        {
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "killer": ["killer_name", "attacker_name", "killer", "attacker"],
            "victim": ["victim_name", "user_name", "victim", "user"],
            "killer_team": [
                "killer_team_name",
                "attacker_team_name",
                "killer_team",
                "attacker_team",
                "killer_side",
                "attacker_side",
            ],
            "victim_team": [
                "victim_team_name",
                "user_team_name",
                "victim_team",
                "user_team",
                "victim_side",
                "user_side",
            ],
            "weapon": ["weapon", "weapon_name"],
        },
    )

    return (
        kills.select(
            [
                pl.col(cols["round_num"]).cast(pl.Int64).alias("round_num"),
                pl.col(cols["tick"]).cast(pl.Int64).alias("tick"),
                pl.col(cols["killer"]).cast(pl.Utf8).alias("player_name"),
                pl.col(cols["victim"]).cast(pl.Utf8).alias("victim"),
                pl.col(cols["killer_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("side"),
                pl.col(cols["victim_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("victim_side"),
                pl.col(cols["weapon"]).cast(pl.Utf8).map_elements(normalize_weapon, return_dtype=pl.Utf8).alias("weapon"),
            ]
        )
        .filter(pl.col("player_name").is_not_null())
        .filter(pl.col("victim").is_not_null())
        .filter(pl.col("side").is_in(["t", "ct"]))
        .filter(pl.col("victim_side").is_in(["t", "ct"]))
        .filter(pl.col("side") != pl.col("victim_side"))
        .filter(pl.col("weapon").is_not_null())
    )


def build_trade_features(demo: Demo, rounds_df: pl.DataFrame, trade_window_seconds: float = 5.0) -> pl.DataFrame:
    trade_kills_raw = calculate_trades(demo, trade_length_in_seconds=trade_window_seconds)

    cols = require_columns(
        trade_kills_raw,
        {
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "killer": ["killer_name", "attacker_name", "killer", "attacker"],
            "victim": ["victim_name", "user_name", "victim", "user"],
            "killer_team": [
                "killer_team_name",
                "attacker_team_name",
                "killer_team",
                "attacker_team",
                "killer_side",
                "attacker_side",
            ],
            "victim_team": [
                "victim_team_name",
                "user_team_name",
                "victim_team",
                "user_team",
                "victim_side",
                "user_side",
            ],
            "was_traded": ["was_traded"],
        },
    )

    kills = (
        trade_kills_raw.select(
            [
                pl.col(cols["round_num"]).cast(pl.Int64).alias("round_num"),
                pl.col(cols["tick"]).cast(pl.Int64).alias("tick"),
                pl.col(cols["killer"]).cast(pl.Utf8).alias("player_name"),
                pl.col(cols["victim"]).cast(pl.Utf8).alias("victim"),
                pl.col(cols["killer_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("side"),
                pl.col(cols["victim_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("victim_side"),
                pl.col(cols["was_traded"]).cast(pl.Boolean).alias("was_traded"),
            ]
        )
        .filter(pl.col("player_name").is_not_null())
        .filter(pl.col("victim").is_not_null())
        .filter(pl.col("side").is_in(["t", "ct"]))
        .filter(pl.col("victim_side").is_in(["t", "ct"]))
        .filter(pl.col("side") != pl.col("victim_side"))
    )

    # 1) traded deaths:
    # if A kills B and that kill was_traded=True, then B's death was traded.
    traded_deaths = (
        kills.filter(pl.col("was_traded"))
        .group_by(
            [
                pl.col("victim").alias("player_name"),
                pl.col("victim_side").alias("side"),
            ]
        )
        .agg(pl.len().alias("traded_deaths"))
    )

    # 2) trade kills:
    # K2 is a trade kill if its victim is the killer from an earlier K1
    # in the same round, on the opposite team, within the trade window.
    first_kills = kills.select(
        [
            "round_num",
            pl.col("tick").alias("first_tick"),
            pl.col("player_name").alias("first_killer"),
            pl.col("victim").alias("first_victim"),
            pl.col("side").alias("first_killer_side"),
            pl.col("victim_side").alias("first_victim_side"),
        ]
    )

    second_kills = kills.select(
        [
            "round_num",
            pl.col("tick").alias("trade_tick"),
            pl.col("player_name").alias("trade_killer"),
            pl.col("victim").alias("trade_victim"),
            pl.col("side").alias("trade_killer_side"),
            pl.col("victim_side").alias("trade_victim_side"),
        ]
    )

    tickrate = getattr(demo, "tickrate", 64)
    trade_window_ticks = int(round(trade_window_seconds * tickrate))

    trade_kill_events = (
        second_kills.join(first_kills, on="round_num", how="inner")
        .filter(pl.col("trade_victim") == pl.col("first_killer"))
        .filter(pl.col("trade_killer_side") == pl.col("first_victim_side"))
        .filter(pl.col("trade_victim_side") == pl.col("first_killer_side"))
        .filter(pl.col("trade_tick") > pl.col("first_tick"))
        .filter((pl.col("trade_tick") - pl.col("first_tick")) <= trade_window_ticks)
        .select(
            [
                pl.col("trade_killer").alias("player_name"),
                pl.col("trade_killer_side").alias("side"),
                "round_num",
                "trade_tick",
            ]
        )
        .unique()
    )

    trade_kills = (
        trade_kill_events.group_by(["player_name", "side"])
        .agg(pl.len().alias("trade_kills"))
    )

    joined = rounds_df.join(trade_kills, on=["player_name", "side"], how="left")
    joined = joined.join(traded_deaths, on=["player_name", "side"], how="left")
    joined = joined.fill_null(0)

    return joined.with_columns(
        [
            (pl.col("trade_kills") / pl.col("rounds_played")).fill_nan(0).alias("trade_kills_per_round"),
            (pl.col("traded_deaths") / pl.col("rounds_played")).fill_nan(0).alias("traded_deaths_per_round"),
            (pl.col("traded_deaths") / pl.col("rounds_played")).fill_nan(0).alias("death_traded_rate"),
            (pl.col("trade_kills") + pl.col("traded_deaths")).alias("trade_participation"),
            ((pl.col("trade_kills") + pl.col("traded_deaths")) / pl.col("rounds_played"))
            .fill_nan(0)
            .alias("trade_participation_per_round"),
        ]
    ).select(
        [
            "player_name",
            "side",
            "trade_kills",
            "traded_deaths",
            "trade_kills_per_round",
            "traded_deaths_per_round",
            "death_traded_rate",
            "trade_participation",
            "trade_participation_per_round",
        ]
    )


def build_opening_features(kills_df: pl.DataFrame, rounds_df: pl.DataFrame) -> pl.DataFrame:
    cols_needed = {"player_name", "victim", "side", "victim_side", "round_num", "tick"}
    missing = [c for c in cols_needed if c not in kills_df.columns]
    if missing:
        raise KeyError(f"build_opening_features missing kill columns: {missing}")

    opening_kills_df = (
        kills_df.group_by("round_num")
        .agg(pl.col("tick").min().alias("opening_tick"))
        .join(kills_df, on="round_num", how="inner")
        .filter(pl.col("tick") == pl.col("opening_tick"))
        .sort(["round_num", "tick"])
        .unique(subset=["round_num"], keep="first")
    )

    opening_kills = (
        opening_kills_df.group_by(["player_name", "side"])
        .agg(pl.len().alias("opening_kills"))
    )

    opening_deaths = (
        opening_kills_df.group_by(
            [
                pl.col("victim").alias("player_name"),
                pl.col("victim_side").alias("side"),
            ]
        )
        .agg(pl.len().alias("opening_deaths"))
    )

    joined = rounds_df.join(opening_kills, on=["player_name", "side"], how="left")
    joined = joined.join(opening_deaths, on=["player_name", "side"], how="left")
    joined = joined.fill_null(0).with_columns(
        [
            (pl.col("opening_kills") + pl.col("opening_deaths")).alias("opening_duels"),
        ]
    )

    joined = joined.with_columns(
        [
            (pl.col("opening_kills") / pl.col("rounds_played")).fill_nan(0).alias("opening_kills_per_round"),
            (pl.col("opening_deaths") / pl.col("rounds_played")).fill_nan(0).alias("opening_deaths_per_round"),
            (pl.col("opening_duels") / pl.col("rounds_played")).fill_nan(0).alias("opening_duels_per_round"),
            pl.when(pl.col("opening_duels") > 0)
            .then(pl.col("opening_kills") / pl.col("opening_duels"))
            .otherwise(0.0)
            .alias("opening_duel_success_rate"),
        ]
    )

    return joined.select(
        [
            "player_name",
            "side",
            "opening_kills",
            "opening_deaths",
            "opening_duels",
            "opening_kills_per_round",
            "opening_deaths_per_round",
            "opening_duels_per_round",
            "opening_duel_success_rate",
        ]
    )

def build_weapon_category_kills(kills_df: pl.DataFrame, rounds_df: pl.DataFrame) -> pl.DataFrame:
    categorized = (
        kills_df.with_columns(
            pl.col("weapon").map_elements(categorize_weapon, return_dtype=pl.Utf8).alias("weapon_category")
        )
        .filter(pl.col("weapon_category").is_not_null())
    )

    base = rounds_df.select(["player_name", "side"])

    if categorized.height == 0:
        return base.with_columns(
            [
                pl.lit(0).alias("rifle_kills"),
                pl.lit(0).alias("awp_kills"),
                pl.lit(0).alias("smg_kills"),
                pl.lit(0).alias("pistol_kills"),
                pl.lit(0).alias("knife_kills"),
                pl.lit(0).alias("shotgun_kills"),
                pl.lit(0).alias("scout_kills"),
                pl.lit(0.0).alias("rifle_kills_per_round"),
                pl.lit(0.0).alias("awp_kills_per_round"),
                pl.lit(0.0).alias("smg_kills_per_round"),
                pl.lit(0.0).alias("pistol_kills_per_round"),
                pl.lit(0.0).alias("knife_kills_per_round"),
                pl.lit(0.0).alias("shotgun_kills_per_round"),
                pl.lit(0.0).alias("scout_kills_per_round"),
            ]
        )

    long_counts = (
        categorized.group_by(["player_name", "side", "weapon_category"])
        .agg(pl.len().alias("kills"))
    )

    wide_counts = (
        long_counts.pivot(
            values="kills",
            index=["player_name", "side"],
            on="weapon_category",
            aggregate_function="first",
        )
        .fill_null(0)
    )

    rename_map = {}
    for c in wide_counts.columns:
        if c not in {"player_name", "side"}:
            rename_map[c] = f"{c}_kills"
    wide_counts = wide_counts.rename(rename_map)

    joined = rounds_df.join(wide_counts, on=["player_name", "side"], how="left").fill_null(0)

    required_kill_cols = [
        "rifle_kills",
        "awp_kills",
        "smg_kills",
        "pistol_kills",
        "knife_kills",
        "shotgun_kills",
        "scout_kills",
    ]
    for col_name in required_kill_cols:
        if col_name not in joined.columns:
            joined = joined.with_columns(pl.lit(0).alias(col_name))

    joined = joined.with_columns(
        [
            (pl.col("rifle_kills") / pl.col("rounds_played")).fill_nan(0).alias("rifle_kills_per_round"),
            (pl.col("awp_kills") / pl.col("rounds_played")).fill_nan(0).alias("awp_kills_per_round"),
            (pl.col("smg_kills") / pl.col("rounds_played")).fill_nan(0).alias("smg_kills_per_round"),
            (pl.col("pistol_kills") / pl.col("rounds_played")).fill_nan(0).alias("pistol_kills_per_round"),
            (pl.col("knife_kills") / pl.col("rounds_played")).fill_nan(0).alias("knife_kills_per_round"),
            (pl.col("shotgun_kills") / pl.col("rounds_played")).fill_nan(0).alias("shotgun_kills_per_round"),
            (pl.col("scout_kills") / pl.col("rounds_played")).fill_nan(0).alias("scout_kills_per_round"),
        ]
    )

    return joined.select(
        [
            "player_name",
            "side",
            "rifle_kills",
            "awp_kills",
            "smg_kills",
            "pistol_kills",
            "knife_kills",
            "shotgun_kills",
            "scout_kills",
            "rifle_kills_per_round",
            "awp_kills_per_round",
            "smg_kills_per_round",
            "pistol_kills_per_round",
            "knife_kills_per_round",
            "shotgun_kills_per_round",
            "scout_kills_per_round",
        ]
    )


def build_utility_kills(kills_df: pl.DataFrame, rounds_df: pl.DataFrame) -> pl.DataFrame:
    util_kills = (
        kills_df.filter(pl.col("weapon").is_in(UTILITY_KILL_WEAPONS))
        .group_by(["player_name", "side"])
        .agg(
            [
                pl.len().alias("utility_kills"),
                (pl.col("weapon") == "hegrenade").cast(pl.Int64).sum().alias("he_kills"),
                pl.col("weapon").is_in(["molotov", "incendiary", "inferno"]).cast(pl.Int64).sum().alias("fire_kills"),
            ]
        )
    )

    joined = rounds_df.join(util_kills, on=["player_name", "side"], how="left").fill_null(0)

    return joined.with_columns(
        [
            (pl.col("utility_kills") / pl.col("rounds_played")).fill_nan(0).alias("utility_kills_per_round"),
            (pl.col("he_kills") / pl.col("rounds_played")).fill_nan(0).alias("he_kills_per_round"),
            (pl.col("fire_kills") / pl.col("rounds_played")).fill_nan(0).alias("fire_kills_per_round"),
        ]
    ).select(
        [
            "player_name",
            "side",
            "utility_kills",
            "he_kills",
            "fire_kills",
            "utility_kills_per_round",
            "he_kills_per_round",
            "fire_kills_per_round",
        ]
    )


def compute_player_map_rows(demo_path: Path) -> pl.DataFrame:
    demo = Demo(str(demo_path))
    demo.parse()

    map_name = map_name_from_demo(demo, demo_path)

    rounds_df = get_player_round_totals_by_side(demo)
    ticks_df = get_ticks_for_side_lookup(demo)

    grenades = get_grenade_events(demo)
    grenades = normalize_grenade_types(grenades)
    grenades = attach_grenade_side(grenades, ticks_df)

    kills_df = get_kills(demo)
    weapon_category_df = build_weapon_category_kills(kills_df, rounds_df)
    utility_kills_df = build_utility_kills(kills_df, rounds_df)
    trade_df = build_trade_features(demo, rounds_df, trade_window_seconds=5.0)
    opening_df = build_opening_features(kills_df, rounds_df)

    adr_df = (
        adr(demo, team_dmg=True, self_dmg=True)
        .rename({"name": "player_name"})
        .with_columns(pl.col("side").cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("side"))
        .filter(pl.col("side").is_in(["t", "ct"]))
        .select(["player_name", "side", "adr"])
    )

    kast_df = (
        kast(demo)
        .rename({"name": "player_name"})
        .with_columns(pl.col("side").cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("side"))
        .filter(pl.col("side").is_in(["t", "ct"]))
        .select(["player_name", "side", "kast"])
    )

    rating_df = (
        rating(demo)
        .rename({"name": "player_name"})
        .with_columns(pl.col("side").cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("side"))
        .filter(pl.col("side").is_in(["t", "ct"]))
        .select(["player_name", "side", "rating"])
    )

    utility_df = (
        grenades.group_by([pl.col("thrower").alias("player_name"), "side"])
        .agg(
            [
                pl.len().alias("grenades_thrown"),
                (pl.col("grenade_type") == "flashbang").cast(pl.Int64).sum().alias("flashbangs_thrown"),
                (pl.col("grenade_type") == "smoke").cast(pl.Int64).sum().alias("smokes_thrown"),
                ((pl.col("grenade_type") == "molotov") | (pl.col("grenade_type") == "incendiary"))
                .cast(pl.Int64)
                .sum()
                .alias("fire_nades_thrown"),
                (pl.col("grenade_type") == "hegrenade").cast(pl.Int64).sum().alias("hegrenades_thrown"),
                (pl.col("grenade_type") == "decoy").cast(pl.Int64).sum().alias("decoys_thrown"),
            ]
        )
    )

    df = rounds_df.join(adr_df, on=["player_name", "side"], how="left")
    df = df.join(kast_df, on=["player_name", "side"], how="left")
    df = df.join(rating_df, on=["player_name", "side"], how="left")
    df = df.join(utility_df, on=["player_name", "side"], how="left")
    df = df.join(weapon_category_df, on=["player_name", "side"], how="left")
    df = df.join(utility_kills_df, on=["player_name", "side"], how="left")
    df = rounds_df.join(adr_df, on=["player_name", "side"], how="left")
    df = df.join(kast_df, on=["player_name", "side"], how="left")
    df = df.join(rating_df, on=["player_name", "side"], how="left")
    df = df.join(utility_df, on=["player_name", "side"], how="left")
    df = df.join(weapon_category_df, on=["player_name", "side"], how="left")
    df = df.join(utility_kills_df, on=["player_name", "side"], how="left")
    df = df.join(trade_df, on=["player_name", "side"], how="left")
    df = df.join(opening_df, on=["player_name", "side"], how="left")

    df = df.fill_null(0).with_columns(
        [
            pl.col("adr").fill_nan(0).alias("adr"),
            pl.col("kast").fill_nan(0).alias("kast"),
            pl.col("rating").fill_nan(0).alias("rating"),
            (pl.col("grenades_thrown") / pl.col("rounds_played")).fill_nan(0).alias("grenades_per_round"),
            (pl.col("flashbangs_thrown") / pl.col("rounds_played")).fill_nan(0).alias("flashbangs_per_round"),
            (pl.col("smokes_thrown") / pl.col("rounds_played")).fill_nan(0).alias("smokes_per_round"),
            (pl.col("fire_nades_thrown") / pl.col("rounds_played")).fill_nan(0).alias("fire_nades_per_round"),
            (pl.col("hegrenades_thrown") / pl.col("rounds_played")).fill_nan(0).alias("hegrenades_per_round"),
            (pl.col("decoys_thrown") / pl.col("rounds_played")).fill_nan(0).alias("decoys_per_round"),
            pl.lit(map_name).alias("map_name"),
        ]
    )

    base_cols = [
        "player_name",
        "map_name",
        "side",
        "rounds_played",
        "adr",
        "kast",
        "rating",
        "grenades_thrown",
        "flashbangs_thrown",
        "smokes_thrown",
        "fire_nades_thrown",
        "hegrenades_thrown",
        "decoys_thrown",
        "grenades_per_round",
        "flashbangs_per_round",
        "smokes_per_round",
        "fire_nades_per_round",
        "hegrenades_per_round",
        "decoys_per_round",
        "rifle_kills",
        "awp_kills",
        "smg_kills",
        "pistol_kills",
        "knife_kills",
        "shotgun_kills",
        "scout_kills",
        "rifle_kills_per_round",
        "awp_kills_per_round",
        "smg_kills_per_round",
        "pistol_kills_per_round",
        "knife_kills_per_round",
        "shotgun_kills_per_round",
        "scout_kills_per_round",
        "utility_kills",
        "he_kills",
        "fire_kills",
        "utility_kills_per_round",
        "he_kills_per_round",
        "fire_kills_per_round",
        "trade_kills",
        "traded_deaths",
        "trade_kills_per_round",
        "traded_deaths_per_round",
        "death_traded_rate",
        "trade_participation",
        "trade_participation_per_round",
        "opening_kills",
        "opening_deaths",
        "opening_duels",
        "opening_kills_per_round",
        "opening_deaths_per_round",
        "opening_duels_per_round",
        "opening_duel_success_rate",
    ]

    existing_cols = [c for c in base_cols if c in df.columns]
    df = df.select(existing_cols)

    pivot = df.pivot(
        values=[c for c in df.columns if c not in {"player_name", "map_name", "side"}],
        index=["player_name", "map_name"],
        on="side",
        aggregate_function="first",
    )

    return pivot.sort(["player_name", "map_name"])


def iter_demo_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".dem":
        yield root
        return
    for path in root.rglob("*.dem"):
        yield path


def resolve_input_path(
    explicit_input: Path | None,
    use_test: bool,
    use_main_demos: bool,
) -> Path:
    script_dir = Path(__file__).resolve().parent

    if explicit_input is not None:
        return explicit_input

    if use_test and use_main_demos:
        raise ValueError("Use only one of --test or --use-main-demos.")

    if use_test:
        candidates = [
            script_dir / "demos_test",
            Path.cwd() / "demos_test",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not find a 'demos_test' folder next to the script or in the current working directory.")

    if use_main_demos:
        candidates = [
            script_dir / "demos",
            Path.cwd() / "demos",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not find a 'demos' folder next to the script or in the current working directory.")

    raise ValueError("Provide input_path, or use --test, or use --use-main-demos.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ADR, KAST, Rating, utility usage, utility kills, and condensed weapon-category kills from CS2 demos."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to a .dem file or folder of .dem files. Optional if using --test or --use-main-demos.",
    )
    parser.add_argument("output_path", type=Path, help="Output .csv or .parquet")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use the local 'demos_test' folder for quicker testing.",
    )
    parser.add_argument(
        "--use-main-demos",
        action="store_true",
        help="Use the local 'demos' folder automatically.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(
        explicit_input=args.input_path,
        use_test=args.test,
        use_main_demos=args.use_main_demos,
    )

    print(f"[info] input path: {input_path}")

    demo_files = list(iter_demo_files(input_path))
    total = len(demo_files)

    if total == 0:
        raise RuntimeError(f"No demo files found in: {input_path}")

    print(f"[info] found {total} demos to process\n")

    rows: list[pl.DataFrame] = []
    failures: list[tuple[str, str]] = []

    for i, demo_path in enumerate(demo_files, start=1):
        try:
            print(f"[{i}/{total}] parsing: {demo_path}")
            rows.append(compute_player_map_rows(demo_path))
        except Exception as exc:
            failures.append((str(demo_path), repr(exc)))
            print(f"[skip {i}/{total}] {demo_path}: {exc}")

    if not rows:
        raise RuntimeError("No demos parsed successfully.")

    out = pl.concat(rows, how="diagonal_relaxed")
    out = out.fill_null(0).fill_nan(0)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_path.suffix.lower() == ".csv":
        out.write_csv(args.output_path)
    else:
        out.write_parquet(args.output_path)

    print(f"\nWrote {out.height} rows to {args.output_path}")
    print(f"[summary] success={len(rows)}, failed={len(failures)}, total={total}")

    if failures:
        print("\nFailures:")
        for demo_path, err in failures:
            print(f"- {demo_path}: {err}")


if __name__ == "__main__":
    main()