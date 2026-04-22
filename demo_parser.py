from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from awpy import Demo


def pick_col(df: pl.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the candidate columns exist: {list(candidates)}. Available: {df.columns}")
    return None


def require_columns(df: pl.DataFrame, mapping: dict[str, Sequence[str]]) -> dict[str, str]:
    return {alias: pick_col(df, cols, required=True) for alias, cols in mapping.items()}


TEAM_ALIASES = {
    "t": "T",
    "terrorist": "T",
    "terrorists": "T",
    "ct": "CT",
    "counterterrorist": "CT",
    "counterterrorists": "CT",
    "counter-terrorist": "CT",
}

RIFLE_REGEX = r"ak47|m4a1|m4a1_silencer|aug|sg556|galilar|famas"
PISTOL_REGEX = r"glock|usp|usp_silencer|p2000|p250|deagle|elite|fiveseven|tec9|cz75|revolver"
UTILITY_DAMAGE_REGEX = r"hegrenade|molotov|incendiary|inferno"


def normalize_team(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return TEAM_ALIASES.get(s, str(value))


def get_round_table(demo: Demo) -> pl.DataFrame:
    rounds = demo.rounds
    cols = require_columns(
        rounds,
        {
            "round_num": ["round_num", "round", "round_number"],
            "start_tick": ["start_tick", "start", "round_start_tick"],
            "end_tick": ["end_tick", "end", "round_end_tick"],
        },
    )
    return (
        rounds.select(
            [
                pl.col(cols["round_num"]).alias("round_num"),
                pl.col(cols["start_tick"]).alias("start_tick"),
                pl.col(cols["end_tick"]).alias("end_tick"),
            ]
        )
        .unique(subset=["round_num"])
        .sort("round_num")
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
            "headshot": ["is_headshot", "headshot"],
            "assister": ["assister_name", "assister"],
            "assistedflash": ["is_assisted_by_flash", "assistedflash", "flash_assist"],
        },
    )
    return kills.select(
        [
            pl.col(cols["round_num"]).alias("round_num"),
            pl.col(cols["tick"]).alias("tick"),
            pl.col(cols["killer"]).cast(pl.Utf8).alias("killer"),
            pl.col(cols["victim"]).cast(pl.Utf8).alias("victim"),
            pl.col(cols["killer_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("killer_team"),
            pl.col(cols["victim_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("victim_team"),
            pl.col(cols["weapon"]).cast(pl.Utf8).alias("weapon"),
            pl.col(cols["headshot"]).cast(pl.Int8).fill_null(0).alias("headshot"),
            pl.col(cols["assister"]).cast(pl.Utf8).alias("assister"),
            pl.col(cols["assistedflash"]).cast(pl.Int8).fill_null(0).alias("flash_assist_flag"),
        ]
    )


def get_damages(demo: Demo) -> pl.DataFrame:
    damages = demo.damages
    cols = require_columns(
        damages,
        {
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "attacker": ["attacker_name", "attacker"],
            "victim": ["victim_name", "user_name", "victim", "user"],
            "attacker_team": ["attacker_team_name", "attacker_team", "attacker_side"],
            "victim_team": ["victim_team_name", "user_team_name", "victim_team", "user_team", "victim_side", "user_side"],
            "hp_damage": ["hp_damage", "dmg_health", "health_damage"],
            "armor_damage": ["armor_damage", "dmg_armor"],
            "weapon": ["weapon", "weapon_name"],
        },
    )
    return damages.select(
        [
            pl.col(cols["round_num"]).alias("round_num"),
            pl.col(cols["tick"]).alias("tick"),
            pl.col(cols["attacker"]).cast(pl.Utf8).alias("attacker"),
            pl.col(cols["victim"]).cast(pl.Utf8).alias("victim"),
            pl.col(cols["attacker_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("attacker_team"),
            pl.col(cols["victim_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("victim_team"),
            pl.col(cols["hp_damage"]).cast(pl.Float64).fill_null(0).alias("hp_damage"),
            pl.col(cols["armor_damage"]).cast(pl.Float64).fill_null(0).alias("armor_damage"),
            pl.col(cols["weapon"]).cast(pl.Utf8).alias("weapon"),
        ]
    )


def get_shots(demo: Demo) -> pl.DataFrame:
    shots = demo.shots
    cols = require_columns(
        shots,
        {
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "shooter": ["shooter_name", "player_name", "attacker_name", "shooter", "player", "attacker"],
            "shooter_team": [
                "shooter_team_name",
                "player_team_name",
                "attacker_team_name",
                "shooter_team",
                "player_team",
                "attacker_team",
                "shooter_side",
                "player_side",
                "attacker_side",
            ],
            "weapon": ["weapon", "weapon_name"],
        },
    )
    return shots.select(
        [
            pl.col(cols["round_num"]).alias("round_num"),
            pl.col(cols["tick"]).alias("tick"),
            pl.col(cols["shooter"]).cast(pl.Utf8).alias("shooter"),
            pl.col(cols["shooter_team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("shooter_team"),
            pl.col(cols["weapon"]).cast(pl.Utf8).alias("weapon"),
        ]
    )


def get_ticks(demo: Demo) -> pl.DataFrame:
    ticks = demo.ticks
    cols = require_columns(
        ticks,
        {
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "name": ["name", "player_name"],
            "team": ["team_name", "side", "player_side", "team"],
            "x": ["X", "x"],
            "y": ["Y", "y"],
            "health": ["health", "hp"],
        },
    )
    return ticks.select(
        [
            pl.col(cols["round_num"]).alias("round_num"),
            pl.col(cols["tick"]).alias("tick"),
            pl.col(cols["name"]).cast(pl.Utf8).alias("player_name"),
            pl.col(cols["team"]).cast(pl.Utf8).map_elements(normalize_team, return_dtype=pl.Utf8).alias("team"),
            pl.col(cols["x"]).cast(pl.Float64).alias("x"),
            pl.col(cols["y"]).cast(pl.Float64).alias("y"),
            (pl.col(cols["health"]).cast(pl.Float64).fill_null(0) > 0).cast(pl.Int8).alias("is_alive"),
        ]
    )


def get_grenades(demo: Demo, ticks_df: pl.DataFrame) -> pl.DataFrame:
    gren = demo.grenades
    cols = require_columns(
        gren,
        {
            "round_num": ["round_num", "round", "round_number"],
            "tick": ["tick", "game_tick"],
            "thrower": ["thrower_name", "player_name", "thrower", "player"],
            "grenade_type": ["grenade_type", "weapon", "grenade"],
        },
    )

    gren_df = gren.select(
        [
            pl.col(cols["round_num"]).alias("round_num"),
            pl.col(cols["tick"]).alias("tick"),
            pl.col(cols["thrower"]).cast(pl.Utf8).alias("thrower"),
            pl.col(cols["grenade_type"]).cast(pl.Utf8).alias("grenade_type"),
        ]
    )

    tick_lookup = (
        ticks_df.select(
            [
                "round_num",
                "tick",
                pl.col("player_name").alias("thrower"),
                pl.col("team").alias("thrower_team"),
            ]
        )
        .unique()
    )

    return gren_df.join(
        tick_lookup,
        on=["round_num", "tick", "thrower"],
        how="left",
    )


def map_name_from_demo(demo: Demo, demo_path: Path) -> str:
    header = getattr(demo, "header", None)
    if isinstance(header, dict):
        for key in ("map_name", "map", "mapName"):
            if key in header and header[key]:
                return str(header[key])
    return demo_path.stem


def compute_player_map_rows(demo_path: Path) -> pl.DataFrame:
    demo = Demo(str(demo_path))
    demo.parse()

    map_name = map_name_from_demo(demo, demo_path)
    round_table = get_round_table(demo)
    kills = get_kills(demo)
    damages = get_damages(demo)
    shots = get_shots(demo)
    ticks = get_ticks(demo)
    grenades = get_grenades(demo, ticks)

    round_starts = round_table.select(
        [
            "round_num",
            pl.col("start_tick").alias("round_start_tick"),
            pl.col("end_tick").alias("round_end_tick"),
        ]
    )

    round_start_ticks = (
        ticks.join(round_starts.select(["round_num", "round_start_tick"]), on="round_num", how="left")
        .filter(pl.col("tick") == pl.col("round_start_tick"))
        .filter(pl.col("team").is_in(["T", "CT"]))
    )

    players = (
        round_start_ticks.select([pl.col("player_name"), pl.col("team").alias("side")])
        .filter(pl.col("player_name").is_not_null())
        .unique()
        .sort(["player_name", "side"])
    )

    rounds_per_player = (
        round_start_ticks
        .filter(pl.col("is_alive") == 1)
        .select(["round_num", "player_name", pl.col("team").alias("side")])
        .unique()
        .group_by(["player_name", "side"])
        .agg(pl.len().alias("rounds_played"))
    )

    kills_clean = (
        kills.filter(
            pl.col("killer").is_not_null()
            & pl.col("victim").is_not_null()
            & pl.col("killer_team").is_in(["T", "CT"])
            & pl.col("victim_team").is_in(["T", "CT"])
            & (pl.col("killer") != pl.col("victim"))
            & (pl.col("killer_team") != pl.col("victim_team"))
        )
        .with_columns(pl.col("weapon").cast(pl.Utf8).str.to_lowercase().alias("weapon_lc"))
    )

    kills_by_player = kills_clean.group_by([pl.col("killer").alias("player_name"), pl.col("killer_team").alias("side")]).agg(
        [
            pl.len().alias("kills"),
            pl.col("headshot").sum().alias("headshot_kills"),
            pl.col("weapon_lc").str.contains("awp").cast(pl.Int64).sum().alias("awp_kills"),
            pl.col("weapon_lc").str.contains(RIFLE_REGEX).cast(pl.Int64).sum().alias("rifle_kills"),
            pl.col("weapon_lc").str.contains(PISTOL_REGEX).cast(pl.Int64).sum().alias("pistol_kills"),
        ]
    )

    deaths_by_player = kills_clean.group_by([pl.col("victim").alias("player_name"), pl.col("victim_team").alias("side")]).agg(
        [pl.len().alias("deaths")]
    )

    assists_by_player = (
        kills_clean.filter(pl.col("assister").is_not_null())
        .group_by([pl.col("assister").alias("player_name"), pl.col("killer_team").alias("side")])
        .agg(pl.len().alias("assists_raw"))
    )

    flash_assists_by_player = (
        kills_clean.filter(pl.col("assister").is_not_null() & (pl.col("flash_assist_flag") == 1))
        .group_by([pl.col("assister").alias("player_name"), pl.col("killer_team").alias("side")])
        .agg(pl.len().alias("flash_assists"))
    )

    first_kills = kills_clean.sort(["round_num", "tick"]).group_by("round_num").first()

    opening_killers = first_kills.group_by([pl.col("killer").alias("player_name"), pl.col("killer_team").alias("side")]).agg(
        pl.len().alias("opening_kills")
    )

    opening_victims = first_kills.group_by([pl.col("victim").alias("player_name"), pl.col("victim_team").alias("side")]).agg(
        pl.len().alias("opening_deaths")
    )

    dmg_by_player = damages.group_by([pl.col("attacker").alias("player_name"), pl.col("attacker_team").alias("side")]).agg(
        [
            pl.col("hp_damage").sum().alias("hp_damage_total"),
            pl.col("armor_damage").sum().alias("armor_damage_total"),
        ]
    )

    utility_damage_by_player = (
        damages.filter(
            pl.col("attacker").is_not_null()
            & pl.col("attacker_team").is_in(["T", "CT"])
            & pl.col("weapon").cast(pl.Utf8).str.to_lowercase().str.contains(UTILITY_DAMAGE_REGEX)
        )
        .group_by([pl.col("attacker").alias("player_name"), pl.col("attacker_team").alias("side")])
        .agg((pl.col("hp_damage").sum() + pl.col("armor_damage").sum()).alias("utility_damage_total"))
    )

    first_damage = (
        damages.filter(pl.col("attacker").is_not_null())
        .sort(["round_num", "tick"])
        .group_by(["round_num", pl.col("attacker").alias("player_name"), pl.col("attacker_team").alias("side")])
        .first()
        .select(["round_num", "player_name", "side", pl.col("tick").alias("first_damage_tick")])
    )

    first_shot = (
        shots.filter(pl.col("shooter").is_not_null())
        .sort(["round_num", "tick"])
        .group_by(["round_num", pl.col("shooter").alias("player_name"), pl.col("shooter_team").alias("side")])
        .first()
        .select(["round_num", "player_name", "side", pl.col("tick").alias("first_shot_tick")])
    )

    engagement_times = (
        first_damage.join(first_shot, on=["round_num", "player_name", "side"], how="full")
        .join(round_starts, on="round_num", how="left")
        .with_columns(pl.min_horizontal(["first_damage_tick", "first_shot_tick"]).alias("first_engagement_tick"))
        .with_columns((pl.col("first_engagement_tick") - pl.col("round_start_tick")).cast(pl.Float64).alias("engagement_ticks"))
        .group_by(["player_name", "side"])
        .agg(pl.col("engagement_ticks").mean().alias("avg_time_to_first_engagement_ticks"))
    )

    death_ticks = kills_clean.group_by(["round_num", pl.col("victim").alias("player_name"), pl.col("victim_team").alias("side")]).agg(
        pl.col("tick").min().alias("death_tick")
    )

    player_round_presence = (
        round_start_ticks
        .filter(pl.col("is_alive") == 1)
        .select(["round_num", "player_name", pl.col("team").alias("side")])
        .unique()
    )

    survival = (
        player_round_presence.join(round_starts, on="round_num", how="left")
        .join(death_ticks, on=["round_num", "player_name", "side"], how="left")
        .with_columns(
            [
                pl.when(pl.col("death_tick").is_null()).then(1).otherwise(0).alias("survived_round"),
                pl.when(pl.col("death_tick").is_null()).then(pl.col("round_end_tick")).otherwise(pl.col("death_tick")).alias("exit_tick"),
            ]
        )
        .with_columns((pl.col("exit_tick") - pl.col("round_start_tick")).cast(pl.Float64).alias("survival_ticks"))
        .group_by(["player_name", "side"])
        .agg(
            [
                pl.col("survival_ticks").mean().alias("avg_survival_ticks"),
                pl.col("survival_ticks").std().fill_null(0).alias("std_survival_ticks"),
                pl.col("survived_round").sum().alias("survived_rounds"),
            ]
        )
    )

    utility = (
        grenades.filter(pl.col("thrower_team").is_in(["T", "CT"]))
        .group_by([pl.col("thrower").alias("player_name"), pl.col("thrower_team").alias("side")])
        .agg(
            [
                pl.len().alias("grenades_thrown"),
                (pl.col("grenade_type").str.to_lowercase().str.contains("flash").cast(pl.Int64)).sum().alias("flashbangs_thrown"),
                (pl.col("grenade_type").str.to_lowercase().str.contains("smoke").cast(pl.Int64)).sum().alias("smokes_thrown"),
                (pl.col("grenade_type").str.to_lowercase().str.contains("molotov|incendiary").cast(pl.Int64)).sum().alias("fire_nades_thrown"),
            ]
        )
    )

    teammate_deaths = kills_clean.select(
        [
            "round_num",
            pl.col("tick").alias("death_tick"),
            pl.col("victim_team").alias("side"),
            pl.col("victim").alias("dead_teammate"),
            pl.col("killer").alias("enemy_killer"),
        ]
    )

    candidate_trades = kills_clean.select(
        [
            "round_num",
            pl.col("tick").alias("trade_tick"),
            pl.col("killer").alias("player_name"),
            pl.col("killer_team").alias("side"),
            pl.col("victim").alias("trade_victim"),
        ]
    )

    trade_window = 5 * 64

    trade_pairs = (
        candidate_trades.join(teammate_deaths, on=["round_num", "side"], how="inner")
        .filter(
            (pl.col("trade_victim") == pl.col("enemy_killer"))
            & (pl.col("trade_tick") > pl.col("death_tick"))
            & ((pl.col("trade_tick") - pl.col("death_tick")) <= trade_window)
            & (pl.col("player_name") != pl.col("dead_teammate"))
        )
    )

    trade_events = (
        trade_pairs
        .sort(["round_num", "death_tick", "trade_tick"])
        .group_by(["round_num", "death_tick", "side", "dead_teammate", "enemy_killer"])
        .first()
    )

    trade_kills = trade_events.group_by(["player_name", "side"]).agg(pl.len().alias("trade_kills"))

    traded_deaths = (
        trade_events.group_by([pl.col("dead_teammate").alias("player_name"), "side"]).agg(pl.len().alias("traded_deaths"))
    )

    trade_attempts = (
        player_round_presence.join(
            teammate_deaths.select(["round_num", "side", "dead_teammate"]).unique(),
            on=["round_num", "side"],
            how="inner",
        )
        .filter(pl.col("player_name") != pl.col("dead_teammate"))
        .group_by(["player_name", "side"])
        .agg(pl.len().alias("trade_attempts"))
    )

    alive_ticks = ticks.filter((pl.col("is_alive") == 1) & pl.col("team").is_in(["T", "CT"]))

    centroids = alive_ticks.group_by(["round_num", "tick", "team"]).agg(
        [
            pl.col("x").mean().alias("team_cx"),
            pl.col("y").mean().alias("team_cy"),
        ]
    )

    pos = (
        alive_ticks.join(centroids, on=["round_num", "tick", "team"], how="left")
        .with_columns((((pl.col("x") - pl.col("team_cx")) ** 2 + (pl.col("y") - pl.col("team_cy")) ** 2).sqrt()).alias("dist_from_team_centroid"))
    )

    pos_by_player = pos.group_by([pl.col("player_name"), pl.col("team").alias("side")]).agg(
        pl.col("dist_from_team_centroid").mean().alias("avg_distance_from_team")
    )

    global_dist_threshold = pos.select(pl.col("dist_from_team_centroid").quantile(0.75).alias("q75")).item()

    isolation = (
        pos.with_columns((pl.col("dist_from_team_centroid") > float(global_dist_threshold)).cast(pl.Int8).alias("isolated_flag"))
        .group_by([pl.col("player_name"), pl.col("team").alias("side")])
        .agg(pl.col("isolated_flag").mean().alias("isolation_rate"))
    )

    df = players.join(rounds_per_player, on=["player_name", "side"], how="left")

    for piece in [
        kills_by_player,
        deaths_by_player,
        assists_by_player,
        flash_assists_by_player,
        opening_killers,
        opening_victims,
        dmg_by_player,
        utility_damage_by_player,
        engagement_times,
        survival,
        utility,
        trade_kills,
        traded_deaths,
        trade_attempts,
        pos_by_player,
        isolation,
    ]:
        join_keys = [k for k in ["player_name", "side"] if k in piece.columns and k in df.columns]
        if join_keys:
            df = df.join(piece, on=join_keys, how="left")
        else:
            df = df.join(piece, on=["player_name"], how="left")

    df = df.fill_null(0)

    df = df.with_columns(
        [
            pl.lit(map_name).alias("map_name"),

            pl.when((pl.col("opening_kills") + pl.col("opening_deaths")) > 0)
            .then(pl.col("opening_kills") / (pl.col("opening_kills") + pl.col("opening_deaths")))
            .otherwise(0.0)
            .alias("opening_success_rate"),

            pl.when(pl.col("rounds_played") > 0)
            .then(pl.col("survived_rounds") / pl.col("rounds_played"))
            .otherwise(0.0)
            .alias("survival_rate"),

            pl.when(pl.col("trade_attempts") > 0)
            .then(pl.col("trade_kills") / pl.col("trade_attempts"))
            .otherwise(0.0)
            .alias("trade_success_rate"),

            pl.when(pl.col("kills") > 0)
            .then(pl.col("headshot_kills") / pl.col("kills"))
            .otherwise(0.0)
            .alias("headshot_rate"),

            pl.when(pl.col("kills") > 0)
            .then(pl.col("awp_kills") / pl.col("kills"))
            .otherwise(0.0)
            .alias("awp_kill_share"),

            pl.when(pl.col("kills") > 0)
            .then(pl.col("rifle_kills") / pl.col("kills"))
            .otherwise(0.0)
            .alias("rifle_kill_share"),

            pl.when(pl.col("kills") > 0)
            .then(pl.col("pistol_kills") / pl.col("kills"))
            .otherwise(0.0)
            .alias("pistol_kill_share"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("opening_kills") / pl.col("rounds_played")).fill_nan(0).alias("opening_kills_per_round"),
            ((pl.col("opening_kills") + pl.col("opening_deaths")) / pl.col("rounds_played")).fill_nan(0).alias("opening_attempts_per_round"),
            ((pl.col("opening_kills") + pl.col("opening_deaths")) / pl.col("rounds_played")).fill_nan(0).alias("opening_duels_per_round"),

            (pl.col("kills") / pl.col("rounds_played")).fill_nan(0).alias("kills_per_round"),
            (pl.col("deaths") / pl.col("rounds_played")).fill_nan(0).alias("deaths_per_round"),
            (pl.col("assists_raw") / pl.col("rounds_played")).fill_nan(0).alias("assists_per_round"),
            (pl.col("flash_assists") / pl.col("rounds_played")).fill_nan(0).alias("flash_assists_per_round"),

            (pl.col("hp_damage_total") / pl.col("rounds_played")).fill_nan(0).alias("hp_damage_per_round"),
            (pl.col("armor_damage_total") / pl.col("rounds_played")).fill_nan(0).alias("armor_damage_per_round"),
            (pl.col("utility_damage_total") / pl.col("rounds_played")).fill_nan(0).alias("utility_damage_per_round"),

            pl.col("avg_time_to_first_engagement_ticks").fill_nan(0).alias("avg_time_to_first_engagement_ticks"),
            pl.col("avg_survival_ticks").fill_nan(0).alias("avg_survival_ticks"),
            pl.col("std_survival_ticks").fill_nan(0).alias("std_survival_ticks"),

            (pl.col("trade_kills") / pl.col("kills")).fill_nan(0).alias("trade_kill_rate"),
            (pl.col("trade_kills") / pl.col("rounds_played")).fill_nan(0).alias("trade_kills_per_round"),
            (pl.col("traded_deaths") / pl.col("rounds_played")).fill_nan(0).alias("traded_deaths_per_round"),
            (pl.col("trade_attempts") / pl.col("rounds_played")).fill_nan(0).alias("trade_attempts_per_round"),

            (pl.col("grenades_thrown") / pl.col("rounds_played")).fill_nan(0).alias("grenades_per_round"),
            (pl.col("flashbangs_thrown") / pl.col("rounds_played")).fill_nan(0).alias("flashbangs_per_round"),
            (pl.col("smokes_thrown") / pl.col("rounds_played")).fill_nan(0).alias("smokes_per_round"),
            (pl.col("fire_nades_thrown") / pl.col("rounds_played")).fill_nan(0).alias("fire_nades_per_round"),

            pl.col("avg_distance_from_team").fill_nan(0).alias("avg_distance_from_team"),
            pl.col("isolation_rate").fill_nan(0).alias("isolation_rate"),
        ]
    )

    feature_cols = [
        "opening_kills_per_round",
        "opening_attempts_per_round",
        "opening_duels_per_round",
        "opening_success_rate",
        "kills_per_round",
        "deaths_per_round",
        "assists_per_round",
        "flash_assists_per_round",
        "hp_damage_per_round",
        "armor_damage_per_round",
        "utility_damage_per_round",
        "headshot_rate",
        "awp_kill_share",
        "rifle_kill_share",
        "pistol_kill_share",
        "avg_time_to_first_engagement_ticks",
        "avg_survival_ticks",
        "std_survival_ticks",
        "survival_rate",
        "trade_kill_rate",
        "trade_kills_per_round",
        "traded_deaths_per_round",
        "trade_attempts_per_round",
        "trade_success_rate",
        "grenades_per_round",
        "flashbangs_per_round",
        "smokes_per_round",
        "fire_nades_per_round",
        "avg_distance_from_team",
        "isolation_rate",
    ]

    df = df.select(["player_name", "map_name", "side", "rounds_played", *feature_cols])

    pivot = df.pivot(
        values=["rounds_played", *feature_cols],
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
    parser = argparse.ArgumentParser(description="Extract side-split CS2 features from demos.")
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