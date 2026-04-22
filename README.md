# Counter Strike 2 Role Classification

This project attempts to infer player roles from gameplay data by extracting behavioral features from professional match demos and applying unsupervised clustering techniques.

Counter-Strike 2 (CS2) is a competitive, team-based first-person shooter where two teams of five players, Terrorists (T) and Counter-Terrorists (CT), compete in round-based matches. The T side aims to plant a bomb at designated bomb sites, then defend the bomb until detonation, while the CT side aims to prevent the bomb from being planted, or to defuse it. This game is very complex, and involves tactical positioning, smart utility usage of grenades, flashbangs, and smoke grenades, economic decision-making, and of course, mechanical shooting skill and teamwork.

Although all players share the same core mechanics, at the highest level of play, teams develop formal roles. Some are quite easy to define, like that of IGLs(In-Game Leaders), who make calls and captain the team, or the AWPer(the player who uses the AWP, the most powerful weapon in the game), who holds angles, punishes peeks, controls space. However, other roles, such as lurkers, anchors, entries, and support riflers, are much less well defined. 

## Goal

The goal is not to force players into the traditional rigid labels, but to test whether recognizable professional roles emerge naturally from behavioral gameplay data.

## Methods

The pipeline consists of three main stages:
1. CS2 demos are parsed using awpy to extract round-level and event-level data
2. Data is aggregated across multiple demos by player
3. A PCA is applied and 3 clustering algorithms(K-Means, GMM, HDBSCAN) are applied to try and group the players


## Results(WIP)

Some roles are easily recovered from this gameplay data

- **AWPers** such as **broky**, **m0NESY**, and **SunPayus** are very easy to identify because of AWP-related features.
- **IGLs** such as **karrigan**, **kyxsan**, and **apEX** appear much harder to identify from gameplay data, as a large portion of their impact comes from communication, mid-round calling, and team leadership rather than measurable in-game actions.
- **Entry players** such as **donk**, **YEKINDAR**, and **EliGE** WIP
- **Lurkers** such as **ropz**, **blameF**, and **Spinx** WIP
- Some players are simply extreme outliers. A player like **ZywOo**, often considered one of the greatest Counter-Strike players of all time, may cluster unusually because his extremely high scores in many regions.

## Usage

python demo_parser.py --test outputs/test_output.csv

python demo_parser.py --use-main-demos outputs/full_output.csv

## Feature Definitions

All features are computed separately by side and map, then pivoted into side-specific columns. Most features are normalized by rounds played so players can be compared across demos with different lengths.

### Opening 
| Feature                      | Description                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `first_kill_rate`            | Opening kills per round played. Higher values usually indicate more entry involvement.                 |
| `first_death_rate`           | Opening deaths per round played. Higher values usually indicate higher-risk entry behavior.            |
| `entry_attempt_rate`         | Opening duels per round, defined as rounds where the player gets either the first kill or first death. |
| `entry_success_rate`         | Fraction of opening duels won by the player.                                                           |
| `opening_kills_per_round`    | Opening kills per round played.                                                                        |
| `opening_attempts_per_round` | Opening duel attempts per round played.                                                                |
| `opening_duels_per_round`    | Same as `opening_attempts_per_round`; kept as an explicit duel-volume feature.                         |
| `opening_success_rate`       | Fraction of opening duels won. Equivalent in meaning to `entry_success_rate`.                          |

### Time

| Feature                                | Description                                                                                          |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `early_round_kill_rate`                | Kills per round that occur in the early-round window. Helps identify aggressive early-round players. |
| `late_round_kill_rate`                 | Kills per round that occur late in the round. Helps identify closers and late-round impact players.  |
| `late_round_presence_rate`             | Fraction of rounds where the player survives into the late-round portion of the round.               |
| `avg_time_to_first_engagement_ticks`   | Average ticks from round start to the player’s first shot or first damage event.                     |
| `avg_time_to_first_engagement_seconds` | Same as above, expressed in seconds.                                                                 |

### Damage Output
| Feature                    | Description                                           |
| -------------------------- | ----------------------------------------------------- |
| `kills_per_round`          | Total kills divided by rounds played.                 |
| `deaths_per_round`         | Total deaths divided by rounds played.                |
| `assists_per_round`        | Total assists divided by rounds played.               |
| `flash_assists_per_round`  | Flash assists divided by rounds played.               |
| `hp_damage_per_round`      | Health damage dealt per round.                        |
| `armor_damage_per_round`   | Armor damage dealt per round.                         |
| `utility_damage_per_round` | Damage dealt with grenades or fire utility per round. |
| `headshot_rate`            | Fraction of kills that were headshots.                |

### Weapon Usage

| Feature             | Description                          |
| ------------------- | ------------------------------------ |
| `awp_kill_share`    | Fraction of kills made with the AWP. |
| `rifle_kill_share`  | Fraction of kills made with rifles.  |
| `pistol_kill_share` | Fraction of kills made with pistols. |

### Survival

| Feature                | Description                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| `avg_survival_ticks`   | Average number of ticks the player stays alive in a round.                                    |
| `avg_survival_seconds` | Same as above, expressed in seconds.                                                          |
| `std_survival_ticks`   | Variation in survival time across rounds. Higher values indicate less consistent life length. |
| `survival_rate`        | Fraction of rounds the player survives until round end.                                       |

### Trading

| Feature                    | Description                                                                     |
| -------------------------- | ------------------------------------------------------------------------------- |
| `trade_kill_rate`          | Fraction of the player’s kills that are classified as trade kills.              |
| `trade_kills_per_round`    | Trade kills divided by rounds played.                                           |
| `traded_deaths_per_round`  | Deaths that were traded by a teammate, divided by rounds played.                |
| `trade_attempts_per_round` | Opportunities to trade a teammate, divided by rounds played.                    |
| `trade_success_rate`       | Fraction of trade opportunities converted into a trade kill.                    |
| `trade_participation_rate` | Combined trade involvement per round, using both trade kills and traded deaths. |

### Utility

| Feature                | Description                                 |
| ---------------------- | ------------------------------------------- |
| `grenades_per_round`   | Total grenades thrown per round.            |
| `flashbangs_per_round` | Flashbangs thrown per round.                |
| `smokes_per_round`     | Smokes thrown per round.                    |
| `fire_nades_per_round` | Molotovs and incendiaries thrown per round. |

### Positioning

| Feature                              | Description                                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `avg_distance_from_team`             | Average distance from the team’s alive-player centroid. Higher values suggest more independent positioning.        |
| `isolation_rate`                     | Fraction of alive ticks where the player is far from the team centroid.                                            |
| `avg_nearby_teammates`               | Average number of nearby teammates while alive, based on sampled tick positions.                                   |
| `time_spent_alone_ratio`             | Fraction of sampled alive ticks where no teammate is within the proximity threshold.                               |
| `avg_nearby_teammates_at_engagement` | Average number of nearby teammates at the moment of first engagement.                                              |
| `engagement_isolation_rate`          | Fraction of first engagements where no teammate is nearby.                                                         |
| `position_entropy`                   | Spatial unpredictability of the player’s positioning over the map. Higher values indicate more varied positioning. |

## Dataset

The dataset consists of professional CS2 match demos collected from HLTV.org and parsed using `awpy`.

### Summary Statistics

| Metric              | Value |
| ------------------- | ----- |
| Total demos         | 247   |
| Total rounds        | 46562 |
| Unique players      | 170   |
| Avg rounds per map  | 31.43 |
| Avg demos per player| 5.00  |

### Notes

- The dataset consists of all matches played during IEM Rio 2026, PGL Bucharest 2026, and BLAST Open Rotterdam 2026.
- All features are computed directly from parsed demo data.
- Player statistics are aggregated across multiple matches to reduce variance.