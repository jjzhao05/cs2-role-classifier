# Counter Strike 2 Role Classification

This project attempts to infer player roles from gameplay data by extracting behavioral features from professional match demos and applying unsupervised clustering techniques.

Counter-Strike 2 (CS2) is a competitive, team-based first-person shooter where two teams of five players, Terrorists (T) and Counter-Terrorists (CT), compete in round-based matches. The T side aims to plant a bomb at designated bomb sites, then defend the bomb until detonation, while the CT side aims to prevent the bomb from being planted, or to defuse it. This game is very complex, and involves tactical positioning, smart utility usage of grenades, flashbangs, and smoke grenades, economic decision-making, and of course, mechanical shooting skill and teamwork.

Although all players share the same core mechanics, at the highest level of play, teams develop formal roles. Some are quite easy to define, like that of IGLs(In-Game Leaders), who make calls and captain the team, or the AWPer(the player who uses the AWP, the most powerful weapon in the game), who holds angles, punishes peeks, controls space. However, other roles, such as lurkers, anchors, entries, and support riflers, are much less well defined. 

## Goal

The goal is not to force players into the traditional rigid labels, but to test whether recognizable roles emerge from gameplay data.

## Methods

The pipeline consists of three main stages:
1. CS2 demos are parsed using `awpy` to extract round-level and event-level data
2. Data is aggregated across multiple demos by player
3. A PCA is applied and 2 clustering algorithms(K-Means, GMM) are applied to try and group the players


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

## Schema

Data is stored in long format, where each row represents a player–side observation. All features are computed per player per side. Most features are normalized by rounds played, enabling comparison across demos of different lengths. 

### Identifiers

| Feature         | Description                     |
| --------------- | ------------------------------- |
| **player_name** | Player identifier (categorical) |
| **side**        | Team side (categorical: T / CT) |

### Generic 

| Feature           | Description                       |
| ----------------- | --------------------------------- |
| **adr**           | Mean damage per round             |
| **kpr**           | Mean kills per round              |
| **survival_rate** | Fraction of rounds survived [0–1] |


### Combat output

| Feature                    | Description                                    |
| -------------------------- | ---------------------------------------------- |
| **damage_per_round**       | Mean damage dealt per round                    |
| **damage_taken_per_round** | Mean damage received per round                 |
| **damage_diff_per_round**  | Mean net damage per round                      |
| **assists_per_round**      | Mean assists per round                         |
| **multi_kill_rate**        | Fraction of rounds with multiple kills `[0–1]` |
| **rifle_kill_share**       | Fraction of kills with rifles `[0–1]`          |
| **awp_kill_share**         | Fraction of kills with AWP `[0–1]`             |


### Opening

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| **opening_kill_rate**    | Fraction of rounds with opening kill `[0–1]`     |
| **opening_death_rate**   | Fraction of rounds with opening death `[0–1]`    |
| **opening_duel_success** | Opening duel win rate `[0–1]`                    |
| **first_contact_rate**   | Fraction of rounds with first engagement `[0–1]` |

### Trading
**Trading** refers to a kill that occurs shortly after a teammate’s death, where the killer eliminates the opponent responsible for that death within 5 seconds.

| Feature                 | Description                                    |
| ----------------------- | ---------------------------------------------- |
| **trade_kill_rate**     | Fraction of kills that are trades* `[0–1]`      |
| **death_traded_rate**   | Fraction of deaths traded by teammates `[0–1]` |
| **trade_participation** | Fraction of trade involvement `[0–1]`          |

### Utility

| Feature                     | Description                                     |
| --------------------------- | ----------------------------------------------- |
| **grenades_per_round**      | Mean grenades used per round                    |
| **he_grenades_per_round**   | Mean HE grenades used per round                 |
| **flashbangs_per_round**    | Mean flashbangs used per round                  |
| **smokes_per_round**        | Mean smokes used per round                      |
| **fire_nades_per_round**    | Mean molotov/incendiary grenades used per round |
| **decoys_per_round**        | Mean decoys used per round                      |
| **flash_assists_per_round** | Mean flash assists per round                    |
| **util_damage_per_round**   | Mean damage dealt via utility                   |

### Movement and Positioning

| Feature                              | Description                            |
| ------------------------------------ | -------------------------------------- |
| **avg_distance_to_enemy**            | Mean distance to nearest enemy         |
| **avg_distance_to_team_centroid**    | Mean distance to team centroid         |
| **relative_team_centroid_distance**  | Normalized distance from team centroid |
| **avg_distance_moved_per_round**     | Mean distance traveled per round       |
| **avg_distance_to_closest_teammate** | Mean distance to nearest teammate      |
| **time_near_enemy_rate**             | Fraction of time near enemies `[0–1]`  |
| **time_stationary_rate**             | Fraction of time stationary `[0–1]`    |

## Dataset

The dataset consists of professional CS2 match demos collected from HLTV.org and parsed using `awpy`.

### Notes

- The dataset consists of all matches played during IEM Rio 2026.
- All features are computed directly from parsed demo data.
- I plan on using data from IEM Cologne Major, IEM Atlanta, and PGL Astana, as that represents a period with limited roster and positional changes due to roster locks
