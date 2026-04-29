# Counter Strike 2 Role Classification

This project attempts to infer player roles from gameplay data by extracting behavioral features from professional match demos and applying unsupervised clustering techniques.

Counter-Strike 2 (CS2) is a competitive, team-based first-person shooter where two teams of five players, Terrorists (T) and Counter-Terrorists (CT), compete in round-based matches. The T side aims to plant a bomb at designated bomb sites, then defend the bomb until detonation, while the CT side aims to prevent the bomb from being planted, or to defuse it. This game is very complex, and involves tactical positioning, smart utility usage of grenades, flashbangs, and smoke grenades, economic decision-making, and of course, mechanical shooting skill and teamwork.

Although all players share the same core mechanics, at the highest level of play, teams develop formal roles. Some are quite easy to define, like that of IGLs(In-Game Leaders), who make calls and captain the team, or the AWPer(the player who uses the AWP, the most powerful weapon in the game), who holds angles, punishes peeks, controls space. However, other roles, such as lurkers, anchors, entries, and support riflers, are much less well defined. 

## Goal

The goal is not to force players into the traditional rigid labels, but to test whether recognizable professional roles emerge naturally from behavioral gameplay data.

## Methods

The pipeline consists of three main stages:
1. CS2 demos are parsed using `awpy` to extract round-level and event-level data
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

All features are computed separately by side, then aggregated and pivoted into side-specific columns. Most features are normalized by rounds played so players can be compared across demos with different lengths.

### Generic 

| Feature             | Description                            |
| ------------------- | -------------------------------------- |
| **kpr**             | Kills per round                        |
| **dpr**             | Deaths per round                       |
| **kdr**             | Kill-to-death ratio                    |
| **survival_rate**   | Fraction of rounds the player survives |
| **multi_kill_rate** | Frequency of rounds with 2+ kills      |

### Damage

| Feature                    | Description                              |
| -------------------------- | ---------------------------------------- |
| **damage_per_round**       | Average damage dealt per round           |
| **damage_taken_per_round** | Average damage taken per round           |
| **damage_diff_per_round**  | Net damage advantage per round           |
| **adr**                    | Average damage per round                 |
| **util_damage_per_round**  | Utility damage per round                 |

### Opening

| Feature                   | Description                                    |
| ------------------------- | ---------------------------------------------- |
| **opening_kill_rate** | Fraction of kills that are opening kills |
| **opening_death_rate** | Fraction of deaths that are opening deaths |
| **opening_duel_attempts** | Number of opening duels taken per round |
| **opening_duel_success**  | How often the player wins opening fights       |

### Trading

| Feature                 | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| **trade_kill_rate**     | How often the player’s kills are trades               |
| **death_traded_rate**   | How often the player’s deaths are traded by teammates |
| **trade_participation** | Trades participated in per round                      |

### Weapon Usage

| Feature             | Description                          |
| ------------------- | ------------------------------------ |
| **awp_kill_share**    | Fraction of kills made with the AWP. |
| **rifle_kill_share**  | Fraction of kills made with rifles.  |

### Utility

| Feature                   | Description                     |
| ------------------------- | ------------------------------- |
| **he_grenades_per_round** | HE grenades thrown per round |
| **flashbangs_per_round**  | Flashbangs thrown per round |
| **smokes_per_round**      | Smokes thrown per round |
| **fire_nades_per_round**  | Molotovs and incendiaries thrown per round |
| **decoys_per_round**      | Decoys thrown per round |

### Other

| Feature    | Description                                                |
| ---------- | ---------------------------------------------------------- |
| **kast**   | Percentage of rounds with kill, assist, survival, or trade |
| **impact** | Overall round impact score                                 |
| **rating** | Overall performance rating                                 |


## Dataset

The dataset consists of professional CS2 match demos collected from HLTV.org and parsed using `awpy`.

### Summary Statistics

### Notes

- The dataset consists of all matches played during IEM Rio 2026.
- All features are computed directly from parsed demo data.
- Player statistics are aggregated across multiple matches to reduce variance.