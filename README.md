# Counter Strike 2 Role Classifier

This project is an attempt and a fun way to try and categorize these players using gameplay data by extracting features from professional match demos and applying unsupervised clustering techniques.

Since middle school, I've loved to watch CS:GO esports. One thing that stands out is how loosely defined player roles are. Some are quite easy to define, like that of IGLs(In-Game Leaders), or the AWPer(the player who uses the AWP, the most powerful weapon in the game). However, other roles, lurkers, anchors, entries, support riflers, are less well defined. 

## Methods

The pipeline consists of three main stages:
1. CS2 demos are parsed using awpy to extract round-level and event-level data
2. Data is aggregated across multiple demos by player
3. A PCA is applied and 3 clustering algorithms(K-Means, GMM, HDBSCAN) are applied to try and group the players


## Findings(WIP)

It apears AWPers(such as: Broky, m0NESY, SunPayus) are extremely easy to identify, due to their unique awp kills features

IGLs(such as: Karrigan, kyxsan, aPeX) seem to be impossible to quantify, as much of their contribution is outside of gameplay data. IGL's typically create impact by making calls and communicating with the team.

Entries(such as: donk, YEKINDAR, EliGE)

Lurkers(such as: Ropz, BlameF, Spinx)

## Feature Definitions

python demo_parser.py --test outputs/test_output.csv

python demo_parser.py --use-main-demos outputs/full_output.csv