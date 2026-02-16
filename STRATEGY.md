# Dota 2 Match Prediction Strategy

## Goal
Predict outcomes of professional Dota 2 matches with **70%+ accuracy**.

---

## Data Sources

| Source | URL | What We Use |
|--------|-----|-------------|
| **OpenDota API** | api.opendota.com | Team ratings, match history, hero stats, draft data, head-to-head records |
| **STRATZ API** | api.stratz.com | Hero meta trends, synergy/counter data (supplementary) |
| **Liquipedia** | liquipedia.net/dota2 | Upcoming match schedules, roster changes, tournament context |
| **PandaScore** | api.pandascore.co | Upcoming match schedules, live scores |

Primary data source: **OpenDota API** (free, well-documented, SQL explorer for custom queries).

---

## Prediction Model

### Architecture
**Ensemble of Gradient Boosting + Logistic Regression** (soft voting, 60/40 weight).

- **Gradient Boosting (GBM):** Captures non-linear interactions between features (e.g., strong team + weak draft).
- **Logistic Regression (LR):** Provides well-calibrated probability estimates.
- **Ensemble:** Combines both for robust predictions.

### Features (33 total, 7 groups)

#### 1. Team Rating (4 features) — Weight: 25%
- Rating difference (OpenDota Elo-like rating)
- Elo expected win probability
- Normalized radiant/dire ratings

*Rationale:* Team Elo rating is the single strongest long-term predictor.

#### 2. Recent Form (7 features) — Weight: 20%
- Win rate (last 20 matches)
- Time-weighted win rate (exponential decay, half-life = 30 days)
- Win/loss streak

*Rationale:* Recent form captures current meta adaptation, team chemistry, and momentum.

#### 3. Head-to-Head (3 features) — Weight: 10%
- Number of H2H games
- Overall H2H win rate
- Recent H2H win rate (last 10 games)

*Rationale:* Some matchups consistently favor one team due to playstyle clashes.

#### 4. Hero Draft (9 features) — Weight: 25%
- Average draft win rate per team
- Draft win rate difference
- Hero role balance/diversity
- Hero synergy (role diversity within team)
- Average hero pick popularity

*Rationale:* Draft is one of the top predictors. Pro hero win rates in the current patch + team role coverage strongly correlate with outcomes.

#### 5. Map Side (2 features) — Weight: 5%
- Radiant advantage (~2% in pro matches)
- First pick advantage

*Rationale:* Radiant side has a small but consistent advantage (Roshan access, map layout).

#### 6. Match Format (5 features) — Weight: 5%
- BO1/BO3/BO5 encoding
- Game number in series
- Is decider game

*Rationale:* BO1 favors upsets/cheese strategies; BO3/BO5 favors consistent teams. Decider games have unique dynamics.

#### 7. Roster Stability (3 features) — Weight: 10%
- Roster stability score per team
- Stability difference

*Rationale:* Teams with stable rosters consistently outperform recently shuffled teams.

---

## How to Achieve 70%+ Accuracy

Based on academic research:
- **Team Elo alone:** ~58-62% accuracy
- **Elo + recent form:** ~64-67%
- **Elo + form + draft:** ~68-72%
- **Full feature set + ensemble:** ~70-75%
- **With in-game data (live):** up to 90%+

### Key Strategies for High Accuracy

1. **Focus on high-confidence predictions.** Skip matches where probability is close to 50/50. Predicting only when confidence > 60% significantly improves hit rate.

2. **Recency weighting.** Recent results (last 2-4 weeks) are far more predictive than career averages.

3. **Draft-aware predictions.** When draft data is available, accuracy jumps significantly.

4. **Avoid BO1.** BO1 matches have inherently higher variance. Focus predictions on BO3/BO5 for better accuracy.

5. **Track the meta.** Hero win rates change with every patch. Regular data refreshes are essential.

---

## Usage

```bash
# Predict a match
python main.py predict "Team Spirit" "Tundra Esports" --format BO3

# Predict all live matches
python main.py live

# Train the model (requires API access, takes time)
python main.py train --matches 300

# Look up a team
python main.py team "Gaimin Gladiators"

# Refresh cached hero/team data
python main.py refresh
```

### Without Training (Heuristic Mode)
The tool works immediately using a heuristic predictor based on the weighted factors above. No training required — just run `predict`.

### With ML Training
Run `python main.py train` to collect pro match data and train the ensemble model for improved accuracy. This requires API access and takes ~30-60 minutes due to rate limits.

---

## Confidence Levels

| Level | Probability | Recommendation |
|-------|-------------|----------------|
| **HIGH** | > 65% | Strong bet |
| **MEDIUM** | 55-65% | Moderate confidence |
| **LOW** | < 55% | Skip — too close to call |

---

## References

- OpenDota API: https://docs.opendota.com/
- "DotA 2 Match Outcome Prediction Using Decision Tree Ensemble" (MDPI, 2025) — 91-98% with in-game features
- "Real-time eSports Match Result Prediction" (arXiv, 2017) — 71.49% pre-match accuracy
- "Performance of ML Algorithms in Predicting Game Outcome from Drafts" — Factorization Machines best for draft-only
- STRATZ neural network achieved ~77% from draft alone at TI8
