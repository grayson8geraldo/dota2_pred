"""
Feature engineering pipeline for Dota 2 match prediction (pre-match only).

Extracts numerical features from raw team/player data that can be
fed into ML models. Draft features are excluded since picks are
unknown before the match starts.

Feature groups:
  1. Team Rating (4)     - Elo-based rating comparison
  2. Recent Form (9)     - Win rates, streaks, momentum
  3. Head-to-Head (3)    - Historical matchup record
  4. Player Strength (6) - Individual player metrics & hero pools
  5. Map Side (2)        - Radiant/Dire advantage
  6. Match Format (5)    - BO1/BO3/BO5, game number
  7. Roster Stability (3)- How stable the team roster is
  8. Meta (1)            - Match recency / meta stability
  Total: 33 features
"""

import math
import time
import logging
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Elo expected score for player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _decay_weight(days_ago: float, half_life: float = 30.0) -> float:
    """Exponential decay weight; half-life in days."""
    return math.exp(-math.log(2) * days_ago / half_life)


# ---------------------------------------------------------------------------
# 1. Team Rating Features
# ---------------------------------------------------------------------------

def extract_team_rating_features(
    radiant_team: dict,
    dire_team: dict,
) -> list[float]:
    """
    Features based on team Elo/rating from OpenDota.

    Returns:
        [rating_diff, elo_win_prob, rad_rating_norm, dire_rating_norm]
    """
    rad_rating = radiant_team.get("info", {}).get("rating", 1200)
    dire_rating = dire_team.get("info", {}).get("rating", 1200)

    rating_diff = rad_rating - dire_rating
    elo_prob = _elo_expected(rad_rating, dire_rating)

    # Normalize ratings to [0, 1] (pro range: ~800-2000)
    rad_norm = max(0.0, min(1.0, (rad_rating - 800) / 1200))
    dire_norm = max(0.0, min(1.0, (dire_rating - 800) / 1200))

    return [rating_diff, elo_prob, rad_norm, dire_norm]


# ---------------------------------------------------------------------------
# 2. Recent Form Features
# ---------------------------------------------------------------------------

def extract_recent_form_features(
    radiant_team: dict,
    dire_team: dict,
    n_recent: int = 20,
) -> list[float]:
    """
    Features based on recent match results.

    Returns:
        [rad_winrate, dire_winrate, form_diff,
         rad_weighted_wr, dire_weighted_wr,
         rad_streak, dire_streak,
         rad_momentum, dire_momentum]
    """
    def _form(team_data: dict) -> tuple[float, float, float, float]:
        matches = team_data.get("matches", [])[:n_recent]
        if not matches:
            return 0.5, 0.5, 0.0, 0.5

        matches = sorted(matches, key=lambda m: m.get("start_time", 0), reverse=True)

        now = time.time()
        wins = 0
        weighted_wins = 0.0
        weighted_total = 0.0
        momentum_wins = 0.0
        momentum_total = 0.0
        streak = 0
        streak_done = False

        for i, m in enumerate(matches):
            is_win = m.get("win") == 1 if "win" in m else m.get("radiant_win", False)
            if is_win:
                wins += 1

            start_time = m.get("start_time", now)
            days_ago = max(0, (now - start_time) / 86400)
            w = _decay_weight(days_ago)
            weighted_total += w
            if is_win:
                weighted_wins += w

            if i < 5:
                mw = _decay_weight(days_ago, half_life=7.0)
                momentum_total += mw
                if is_win:
                    momentum_wins += mw

            if not streak_done:
                if streak == 0:
                    streak = 1 if is_win else -1
                elif (streak > 0 and is_win) or (streak < 0 and not is_win):
                    streak += 1 if is_win else -1
                else:
                    streak_done = True

        winrate = _safe_div(wins, len(matches), 0.5)
        weighted_wr = _safe_div(weighted_wins, weighted_total, 0.5)
        momentum = _safe_div(momentum_wins, momentum_total, 0.5)
        return winrate, weighted_wr, streak, momentum

    rad_wr, rad_wwr, rad_streak, rad_mom = _form(radiant_team)
    dire_wr, dire_wwr, dire_streak, dire_mom = _form(dire_team)

    return [
        rad_wr, dire_wr, rad_wr - dire_wr,
        rad_wwr, dire_wwr,
        rad_streak / 10.0, dire_streak / 10.0,
        rad_mom, dire_mom,
    ]


# ---------------------------------------------------------------------------
# 3. Head-to-Head Features
# ---------------------------------------------------------------------------

def extract_h2h_features(
    radiant_team: dict,
    dire_team: dict,
    radiant_team_id: int,
    dire_team_id: int,
) -> list[float]:
    """
    Features based on head-to-head record between two teams.

    Returns:
        [h2h_games, h2h_rad_winrate, h2h_recent_rad_wr]
    """
    rad_matches = radiant_team.get("matches", [])
    rad_matches = sorted(rad_matches, key=lambda m: m.get("start_time", 0), reverse=True)

    h2h_matches: list[bool] = []
    for m in rad_matches:
        opposing_id = m.get("opposing_team_id")
        if opposing_id == dire_team_id:
            is_win = m.get("win") == 1 if "win" in m else m.get("radiant_win", False)
            h2h_matches.append(is_win)

    h2h_total = len(h2h_matches)
    h2h_rad_wins = sum(h2h_matches)
    h2h_wr = _safe_div(h2h_rad_wins, h2h_total, 0.5)

    recent = h2h_matches[:10]
    h2h_recent_wr = _safe_div(sum(recent), len(recent), 0.5) if recent else h2h_wr

    return [
        min(h2h_total, 30) / 30.0,
        h2h_wr,
        h2h_recent_wr,
    ]


# ---------------------------------------------------------------------------
# 4. Player Strength Features (NEW - replaces draft)
# ---------------------------------------------------------------------------

def extract_player_features(
    radiant_team: dict,
    dire_team: dict,
) -> list[float]:
    """
    Features based on individual player performance and hero pools.

    Uses team player data (win rates of current roster) and
    team hero data (diversity of hero pool) as pre-match signals.

    Returns:
        [rad_player_winrate, dire_player_winrate, player_wr_diff,
         rad_hero_pool, dire_hero_pool, hero_pool_diff]
    """
    def _player_stats(team_data: dict) -> tuple[float, float]:
        # Player win rate from current roster
        players = team_data.get("players", [])
        current = [p for p in players if p.get("is_current_team_member", False)]
        if not current:
            current = players[:5]

        if current:
            winrates = []
            for p in current:
                games = p.get("games_played", 0)
                wins = p.get("wins", 0)
                if games >= 5:
                    winrates.append(wins / games)
            player_wr = np.mean(winrates) if winrates else 0.5
        else:
            player_wr = 0.5

        # Hero pool depth: how many heroes the team plays competently
        heroes = team_data.get("heroes", [])
        if heroes:
            # Count heroes with >= 3 games and > 40% win rate
            viable_heroes = sum(
                1 for h in heroes
                if h.get("games_played", 0) >= 3
                and _safe_div(h.get("wins", 0), h.get("games_played", 1), 0) > 0.4
            )
            # Normalize: 30 viable heroes = 0.5, 60+ = 1.0
            hero_pool = min(viable_heroes / 60.0, 1.0)
        else:
            hero_pool = 0.3  # neutral default

        return float(player_wr), float(hero_pool)

    rad_wr, rad_pool = _player_stats(radiant_team)
    dire_wr, dire_pool = _player_stats(dire_team)

    return [
        rad_wr, dire_wr, rad_wr - dire_wr,
        rad_pool, dire_pool, rad_pool - dire_pool,
    ]


# ---------------------------------------------------------------------------
# 5. Map Side Features
# ---------------------------------------------------------------------------

def extract_map_side_features(is_radiant_first_pick: bool = True) -> list[float]:
    """
    Features based on Radiant/Dire side advantages.

    Returns:
        [radiant_advantage, first_pick_advantage]
    """
    radiant_adv = 0.015
    first_pick_adv = 0.008 if is_radiant_first_pick else -0.008
    return [radiant_adv, first_pick_adv]


# ---------------------------------------------------------------------------
# 6. Match Format Features
# ---------------------------------------------------------------------------

def extract_format_features(
    series_type: int = 1,
    game_number: int = 1,
) -> list[float]:
    """
    Features based on match format (BO1/BO2/BO3/BO5) and game number in series.

    series_type: 0=BO1, 1=BO3, 2=BO5, 3=BO2
    game_number: which game in the series (1-indexed)

    Returns:
        [is_bo1, is_bo3, is_bo5, game_number_norm, is_decider]
    """
    is_bo1 = 1.0 if series_type == 0 else 0.0
    is_bo3 = 1.0 if series_type == 1 else 0.0
    is_bo5 = 1.0 if series_type == 2 else 0.0

    max_games = {0: 1, 1: 3, 2: 5, 3: 2}.get(series_type, 3)
    game_norm = game_number / max_games
    is_decider = 1.0 if game_number == max_games else 0.0

    return [is_bo1, is_bo3, is_bo5, game_norm, is_decider]


# ---------------------------------------------------------------------------
# 7. Roster Stability Features
# ---------------------------------------------------------------------------

def extract_roster_features(
    radiant_team: dict,
    dire_team: dict,
) -> list[float]:
    """
    Features based on roster stability.

    Returns:
        [rad_roster_stability, dire_roster_stability, stability_diff]
    """
    def _stability(team_data: dict) -> float:
        players = team_data.get("players", [])
        if not players:
            return 0.5

        current_players = [p for p in players if p.get("is_current_team_member", False)]
        if not current_players:
            current_players = players[:5]

        games_together = sum(p.get("games_played", 0) for p in current_players)
        stability = min(math.log1p(games_together) / math.log1p(2500), 1.0)
        return stability

    rad_stab = _stability(radiant_team)
    dire_stab = _stability(dire_team)

    return [rad_stab, dire_stab, rad_stab - dire_stab]


# ---------------------------------------------------------------------------
# 8. Meta / Recency Features (NEW)
# ---------------------------------------------------------------------------

def extract_meta_features(
    match_data: dict = None,
    reference_time: float = None,
) -> list[float]:
    """
    Feature capturing how recent the data is relative to current meta.

    For training: uses match start_time to weight recent matches higher.
    For prediction: returns a neutral value (recency handled via sample weights).

    Returns:
        [match_recency]
    """
    if reference_time is None:
        reference_time = time.time()

    if match_data and match_data.get("start_time"):
        days_ago = max(0, (reference_time - match_data["start_time"]) / 86400)
        recency = _decay_weight(days_ago, half_life=config.PATCH_HALF_LIFE_DAYS)
    else:
        recency = 1.0  # prediction time = most recent

    return [recency]


# ---------------------------------------------------------------------------
# Full Feature Vector
# ---------------------------------------------------------------------------

def extract_features(
    match_data: dict,
    radiant_team: dict,
    dire_team: dict,
    hero_stats: dict,
    radiant_team_id: int,
    dire_team_id: int,
    series_type: int = 1,
    game_number: int = 1,
) -> np.ndarray:
    """
    Extract complete feature vector for a match (pre-match only).

    Returns numpy array of shape (N_FEATURES,).
    """
    features = []

    # 1. Team rating (4 features)
    features.extend(extract_team_rating_features(radiant_team, dire_team))

    # 2. Recent form (9 features)
    features.extend(extract_recent_form_features(radiant_team, dire_team))

    # 3. Head-to-head (3 features)
    features.extend(extract_h2h_features(
        radiant_team, dire_team, radiant_team_id, dire_team_id
    ))

    # 4. Player strength (6 features) — replaces draft
    features.extend(extract_player_features(radiant_team, dire_team))

    # 5. Map side (2 features)
    features.extend(extract_map_side_features())

    # 6. Match format (5 features)
    features.extend(extract_format_features(series_type, game_number))

    # 7. Roster stability (3 features)
    features.extend(extract_roster_features(radiant_team, dire_team))

    # 8. Meta recency (1 feature)
    features.extend(extract_meta_features(match_data))

    return np.array(features, dtype=np.float64)


def get_feature_names() -> list[str]:
    """Return names for all features in the same order as extract_features."""
    return [
        # Team rating (4)
        "rating_diff", "elo_win_prob", "rad_rating_norm", "dire_rating_norm",
        # Recent form (9)
        "rad_winrate", "dire_winrate", "form_diff",
        "rad_weighted_wr", "dire_weighted_wr",
        "rad_streak", "dire_streak",
        "rad_momentum", "dire_momentum",
        # Head-to-head (3)
        "h2h_games", "h2h_rad_winrate", "h2h_recent_rad_wr",
        # Player strength (6)
        "rad_player_winrate", "dire_player_winrate", "player_wr_diff",
        "rad_hero_pool", "dire_hero_pool", "hero_pool_diff",
        # Map side (2)
        "radiant_advantage", "first_pick_advantage",
        # Match format (5)
        "is_bo1", "is_bo3", "is_bo5", "game_number_norm", "is_decider",
        # Roster stability (3)
        "rad_roster_stability", "dire_roster_stability", "stability_diff",
        # Meta (1)
        "match_recency",
    ]


N_FEATURES = len(get_feature_names())  # 33 features
