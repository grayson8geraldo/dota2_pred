"""
Feature engineering pipeline for Dota 2 match prediction.

Extracts numerical features from raw match/team/hero data that can be
fed into ML models. Each feature group captures a different aspect
of what determines match outcome.
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

    # Normalize ratings (typical pro range: 800-2000)
    rad_norm = (rad_rating - 800) / 1200
    dire_norm = (dire_rating - 800) / 1200

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
         rad_streak, dire_streak]
    """
    def _form(team_data: dict) -> tuple[float, float, float]:
        matches = team_data.get("matches", [])[:n_recent]
        if not matches:
            return 0.5, 0.5, 0.0

        now = time.time()
        wins = 0
        weighted_wins = 0.0
        weighted_total = 0.0
        streak = 0
        streak_counting = True

        for m in matches:
            is_win = m.get("win") == 1 if "win" in m else m.get("radiant_win", False)
            if is_win:
                wins += 1

            # Time-weighted form
            start_time = m.get("start_time", now)
            days_ago = max(0, (now - start_time) / 86400)
            w = _decay_weight(days_ago)
            weighted_total += w
            if is_win:
                weighted_wins += w

            # Win/loss streak
            if streak_counting:
                if is_win:
                    streak += 1
                else:
                    if streak == 0:
                        streak -= 1
                    streak_counting = False

        winrate = _safe_div(wins, len(matches), 0.5)
        weighted_wr = _safe_div(weighted_wins, weighted_total, 0.5)
        return winrate, weighted_wr, streak

    rad_wr, rad_wwr, rad_streak = _form(radiant_team)
    dire_wr, dire_wwr, dire_streak = _form(dire_team)

    return [
        rad_wr, dire_wr, rad_wr - dire_wr,
        rad_wwr, dire_wwr,
        rad_streak / 10.0, dire_streak / 10.0,  # normalize streak
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

    h2h_total = 0
    h2h_rad_wins = 0
    h2h_recent_wins = 0
    h2h_recent_total = 0

    for m in rad_matches:
        opposing_id = m.get("opposing_team_id")
        if opposing_id == dire_team_id:
            h2h_total += 1
            is_win = m.get("win") == 1 if "win" in m else m.get("radiant_win", False)
            if is_win:
                h2h_rad_wins += 1
            if h2h_total <= 10:
                h2h_recent_total += 1
                if is_win:
                    h2h_recent_wins += 1

    h2h_wr = _safe_div(h2h_rad_wins, h2h_total, 0.5)
    h2h_recent_wr = _safe_div(h2h_recent_wins, h2h_recent_total, 0.5)

    return [
        min(h2h_total, 30) / 30.0,  # normalize game count
        h2h_wr,
        h2h_recent_wr,
    ]


# ---------------------------------------------------------------------------
# 4. Hero Draft Features
# ---------------------------------------------------------------------------

def extract_draft_features(
    match_data: dict,
    hero_stats: dict,
) -> list[float]:
    """
    Features based on hero picks and bans.

    Returns:
        [rad_draft_wr, dire_draft_wr, draft_wr_diff,
         rad_synergy, dire_synergy,
         rad_roles_balanced, dire_roles_balanced,
         rad_avg_pro_pickrate, dire_avg_pro_pickrate]
    """
    picks_bans = match_data.get("picks_bans") or []
    heroes_data = hero_stats.get("heroes", {})

    radiant_picks = []
    dire_picks = []

    for pb in picks_bans:
        if not pb.get("is_pick"):
            continue
        hero_id = pb.get("hero_id")
        team = pb.get("team")  # 0 = radiant, 1 = dire
        if team == 0:
            radiant_picks.append(hero_id)
        else:
            dire_picks.append(hero_id)

    # Also try to extract from player data if picks_bans is empty
    if not radiant_picks and not dire_picks:
        players = match_data.get("players", [])
        for p in players:
            hero_id = p.get("hero_id")
            if hero_id:
                slot = p.get("player_slot", 0)
                if slot < 128:
                    radiant_picks.append(hero_id)
                else:
                    dire_picks.append(hero_id)

    def _draft_stats(picks: list[int]) -> tuple[float, float, float]:
        if not picks:
            return 0.5, 0.0, 0.0
        winrates = []
        pickrates = []
        roles = set()
        for hid in picks:
            h = heroes_data.get(hid, heroes_data.get(str(hid), {}))
            winrates.append(h.get("pro_winrate", 0.5))
            total_games = h.get("pro_pick", 0) + h.get("pro_ban", 0)
            pickrates.append(min(total_games / 1000.0, 1.0))
            for r in h.get("roles", []):
                roles.add(r)

        avg_wr = np.mean(winrates) if winrates else 0.5
        avg_pr = np.mean(pickrates) if pickrates else 0.0
        # Role balance: more unique roles = more balanced
        role_balance = min(len(roles), 8) / 8.0
        return avg_wr, role_balance, avg_pr

    rad_wr, rad_roles, rad_pr = _draft_stats(radiant_picks)
    dire_wr, dire_roles, dire_pr = _draft_stats(dire_picks)

    # Simple synergy: hero pair win rate correlation
    def _pair_synergy(picks: list[int]) -> float:
        if len(picks) < 2:
            return 0.0
        # Use role diversity as a proxy for synergy
        roles_list = []
        for hid in picks:
            h = heroes_data.get(hid, heroes_data.get(str(hid), {}))
            roles_list.extend(h.get("roles", []))
        # More diverse roles = better synergy
        unique_ratio = len(set(roles_list)) / max(len(roles_list), 1)
        return unique_ratio

    rad_syn = _pair_synergy(radiant_picks)
    dire_syn = _pair_synergy(dire_picks)

    return [
        rad_wr, dire_wr, rad_wr - dire_wr,
        rad_syn, dire_syn,
        rad_roles, dire_roles,
        rad_pr, dire_pr,
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
    # Historical Radiant advantage in pro matches (~1-3% depending on patch)
    radiant_adv = 0.02
    first_pick_adv = 0.01 if is_radiant_first_pick else -0.01
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

    Note: BO2 is treated as closest to BO3 for model compatibility
    (all three binary flags are 0, so the model sees a neutral format signal).
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
            # Fallback: use last 5 players
            current_players = players[:5]

        # How many games played together as a team
        games_together = sum(p.get("games_played", 0) for p in current_players)
        # Normalize (100+ games together = very stable)
        stability = min(games_together / 500.0, 1.0)
        return stability

    rad_stab = _stability(radiant_team)
    dire_stab = _stability(dire_team)

    return [rad_stab, dire_stab, rad_stab - dire_stab]


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
    Extract complete feature vector for a match.

    Returns numpy array of shape (N_FEATURES,).
    """
    features = []

    # 1. Team rating (4 features)
    features.extend(extract_team_rating_features(radiant_team, dire_team))

    # 2. Recent form (7 features)
    features.extend(extract_recent_form_features(radiant_team, dire_team))

    # 3. Head-to-head (3 features)
    features.extend(extract_h2h_features(
        radiant_team, dire_team, radiant_team_id, dire_team_id
    ))

    # 4. Hero draft (9 features)
    features.extend(extract_draft_features(match_data, hero_stats))

    # 5. Map side (2 features)
    features.extend(extract_map_side_features())

    # 6. Match format (5 features)
    features.extend(extract_format_features(series_type, game_number))

    # 7. Roster stability (3 features)
    features.extend(extract_roster_features(radiant_team, dire_team))

    return np.array(features, dtype=np.float64)


def get_feature_names() -> list[str]:
    """Return names for all features in the same order as extract_features."""
    return [
        # Team rating
        "rating_diff", "elo_win_prob", "rad_rating_norm", "dire_rating_norm",
        # Recent form
        "rad_winrate", "dire_winrate", "form_diff",
        "rad_weighted_wr", "dire_weighted_wr",
        "rad_streak", "dire_streak",
        # Head-to-head
        "h2h_games", "h2h_rad_winrate", "h2h_recent_rad_wr",
        # Hero draft
        "rad_draft_wr", "dire_draft_wr", "draft_wr_diff",
        "rad_synergy", "dire_synergy",
        "rad_roles_balanced", "dire_roles_balanced",
        "rad_avg_pro_pickrate", "dire_avg_pro_pickrate",
        # Map side
        "radiant_advantage", "first_pick_advantage",
        # Match format
        "is_bo1", "is_bo3", "is_bo5", "game_number_norm", "is_decider",
        # Roster stability
        "rad_roster_stability", "dire_roster_stability", "stability_diff",
    ]


N_FEATURES = len(get_feature_names())  # 33 features total
