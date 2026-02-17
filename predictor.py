"""
Main prediction interface that ties together data collection,
feature engineering, and model prediction.
"""

import json
import os
import logging
import time
from typing import Optional

import numpy as np

import config
from data_collector import OpenDotaClient, collect_hero_stats, save_data, load_data
from feature_engine import (
    extract_features,
    extract_team_rating_features,
    extract_recent_form_features,
    extract_h2h_features,
    extract_draft_features,
    extract_map_side_features,
    extract_format_features,
    extract_roster_features,
    get_feature_names,
    N_FEATURES,
)
from model import Dota2Predictor, HeuristicPredictor

logger = logging.getLogger(__name__)


class MatchPredictor:
    """High-level interface for predicting Dota 2 matches."""

    # Cache entries older than this are re-fetched from API
    CACHE_TTL = 3600  # 1 hour

    def __init__(self, use_ml_model: bool = True):
        self.client = OpenDotaClient()
        self.ml_model = Dota2Predictor()
        self.heuristic = HeuristicPredictor()
        self.use_ml = use_ml_model
        self.hero_stats = None
        self.team_cache = {}        # team_id -> team data
        self._cache_timestamps = {}  # team_id -> fetch timestamp
        self._api_failures = 0       # consecutive API failures (circuit breaker)
        self._load_caches()

    def _load_caches(self):
        """Load cached data if available."""
        hero_path = os.path.join(config.MODEL_PATH, config.HERO_STATS_FILE)
        team_path = os.path.join(config.MODEL_PATH, config.TEAM_CACHE_FILE)

        self.hero_stats = load_data(hero_path)
        team_data = load_data(team_path)
        if team_data:
            self.team_cache = {int(k): v for k, v in team_data.items()}

        # Pre-load teams list from disk (avoids API call on first name lookup)
        self._teams_list = self._load_teams_list()

        if self.use_ml:
            try:
                self.ml_model.load()
                logger.info("ML model loaded successfully")
            except FileNotFoundError:
                logger.warning("No trained ML model found, using heuristic predictor")
                self.use_ml = False

    def refresh_hero_stats(self):
        """Fetch fresh hero statistics from API."""
        logger.info("Refreshing hero stats...")
        self.hero_stats = collect_hero_stats(self.client)
        save_data(self.hero_stats, os.path.join(config.MODEL_PATH, config.HERO_STATS_FILE))
        logger.info(f"Cached {len(self.hero_stats.get('heroes', {}))} heroes")

    def get_team_data(self, team_id: int, force_refresh: bool = False) -> dict:
        """Get team data, fetching from API if not cached.

        Uses a circuit breaker: after 3 consecutive API failures, stops
        trying the API and returns cached/default data instantly.
        """
        default = {"info": {"rating": 1200}, "matches": [], "heroes": [], "players": []}

        # Return cached data if available (ignore TTL when API is down)
        if not force_refresh and team_id in self.team_cache:
            cached_at = self._cache_timestamps.get(team_id, 0)
            # If API is healthy and cache is fresh, use it
            # If API is down (circuit open), use any cached data regardless of age
            if self._api_failures >= 3 or (time.time() - cached_at < self.CACHE_TTL):
                return self.team_cache[team_id]

        # Circuit breaker: don't attempt API if it's been failing
        if self._api_failures >= 3:
            if team_id in self.team_cache:
                return self.team_cache[team_id]
            return default

        logger.info(f"Fetching data for team {team_id}...")
        team_info = self.client.get_team(team_id)
        if not team_info:
            self._api_failures += 1
            if self._api_failures >= 3:
                logger.warning(
                    "API circuit breaker OPEN: too many failures. "
                    "Using cached data only."
                )
            # Return stale cache if available, otherwise default
            if team_id in self.team_cache:
                return self.team_cache[team_id]
            return default

        # API success — reset circuit breaker
        self._api_failures = 0

        matches = self.client.get_team_matches(team_id) or []
        heroes = self.client.get_team_heroes(team_id) or []
        players = self.client.get_team_players(team_id) or []

        data = {
            "info": team_info,
            "matches": matches[:config.TEAM_MATCH_HISTORY],
            "heroes": heroes,
            "players": players,
        }

        self.team_cache[team_id] = data
        self._cache_timestamps[team_id] = time.time()
        return data

    def save_caches(self):
        """Save cached data to disk."""
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        if self.hero_stats:
            save_data(self.hero_stats, os.path.join(config.MODEL_PATH, config.HERO_STATS_FILE))
        if self.team_cache:
            save_data(
                {str(k): v for k, v in self.team_cache.items()},
                os.path.join(config.MODEL_PATH, config.TEAM_CACHE_FILE),
            )

    def _load_teams_list(self) -> list[dict]:
        """Load the /teams list, trying multiple sources in order.

        Priority:
        1. Disk cache (teams_list.json) — fast, no API call
        2. OpenDota /teams API — fresh data, saves to disk
        3. Stale disk cache — if API fails but old file exists
        4. Build from team_cache — if API fails and no teams_list on disk
        """
        teams_path = os.path.join(config.MODEL_PATH, config.TEAMS_LIST_FILE)
        teams_list = []

        # 1. Try fresh disk cache first
        if os.path.exists(teams_path):
            cache_age = time.time() - os.path.getmtime(teams_path)
            if cache_age < config.TEAMS_LIST_TTL:
                try:
                    with open(teams_path, "r") as f:
                        teams_list = json.load(f)
                    logger.info(
                        f"Loaded {len(teams_list)} teams from disk cache "
                        f"(age: {cache_age / 3600:.1f}h)"
                    )
                    return teams_list
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Could not read teams cache: {e}")

        # 2. Fetch from API
        logger.info("Fetching teams list from OpenDota API...")
        teams_list = self.client.get_teams() or []

        if teams_list:
            # Save to disk for next time
            os.makedirs(config.MODEL_PATH, exist_ok=True)
            try:
                with open(teams_path, "w") as f:
                    json.dump(teams_list, f)
                logger.info(f"Cached {len(teams_list)} teams to disk")
            except OSError as e:
                logger.warning(f"Could not save teams cache: {e}")
            return teams_list

        # 3. API failed — try stale disk cache
        if os.path.exists(teams_path):
            try:
                with open(teams_path, "r") as f:
                    teams_list = json.load(f)
                logger.warning(
                    f"API failed, using stale disk cache ({len(teams_list)} teams)"
                )
                return teams_list
            except (json.JSONDecodeError, OSError):
                pass

        # 4. Last resort — build from team_cache (training data)
        if self.team_cache:
            teams_list = []
            for tid, data in self.team_cache.items():
                info = data.get("info", {})
                if info.get("name"):
                    teams_list.append({
                        "team_id": int(tid),
                        "name": info.get("name", ""),
                        "tag": info.get("tag", ""),
                        "rating": info.get("rating", 0),
                        "last_match_time": info.get("last_match_time"),
                    })
            if teams_list:
                logger.warning(
                    f"API unavailable, built teams list from team_cache "
                    f"({len(teams_list)} teams)"
                )
                # Save this so next startup doesn't need API either
                os.makedirs(config.MODEL_PATH, exist_ok=True)
                try:
                    with open(teams_path, "w") as f:
                        json.dump(teams_list, f)
                except OSError:
                    pass
                return teams_list

        logger.error(
            "No teams data available! Run 'python main.py refresh' "
            "when OpenDota rate limits reset."
        )
        return []

    def find_team_by_name(self, name: str) -> Optional[dict]:
        """Search for a team by name using the teams endpoint.

        Strategy: exact name/tag -> partial match -> fuzzy match (>= 80 score).
        The team list is loaded from disk cache (refreshed every 24h).
        Only considers active teams (played in last 6 months) for fuzzy matching
        to avoid matching defunct teams with similar names.
        """
        if not hasattr(self, "_teams_list") or self._teams_list is None:
            self._teams_list = self._load_teams_list()

        if not self._teams_list:
            return None

        name_lower = name.lower().strip()

        # 1. Exact match on name or tag
        for t in self._teams_list:
            if t.get("name", "").lower() == name_lower or t.get("tag", "").lower() == name_lower:
                return t

        # 2. Partial match (query is substring of team name or vice versa)
        for t in self._teams_list:
            t_name = t.get("name", "").lower()
            t_tag = t.get("tag", "").lower()
            if (name_lower in t_name or t_name in name_lower or
                    name_lower in t_tag or t_tag in name_lower):
                return t

        # 3. Fuzzy match (handles typos, abbreviations, slight name variations)
        try:
            from thefuzz import fuzz
        except ImportError:
            return None

        # Only consider teams with recent activity to avoid ghost matches
        import time as _time
        six_months_ago = _time.time() - 180 * 86400

        best_score = 0
        best_team = None
        for t in self._teams_list:
            # Skip teams with no recent matches
            last_match = t.get("last_match_time")
            if last_match and last_match < six_months_ago:
                continue

            t_name = t.get("name", "")
            if not t_name:
                continue

            # token_sort_ratio handles word reordering ("Team Spirit" vs "Spirit Team")
            score = fuzz.token_sort_ratio(name_lower, t_name.lower())
            if score > best_score:
                best_score = score
                best_team = t

        if best_score >= 80 and best_team:
            logger.debug(
                f"Fuzzy matched '{name}' -> '{best_team.get('name')}' (score={best_score})"
            )
            return best_team

        return None

    def predict_match(
        self,
        radiant_team_id: int,
        dire_team_id: int,
        match_data: Optional[dict] = None,
        series_type: int = 1,
        game_number: int = 1,
    ) -> dict:
        """
        Predict the outcome of a match between two teams.

        Args:
            radiant_team_id: OpenDota team ID for radiant
            dire_team_id: OpenDota team ID for dire
            match_data: Optional dict with picks_bans data for draft analysis
            series_type: 0=BO1, 1=BO3, 2=BO5, 3=BO2
            game_number: Game number in series

        Returns:
            Dictionary with prediction details.
        """
        # Ensure hero stats are loaded
        if not self.hero_stats:
            self.refresh_hero_stats()

        # Get team data
        rad_team = self.get_team_data(radiant_team_id)
        dire_team = self.get_team_data(dire_team_id)

        if not match_data:
            match_data = {}

        # Extract features
        features = extract_features(
            match_data=match_data,
            radiant_team=rad_team,
            dire_team=dire_team,
            hero_stats=self.hero_stats,
            radiant_team_id=radiant_team_id,
            dire_team_id=dire_team_id,
            series_type=series_type,
            game_number=game_number,
        )

        feature_names = get_feature_names()

        # Predict
        if self.use_ml and self.ml_model.is_trained:
            prediction, probability = self.ml_model.predict_single(features)
        else:
            prediction, probability = self.heuristic.predict(features, feature_names)

        # Build result
        rad_name = rad_team.get("info", {}).get("name", f"Team {radiant_team_id}")
        dire_name = dire_team.get("info", {}).get("name", f"Team {dire_team_id}")

        winner = rad_name if prediction == 1 else dire_name
        confidence = probability if prediction == 1 else (1 - probability)

        # Determine confidence level
        if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
            confidence_label = "HIGH"
        elif confidence >= config.LOW_CONFIDENCE_THRESHOLD:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        # Feature breakdown (use name lookup instead of fragile indices)
        f = dict(zip(feature_names, features))
        breakdown = {
            "team_ratings": {
                "radiant": rad_team.get("info", {}).get("rating", "N/A"),
                "dire": dire_team.get("info", {}).get("rating", "N/A"),
                "elo_win_prob": f"{f.get('elo_win_prob', 0.5):.1%}",
            },
            "recent_form": {
                "radiant_winrate": f"{f.get('rad_winrate', 0.5):.1%}",
                "dire_winrate": f"{f.get('dire_winrate', 0.5):.1%}",
                "radiant_momentum": f"{f.get('rad_momentum', 0.5):.1%}",
                "dire_momentum": f"{f.get('dire_momentum', 0.5):.1%}",
            },
            "h2h": {
                "games": int(f.get('h2h_games', 0) * 30),
                "radiant_winrate": f"{f.get('h2h_rad_winrate', 0.5):.1%}",
            },
        }

        result = {
            "radiant_team": rad_name,
            "dire_team": dire_name,
            "predicted_winner": winner,
            "win_probability": round(confidence, 4),
            "confidence": confidence_label,
            "radiant_win_prob": round(probability, 4),
            "dire_win_prob": round(1 - probability, 4),
            "model_type": "ML Ensemble" if self.use_ml else "Heuristic",
            "series_format": {0: "BO1", 1: "BO3", 2: "BO5", 3: "BO2"}.get(series_type, "BO3"),
            "breakdown": breakdown,
        }

        return result

    def predict_by_names(
        self,
        radiant_name: str,
        dire_name: str,
        series_type: int = 1,
    ) -> dict:
        """Predict match by team names."""
        rad_info = self.find_team_by_name(radiant_name)
        if not rad_info:
            return {"error": f"Team not found: {radiant_name}"}

        dire_info = self.find_team_by_name(dire_name)
        if not dire_info:
            return {"error": f"Team not found: {dire_name}"}

        return self.predict_match(
            radiant_team_id=rad_info["team_id"],
            dire_team_id=dire_info["team_id"],
            series_type=series_type,
        )

    def predict_live_matches(self) -> list[dict]:
        """Predict outcomes for all currently live pro matches."""
        live = self.client.get_live_matches()
        if not live:
            logger.info("No live matches found")
            return []

        predictions = []
        for match in live:
            # Only pro matches with team data
            rad_id = match.get("radiant_team", {}).get("team_id")
            dire_id = match.get("dire_team", {}).get("team_id")
            if not rad_id or not dire_id:
                continue

            try:
                match_data = {
                    "players": match.get("players", []),
                }
                pred = self.predict_match(
                    radiant_team_id=rad_id,
                    dire_team_id=dire_id,
                    match_data=match_data,
                )
                pred["match_id"] = match.get("match_id")
                pred["league"] = match.get("league", {}).get("name", "Unknown")
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting live match: {e}")

        return predictions

    def predict_today_matches(self, min_rating: int = 0, show_all: bool = False) -> list[dict]:
        """
        Predict matches scheduled for today.

        Sources (in order):
        1. OpenDota /live  — currently in-progress matches
        2. OpenDota /proMatches (paginated) — recently completed today
        3. Upcoming match API (Liquipedia-based) — scheduled matches

        Args:
            min_rating: Minimum team rating to include (both teams must meet).
                        0 means use config.MIN_TEAM_RATING.
            show_all: If True, ignore rating filter and show all matches.
        """
        from datetime import datetime, timezone, timedelta
        from upcoming_matches import fetch_upcoming_matches

        if not min_rating:
            min_rating = config.MIN_TEAM_RATING

        # Use UTC for all timestamp comparisons
        now_utc = datetime.now(timezone.utc)
        today_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end_utc = today_start_utc + timedelta(days=1)
        ts_start = int(today_start_utc.timestamp())
        ts_end = int(today_end_utc.timestamp())

        seen_pairs: set[tuple[int, int]] = set()
        match_list: list[dict] = []

        # ---- Source 1: Live matches ----
        logger.info("Fetching live matches...")
        live = self.client.get_live_matches() or []
        for m in live:
            rad = m.get("radiant_team", {})
            dire = m.get("dire_team", {})
            rad_id = rad.get("team_id")
            dire_id = dire.get("team_id")
            if not rad_id or not dire_id:
                continue
            pair = (min(rad_id, dire_id), max(rad_id, dire_id))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            match_list.append({
                "radiant_team_id": rad_id,
                "dire_team_id": dire_id,
                "radiant_name": rad.get("team_name", f"Team {rad_id}"),
                "dire_name": dire.get("team_name", f"Team {dire_id}"),
                "league": m.get("league", {}).get("name", "Unknown"),
                "status": "LIVE",
                "match_data": {"players": m.get("players", [])},
            })
        logger.info(f"  Live matches with team data: {len(match_list)}")

        # ---- Source 2: Pro matches (paginated, today only) ----
        logger.info("Fetching recent pro matches (paginated)...")
        pro_matches = self.client.get_pro_matches_paginated(pages=3)
        pro_added = 0
        for m in pro_matches:
            start = m.get("start_time", 0)
            if start < ts_start or start > ts_end:
                continue
            rad_id = m.get("radiant_team_id")
            dire_id = m.get("dire_team_id")
            if not rad_id or not dire_id:
                continue
            pair = (min(rad_id, dire_id), max(rad_id, dire_id))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            match_list.append({
                "radiant_team_id": rad_id,
                "dire_team_id": dire_id,
                "radiant_name": m.get("radiant_name", f"Team {rad_id}"),
                "dire_name": m.get("dire_name", f"Team {dire_id}"),
                "league": m.get("league_name", "Unknown"),
                "status": "TODAY",
                "match_data": {},
            })
            pro_added += 1
        logger.info(f"  Pro matches played today: {pro_added}")

        # ---- Source 3: Upcoming scheduled matches ----
        logger.info("Fetching upcoming scheduled matches...")
        upcoming = fetch_upcoming_matches()
        upcoming_added = 0
        for um in upcoming:
            team1_name = um["team1"]
            team2_name = um["team2"]

            # Resolve team names to OpenDota IDs
            t1_info = self.find_team_by_name(team1_name)
            t2_info = self.find_team_by_name(team2_name)

            if not t1_info or not t2_info:
                missing = []
                if not t1_info:
                    missing.append(team1_name)
                if not t2_info:
                    missing.append(team2_name)
                logger.info(
                    f"  Could not resolve: {team1_name} vs {team2_name} "
                    f"(missing: {', '.join(missing)})"
                )
                continue

            t1_id = t1_info["team_id"]
            t2_id = t2_info["team_id"]
            pair = (min(t1_id, t2_id), max(t1_id, t2_id))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            status = um.get("status", "UPCOMING")

            # Convert best_of (1,2,3,5) to series_type (0,1,2,3)
            best_of = um.get("best_of")
            bo_to_st = {1: 0, 2: 3, 3: 1, 5: 2}
            series_type = bo_to_st.get(best_of, 1)  # default BO3

            match_list.append({
                "radiant_team_id": t1_id,
                "dire_team_id": t2_id,
                "radiant_name": t1_info.get("name", team1_name),
                "dire_name": t2_info.get("name", team2_name),
                "league": um.get("league", "Unknown"),
                "status": status,
                "match_data": {},
                "_start_time": um.get("start_time", 0),
                "_series_type": series_type,
            })
            upcoming_added += 1
        logger.info(f"  Upcoming scheduled matches resolved: {upcoming_added}")
        logger.info(f"  Total matches before filtering: {len(match_list)}")

        # ---- Filter by team rating (unless --all) ----
        # Use ratings from _teams_list (already loaded, no API calls needed)
        if not show_all:
            # Build team_id -> rating lookup from teams_list and team_cache
            rating_lookup = {}
            for t in (self._teams_list or []):
                tid = t.get("team_id")
                if tid:
                    rating_lookup[tid] = t.get("rating") or 0
            # Also check team_cache for ratings not in teams_list
            for tid, data in self.team_cache.items():
                if tid not in rating_lookup:
                    rating_lookup[tid] = data.get("info", {}).get("rating", 0) or 0

            filtered = []
            for entry in match_list:
                rad_rating = rating_lookup.get(entry["radiant_team_id"], 0)
                dire_rating = rating_lookup.get(entry["dire_team_id"], 0)
                avg_rating = (rad_rating + dire_rating) / 2

                if avg_rating >= min_rating:
                    entry["_avg_rating"] = avg_rating
                    filtered.append(entry)
                else:
                    logger.debug(
                        f"Skipping {entry['radiant_name']} vs {entry['dire_name']} "
                        f"(avg rating {avg_rating:.0f} < {min_rating})"
                    )
            logger.info(
                f"  After rating filter (>= {min_rating}): "
                f"{len(filtered)}/{len(match_list)}"
            )
            match_list = filtered

        # ---- Sort: LIVE first, then by avg rating descending ----
        status_order = {"LIVE": 0, "UPCOMING": 1, "TODAY": 2}
        match_list.sort(
            key=lambda e: (
                status_order.get(e.get("status", ""), 9),
                -e.get("_avg_rating", 0),
            )
        )

        # ---- Predict each pair ----
        predictions = []
        for entry in match_list:
            try:
                pred = self.predict_match(
                    radiant_team_id=entry["radiant_team_id"],
                    dire_team_id=entry["dire_team_id"],
                    match_data=entry.get("match_data", {}),
                    series_type=entry.get("_series_type", 1),
                )
                pred["league"] = entry["league"]
                pred["status"] = entry["status"]
                pred["avg_team_rating"] = entry.get("_avg_rating", 0)
                if entry.get("_start_time"):
                    pred["start_time"] = entry["_start_time"]
                predictions.append(pred)
            except Exception as e:
                logger.error(
                    f"Error predicting {entry['radiant_name']} vs "
                    f"{entry['dire_name']}: {e}"
                )

        return predictions
