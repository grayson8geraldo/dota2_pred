"""
Data collection module for Dota 2 match statistics.
Uses OpenDota API as the primary data source.
"""

import json
import os
import time
import logging
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


class OpenDotaClient:
    """Client for OpenDota API."""

    def __init__(self, api_key: Optional[str] = None):
        self.base_url = config.OPENDOTA_BASE_URL
        self.api_key = api_key or config.OPENDOTA_API_KEY or os.environ.get("OPENDOTA_API_KEY")
        self.session = requests.Session()
        self._last_request_time = 0

    def _rate_limit(self):
        """Respect API rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < config.REQUEST_DELAY:
            time.sleep(config.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict | list]:
        """Make a GET request to the OpenDota API."""
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error {response.status_code}: {response.text[:200]}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))

        return None

    # ---- Pro Matches ----

    def get_pro_matches(self, less_than_match_id: Optional[int] = None) -> list[dict]:
        """Fetch recent professional matches."""
        params = {}
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        result = self._get("/proMatches", params)
        return result if result else []

    def get_pro_matches_paginated(self, pages: int = 3) -> list[dict]:
        """Fetch multiple pages of recent pro matches for broader coverage.

        Each page returns ~100 matches. Default 3 pages = ~300 matches,
        covering roughly the last 2-3 days of pro matches.
        """
        all_matches = []
        last_match_id = None

        for page in range(pages):
            batch = self.get_pro_matches(less_than_match_id=last_match_id)
            if not batch:
                break
            all_matches.extend(batch)
            # Get the smallest match_id to paginate backwards
            last_match_id = min(m.get("match_id", float("inf")) for m in batch)

        return all_matches

    def get_match_details(self, match_id: int) -> Optional[dict]:
        """Fetch detailed match data."""
        return self._get(f"/matches/{match_id}")

    # ---- Teams ----

    def get_teams(self) -> list[dict]:
        """Fetch all teams."""
        result = self._get("/teams")
        return result if result else []

    def get_team(self, team_id: int) -> Optional[dict]:
        """Fetch team info."""
        return self._get(f"/teams/{team_id}")

    def get_team_matches(self, team_id: int) -> list[dict]:
        """Fetch team match history."""
        result = self._get(f"/teams/{team_id}/matches")
        return result if result else []

    def get_team_heroes(self, team_id: int) -> list[dict]:
        """Fetch team hero performance."""
        result = self._get(f"/teams/{team_id}/heroes")
        return result if result else []

    def get_team_players(self, team_id: int) -> list[dict]:
        """Fetch team player history."""
        result = self._get(f"/teams/{team_id}/players")
        return result if result else []

    # ---- Heroes ----

    def get_heroes(self) -> list[dict]:
        """Fetch all heroes."""
        result = self._get("/heroes")
        return result if result else []

    def get_hero_stats(self) -> list[dict]:
        """Fetch aggregate hero statistics."""
        result = self._get("/heroStats")
        return result if result else []

    def get_hero_matchups(self, hero_id: int) -> list[dict]:
        """Fetch hero matchup win rates."""
        result = self._get(f"/heroes/{hero_id}/matchups")
        return result if result else []

    # ---- Live ----

    def get_live_matches(self) -> list[dict]:
        """Fetch currently live matches."""
        result = self._get("/live")
        return result if result else []

    # ---- SQL Explorer ----

    def explorer_query(self, sql: str) -> Optional[dict]:
        """Run a SQL query on OpenDota's database."""
        return self._get("/explorer", params={"sql": sql})


def collect_pro_matches(client: OpenDotaClient, count: int = 500) -> list[dict]:
    """
    Collect recent pro matches with full details.
    Returns list of match detail dicts.
    """
    logger.info(f"Collecting {count} pro matches...")
    pro_matches = []
    last_match_id = None

    while len(pro_matches) < count:
        batch = client.get_pro_matches(less_than_match_id=last_match_id)
        if not batch:
            break
        pro_matches.extend(batch)
        last_match_id = batch[-1]["match_id"]
        logger.info(f"  Collected {len(pro_matches)} pro match summaries...")

    pro_matches = pro_matches[:count]

    # Fetch detailed data for matches that have both teams
    detailed = []
    for i, m in enumerate(pro_matches):
        if not m.get("radiant_team_id") or not m.get("dire_team_id"):
            continue
        details = client.get_match_details(m["match_id"])
        if details and details.get("radiant_win") is not None:
            details["radiant_team_id"] = m.get("radiant_team_id")
            details["dire_team_id"] = m.get("dire_team_id")
            details["league_name"] = m.get("league_name", "")
            detailed.append(details)
            if (i + 1) % 20 == 0:
                logger.info(f"  Fetched details for {len(detailed)} matches...")

    logger.info(f"Collected {len(detailed)} detailed pro matches")
    return detailed


def collect_team_data(client: OpenDotaClient, team_ids: list[int]) -> dict:
    """
    Collect data for specified teams.
    Returns dict mapping team_id -> team data with matches and heroes.
    """
    teams_data = {}
    for team_id in team_ids:
        logger.info(f"Collecting data for team {team_id}...")
        team_info = client.get_team(team_id)
        if not team_info:
            continue

        matches = client.get_team_matches(team_id)
        heroes = client.get_team_heroes(team_id)
        players = client.get_team_players(team_id)

        teams_data[team_id] = {
            "info": team_info,
            "matches": matches[:config.TEAM_MATCH_HISTORY],
            "heroes": heroes,
            "players": players,
        }

    return teams_data


def collect_hero_stats(client: OpenDotaClient) -> dict:
    """Collect hero statistics and matchup data."""
    heroes = client.get_heroes()
    hero_stats = client.get_hero_stats()

    hero_data = {}
    for h in hero_stats:
        hero_id = h["id"]
        pro_pick = h.get("pro_pick", 0)
        pro_win = h.get("pro_win", 0)
        pro_ban = h.get("pro_ban", 0)
        hero_data[hero_id] = {
            "id": hero_id,
            "name": h.get("localized_name", f"hero_{hero_id}"),
            "pro_pick": pro_pick,
            "pro_win": pro_win,
            "pro_ban": pro_ban,
            "pro_winrate": pro_win / pro_pick if pro_pick > 0 else 0.5,
            "primary_attr": h.get("primary_attr", ""),
            "attack_type": h.get("attack_type", ""),
            "roles": h.get("roles", []),
        }

    # Build hero name -> id mapping
    hero_name_map = {}
    for h in heroes:
        hero_name_map[h["localized_name"].lower()] = h["id"]
        hero_name_map[str(h["id"])] = h["id"]

    return {"heroes": hero_data, "name_map": hero_name_map}


def save_data(data: dict, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_data(filepath: str) -> Optional[dict]:
    """Load data from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None
