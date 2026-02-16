"""
Fetch upcoming/scheduled Dota 2 professional matches from multiple sources.

Primary source: dota-matches-api (Liquipedia-based, free, no auth required)
Fallback: OpenDota proMatches with expanded pagination
"""

import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# --- Source 1: dota-matches-api (Liquipedia scraper by beeequeue) ---
DOTA_MATCHES_API_URL = "https://dota.haglund.dev/v1/matches"

# --- Source 2: Liquipedia MediaWiki API ---
LIQUIPEDIA_API_URL = "https://liquipedia.net/dota2/api.php"
LIQUIPEDIA_USER_AGENT = (
    "Dota2PredTool/1.0 (https://github.com/dota2pred; personal-use)"
)

# Rate limiting
_last_request_time = 0
_REQUEST_DELAY = 2.0


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _REQUEST_DELAY:
        time.sleep(_REQUEST_DELAY - elapsed)
    _last_request_time = time.time()


def fetch_upcoming_matches() -> list[dict]:
    """
    Fetch upcoming Dota 2 matches from available sources.

    Returns list of dicts:
        {
            "team1": str,          # Team name
            "team2": str,          # Team name
            "league": str,         # Tournament/league name
            "start_time": int,     # Unix timestamp (0 if unknown)
            "status": "UPCOMING" | "LIVE",
        }
    """
    # Try primary source first
    matches = _fetch_from_dota_matches_api()
    if matches:
        logger.info(f"Found {len(matches)} upcoming matches from dota-matches-api")
        return matches

    # Try Liquipedia match ticker as fallback
    matches = _fetch_from_liquipedia()
    if matches:
        logger.info(f"Found {len(matches)} upcoming matches from Liquipedia")
        return matches

    logger.warning("No upcoming match sources available")
    return []


def _fetch_from_dota_matches_api() -> list[dict]:
    """
    Fetch from dota-matches-api (https://dota.haglund.dev/v1/matches).
    Returns structured match data from Liquipedia.
    """
    _rate_limit()
    try:
        resp = requests.get(
            DOTA_MATCHES_API_URL,
            headers={"User-Agent": "Dota2PredTool/1.0"},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"dota-matches-api returned {resp.status_code}")
            return []

        data = resp.json()
        if not isinstance(data, list):
            logger.warning("Unexpected response format from dota-matches-api")
            return []

        matches = []
        now = time.time()
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        tomorrow_end = today_start + timedelta(days=2)

        for entry in data:
            teams = entry.get("teams", [None, None])
            if not teams or len(teams) < 2:
                continue

            team1 = teams[0]
            team2 = teams[1]
            if not team1 or not team2:
                continue

            team1_name = team1.get("name")
            team2_name = team2.get("name")
            if not team1_name or not team2_name:
                continue

            # Parse start time
            starts_at = entry.get("startsAt")
            start_time = 0
            if starts_at:
                try:
                    dt = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
                    start_time = int(dt.timestamp())
                except (ValueError, TypeError):
                    pass

            # Filter: only today's and upcoming matches (within ~48h)
            if start_time:
                match_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
                if match_dt > tomorrow_end:
                    continue
                # Skip matches that ended long ago (more than 6h ago)
                if start_time < now - 6 * 3600:
                    continue

            # Determine status
            if start_time and start_time <= now:
                status = "LIVE"
            else:
                status = "UPCOMING"

            league_name = entry.get("leagueName") or "Unknown"

            matches.append({
                "team1": team1_name,
                "team2": team2_name,
                "league": league_name,
                "start_time": start_time,
                "status": status,
            })

        return matches

    except requests.RequestException as e:
        logger.warning(f"dota-matches-api request failed: {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.warning(f"dota-matches-api parse error: {e}")
        return []


def _fetch_from_liquipedia() -> list[dict]:
    """
    Fetch upcoming matches from Liquipedia's match ticker page
    via the MediaWiki API.
    """
    _rate_limit()
    try:
        headers = {
            "User-Agent": LIQUIPEDIA_USER_AGENT,
            "Accept-Encoding": "gzip",
        }
        params = {
            "action": "parse",
            "page": "Liquipedia:Upcoming_and_ongoing_matches",
            "format": "json",
            "prop": "text",
        }

        resp = requests.get(
            LIQUIPEDIA_API_URL,
            params=params,
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"Liquipedia API returned {resp.status_code}")
            return []

        data = resp.json()
        html = data.get("parse", {}).get("text", {}).get("*", "")
        if not html:
            return []

        return _parse_liquipedia_ticker(html)

    except requests.RequestException as e:
        logger.warning(f"Liquipedia request failed: {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.warning(f"Liquipedia parse error: {e}")
        return []


def _parse_liquipedia_ticker(html: str) -> list[dict]:
    """
    Parse Liquipedia match ticker HTML to extract team names,
    timestamps, and tournament info.
    """
    matches = []
    now = time.time()

    # Split into table rows
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)

    for row in rows:
        # Extract team names from team-template-text spans
        teams = re.findall(
            r'<span[^>]*class="[^"]*team-template-text[^"]*"[^>]*>'
            r'\s*<a[^>]*title="([^"]+)"',
            row,
            re.DOTALL,
        )
        if len(teams) < 2:
            continue

        # Extract timestamp
        ts_match = re.search(r'data-timestamp="(\d+)"', row)
        start_time = int(ts_match.group(1)) if ts_match else 0

        # Skip old matches (more than 6h ago)
        if start_time and start_time < now - 6 * 3600:
            continue

        # Extract tournament name from links
        league = "Unknown"
        # Try tournament-specific patterns first
        tourn_match = re.search(
            r'<a[^>]*title="([^"]*(?:Major|Minor|League|Championship|'
            r"Tournament|Cup|Series|DPC|Masters|Open|Qualifier|"
            r'Invitational|Division|Season|Circuit)[^"]*)"',
            row,
            re.IGNORECASE,
        )
        if tourn_match:
            league = tourn_match.group(1)
        else:
            # Try any non-team link
            all_links = re.findall(r'<a[^>]*title="([^"]+)"', row)
            for link in all_links:
                if link not in teams and not link.startswith("Special:"):
                    league = link
                    break

        # Determine status
        if start_time and start_time <= now:
            status = "LIVE"
        else:
            status = "UPCOMING"

        matches.append({
            "team1": teams[0],
            "team2": teams[1],
            "league": league,
            "start_time": start_time,
            "status": status,
        })

    logger.info(f"Parsed {len(matches)} matches from Liquipedia ticker")
    return matches
