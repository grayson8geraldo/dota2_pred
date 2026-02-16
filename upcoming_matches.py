"""
Fetch upcoming/scheduled Dota 2 professional matches from multiple sources.

Sources tried in order:
1. dota-matches-api (Liquipedia-based, free, no auth)
2. Liquipedia MediaWiki API (match ticker page)
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
# Pages to try (the name may change over time)
LIQUIPEDIA_MATCH_PAGES = [
    "Liquipedia:Matches",
    "Liquipedia:Upcoming_and_ongoing_matches",
]

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
    Returns structured match data scraped from Liquipedia.
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

        logger.info(f"dota-matches-api returned {len(data)} items")
        if not data:
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
    via the MediaWiki API.  Tries multiple known page names
    and follows redirects.
    """
    for page_name in LIQUIPEDIA_MATCH_PAGES:
        _rate_limit()
        try:
            headers = {
                "User-Agent": LIQUIPEDIA_USER_AGENT,
                "Accept-Encoding": "gzip",
            }
            params = {
                "action": "parse",
                "page": page_name,
                "format": "json",
                "prop": "text",
                "redirects": "true",  # Auto-follow redirects
            }

            resp = requests.get(
                LIQUIPEDIA_API_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"Liquipedia API returned {resp.status_code} for {page_name}"
                )
                continue

            data = resp.json()

            # Check for API error
            if "error" in data:
                logger.warning(
                    f"Liquipedia API error for {page_name}: "
                    f"{data['error'].get('info', 'unknown')}"
                )
                continue

            html = data.get("parse", {}).get("text", {}).get("*", "")
            if not html:
                logger.warning(f"Empty HTML from Liquipedia for {page_name}")
                continue

            # Check if the response is just a redirect stub
            if "redirectMsg" in html and len(html) < 2000:
                logger.info(f"Page {page_name} is a redirect, trying next...")
                continue

            logger.info(
                f"Liquipedia HTML from {page_name}: {len(html)} chars"
            )

            matches = _parse_liquipedia_ticker(html)
            if matches:
                return matches

            logger.info(f"No matches parsed from {page_name}")

        except requests.RequestException as e:
            logger.warning(f"Liquipedia request failed for {page_name}: {e}")
        except (ValueError, KeyError) as e:
            logger.warning(f"Liquipedia parse error for {page_name}: {e}")

    return []


def _parse_liquipedia_ticker(html: str) -> list[dict]:
    """
    Parse Liquipedia match ticker HTML.

    Tries multiple strategies since the HTML structure can vary:
    1. Look for team-template-text in any element (span, a, div)
    2. Pair consecutive team names as opponents
    3. Find nearest timestamp for each pair
    """
    now = time.time()

    # ---- Strategy 1: Find all team names ----
    # team-template-text can appear in <span>, <a>, or other elements
    # Pattern: class containing "team-template-text" followed by an <a> with title
    team_pattern_a = re.compile(
        r'class="[^"]*team-template-text[^"]*"[^>]*>'
        r'[^<]*<a[^>]*title="([^"]+)"',
        re.DOTALL,
    )
    # Alternative: team-template-text as a direct <a> class
    team_pattern_b = re.compile(
        r'<a[^>]*class="[^"]*team-template-text[^"]*"[^>]*title="([^"]+)"',
        re.DOTALL,
    )
    # Alternative: team name in a span with data-highlightingclass
    team_pattern_c = re.compile(
        r'data-highlightingclass="([^"]+)"',
        re.DOTALL,
    )

    # Collect team names with their positions in HTML
    team_occurrences = []  # list of (position, name)

    for m in team_pattern_a.finditer(html):
        team_occurrences.append((m.start(), m.group(1)))
    for m in team_pattern_b.finditer(html):
        team_occurrences.append((m.start(), m.group(1)))

    # If we found no teams with patterns a/b, try pattern c
    if not team_occurrences:
        for m in team_pattern_c.finditer(html):
            team_occurrences.append((m.start(), m.group(1)))

    # Sort by position in HTML
    team_occurrences.sort(key=lambda x: x[0])

    logger.info(f"Liquipedia: found {len(team_occurrences)} team name occurrences")

    if len(team_occurrences) < 2:
        # Last resort: try to find team names in any <a> with title inside
        # match-related containers
        generic_teams = re.findall(
            r'class="[^"]*(?:team|opponent)[^"]*"[^>]*>.*?'
            r'<a[^>]*title="([^"]+)"',
            html,
            re.DOTALL,
        )
        logger.info(
            f"Liquipedia fallback: found {len(generic_teams)} team names"
        )
        for i, name in enumerate(generic_teams):
            team_occurrences.append((i * 1000, name))

    if len(team_occurrences) < 2:
        return []

    # ---- Find all timestamps ----
    timestamps = []  # list of (position, unix_timestamp)
    for m in re.finditer(r'data-timestamp="(\d+)"', html):
        timestamps.append((m.start(), int(m.group(1))))

    # ---- Find all tournament/league names ----
    tournaments = []  # list of (position, name)
    for m in re.finditer(
        r'<a[^>]*title="([^"]+)"[^>]*>[^<]*</a>',
        html,
    ):
        name = m.group(1)
        # Filter out team names and special pages
        if name.startswith("Special:") or name.startswith("Category:"):
            continue
        # Check if this looks like a tournament
        tournaments.append((m.start(), name))

    # ---- Pair teams into matches ----
    matches = []
    team_names_set = {name for _, name in team_occurrences}

    i = 0
    while i < len(team_occurrences) - 1:
        pos1, team1 = team_occurrences[i]
        pos2, team2 = team_occurrences[i + 1]

        # Skip if same team appears twice (might be a header/logo)
        if team1 == team2:
            i += 1
            continue

        i += 2  # Move to next pair

        # Find nearest timestamp (between the two team positions or shortly after)
        start_time = 0
        search_start = pos1
        search_end = pos2 + 2000  # Look up to 2000 chars after team2
        for ts_pos, ts_val in timestamps:
            if search_start <= ts_pos <= search_end:
                start_time = ts_val
                break

        # Skip old matches (more than 6h ago)
        if start_time and start_time < now - 6 * 3600:
            continue

        # Find nearest tournament name (before team1, within 3000 chars)
        league = "Unknown"
        best_dist = float("inf")
        for t_pos, t_name in tournaments:
            if t_name in team_names_set:
                continue  # Skip team names
            dist = pos1 - t_pos
            if 0 < dist < 3000 and dist < best_dist:
                best_dist = dist
                league = t_name

        # If no tournament found before, look after
        if league == "Unknown":
            for t_pos, t_name in tournaments:
                if t_name in team_names_set:
                    continue
                dist = t_pos - pos1
                if 0 < dist < 2000 and dist < best_dist:
                    best_dist = dist
                    league = t_name

        # Determine status
        if start_time and start_time <= now:
            status = "LIVE"
        else:
            status = "UPCOMING"

        matches.append({
            "team1": team1,
            "team2": team2,
            "league": league,
            "start_time": start_time,
            "status": status,
        })

    logger.info(f"Parsed {len(matches)} matches from Liquipedia ticker")
    return matches
