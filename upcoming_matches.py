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
    Parse Liquipedia Matches page HTML.

    Actual HTML structure (as of 2026):
      <div class="new-match-style">
        <div class="match-info">
          <span class="match-info-countdown">
            <span class="timer-object" data-timestamp="...">...</span>
          </span>
          <div class="match-info-header">
            <div class="match-info-header-opponent match-info-header-opponent-left">
              <div class="block-team flipped">
                <span class="team-template-image-icon ...">
                  <a href="..." title="Team Name">...</a>
                </span>
              </div>
            </div>
            ...right opponent same structure...
          </div>
        </div>
      </div>

    Tournament headers are in <div class="match-section-header">.
    """
    now = time.time()

    # ---- Split into individual match blocks ----
    # Each match is inside <div class="new-match-style">
    match_blocks = re.split(r'<div\s+class="new-match-style">', html)
    # First element is everything before the first match block
    preamble = match_blocks[0] if match_blocks else ""
    match_blocks = match_blocks[1:]  # Remove preamble

    logger.info(f"Liquipedia: found {len(match_blocks)} match blocks")

    if not match_blocks:
        return []

    # ---- Track current tournament from section headers ----
    # Section headers appear between match blocks in the full HTML.
    # Find all section headers with positions.
    section_headers = []
    for m in re.finditer(
        r'<div\s+class="match-section-header"[^>]*>.*?'
        r'<a[^>]*title="([^"]+)"',
        html,
        re.DOTALL,
    ):
        section_headers.append((m.start(), m.group(1)))

    # Also find positions of each match block in the original HTML
    block_positions = []
    search_start = 0
    for block in match_blocks:
        pos = html.find(block[:100], search_start)
        block_positions.append(pos if pos >= 0 else search_start)
        search_start = pos + 1 if pos >= 0 else search_start + 1

    # ---- Parse each match block ----
    matches = []

    for idx, block in enumerate(match_blocks):
        # Extract team names from <a> tags inside block-team divs
        # The title attribute of <a> inside team-template-image-icon has the team name
        teams = re.findall(
            r'<div\s+class="block-team[^"]*"[^>]*>.*?'
            r'<a[^>]*\btitle="([^"]+)"',
            block,
            re.DOTALL,
        )

        if len(teams) < 2:
            # Fallback: find any <a> with title inside the block
            # that's near a team-template class
            teams = re.findall(
                r'team-template-image-icon[^>]*>\s*'
                r'<a[^>]*\btitle="([^"]+)"',
                block,
                re.DOTALL,
            )

        if len(teams) < 2:
            continue

        team1 = teams[0]
        team2 = teams[1]

        # Skip if same team (shouldn't happen)
        if team1 == team2:
            continue

        # Extract timestamp
        ts_match = re.search(r'data-timestamp="(\d+)"', block)
        start_time = int(ts_match.group(1)) if ts_match else 0

        # Skip old matches (more than 6h ago)
        if start_time and start_time < now - 6 * 3600:
            continue

        # Determine current tournament from nearest section header before this block
        league = "Unknown"
        block_pos = block_positions[idx] if idx < len(block_positions) else 0
        for hdr_pos, hdr_name in reversed(section_headers):
            if hdr_pos < block_pos:
                league = hdr_name
                break

        # If no section header found, try to find tournament link inside block
        if league == "Unknown":
            tourn_match = re.search(
                r'<a[^>]*title="([^"]+(?:League|Major|Minor|Championship|'
                r'Tournament|Cup|Series|DPC|Masters|Division|Season|'
                r'Qualifier|Invitational|Circuit|Pro)[^"]*)"',
                block,
                re.IGNORECASE,
            )
            if tourn_match:
                league = tourn_match.group(1)

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
