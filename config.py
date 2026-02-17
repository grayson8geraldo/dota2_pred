"""
Configuration for Dota 2 Prediction Tool.
"""

# OpenDota API
OPENDOTA_BASE_URL = "https://api.opendota.com/api"
OPENDOTA_API_KEY = None  # Set your API key here or via environment variable

# Request settings
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 1.2  # seconds between requests (OpenDota: 60/min free, 1200/min with key)
MAX_RETRIES = 3

# Data collection settings
PRO_MATCHES_LIMIT = 1500  # Number of recent pro matches to fetch for training
TEAM_MATCH_HISTORY = 50  # Last N matches per team for form calculation
H2H_LOOKBACK = 18  # Number of months to look back for head-to-head

# Feature weights (tuned empirically)
# These represent relative importance of each factor group
FEATURE_GROUPS = {
    "team_rating": 0.25,       # OpenDota Elo-like rating
    "recent_form": 0.20,       # Win rate in last N matches
    "head_to_head": 0.10,      # Historical matchup record
    "hero_draft": 0.25,        # Draft composition analysis
    "map_side": 0.05,          # Radiant/Dire advantage
    "match_format": 0.05,      # BO1/BO3/BO5
    "roster_stability": 0.10,  # How stable the roster is
}

# Hero data
TOTAL_HEROES = 145  # Approximate number of Dota 2 heroes (updated 2026)

# Model settings
MODEL_PATH = "model_data/"
TRAINED_MODEL_FILE = "dota2_predictor.joblib"
SCALER_FILE = "feature_scaler.joblib"
HERO_STATS_FILE = "hero_stats.json"
TEAM_CACHE_FILE = "team_cache.json"

# Prediction thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.65  # Above this = high confidence prediction
LOW_CONFIDENCE_THRESHOLD = 0.55   # Below this = skip/low confidence

# Team tier filtering for "today" command
MIN_TEAM_RATING = 1100  # Minimum team rating to include in today's predictions
# Teams below this are typically tier-3+ amateur teams
# Tier 1 teams: ~1400+, Tier 2: ~1200-1400, Tier 3: ~1100-1200
