"""
Configuration for Dota 2 Prediction Tool.
"""

import os

# OpenDota API
OPENDOTA_BASE_URL = "https://api.opendota.com/api"
OPENDOTA_API_KEY = None  # Set your API key here or via environment variable

# Request settings
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 2.0  # seconds between requests (safe for OpenDota free tier: 30 req/min)
MAX_RETRIES = 3

# Data collection settings
PRO_MATCHES_LIMIT = 1500  # Number of recent pro matches to fetch for training
TEAM_MATCH_HISTORY = 50  # Last N matches per team for form calculation
H2H_LOOKBACK = 18  # Number of months to look back for head-to-head

# Feature weights (tuned empirically, draft removed for pre-match focus)
FEATURE_GROUPS = {
    "team_rating": 0.30,       # OpenDota Elo-like rating
    "recent_form": 0.25,       # Win rate in last N matches
    "head_to_head": 0.10,      # Historical matchup record
    "player_strength": 0.15,   # Individual player metrics
    "map_side": 0.03,          # Radiant/Dire advantage
    "match_format": 0.05,      # BO1/BO3/BO5
    "roster_stability": 0.10,  # How stable the roster is
    "meta": 0.02,              # Patch/meta recency
}

# Hero data
TOTAL_HEROES = 145  # Approximate number of Dota 2 heroes (updated 2026)

# Model settings
MODEL_PATH = "model_data/"
TRAINED_MODEL_FILE = "dota2_predictor.joblib"
SCALER_FILE = "feature_scaler.joblib"
HERO_STATS_FILE = "hero_stats.json"
TEAM_CACHE_FILE = "team_cache.json"
TEAMS_LIST_FILE = "teams_list.json"
TEAMS_LIST_TTL = 86400  # Refresh teams list every 24 hours

# Database (prediction tracking)
DB_PATH = os.path.join(MODEL_PATH, "predictions.db")

# Prediction thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.65  # Above this = high confidence prediction
LOW_CONFIDENCE_THRESHOLD = 0.55   # Below this = skip/low confidence

# Team tier filtering for "today" command
MIN_TEAM_RATING = 1100  # Minimum team rating to include in today's predictions

# Patch-aware decay
PATCH_HALF_LIFE_DAYS = 45  # How quickly old data loses relevance after patches

# Hyperparameter tuning
OPTUNA_N_TRIALS = 50  # Number of Optuna trials for hyperparameter search
OPTUNA_TIMEOUT = 300  # Timeout in seconds for tuning
