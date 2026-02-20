"""
Flask web application for Dota 2 Match Prediction (pre-match only).
"""

import json
import logging
import os
import time
from collections import defaultdict

from flask import Flask, render_template, request, jsonify

from predictor import MatchPredictor
from prediction_tracker import PredictionTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Lazy-loaded predictor
_predictor = None

# Simple in-memory rate limiter
_rate_limits = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30     # max requests per window


def get_predictor() -> MatchPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MatchPredictor(use_ml_model=True)
    return _predictor


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate limited."""
    now = time.time()
    times = _rate_limits[ip]
    # Remove old entries
    _rate_limits[ip] = [t for t in times if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limits[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limits[ip].append(now)
    return True


@app.before_request
def rate_limit_check():
    ip = request.remote_addr or "unknown"
    if request.path.startswith("/api/") and not _check_rate_limit(ip):
        return jsonify({"error": "Rate limited. Try again in a minute."}), 429


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict a match outcome (pre-match)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    radiant = data.get("radiant", "").strip()
    dire = data.get("dire", "").strip()
    series_format = data.get("format", "BO3").upper()

    if not radiant or not dire:
        return jsonify({"error": "Both team names are required"}), 400

    format_map = {"BO1": 0, "BO2": 3, "BO3": 1, "BO5": 2}
    series_type = format_map.get(series_format, 1)

    predictor = get_predictor()
    result = predictor.predict_by_names(radiant, dire, series_type=series_type)
    predictor.save_caches()

    if "error" in result:
        return jsonify(result), 404

    return jsonify(result)


@app.route("/api/team", methods=["GET"])
def api_team():
    """Look up a team by name."""
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "Team name is required"}), 400

    predictor = get_predictor()
    info = predictor.find_team_by_name(name)

    if not info:
        return jsonify({"error": f"Team not found: {name}"}), 404

    total = (info.get("wins") or 0) + (info.get("losses") or 0)
    winrate = (info.get("wins") or 0) / total if total > 0 else 0

    return jsonify({
        "name": info.get("name"),
        "tag": info.get("tag"),
        "team_id": info.get("team_id"),
        "rating": info.get("rating"),
        "wins": info.get("wins"),
        "losses": info.get("losses"),
        "winrate": round(winrate, 4),
    })


@app.route("/api/today", methods=["GET"])
def api_today():
    """Predict all of today's matches (pre-match)."""
    show_all = request.args.get("all", "").lower() in ("1", "true", "yes")
    min_rating = int(request.args.get("min_rating", 0))
    predictor = get_predictor()
    predictions = predictor.predict_today_matches(
        min_rating=min_rating,
        show_all=show_all,
    )
    predictor.save_caches()
    return jsonify(predictions)


@app.route("/api/accuracy", methods=["GET"])
def api_accuracy():
    """Get prediction accuracy statistics."""
    days = int(request.args.get("days", 30))
    tracker = PredictionTracker()
    stats = tracker.get_accuracy_stats(days=days)
    drift = tracker.check_drift()
    return jsonify({
        "stats": stats,
        "drift": {
            "detected": drift["drift_detected"],
            "reason": drift["reason"],
        },
        "recent_predictions": tracker.get_recent_predictions(limit=20),
        "model_history": tracker.get_model_history(limit=5),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")
    app.run(host="0.0.0.0", port=port, debug=debug)
