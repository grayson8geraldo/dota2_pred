"""
Flask web application for Dota 2 Match Prediction.
"""

import json
import logging
import os

from flask import Flask, render_template, request, jsonify

from predictor import MatchPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Lazy-loaded predictor
_predictor = None


def get_predictor() -> MatchPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MatchPredictor(use_ml_model=True)
    return _predictor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict a match outcome."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    radiant = data.get("radiant", "").strip()
    dire = data.get("dire", "").strip()
    series_format = data.get("format", "BO3").upper()

    if not radiant or not dire:
        return jsonify({"error": "Both team names are required"}), 400

    format_map = {"BO1": 0, "BO3": 1, "BO5": 2}
    series_type = format_map.get(series_format, 1)

    predictor = get_predictor()
    result = predictor.predict_by_names(radiant, dire, series_type=series_type)
    predictor.save_caches()

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


@app.route("/api/live", methods=["GET"])
def api_live():
    """Predict live matches."""
    predictor = get_predictor()
    predictions = predictor.predict_live_matches()
    predictor.save_caches()
    return jsonify(predictions)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
