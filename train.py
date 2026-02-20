"""
Training pipeline for the Dota 2 prediction model.

Collects pro match data, extracts features, trains the ensemble model
with temporal validation, and evaluates performance.
"""

import os
import json
import logging
import time

import numpy as np

import config
from data_collector import (
    OpenDotaClient,
    collect_pro_matches,
    collect_team_data,
    collect_hero_stats,
    save_data,
    load_data,
)
from feature_engine import extract_features, get_feature_names, N_FEATURES
from model import Dota2Predictor
from prediction_tracker import PredictionTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_training_data(
    matches: list[dict],
    teams_data: dict,
    hero_stats: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix, label vector, and timestamps from pro match data.

    Returns:
        (X, y, timestamps) where timestamps are start_time for each sample.
    """
    X_list = []
    y_list = []
    ts_list = []
    skipped = 0

    for match in matches:
        try:
            rad_id = match.get("radiant_team_id")
            dire_id = match.get("dire_team_id")
            radiant_win = match.get("radiant_win")

            if not rad_id or not dire_id or radiant_win is None:
                skipped += 1
                continue

            rad_team = teams_data.get(rad_id, {
                "info": {"rating": 1200}, "matches": [], "heroes": [], "players": [],
            })
            dire_team = teams_data.get(dire_id, {
                "info": {"rating": 1200}, "matches": [], "heroes": [], "players": [],
            })

            series_type = match.get("series_type", 1)

            features = extract_features(
                match_data=match,
                radiant_team=rad_team,
                dire_team=dire_team,
                hero_stats=hero_stats,
                radiant_team_id=rad_id,
                dire_team_id=dire_id,
                series_type=series_type,
            )

            if features.shape[0] != N_FEATURES:
                skipped += 1
                continue

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            X_list.append(features)
            y_list.append(1 if radiant_win else 0)
            ts_list.append(match.get("start_time", 0))

        except Exception as e:
            logger.warning(f"Error processing match {match.get('match_id', '?')}: {e}")
            skipped += 1

    logger.info(f"Built {len(X_list)} training samples, skipped {skipped}")

    if not X_list:
        return np.empty((0, N_FEATURES)), np.empty(0), np.empty(0)

    return np.array(X_list), np.array(y_list), np.array(ts_list)


def train_model(n_matches: int = config.PRO_MATCHES_LIMIT,
                tune: bool = False):
    """
    Full training pipeline:
    1. Collect pro match data
    2. Collect team data for all teams in matches
    3. Collect hero stats
    4. Extract features (sorted by time)
    5. Train model with temporal validation and sample weights
    6. Save model, caches, and metrics
    """
    client = OpenDotaClient()
    os.makedirs(config.MODEL_PATH, exist_ok=True)

    # Step 1: Collect match data (incremental)
    matches_file = os.path.join(config.MODEL_PATH, "training_matches.json")
    cached = load_data(matches_file)
    if cached:
        logger.info(f"Found cached training data ({len(cached)} matches)")
        existing_ids = {m.get("match_id") for m in cached if m.get("match_id")}

        need = max(0, n_matches - len(cached))
        if need > 0 or len(cached) < n_matches * 0.8:
            fetch_count = max(need, n_matches // 3)
            logger.info(
                f"Fetching {fetch_count} new matches to supplement "
                f"{len(cached)} cached..."
            )
            new_matches = collect_pro_matches(client, fetch_count)
            added = 0
            for m in new_matches:
                mid = m.get("match_id")
                if mid and mid not in existing_ids:
                    cached.append(m)
                    existing_ids.add(mid)
                    added += 1
            logger.info(f"  Added {added} new unique matches (total: {len(cached)})")
            cached.sort(key=lambda m: m.get("match_id", 0), reverse=True)
            cached = cached[:n_matches]
            save_data(cached, matches_file)
        matches = cached
    else:
        logger.info(f"Collecting {n_matches} pro matches from OpenDota...")
        matches = collect_pro_matches(client, n_matches)
        if matches:
            save_data(matches, matches_file)
            logger.info(f"Saved {len(matches)} matches to cache")

    if not matches:
        logger.error("No match data collected. Check API connectivity.")
        return None

    # Step 2: Collect team data
    team_ids = set()
    for m in matches:
        if m.get("radiant_team_id"):
            team_ids.add(m["radiant_team_id"])
        if m.get("dire_team_id"):
            team_ids.add(m["dire_team_id"])

    logger.info(f"Found {len(team_ids)} unique teams")

    teams_file = os.path.join(config.MODEL_PATH, "teams_data.json")
    teams_data = load_data(teams_file)
    if teams_data:
        teams_data = {int(k): v for k, v in teams_data.items()}
        logger.info(f"Loaded cached team data ({len(teams_data)} teams)")
    else:
        teams_data = {}

    missing_teams = [tid for tid in team_ids if tid not in teams_data]
    if missing_teams:
        logger.info(f"Fetching data for {len(missing_teams)} teams...")
        new_data = collect_team_data(client, missing_teams[:300])
        teams_data.update(new_data)
        save_data({str(k): v for k, v in teams_data.items()}, teams_file)

    # Step 3: Hero stats (still used for team hero pool analysis)
    hero_stats_file = os.path.join(config.MODEL_PATH, config.HERO_STATS_FILE)
    hero_stats = load_data(hero_stats_file)
    if not hero_stats:
        logger.info("Collecting hero stats...")
        hero_stats = collect_hero_stats(client)
        save_data(hero_stats, hero_stats_file)

    # Step 4: Build training data
    logger.info("Building training features...")
    X, y, timestamps = build_training_data(matches, teams_data, hero_stats)

    if X.shape[0] < 50:
        logger.error(f"Not enough training data: {X.shape[0]} samples")
        return None

    # Sort by time (critical for temporal validation)
    sort_idx = np.argsort(timestamps)
    X = X[sort_idx]
    y = y[sort_idx]
    timestamps = timestamps[sort_idx]

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class balance: {y.mean():.2%} radiant wins")
    if timestamps[0] > 0:
        from datetime import datetime, timezone
        oldest = datetime.fromtimestamp(timestamps[0], tz=timezone.utc)
        newest = datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)
        logger.info(f"Date range: {oldest:%Y-%m-%d} to {newest:%Y-%m-%d}")

    # Step 5: Train model
    model = Dota2Predictor()
    metrics = model.train(X, y, start_times=timestamps, tune_hyperparams=tune)

    # Step 6: Save model and metrics
    model.save()
    logger.info("Model training complete!")

    # Log metrics to SQLite
    try:
        tracker = PredictionTracker()
        tracker.log_model_metrics(metrics, validation_type="temporal")
    except Exception as e:
        logger.warning(f"Could not log metrics to DB: {e}")

    # Print feature importances
    feature_names = get_feature_names()
    importances = model.get_feature_importances(feature_names)
    if importances:
        logger.info("\nTop 15 feature importances:")
        for name, imp in importances[:15]:
            logger.info(f"  {name:30s} {imp:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=1500)
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()

    metrics = train_model(n_matches=args.matches, tune=args.tune)
    if metrics:
        print("\n=== Training Results ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
