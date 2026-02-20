#!/usr/bin/env python3
"""
Dota 2 Match Prediction Tool (Pre-Match Only)
==============================================
Predicts outcomes of professional Dota 2 matches using team statistics,
recent form, head-to-head records, player analysis, and more.

Usage:
    python main.py predict <radiant_team> <dire_team> [--format BO1|BO2|BO3|BO5]
    python main.py today [--all] [--min-rating N]
    python main.py train [--matches N] [--tune]
    python main.py team <team_name>
    python main.py accuracy [--days N]
    python main.py drift
    python main.py refresh
"""

import argparse
import logging
import sys

from tabulate import tabulate

import config
from predictor import MatchPredictor
from train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def format_prediction(pred: dict) -> str:
    """Format a prediction result for display."""
    if "error" in pred:
        return f"\n  ERROR: {pred['error']}\n"

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  MATCH PREDICTION ({pred.get('series_format', 'BO3')})")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  {pred['radiant_team']:>25s}  (Radiant)")
    lines.append(f"  {'vs':>25s}")
    lines.append(f"  {pred['dire_team']:>25s}  (Dire)")
    lines.append("")
    lines.append(f"  Model: {pred.get('model_type', 'Heuristic')}")
    lines.append("-" * 60)
    lines.append(f"  PREDICTED WINNER: {pred['predicted_winner']}")
    lines.append(f"  Win probability:  {pred['win_probability']:.1%}")
    lines.append(f"  Confidence:       {pred['confidence']}")
    lines.append("")

    rad_prob = pred["radiant_win_prob"]
    dire_prob = pred["dire_win_prob"]
    rad_bar = "#" * int(rad_prob * 40)
    dire_bar = "#" * int(dire_prob * 40)
    lines.append(f"  {pred['radiant_team'][:15]:>15s} [{rad_bar:<40s}] {rad_prob:.1%}")
    lines.append(f"  {pred['dire_team'][:15]:>15s} [{dire_bar:<40s}] {dire_prob:.1%}")
    lines.append("")

    bd = pred.get("breakdown", {})
    if bd:
        lines.append("  KEY FACTORS:")
        lines.append("-" * 60)

        tr = bd.get("team_ratings", {})
        if tr:
            lines.append(f"  Team Ratings:   Radiant={tr.get('radiant', 'N/A')}"
                         f"  Dire={tr.get('dire', 'N/A')}"
                         f"  (Elo prob: {tr.get('elo_win_prob', 'N/A')})")

        rf = bd.get("recent_form", {})
        if rf:
            lines.append(f"  Recent Form:    Radiant={rf.get('radiant_winrate', 'N/A')}"
                         f"  Dire={rf.get('dire_winrate', 'N/A')}")
            rad_mom = rf.get("radiant_momentum")
            dire_mom = rf.get("dire_momentum")
            if rad_mom and dire_mom:
                lines.append(f"  Momentum (5g):  Radiant={rad_mom}  Dire={dire_mom}")

        h2h = bd.get("h2h", {})
        if h2h:
            games = h2h.get("games", 0)
            if games > 0:
                lines.append(f"  Head-to-Head:   {games} games"
                             f"  (Radiant WR: {h2h.get('radiant_winrate', 'N/A')})")
            else:
                lines.append("  Head-to-Head:   No previous matches")

        ps = bd.get("player_strength", {})
        if ps:
            lines.append(f"  Player WR:      Radiant={ps.get('radiant_wr', 'N/A')}"
                         f"  Dire={ps.get('dire_wr', 'N/A')}")
            lines.append(f"  Hero Pool:      Radiant={ps.get('rad_hero_pool', 'N/A')}"
                         f"  Dire={ps.get('dire_hero_pool', 'N/A')}")

        df = bd.get("data_freshness", {})
        if df:
            lines.append(f"  Data Age:       Radiant={df.get('radiant', '?')}"
                         f"  Dire={df.get('dire', '?')}")

    lines.append("")
    lines.append("=" * 60)

    if pred["confidence"] == "HIGH":
        lines.append(f"  >> STRONG PREDICTION: {pred['predicted_winner']} ({pred['win_probability']:.1%})")
    elif pred["confidence"] == "MEDIUM":
        lines.append(f"  >> MODERATE PREDICTION: {pred['predicted_winner']} ({pred['win_probability']:.1%})")
    else:
        lines.append(f"  >> LOW CONFIDENCE - consider skipping this match")

    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def cmd_predict(args):
    """Predict a match between two teams."""
    predictor = MatchPredictor(use_ml_model=True)

    format_map = {"BO1": 0, "BO2": 3, "BO3": 1, "BO5": 2}
    series_type = format_map.get(args.format.upper(), 1)

    print(f"\nLooking up teams: {args.radiant} vs {args.dire}...")
    result = predictor.predict_by_names(args.radiant, args.dire, series_type=series_type)
    print(format_prediction(result))

    predictor.save_caches()


def cmd_train(args):
    """Train the prediction model on pro match data."""
    print(f"\nStarting model training with {args.matches} matches...")
    if args.tune:
        print("Hyperparameter tuning enabled (this will take longer).")
    print("This will take a while due to API rate limits.\n")

    metrics = train_model(n_matches=args.matches, tune=args.tune)

    if metrics:
        print("\n" + "=" * 50)
        print("  TRAINING COMPLETE")
        print("=" * 50)

        table_data = []
        for k, v in metrics.items():
            if isinstance(v, float):
                table_data.append([k, f"{v:.4f}"])
            else:
                table_data.append([k, str(v)])

        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="simple"))

        cv_acc = metrics.get("cv_accuracy_ensemble", 0)
        if cv_acc >= 0.70:
            print(f"\n  Model achieves {cv_acc:.1%} temporal CV accuracy (target: 70%+)")
        else:
            print(f"\n  Model accuracy: {cv_acc:.1%}. Consider collecting more data.")
        print(f"  Validation: {metrics.get('validation', 'temporal')}")
        print(f"  Engine: {metrics.get('model_type', 'unknown')}")
    else:
        print("\n  Training failed. Check logs for details.")


def cmd_team(args):
    """Look up team information."""
    predictor = MatchPredictor(use_ml_model=False)
    info = predictor.find_team_by_name(args.name)

    if not info:
        print(f"\n  Team not found: {args.name}")
        return

    print(f"\n  Team: {info.get('name', 'N/A')}")
    print(f"  Tag: {info.get('tag', 'N/A')}")
    print(f"  ID: {info.get('team_id', 'N/A')}")
    print(f"  Rating: {info.get('rating', 'N/A')}")
    print(f"  Wins: {info.get('wins', 'N/A')}")
    print(f"  Losses: {info.get('losses', 'N/A')}")
    total = (info.get("wins", 0) or 0) + (info.get("losses", 0) or 0)
    if total > 0:
        wr = (info.get("wins", 0) or 0) / total
        print(f"  Win Rate: {wr:.1%}")
    print()


def cmd_today(args):
    """Predict all matches scheduled for today."""
    from datetime import datetime, timezone

    predictor = MatchPredictor(use_ml_model=True)

    show_all = getattr(args, "all", False)
    min_rating = getattr(args, "min_rating", 0)

    if show_all:
        print("\nFetching ALL today's matches (no rating filter)...")
    else:
        print(f"\nFetching today's matches (min avg rating: {min_rating or config.MIN_TEAM_RATING})...")
        print("  Use --all to include low-tier matches\n")

    print("  Sources: OpenDota (pro), Liquipedia (upcoming)")
    print("  This may take a minute...\n")

    predictions = predictor.predict_today_matches(
        min_rating=min_rating,
        show_all=show_all,
    )

    if not predictions:
        if not show_all:
            print("  No matches from notable teams found for today.")
            print("  Try --all to see all matches, or --min-rating 900 to lower the bar.")
        else:
            print("  No professional matches found for today.")
        print("  Or use 'python main.py predict <team1> <team2>' for a manual prediction.")
        return

    by_league = {}
    for p in predictions:
        league = p.get("league", "Unknown")
        by_league.setdefault(league, []).append(p)

    total = len(predictions)
    high = sum(1 for p in predictions if p["confidence"] == "HIGH")
    medium = sum(1 for p in predictions if p["confidence"] == "MEDIUM")

    print(f"\n{'=' * 60}")
    print(f"  TODAY'S PRE-MATCH PREDICTIONS  ({total} matches)")
    print(f"  High confidence: {high}  |  Medium: {medium}  |  Low: {total - high - medium}")
    print(f"{'=' * 60}")

    for league, preds in by_league.items():
        print(f"\n  --- {league} ---")
        for pred in preds:
            status = pred.get("status", "")
            status_tag = f" [{status}]" if status else ""

            time_tag = ""
            if pred.get("start_time"):
                try:
                    st = datetime.fromtimestamp(pred["start_time"], tz=timezone.utc)
                    time_tag = f" @ {st.strftime('%H:%M UTC')}"
                except (ValueError, OSError):
                    pass

            conf_mark = {
                "HIGH": "+++",
                "MEDIUM": "++ ",
                "LOW": "+  ",
            }.get(pred["confidence"], "   ")

            winner = pred["predicted_winner"]
            prob = pred["win_probability"]
            rad = pred["radiant_team"]
            dire = pred["dire_team"]

            avg_r = pred.get("avg_team_rating", 0)
            rating_tag = f"  (avg {avg_r:.0f})" if avg_r else ""

            print(f"  {conf_mark}  {rad} vs {dire}{status_tag}{time_tag}{rating_tag}")
            print(f"        -> {winner} ({prob:.1%})  [{pred['confidence']}]")

    high_conf = [p for p in predictions if p["confidence"] == "HIGH"]
    if high_conf:
        print(f"\n{'=' * 60}")
        print(f"  TOP PICKS (HIGH CONFIDENCE)")
        print(f"{'=' * 60}")
        for pred in sorted(high_conf, key=lambda x: x["win_probability"], reverse=True):
            print(format_prediction(pred))

    predictor.save_caches()


def cmd_accuracy(args):
    """Show prediction accuracy statistics."""
    from prediction_tracker import PredictionTracker
    tracker = PredictionTracker()
    stats = tracker.get_accuracy_stats(days=args.days)

    print(f"\n{'=' * 60}")
    print(f"  PREDICTION ACCURACY (last {args.days} days)")
    print(f"{'=' * 60}")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Resolved:          {stats['resolved']}")

    if stats['accuracy'] is not None:
        print(f"  Overall accuracy:  {stats['accuracy']:.1%}")
    else:
        print("  Overall accuracy:  N/A (no resolved predictions)")

    if stats['brier_score'] is not None:
        print(f"  Brier score:       {stats['brier_score']:.4f}")

    if stats['by_confidence']:
        print(f"\n  By Confidence Level:")
        for c in stats['by_confidence']:
            acc = f"{c['accuracy']:.1%}" if c['accuracy'] is not None else "N/A"
            print(f"    {c['confidence']:>6s}: {c['total']} predictions, "
                  f"{c['resolved']} resolved, accuracy: {acc}")

    if stats['by_model']:
        print(f"\n  By Model Type:")
        for m in stats['by_model']:
            acc = f"{m['accuracy']:.1%}" if m['accuracy'] is not None else "N/A"
            print(f"    {m['model_type'] or 'unknown':>15s}: {m['total']} predictions, accuracy: {acc}")

    trend = stats.get('trend', {})
    if trend.get('last_7d') is not None or trend.get('prev_7d') is not None:
        print(f"\n  Trend:")
        l7 = f"{trend['last_7d']:.1%}" if trend['last_7d'] is not None else "N/A"
        p7 = f"{trend['prev_7d']:.1%}" if trend['prev_7d'] is not None else "N/A"
        print(f"    Last 7 days: {l7}  |  Previous 7 days: {p7}")

    print()

    # Show recent predictions
    recent = tracker.get_recent_predictions(limit=10)
    if recent:
        print(f"  RECENT PREDICTIONS:")
        print("-" * 60)
        for p in recent:
            status = "?" if p["correct"] is None else ("Y" if p["correct"] else "N")
            print(f"  [{status}] {p['radiant_team']} vs {p['dire_team']} "
                  f"-> {p['predicted_winner']} ({p['confidence']}) "
                  f"[{p['predicted_at'][:16]}]")
        print()


def cmd_drift(args):
    """Check for model drift."""
    from prediction_tracker import PredictionTracker
    tracker = PredictionTracker()
    result = tracker.check_drift(window_days=14)

    print(f"\n{'=' * 60}")
    print(f"  MODEL DRIFT CHECK (14-day window)")
    print(f"{'=' * 60}")

    if result["drift_detected"]:
        print(f"  WARNING: Drift detected!")
        print(f"  Reason: {result['reason']}")
        print(f"\n  Recommendation: Retrain the model with fresh data.")
        print(f"    python main.py train --matches 1500")
    else:
        stats = result["stats"]
        if stats["resolved"] > 0:
            print(f"  No drift detected. Model performing normally.")
            if stats["accuracy"] is not None:
                print(f"  Current accuracy: {stats['accuracy']:.1%} ({stats['resolved']} resolved)")
        else:
            print(f"  Not enough resolved predictions to assess drift.")
            print(f"  Total predictions: {stats['total_predictions']}, Resolved: {stats['resolved']}")

    print()


def cmd_refresh(args):
    """Refresh cached data."""
    predictor = MatchPredictor(use_ml_model=False)

    print("\nRefreshing teams list...")
    teams = predictor.client.get_teams() or []
    if teams:
        import os, json
        teams_path = os.path.join(config.MODEL_PATH, config.TEAMS_LIST_FILE)
        os.makedirs(config.MODEL_PATH, exist_ok=True)
        with open(teams_path, "w") as f:
            json.dump(teams, f)
        predictor._teams_list = teams
        print(f"  Cached {len(teams)} teams to disk.")
    else:
        print("  WARNING: Could not fetch teams list.")

    if predictor.team_cache:
        print(f"\nRefreshing match history for {len(predictor.team_cache)} cached teams...")
        predictor.refresh_teams_matches(list(predictor.team_cache.keys()))
        print("  Match history updated.")

    print("\nRefreshing hero stats...")
    predictor.refresh_hero_stats()
    predictor.save_caches()
    print("  Hero stats updated successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dota 2 Match Prediction Tool (Pre-Match)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py predict "Team Spirit" "Tundra Esports" --format BO3
  python main.py predict "Gaimin Gladiators" "BetBoom Team"
  python main.py today
  python main.py today --all --min-rating 900
  python main.py train --matches 1500
  python main.py train --matches 1500 --tune
  python main.py team "Team Spirit"
  python main.py accuracy --days 7
  python main.py drift
  python main.py refresh
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict a match outcome (pre-match)")
    p_pred.add_argument("radiant", help="Radiant team name")
    p_pred.add_argument("dire", help="Dire team name")
    p_pred.add_argument("--format", default="BO3", choices=["BO1", "BO2", "BO3", "BO5"],
                        help="Match format (default: BO3)")
    p_pred.set_defaults(func=cmd_predict)

    # train
    p_train = subparsers.add_parser("train", help="Train the prediction model")
    p_train.add_argument("--matches", type=int, default=1500,
                         help="Number of pro matches to train on (default: 1500)")
    p_train.add_argument("--tune", action="store_true",
                         help="Run Optuna hyperparameter tuning")
    p_train.set_defaults(func=cmd_train)

    # team
    p_team = subparsers.add_parser("team", help="Look up team info")
    p_team.add_argument("name", help="Team name to look up")
    p_team.set_defaults(func=cmd_team)

    # today
    p_today = subparsers.add_parser("today", help="Predict all of today's matches")
    p_today.add_argument("--all", action="store_true",
                         help="Show all matches including low-tier teams")
    p_today.add_argument("--min-rating", type=int, default=0,
                         help="Minimum average team rating to include (default: 1100)")
    p_today.set_defaults(func=cmd_today)

    # accuracy
    p_acc = subparsers.add_parser("accuracy", help="Show prediction accuracy stats")
    p_acc.add_argument("--days", type=int, default=30,
                       help="Number of days to analyze (default: 30)")
    p_acc.set_defaults(func=cmd_accuracy)

    # drift
    p_drift = subparsers.add_parser("drift", help="Check for model drift")
    p_drift.set_defaults(func=cmd_drift)

    # refresh
    p_refresh = subparsers.add_parser("refresh", help="Refresh cached data")
    p_refresh.set_defaults(func=cmd_refresh)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
