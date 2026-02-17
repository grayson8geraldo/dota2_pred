#!/usr/bin/env python3
"""
Dota 2 Match Prediction Tool
=============================
Predicts outcomes of professional Dota 2 matches using team statistics,
recent form, head-to-head records, hero draft analysis, and more.

Usage:
    python main.py predict <radiant_team> <dire_team> [--format BO1|BO2|BO3|BO5]
    python main.py today
    python main.py live
    python main.py train [--matches N]
    python main.py team <team_name>
    python main.py refresh
"""

import argparse
import json
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

    # Probability bars
    rad_prob = pred["radiant_win_prob"]
    dire_prob = pred["dire_win_prob"]
    rad_bar = "#" * int(rad_prob * 40)
    dire_bar = "#" * int(dire_prob * 40)
    lines.append(f"  {pred['radiant_team'][:15]:>15s} [{rad_bar:<40s}] {rad_prob:.1%}")
    lines.append(f"  {pred['dire_team'][:15]:>15s} [{dire_bar:<40s}] {dire_prob:.1%}")
    lines.append("")

    # Breakdown
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

    lines.append("")
    lines.append("=" * 60)

    # Recommendation
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


def cmd_live(args):
    """Predict all live pro matches."""
    predictor = MatchPredictor(use_ml_model=True)

    print("\nFetching live matches...")
    predictions = predictor.predict_live_matches()

    if not predictions:
        print("  No live pro matches with team data found.")
        print("  Try again when professional matches are being played.")
        return

    print(f"\nFound {len(predictions)} live match(es):\n")
    for pred in predictions:
        league = pred.get("league", "Unknown")
        print(f"  League: {league}")
        if pred.get("match_id"):
            print(f"  Match ID: {pred['match_id']}")
        print(format_prediction(pred))

    predictor.save_caches()


def cmd_train(args):
    """Train the prediction model on pro match data."""
    print(f"\nStarting model training with {args.matches} matches...")
    print("This will take a while due to API rate limits.\n")

    metrics = train_model(n_matches=args.matches)

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
            print(f"\n  Model achieves {cv_acc:.1%} cross-validation accuracy (target: 70%+)")
        else:
            print(f"\n  Model accuracy: {cv_acc:.1%}. Consider collecting more data.")
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

    print("  Sources: OpenDota (live + pro), Liquipedia (upcoming)")
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

    # Group by league
    by_league = {}
    for p in predictions:
        league = p.get("league", "Unknown")
        by_league.setdefault(league, []).append(p)

    total = len(predictions)
    live_count = sum(1 for p in predictions if p.get("status") == "LIVE")
    upcoming_count = sum(1 for p in predictions if p.get("status") == "UPCOMING")
    high = sum(1 for p in predictions if p["confidence"] == "HIGH")
    medium = sum(1 for p in predictions if p["confidence"] == "MEDIUM")

    print(f"\n{'=' * 60}")
    print(f"  TODAY'S PREDICTIONS  ({total} matches)")
    if live_count:
        print(f"  Live: {live_count}  |  Upcoming: {upcoming_count}  |  Completed: {total - live_count - upcoming_count}")
    print(f"  High confidence: {high}  |  Medium: {medium}  |  Low: {total - high - medium}")
    print(f"{'=' * 60}")

    for league, preds in by_league.items():
        print(f"\n  --- {league} ---")
        for pred in preds:
            status = pred.get("status", "")
            status_tag = f" [{status}]" if status else ""

            # Show scheduled start time for upcoming matches
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

    # Detailed output for high-confidence picks
    high_conf = [p for p in predictions if p["confidence"] == "HIGH"]
    if high_conf:
        print(f"\n{'=' * 60}")
        print(f"  TOP PICKS (HIGH CONFIDENCE)")
        print(f"{'=' * 60}")
        for pred in sorted(high_conf, key=lambda x: x["win_probability"], reverse=True):
            print(format_prediction(pred))

    predictor.save_caches()


def cmd_refresh(args):
    """Refresh cached hero and team data."""
    predictor = MatchPredictor(use_ml_model=False)
    print("\nRefreshing hero stats...")
    predictor.refresh_hero_stats()
    predictor.save_caches()
    print("  Hero stats updated successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dota 2 Match Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py predict "Team Spirit" "Tundra Esports" --format BO3
  python main.py predict "Gaimin Gladiators" "BetBoom Team"
  python main.py today
  python main.py live
  python main.py train --matches 300
  python main.py team "Team Spirit"
  python main.py refresh
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict a match outcome")
    p_pred.add_argument("radiant", help="Radiant team name")
    p_pred.add_argument("dire", help="Dire team name")
    p_pred.add_argument("--format", default="BO3", choices=["BO1", "BO2", "BO3", "BO5"],
                        help="Match format (default: BO3)")
    p_pred.set_defaults(func=cmd_predict)

    # live
    p_live = subparsers.add_parser("live", help="Predict live matches")
    p_live.set_defaults(func=cmd_live)

    # train
    p_train = subparsers.add_parser("train", help="Train the prediction model")
    p_train.add_argument("--matches", type=int, default=300,
                         help="Number of pro matches to train on (default: 300)")
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
