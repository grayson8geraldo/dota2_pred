"""
SQLite-based prediction tracking and accuracy monitoring.

Logs every prediction, resolves outcomes from match results,
and provides accuracy analytics by confidence level, time period, etc.
"""

import os
import sqlite3
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import config

logger = logging.getLogger(__name__)


class PredictionTracker:
    """Tracks predictions and their outcomes in SQLite."""

    def __init__(self, db_path: str = config.DB_PATH):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    radiant_team TEXT NOT NULL,
                    dire_team TEXT NOT NULL,
                    radiant_team_id INTEGER,
                    dire_team_id INTEGER,
                    predicted_winner TEXT NOT NULL,
                    radiant_win_prob REAL NOT NULL,
                    confidence TEXT NOT NULL,
                    model_type TEXT,
                    series_format TEXT,
                    predicted_at TEXT DEFAULT (datetime('now')),
                    actual_winner TEXT,
                    resolved_at TEXT,
                    correct INTEGER
                );

                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT DEFAULT (datetime('now')),
                    n_samples INTEGER,
                    n_features INTEGER,
                    cv_accuracy REAL,
                    cv_std REAL,
                    train_accuracy REAL,
                    brier_score REAL,
                    log_loss_val REAL,
                    model_config TEXT,
                    validation_type TEXT
                );

                CREATE TABLE IF NOT EXISTS drift_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    checked_at TEXT DEFAULT (datetime('now')),
                    window_days INTEGER,
                    total_predictions INTEGER,
                    resolved_predictions INTEGER,
                    accuracy REAL,
                    high_conf_accuracy REAL,
                    brier_score REAL,
                    drift_detected INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_pred_teams
                    ON predictions(radiant_team_id, dire_team_id);
                CREATE INDEX IF NOT EXISTS idx_pred_date
                    ON predictions(predicted_at);
                CREATE INDEX IF NOT EXISTS idx_pred_resolved
                    ON predictions(correct);
            """)
            conn.commit()
        finally:
            conn.close()

    def log_prediction(self, prediction: dict) -> int:
        """Log a prediction to the database. Returns the prediction ID."""
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                INSERT INTO predictions
                    (match_id, radiant_team, dire_team, radiant_team_id, dire_team_id,
                     predicted_winner, radiant_win_prob, confidence, model_type, series_format)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.get("match_id"),
                prediction.get("radiant_team", ""),
                prediction.get("dire_team", ""),
                prediction.get("radiant_team_id"),
                prediction.get("dire_team_id"),
                prediction.get("predicted_winner", ""),
                prediction.get("radiant_win_prob", 0.5),
                prediction.get("confidence", "LOW"),
                prediction.get("model_type", ""),
                prediction.get("series_format", ""),
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def resolve_prediction(self, prediction_id: int, actual_winner: str):
        """Record the actual outcome of a predicted match."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT predicted_winner FROM predictions WHERE id = ?",
                (prediction_id,)
            ).fetchone()
            if not row:
                return

            correct = 1 if row["predicted_winner"] == actual_winner else 0
            conn.execute("""
                UPDATE predictions
                SET actual_winner = ?, resolved_at = datetime('now'), correct = ?
                WHERE id = ?
            """, (actual_winner, correct, prediction_id))
            conn.commit()
        finally:
            conn.close()

    def resolve_by_teams(self, radiant_team_id: int, dire_team_id: int,
                         actual_winner: str, match_id: str = None):
        """Resolve the most recent unresolved prediction for a team pair."""
        conn = self._get_conn()
        try:
            query = """
                SELECT id, predicted_winner FROM predictions
                WHERE radiant_team_id = ? AND dire_team_id = ?
                    AND correct IS NULL
                ORDER BY predicted_at DESC LIMIT 1
            """
            row = conn.execute(query, (radiant_team_id, dire_team_id)).fetchone()
            if not row:
                return

            correct = 1 if row["predicted_winner"] == actual_winner else 0
            conn.execute("""
                UPDATE predictions
                SET actual_winner = ?, resolved_at = datetime('now'), correct = ?,
                    match_id = COALESCE(?, match_id)
                WHERE id = ?
            """, (actual_winner, correct, match_id, row["id"]))
            conn.commit()
        finally:
            conn.close()

    def get_accuracy_stats(self, days: int = 30) -> dict:
        """Get accuracy statistics for a given time window."""
        conn = self._get_conn()
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Overall stats
            rows = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN correct IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count,
                    AVG(CASE WHEN correct IS NOT NULL THEN correct ELSE NULL END) as accuracy,
                    AVG(CASE WHEN correct IS NOT NULL THEN
                        (CASE WHEN predicted_winner = radiant_team
                         THEN (1 - radiant_win_prob) * (1 - radiant_win_prob)
                         ELSE radiant_win_prob * radiant_win_prob END)
                        ELSE NULL END) as brier_score
                FROM predictions
                WHERE predicted_at >= ?
            """, (cutoff,)).fetchone()

            # By confidence level
            conf_rows = conn.execute("""
                SELECT
                    confidence,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct IS NOT NULL THEN 1 ELSE 0 END) as resolved,
                    AVG(CASE WHEN correct IS NOT NULL THEN correct ELSE NULL END) as accuracy
                FROM predictions
                WHERE predicted_at >= ?
                GROUP BY confidence
                ORDER BY confidence
            """, (cutoff,)).fetchall()

            # By model type
            model_rows = conn.execute("""
                SELECT
                    model_type,
                    COUNT(*) as total,
                    AVG(CASE WHEN correct IS NOT NULL THEN correct ELSE NULL END) as accuracy
                FROM predictions
                WHERE predicted_at >= ?
                GROUP BY model_type
            """, (cutoff,)).fetchall()

            # Recent trend (last 7 days vs previous 7 days)
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            two_weeks_ago = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()

            recent = conn.execute("""
                SELECT AVG(correct) as accuracy
                FROM predictions
                WHERE predicted_at >= ? AND correct IS NOT NULL
            """, (week_ago,)).fetchone()

            previous = conn.execute("""
                SELECT AVG(correct) as accuracy
                FROM predictions
                WHERE predicted_at >= ? AND predicted_at < ? AND correct IS NOT NULL
            """, (two_weeks_ago, week_ago)).fetchone()

            return {
                "period_days": days,
                "total_predictions": rows["total"] or 0,
                "resolved": rows["resolved"] or 0,
                "correct": rows["correct_count"] or 0,
                "accuracy": round(rows["accuracy"], 4) if rows["accuracy"] else None,
                "brier_score": round(rows["brier_score"], 4) if rows["brier_score"] else None,
                "by_confidence": [
                    {
                        "confidence": r["confidence"],
                        "total": r["total"],
                        "resolved": r["resolved"],
                        "accuracy": round(r["accuracy"], 4) if r["accuracy"] else None,
                    }
                    for r in conf_rows
                ],
                "by_model": [
                    {
                        "model_type": r["model_type"],
                        "total": r["total"],
                        "accuracy": round(r["accuracy"], 4) if r["accuracy"] else None,
                    }
                    for r in model_rows
                ],
                "trend": {
                    "last_7d": round(recent["accuracy"], 4) if recent["accuracy"] else None,
                    "prev_7d": round(previous["accuracy"], 4) if previous["accuracy"] else None,
                },
            }
        finally:
            conn.close()

    def log_model_metrics(self, metrics: dict, model_config: dict = None,
                          validation_type: str = "temporal"):
        """Log training metrics to database."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO model_metrics
                    (n_samples, n_features, cv_accuracy, cv_std,
                     train_accuracy, brier_score, log_loss_val,
                     model_config, validation_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.get("n_samples"),
                metrics.get("n_features"),
                metrics.get("cv_accuracy_ensemble"),
                metrics.get("cv_std"),
                metrics.get("train_accuracy"),
                metrics.get("train_brier"),
                metrics.get("train_log_loss"),
                json.dumps(model_config) if model_config else None,
                validation_type,
            ))
            conn.commit()
        finally:
            conn.close()

    def check_drift(self, window_days: int = 14) -> dict:
        """Check for model drift by comparing recent accuracy to baseline."""
        stats = self.get_accuracy_stats(days=window_days)

        drift_detected = False
        reason = ""

        if stats["resolved"] >= 10:
            # Check if accuracy dropped below acceptable threshold
            if stats["accuracy"] is not None and stats["accuracy"] < 0.52:
                drift_detected = True
                reason = f"Accuracy {stats['accuracy']:.1%} below 52% threshold"

            # Check high-confidence accuracy
            for conf in stats["by_confidence"]:
                if conf["confidence"] == "HIGH" and conf["resolved"] >= 5:
                    if conf["accuracy"] is not None and conf["accuracy"] < 0.60:
                        drift_detected = True
                        reason = f"HIGH confidence accuracy {conf['accuracy']:.1%} below 60%"

        # Log drift check
        high_acc = None
        for conf in stats["by_confidence"]:
            if conf["confidence"] == "HIGH":
                high_acc = conf["accuracy"]

        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT INTO drift_log
                    (window_days, total_predictions, resolved_predictions,
                     accuracy, high_conf_accuracy, brier_score, drift_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                window_days,
                stats["total_predictions"],
                stats["resolved"],
                stats["accuracy"],
                high_acc,
                stats["brier_score"],
                1 if drift_detected else 0,
            ))
            conn.commit()
        finally:
            conn.close()

        return {
            "drift_detected": drift_detected,
            "reason": reason,
            "stats": stats,
        }

    def get_recent_predictions(self, limit: int = 20) -> list[dict]:
        """Get recent predictions for display."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM predictions
                ORDER BY predicted_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_model_history(self, limit: int = 10) -> list[dict]:
        """Get training history."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM model_metrics
                ORDER BY trained_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
