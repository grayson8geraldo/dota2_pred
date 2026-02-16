"""
Prediction model for Dota 2 match outcomes.

Uses an ensemble of Gradient Boosting and Logistic Regression.
The ensemble combines the strengths of both:
  - GBM captures non-linear interactions between features
  - LR provides calibrated probability estimates
"""

import os
import logging

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import config

logger = logging.getLogger(__name__)


class Dota2Predictor:
    """Ensemble model for predicting Dota 2 match outcomes."""

    def __init__(self):
        self.scaler = StandardScaler()

        self.gbm = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )

        self.lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )

        self.ensemble = VotingClassifier(
            estimators=[
                ("gbm", self.gbm),
                ("lr", self.lr),
            ],
            voting="soft",
            weights=[0.6, 0.4],  # GBM weighted slightly higher
        )

        self.is_trained = False
        self._feature_importances = None

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the ensemble model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 = radiant win, 0 = dire win)

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training on {X.shape[0]} samples, {X.shape[1]} features...")

        # Cross-validation with proper per-fold scaling (no data leakage)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Wrap each model in a Pipeline so the scaler is re-fit per fold
        gbm_pipe = Pipeline([("scaler", StandardScaler()), ("clf", self.gbm)])
        lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", self.lr)])
        ens_pipe = Pipeline([("scaler", StandardScaler()), ("clf", self.ensemble)])

        gbm_scores = cross_val_score(gbm_pipe, X, y, cv=cv, scoring="accuracy")
        lr_scores = cross_val_score(lr_pipe, X, y, cv=cv, scoring="accuracy")
        ensemble_scores = cross_val_score(ens_pipe, X, y, cv=cv, scoring="accuracy")

        logger.info(f"  GBM CV accuracy: {gbm_scores.mean():.4f} (+/- {gbm_scores.std():.4f})")
        logger.info(f"  LR CV accuracy:  {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")
        logger.info(f"  Ensemble CV:     {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")

        # Train final model on all data (single scaler for production)
        X_scaled = self.scaler.fit_transform(X)
        self.ensemble.fit(X_scaled, y)
        self.gbm.fit(X_scaled, y)  # Also fit individually for feature importances

        self.is_trained = True
        self._feature_importances = self.gbm.feature_importances_

        # Full training metrics
        y_pred = self.ensemble.predict(X_scaled)
        y_proba = self.ensemble.predict_proba(X_scaled)[:, 1]

        metrics = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "cv_accuracy_gbm": float(gbm_scores.mean()),
            "cv_accuracy_lr": float(lr_scores.mean()),
            "cv_accuracy_ensemble": float(ensemble_scores.mean()),
            "cv_std": float(ensemble_scores.std()),
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "train_log_loss": float(log_loss(y, y_proba)),
            "train_brier": float(brier_score_loss(y, y_proba)),
            "class_balance": float(y.mean()),
        }

        logger.info(f"  Training accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"  Brier score: {metrics['train_brier']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict match outcomes.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            (predictions, probabilities)
            predictions: 1 = radiant win, 0 = dire win
            probabilities: P(radiant win)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X_scaled = self.scaler.transform(X)
        predictions = self.ensemble.predict(X_scaled)
        probabilities = self.ensemble.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def predict_single(self, features: np.ndarray) -> tuple[int, float]:
        """Predict a single match."""
        X = features.reshape(1, -1)
        preds, probs = self.predict(X)
        return int(preds[0]), float(probs[0])

    def get_feature_importances(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """Get feature importances sorted by importance."""
        if self._feature_importances is None:
            return []
        pairs = list(zip(feature_names, self._feature_importances))
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def save(self, directory: str = config.MODEL_PATH):
        """Save model to disk."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.ensemble, os.path.join(directory, config.TRAINED_MODEL_FILE))
        joblib.dump(self.scaler, os.path.join(directory, config.SCALER_FILE))
        logger.info(f"Model saved to {directory}")

    def load(self, directory: str = config.MODEL_PATH):
        """Load model from disk."""
        model_path = os.path.join(directory, config.TRAINED_MODEL_FILE)
        scaler_path = os.path.join(directory, config.SCALER_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")

        self.ensemble = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        logger.info(f"Model loaded from {directory}")


class HeuristicPredictor:
    """
    Fallback rule-based predictor for when no trained model is available.
    Uses weighted heuristics based on research findings.
    """

    def predict(self, features: np.ndarray, feature_names: list[str]) -> tuple[int, float]:
        """
        Predict using weighted heuristic rules.

        Returns (prediction, probability) where prediction is 1 for radiant win.
        """
        f = dict(zip(feature_names, features))

        score = 0.5  # Start at 50/50

        # Team rating (most important single factor)
        elo_prob = f.get("elo_win_prob", 0.5)
        score += (elo_prob - 0.5) * 0.35

        # Recent form
        form_diff = f.get("form_diff", 0.0)
        score += form_diff * 0.20

        # Weighted recent form
        wwr_diff = f.get("rad_weighted_wr", 0.5) - f.get("dire_weighted_wr", 0.5)
        score += wwr_diff * 0.10

        # Head-to-head
        h2h_games = f.get("h2h_games", 0.0)
        h2h_wr = f.get("h2h_rad_winrate", 0.5)
        if h2h_games > 0.1:  # At least ~3 games
            score += (h2h_wr - 0.5) * 0.10 * min(h2h_games * 3, 1.0)

        # Draft
        draft_diff = f.get("draft_wr_diff", 0.0)
        score += draft_diff * 0.15

        # Map side
        score += f.get("radiant_advantage", 0.0) * 0.05

        # Roster stability
        stab_diff = f.get("stability_diff", 0.0)
        score += stab_diff * 0.05

        # Clamp probability
        score = max(0.05, min(0.95, score))

        prediction = 1 if score > 0.5 else 0
        return prediction, score
