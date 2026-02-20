"""
Prediction model for Dota 2 match outcomes (pre-match).

Uses an ensemble of LightGBM and Logistic Regression with:
  - Temporal cross-validation (no data leakage from future matches)
  - Probability calibration via isotonic regression
  - Optional hyperparameter tuning via Optuna
  - Recency-based sample weighting (recent matches matter more)
"""

import os
import logging
import time

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import config

logger = logging.getLogger(__name__)

# Try LightGBM first, fall back to sklearn GBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LIGHTGBM = False

# Try Optuna for hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def _make_temporal_splits(n_samples: int, n_splits: int = 5,
                          min_train_size: float = 0.3) -> list[tuple]:
    """
    Create temporal train/test splits where training always precedes test.

    Unlike sklearn TimeSeriesSplit, this uses expanding window with
    a minimum training size to ensure enough data in early folds.
    """
    min_train = int(n_samples * min_train_size)
    test_size = (n_samples - min_train) // n_splits
    if test_size < 10:
        test_size = max(10, n_samples // (n_splits + 1))
        min_train = n_samples - test_size * n_splits

    splits = []
    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = min(test_start + test_size, n_samples)
        if test_start >= n_samples:
            break
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) >= 20 and len(test_idx) >= 5:
            splits.append((train_idx, test_idx))

    return splits


def _compute_sample_weights(start_times: np.ndarray,
                            half_life_days: float = None) -> np.ndarray:
    """Compute recency-based sample weights using exponential decay."""
    if half_life_days is None:
        half_life_days = config.PATCH_HALF_LIFE_DAYS

    if start_times is None or len(start_times) == 0:
        return None

    now = time.time()
    days_ago = np.maximum(0, (now - start_times) / 86400.0)
    weights = np.exp(-np.log(2) * days_ago / half_life_days)

    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


class Dota2Predictor:
    """Ensemble model for predicting Dota 2 match outcomes."""

    def __init__(self):
        self.scaler = StandardScaler()

        if HAS_LIGHTGBM:
            self.gbm = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=15,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        else:
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

        self.gbm_weight = 0.65
        self.lr_weight = 0.35
        self.is_trained = False
        self._feature_importances = None

    def train(self, X: np.ndarray, y: np.ndarray,
              start_times: np.ndarray = None,
              tune_hyperparams: bool = False) -> dict:
        """
        Train the ensemble model with temporal validation.

        Args:
            X: Feature matrix (n_samples, n_features), sorted by time
            y: Labels (1 = radiant win, 0 = dire win)
            start_times: Unix timestamps for each sample (for sample weights)
            tune_hyperparams: Whether to run Optuna hyperparameter search
        """
        logger.info(f"Training on {X.shape[0]} samples, {X.shape[1]} features...")

        # Compute sample weights
        sample_weights = _compute_sample_weights(start_times)

        # Hyperparameter tuning (optional)
        if tune_hyperparams and HAS_OPTUNA and HAS_LIGHTGBM:
            logger.info("Running hyperparameter optimization...")
            best_params = self._tune_hyperparams(X, y, start_times)
            if best_params:
                self.gbm.set_params(**best_params)
                logger.info(f"  Best params: {best_params}")

        # Temporal cross-validation
        splits = _make_temporal_splits(X.shape[0])
        logger.info(f"  Temporal CV with {len(splits)} splits")

        gbm_scores = []
        lr_scores = []
        ensemble_scores = []

        for train_idx, test_idx in splits:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            sw = sample_weights[train_idx] if sample_weights is not None else None

            # GBM
            if sw is not None:
                self.gbm.fit(X_tr_s, y_tr, sample_weight=sw)
            else:
                self.gbm.fit(X_tr_s, y_tr)
            gbm_scores.append(accuracy_score(y_te, self.gbm.predict(X_te_s)))

            # LR
            if sw is not None:
                self.lr.fit(X_tr_s, y_tr, sample_weight=sw)
            else:
                self.lr.fit(X_tr_s, y_tr)
            lr_scores.append(accuracy_score(y_te, self.lr.predict(X_te_s)))

            # Ensemble (manual weighted average of probabilities)
            gbm_prob = self.gbm.predict_proba(X_te_s)[:, 1]
            lr_prob = self.lr.predict_proba(X_te_s)[:, 1]
            ens_prob = self.gbm_weight * gbm_prob + self.lr_weight * lr_prob
            ens_pred = (ens_prob >= 0.5).astype(int)
            ensemble_scores.append(accuracy_score(y_te, ens_pred))

        gbm_scores = np.array(gbm_scores)
        lr_scores = np.array(lr_scores)
        ensemble_scores = np.array(ensemble_scores)

        logger.info(f"  GBM temporal CV: {gbm_scores.mean():.4f} (+/- {gbm_scores.std():.4f})")
        logger.info(f"  LR temporal CV:  {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")
        logger.info(f"  Ensemble CV:     {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")

        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        sw = sample_weights

        if sw is not None:
            self.gbm.fit(X_scaled, y, sample_weight=sw)
            self.lr.fit(X_scaled, y, sample_weight=sw)
        else:
            self.gbm.fit(X_scaled, y)
            self.lr.fit(X_scaled, y)

        # Calibrate probabilities
        logger.info("  Calibrating probabilities (isotonic regression)...")
        self.calibrated_gbm = CalibratedClassifierCV(
            self.gbm, cv=5, method="isotonic"
        )
        self.calibrated_lr = CalibratedClassifierCV(
            self.lr, cv=5, method="isotonic"
        )
        self.calibrated_gbm.fit(X_scaled, y)
        self.calibrated_lr.fit(X_scaled, y)

        self.is_trained = True
        self._feature_importances = self.gbm.feature_importances_

        # Compute final metrics
        y_proba = self._predict_proba_internal(X_scaled)
        y_pred = (y_proba >= 0.5).astype(int)

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
            "model_type": "LightGBM" if HAS_LIGHTGBM else "sklearn_GBM",
            "validation": "temporal",
        }

        logger.info(f"  Training accuracy: {metrics['train_accuracy']:.4f}")
        logger.info(f"  Brier score: {metrics['train_brier']:.4f}")

        return metrics

    def _predict_proba_internal(self, X_scaled: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability prediction."""
        gbm_model = getattr(self, "calibrated_gbm", self.gbm)
        lr_model = getattr(self, "calibrated_lr", self.lr)

        gbm_prob = gbm_model.predict_proba(X_scaled)[:, 1]
        lr_prob = lr_model.predict_proba(X_scaled)[:, 1]

        return self.gbm_weight * gbm_prob + self.lr_weight * lr_prob

    def _tune_hyperparams(self, X: np.ndarray, y: np.ndarray,
                          start_times: np.ndarray = None) -> dict:
        """Run Optuna hyperparameter search with temporal validation."""
        splits = _make_temporal_splits(X.shape[0], n_splits=3)
        if not splits:
            return {}

        sample_weights = _compute_sample_weights(start_times)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            scores = []

            for train_idx, test_idx in splits:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[train_idx])
                X_te = scaler.transform(X[test_idx])
                sw = sample_weights[train_idx] if sample_weights is not None else None

                model.fit(X_tr, y[train_idx], sample_weight=sw)
                scores.append(accuracy_score(y[test_idx], model.predict(X_te)))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=config.OPTUNA_N_TRIALS,
            timeout=config.OPTUNA_TIMEOUT,
        )

        logger.info(f"  Optuna best accuracy: {study.best_value:.4f}")
        return study.best_params

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict match outcomes.

        Returns:
            (predictions, probabilities)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        expected = self.scaler.n_features_in_
        actual = X.shape[1]
        if actual > expected:
            logger.warning(
                f"Feature count mismatch: got {actual}, model expects {expected}. "
                f"Truncating extra features."
            )
            X = X[:, :expected]
        elif actual < expected:
            logger.warning(
                f"Feature count mismatch: got {actual}, model expects {expected}. "
                f"Padding with zeros."
            )
            pad = np.zeros((X.shape[0], expected - actual))
            X = np.hstack([X, pad])

        X_scaled = self.scaler.transform(X)

        # Handle legacy model format
        if hasattr(self, "_legacy_ensemble"):
            predictions = self._legacy_ensemble.predict(X_scaled)
            probabilities = self._legacy_ensemble.predict_proba(X_scaled)[:, 1]
            return predictions, probabilities

        probabilities = self._predict_proba_internal(X_scaled)
        predictions = (probabilities >= 0.5).astype(int)

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
        importances = self._feature_importances
        if len(importances) != len(feature_names):
            return []
        pairs = list(zip(feature_names, importances))
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def save(self, directory: str = config.MODEL_PATH):
        """Save model to disk."""
        os.makedirs(directory, exist_ok=True)
        model_state = {
            "gbm": self.gbm,
            "lr": self.lr,
            "scaler": self.scaler,
            "gbm_weight": self.gbm_weight,
            "lr_weight": self.lr_weight,
            "calibrated_gbm": getattr(self, "calibrated_gbm", None),
            "calibrated_lr": getattr(self, "calibrated_lr", None),
            "feature_importances": self._feature_importances,
        }
        joblib.dump(model_state, os.path.join(directory, config.TRAINED_MODEL_FILE))
        logger.info(f"Model saved to {directory}")

    def load(self, directory: str = config.MODEL_PATH):
        """Load model from disk."""
        model_path = os.path.join(directory, config.TRAINED_MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")

        state = joblib.load(model_path)

        if isinstance(state, dict):
            self.gbm = state["gbm"]
            self.lr = state["lr"]
            self.scaler = state["scaler"]
            self.gbm_weight = state.get("gbm_weight", 0.65)
            self.lr_weight = state.get("lr_weight", 0.35)
            if state.get("calibrated_gbm"):
                self.calibrated_gbm = state["calibrated_gbm"]
            if state.get("calibrated_lr"):
                self.calibrated_lr = state["calibrated_lr"]
            self._feature_importances = state.get("feature_importances")
        else:
            # Legacy format: state is the old ensemble VotingClassifier
            logger.warning("Loaded legacy model format. Retrain recommended.")
            self._legacy_ensemble = state
            scaler_path = os.path.join(directory, config.SCALER_FILE)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

        self.is_trained = True
        logger.info(f"Model loaded from {directory}")


class HeuristicPredictor:
    """
    Fallback rule-based predictor for pre-match only (no draft features).
    """

    def predict(self, features: np.ndarray, feature_names: list[str]) -> tuple[int, float]:
        """Predict using weighted heuristic rules."""
        f = dict(zip(feature_names, features))

        score = 0.5

        # Team rating (most important)
        elo_prob = f.get("elo_win_prob", 0.5)
        score += (elo_prob - 0.5) * 0.40

        # Recent form
        form_diff = f.get("form_diff", 0.0)
        score += form_diff * 0.20

        # Weighted recent form
        wwr_diff = f.get("rad_weighted_wr", 0.5) - f.get("dire_weighted_wr", 0.5)
        score += wwr_diff * 0.08

        # Short-term momentum
        mom_diff = f.get("rad_momentum", 0.5) - f.get("dire_momentum", 0.5)
        score += mom_diff * 0.12

        # Head-to-head
        h2h_games = f.get("h2h_games", 0.0)
        h2h_wr = f.get("h2h_rad_winrate", 0.5)
        if h2h_games > 0.1:
            score += (h2h_wr - 0.5) * 0.10 * min(h2h_games * 3, 1.0)

        # Player strength (replaces draft)
        player_diff = f.get("player_wr_diff", 0.0)
        score += player_diff * 0.10

        # Map side
        score += f.get("radiant_advantage", 0.0) * 0.03

        # Roster stability
        stab_diff = f.get("stability_diff", 0.0)
        score += stab_diff * 0.05

        score = max(0.05, min(0.95, score))
        prediction = 1 if score > 0.5 else 0
        return prediction, score
