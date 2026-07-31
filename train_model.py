"""
train_model.py

Trains the win-probability model on historical games:
  feature_i = home_metric_z_i - away_metric_z_i   (per z-scored box-score metric)
  label     = did the home team win

Two models are fit -- a logistic regression baseline and a calibrated
XGBoost model -- and both are scored out-of-sample using a season-based
train/test split (never a random row split: that would leak future-season
information about a team into training).

Run from the project root:
    python python/train_model.py
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "python", "model_artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

# NOTE: this list must exactly match the "_z" columns produced by
# R/03_zscore_features.R. If you add/remove a core_metric there, mirror it here.
FEATURE_METRICS = [
    "offensiveRating", "defensiveRating", "netRating",
    "pace", "trueShootingPercentage", "effectiveFieldGoalPercentage",
    "reboundPercentage", "teamTurnoverPercentage",
    "assistToTurnoverRatio", "freeThrowAttemptRate",
]


def load_data():
    team_season = pd.read_csv(f"{DATA_DIR}/team_season_full.csv")
    games = pd.read_csv(f"{DATA_DIR}/games_clean.csv")
    return team_season, games


def build_training_table(team_season: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per game: feature = home_metric_z - away_metric_z, label = home win.
    Also includes the Elo gap, since Elo captures game-to-game momentum that a
    season-average z-score can't."""
    ts = team_season.set_index(["teamId", "season"])

    rows = []
    for g in games.itertuples(index=False):
        try:
            home = ts.loc[(g.hometeamId, g.season)]
            away = ts.loc[(g.awayteamId, g.season)]
        except KeyError:
            continue  # team-season not present in stats table, skip

        diff = {f"{m}_z_diff": home[f"{m}_z"] - away[f"{m}_z"] for m in FEATURE_METRICS}
        diff["elo_diff"] = home.get("end_of_season_elo", 1500) - away.get("end_of_season_elo", 1500)
        diff["season"] = g.season
        diff["home_win"] = int(g.homeScore > g.awayScore)
        rows.append(diff)

    return pd.DataFrame(rows)


def season_based_split(df: pd.DataFrame, test_seasons_frac=0.2):
    """Hold out the most recent N% of seasons as test, rather than random rows."""
    seasons_sorted = sorted(df["season"].unique())
    cutoff_idx = int(len(seasons_sorted) * (1 - test_seasons_frac))
    train_seasons = seasons_sorted[:cutoff_idx]
    test_seasons = seasons_sorted[cutoff_idx:]

    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test, train_seasons, test_seasons


def main():
    team_season, games = load_data()
    training_table = build_training_table(team_season, games)

    feature_cols = [f"{m}_z_diff" for m in FEATURE_METRICS] + ["elo_diff"]
    train, test, train_seasons, test_seasons = season_based_split(training_table)

    X_train, y_train = train[feature_cols], train["home_win"]
    X_test, y_test = test[feature_cols], test["home_win"]

    print(f"Train seasons: {train_seasons[0]}-{train_seasons[-1]} ({len(X_train)} games)")
    print(f"Test seasons:  {test_seasons[0]}-{test_seasons[-1]} ({len(X_test)} games)")

    # --- baseline: logistic regression ---
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    logreg_probs = logreg.predict_proba(X_test)[:, 1]

    # --- stronger model: XGBoost, then calibrate ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    xgb_probs = calibrated.predict_proba(X_test)[:, 1]

    for name, probs in [("Logistic Regression", logreg_probs), ("XGBoost (calibrated)", xgb_probs)]:
        print(f"\n{name}")
        print(f"  AUC:        {roc_auc_score(y_test, probs):.4f}")
        print(f"  Log Loss:   {log_loss(y_test, probs):.4f}")
        print(f"  Brier:      {brier_score_loss(y_test, probs):.4f}")

    importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importances:\n", importances)

    joblib.dump(calibrated, f"{MODEL_DIR}/win_prob_model.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
    print(f"\nSaved model to {MODEL_DIR}/win_prob_model.pkl")


if __name__ == "__main__":
    main()
