"""
simulate.py

Python port of the R Monte Carlo simulator (R/04_simulation_engine_reference.R),
used directly by the FastAPI backend so the app doesn't have to depend on a
running R/Plumber process in production. Keep this logic in sync with the R
version -- the R version is the "reference" implementation used during
model development / notebooks, this is the deployed version.
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Resolve relative to the project root, not the process's current working
# directory -- this is what was breaking backend.py before: uvicorn's cwd
# depends on where you launch it from, so a bare relative path is fragile.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAM_SEASON_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "team_season_full.csv")


@dataclass
class SimResult:
    team_a_win_prob: float
    team_b_win_prob: float
    team_a_exp_score: float
    team_b_exp_score: float
    team_a_score_sd: float
    team_b_score_sd: float
    margin_mean: float
    margin_sd: float
    n_sims: int
    sample_scores_a: list  # small sample for plotting the distribution in the UI
    sample_scores_b: list


class MatchupSimulator:
    def __init__(self, team_season_path: str = TEAM_SEASON_PATH):
        self.team_season = pd.read_csv(team_season_path).set_index(["teamId", "season"])

    def _get_profile(self, team_id: int, season: int) -> pd.Series:
        try:
            return self.team_season.loc[(team_id, season)]
        except KeyError:
            raise ValueError(f"No data for team {team_id} in season {season}")

    def simulate(self, team_a_id: int, season_a: int, team_b_id: int, season_b: int,
                 n_sims: int = 10000, seed: int | None = None) -> SimResult:
        rng = np.random.default_rng(seed)

        A = self._get_profile(team_a_id, season_a)
        B = self._get_profile(team_b_id, season_b)

        sim_pace = (A["pace"] + B["pace"]) / 2

        exp_ortg_a = A["offensiveRating"] * (1 - 0.05 * B["defensiveRating_z"])
        exp_ortg_b = B["offensiveRating"] * (1 - 0.05 * A["defensiveRating_z"])

        exp_pts_a = exp_ortg_a * sim_pace / 100
        exp_pts_b = exp_ortg_b * sim_pace / 100

        sd_a = A["pts_sd"] if pd.notna(A["pts_sd"]) and A["pts_sd"] > 0 else 10.0
        sd_b = B["pts_sd"] if pd.notna(B["pts_sd"]) and B["pts_sd"] > 0 else 10.0

        scores_a = rng.normal(exp_pts_a, sd_a, n_sims)
        scores_b = rng.normal(exp_pts_b, sd_b, n_sims)

        a_win = scores_a > scores_b

        return SimResult(
            team_a_win_prob=round(float(a_win.mean()), 4),
            team_b_win_prob=round(float(1 - a_win.mean()), 4),
            team_a_exp_score=round(float(scores_a.mean()), 1),
            team_b_exp_score=round(float(scores_b.mean()), 1),
            team_a_score_sd=round(float(scores_a.std()), 1),
            team_b_score_sd=round(float(scores_b.std()), 1),
            margin_mean=round(float((scores_a - scores_b).mean()), 1),
            margin_sd=round(float((scores_a - scores_b).std()), 1),
            n_sims=n_sims,
            sample_scores_a=np.round(scores_a[:500], 1).tolist(),
            sample_scores_b=np.round(scores_b[:500], 1).tolist(),
        )


def elo_win_prob(elo_a: float, elo_b: float, home_adv_to_a: float = 0.0) -> float:
    """Straight logistic Elo win probability. home_adv_to_a is 0 for a neutral-site
    hypothetical (the default for cross-era matchups) or ~65 if you want to model
    team A as the nominal home team."""
    adj_a = elo_a + home_adv_to_a
    return round(1 / (1 + 10 ** ((elo_b - adj_a) / 400)), 4)


def blend_predictions(sim_prob_a: float, ml_prob_a: float, elo_prob_a: float,
                       weights: tuple[float, float, float] = (0.4, 0.4, 0.2)) -> float:
    """Blend simulation, ML model, and Elo into a single headline win probability.
    Weights sum to 1; tune against backtested accuracy of each source individually."""
    w_sim, w_ml, w_elo = weights
    return round(w_sim * sim_prob_a + w_ml * ml_prob_a + w_elo * elo_prob_a, 4)


if __name__ == "__main__":
    sim = MatchupSimulator()
    result = sim.simulate(team_a_id=1610612741, season_a=1996,
                           team_b_id=1610612744, season_b=2017, n_sims=10000)
    print(result)
