library(tidyverse)

team_stats <- read_csv("data/processed/team_stats_clean.csv", show_col_types = FALSE)
team_current_name <- read_csv("data/processed/team_current_name.csv", show_col_types = FALSE)
elo_by_season <- read_csv("data/processed/elo_by_season.csv", show_col_types = FALSE)

core_metrics <- c(
  "offensiveRating", "defensiveRating", "netRating",
  "pace", "trueShootingPercentage", "effectiveFieldGoalPercentage",
  "reboundPercentage", "offensiveReboundPercentage", "defensiveReboundPercentage",
  "teamTurnoverPercentage", "assistPercentage", "assistToTurnoverRatio",
  "freeThrowAttemptRate", "percentPointsFastBreak", "percentPointsInPaint",
  "percentPoints3Point"
)

# Step 1: one row per team-season
team_season <- team_stats %>%
  group_by(teamId, teamCity, teamName, season) %>%
  summarise(
    across(all_of(core_metrics), \(x) mean(x, na.rm = TRUE)),
    games_played = n(),
    win_pct = mean(win, na.rm = TRUE),
    pts_mean = mean(teamScore, na.rm = TRUE),
    pts_sd = sd(teamScore, na.rm = TRUE),
    opp_pts_mean = mean(opponentScore, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(games_played >= 20) # drop shortened seasons (COVID, lockout, etc.)

# Step 2: z-score every metric *within season* -- this is what makes cross-era
# comparison valid: a 1996 team and a 2017 team are compared to their own
# league context, not to raw numbers that drift with era-level pace/rules changes
lower_is_better <- c("defensiveRating", "teamTurnoverPercentage")

team_season_z <- team_season %>%
  group_by(season) %>%
  mutate(across(all_of(core_metrics),
                \(x) as.numeric(scale(x)),
                .names = "{.col}_z")) %>%
  ungroup()

for (m in lower_is_better) {
  zcol <- paste0(m, "_z")
  team_season_z[[zcol]] <- -1 * team_season_z[[zcol]]
}

# Step 3: composite quality index, weights fit by regression rather than hand-picked
weight_model <- lm(
  win_pct ~ netRating_z + effectiveFieldGoalPercentage_z + reboundPercentage_z +
    teamTurnoverPercentage_z + freeThrowAttemptRate_z + assistToTurnoverRatio_z,
  data = team_season_z
)
print(summary(weight_model))

coefs <- coef(weight_model)
team_season_z <- team_season_z %>%
  mutate(
    quality_index = coefs["netRating_z"] * netRating_z +
      coefs["effectiveFieldGoalPercentage_z"] * effectiveFieldGoalPercentage_z +
      coefs["reboundPercentage_z"] * reboundPercentage_z +
      coefs["teamTurnoverPercentage_z"] * teamTurnoverPercentage_z +
      coefs["freeThrowAttemptRate_z"] * freeThrowAttemptRate_z +
      coefs["assistToTurnoverRatio_z"] * assistToTurnoverRatio_z
  )

check_cor <- team_season_z %>%
  group_by(season) %>%
  summarise(cor_with_wins = cor(quality_index, win_pct, use = "complete.obs"))
print(check_cor, n = 50)

# Step 4: bring in team name + end-of-season Elo (previously computed but
# never joined back into the table the app actually reads)
team_season_full <- team_season_z %>%
  left_join(team_current_name, by = "teamId") %>%
  left_join(elo_by_season, by = c("teamId", "season"))

# a handful of teams may be missing playoff-run-only or mid-relocation Elo rows;
# fall back to 1500 (league average) rather than dropping the team-season
team_season_full <- team_season_full %>%
  mutate(end_of_season_elo = coalesce(end_of_season_elo, 1500))

write_csv(team_season_full, "data/processed/team_season_full.csv")

cat("Step 3/4 done -> data/processed/team_season_full.csv (z-scores + quality_index + Elo)\n")
