library(tidyverse)

games <- read_csv("data/processed/games_clean.csv", show_col_types = FALSE) %>%
  arrange(gameDateTimeEst)

K <- 20                 # update speed -- higher = more reactive to recent games
HOME_ADV <- 65           # Elo points added to home team's rating pre-game
REGRESS_TO_MEAN <- 0.25  # fraction each team regresses toward 1500 at season start

initial_rating <- 1500

# storage: named vector of current ratings, keyed by teamId (as character)
ratings <- c()
elo_history <- list()

get_rating <- function(id) {
  id <- as.character(id)
  if (is.null(ratings[id]) || is.na(ratings[id])) initial_rating else ratings[id]
}

current_season <- NA

for (i in seq_len(nrow(games))) {
  g <- games[i, ]

  # regress all ratings toward the mean at the start of a new season --
  # prevents a great '96 team's rating from carrying undiminished into '97
  if (is.na(current_season) || g$season != current_season) {
    if (length(ratings) > 0) {
      ratings <- ratings * (1 - REGRESS_TO_MEAN) + initial_rating * REGRESS_TO_MEAN
    }
    current_season <- g$season
  }

  home_id <- as.character(g$hometeamId)
  away_id <- as.character(g$awayteamId)

  R_home <- get_rating(home_id) + HOME_ADV
  R_away <- get_rating(away_id)

  E_home <- 1 / (1 + 10^((R_away - R_home) / 400))
  home_won <- as.numeric(g$homeScore > g$awayScore)

  new_home <- get_rating(home_id) + K * (home_won - E_home)
  new_away <- get_rating(away_id) + K * ((1 - home_won) - (1 - E_home))

  ratings[home_id] <- new_home
  ratings[away_id] <- new_away

  elo_history[[i]] <- tibble(
    gameId = g$gameId, season = g$season, gameDateTimeEst = g$gameDateTimeEst,
    teamId = as.integer(home_id), opponentId = as.integer(away_id),
    pre_game_elo = R_home - HOME_ADV, post_game_elo = new_home, is_home = TRUE
  )
  elo_history[[length(elo_history) + 1]] <- tibble(
    gameId = g$gameId, season = g$season, gameDateTimeEst = g$gameDateTimeEst,
    teamId = as.integer(away_id), opponentId = as.integer(home_id),
    pre_game_elo = R_away, post_game_elo = new_away, is_home = FALSE
  )
}

elo_history_df <- bind_rows(elo_history)

# end-of-season Elo per team -- this is what the app will actually look up
# for a "Team X in Season Y" query
elo_by_season <- elo_history_df %>%
  group_by(teamId, season) %>%
  slice_max(gameDateTimeEst, n = 1) %>%
  ungroup() %>%
  select(teamId, season, end_of_season_elo = post_game_elo)

write_csv(elo_history_df, "data/processed/elo_history_full.csv")
write_csv(elo_by_season, "data/processed/elo_by_season.csv")

cat("Step 2/4 done -> data/processed/{elo_history_full, elo_by_season}.csv\n")
