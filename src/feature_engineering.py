from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


def expected_score(rating_a, rating_b):
    """Calculate the expected Elo score of team A against team B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating, expected, actual, k=20):
    """Update an Elo rating after a match."""
    return rating + k * (actual - expected)


def get_team_stats(history):
    """
    Calculate statistics from the previous five matches of a team.
    """

    if len(history) == 0:
        return {
            "points_avg": 0.0,
            "goals_avg": 0.0,
            "conceded_avg": 0.0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "games": 0
        }

    points = [game["points"] for game in history]
    goals = [game["goals"] for game in history]
    conceded = [game["conceded"] for game in history]

    wins = sum(game["result"] == "W" for game in history)
    draws = sum(game["result"] == "D" for game in history)
    losses = sum(game["result"] == "L" for game in history)

    return {
        "points_avg": float(np.mean(points)),
        "goals_avg": float(np.mean(goals)),
        "conceded_avg": float(np.mean(conceded)),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": len(history)
    }


def load_match_data():
    """
    Load and chronologically sort historical Bundesliga matches.
    """

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "bundesliga_2010_2026.csv"

    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def calculate_current_team_state(df):
    """
    Calculate the current state of every team from all historical matches.

    Returns:
        team_history
        home_history
        away_history
        elo_ratings
    """

    team_history = defaultdict(lambda: deque(maxlen=5))
    home_history = defaultdict(lambda: deque(maxlen=5))
    away_history = defaultdict(lambda: deque(maxlen=5))

    elo_ratings = defaultdict(lambda: 1500.0)

    for _, row in df.iterrows():

        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        # Store actual ratings BEFORE the current match
        home_elo = elo_ratings[home_team]
        away_elo = elo_ratings[away_team]

        # Expected Elo result
        expected_home = expected_score(home_elo, away_elo)
        expected_away = 1 - expected_home

        # Convert match result into numerical scores
        if row["FTR"] == "H":
            home_points = 3
            away_points = 0
            home_result = "W"
            away_result = "L"

            actual_home = 1.0
            actual_away = 0.0

        elif row["FTR"] == "D":
            home_points = 1
            away_points = 1
            home_result = "D"
            away_result = "D"

            actual_home = 0.5
            actual_away = 0.5

        else:
            home_points = 0
            away_points = 3
            home_result = "L"
            away_result = "W"

            actual_home = 0.0
            actual_away = 1.0

        # Add match to overall history
        team_history[home_team].append({
            "points": home_points,
            "goals": row["FTHG"],
            "conceded": row["FTAG"],
            "result": home_result
        })

        team_history[away_team].append({
            "points": away_points,
            "goals": row["FTAG"],
            "conceded": row["FTHG"],
            "result": away_result
        })

        # Add match to location-specific history
        home_history[home_team].append({
            "points": home_points,
            "goals": row["FTHG"],
            "conceded": row["FTAG"],
            "result": home_result
        })

        away_history[away_team].append({
            "points": away_points,
            "goals": row["FTAG"],
            "conceded": row["FTHG"],
            "result": away_result
        })

        # Update Elo AFTER the match
        elo_ratings[home_team] = update_elo(
            home_elo,
            expected_home,
            actual_home
        )

        elo_ratings[away_team] = update_elo(
            away_elo,
            expected_away,
            actual_away
        )

    return (
        team_history,
        home_history,
        away_history,
        elo_ratings
    )


def create_match_features(home_team, away_team):
    """
    Create the feature vector required by the trained model
    for a new home-team vs. away-team match.
    """

    df = load_match_data()

    (
        team_history,
        home_history,
        away_history,
        elo_ratings
    ) = calculate_current_team_state(df)

    # Prevent invalid matches
    if home_team == away_team:
        raise ValueError("Heimteam und Auswärtsteam müssen unterschiedlich sein.")

    # Overall form
    home_stats = get_team_stats(team_history[home_team])
    away_stats = get_team_stats(team_history[away_team])

    # Home-specific form
    home_home_stats = get_team_stats(home_history[home_team])

    # Away-specific form
    away_away_stats = get_team_stats(away_history[away_team])

    # Current Elo ratings
    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]

    features = {
        "home_points_avg": home_stats["points_avg"],
        "away_points_avg": away_stats["points_avg"],

        "home_goals_avg": home_stats["goals_avg"],
        "away_goals_avg": away_stats["goals_avg"],

        "home_conceded_avg": home_stats["conceded_avg"],
        "away_conceded_avg": away_stats["conceded_avg"],

        "home_wins": home_stats["wins"],
        "away_wins": away_stats["wins"],

        "home_draws": home_stats["draws"],
        "away_draws": away_stats["draws"],

        "home_losses": home_stats["losses"],
        "away_losses": away_stats["losses"],

        "home_games_history": home_stats["games"],
        "away_games_history": away_stats["games"],

        "home_home_points_avg": home_home_stats["points_avg"],
        "away_away_points_avg": away_away_stats["points_avg"],

        "home_home_goals_avg": home_home_stats["goals_avg"],
        "away_away_goals_avg": away_away_stats["goals_avg"],

        "home_home_conceded_avg": home_home_stats["conceded_avg"],
        "away_away_conceded_avg": away_away_stats["conceded_avg"],

        "home_home_wins": home_home_stats["wins"],
        "away_away_wins": away_away_stats["wins"],

        "home_home_games": home_home_stats["games"],
        "away_away_games": away_away_stats["games"],

        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": home_elo - away_elo,

        "points_avg_diff": (
            home_stats["points_avg"] - away_stats["points_avg"]
        ),

        "goals_avg_diff": (
            home_stats["goals_avg"] - away_stats["goals_avg"]
        ),

        "conceded_avg_diff": (
            home_stats["conceded_avg"] - away_stats["conceded_avg"]
        ),

        "wins_diff": (
            home_stats["wins"] - away_stats["wins"]
        )
    }

    return pd.DataFrame([features])