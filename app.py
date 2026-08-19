import streamlit as st
import joblib
import pandas as pd

from src.feature_engineering import create_match_features


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bundesliga Match Predictor",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/logistic_regression_elo.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

@st.cache_data
def load_teams():
    df = pd.read_csv(
        "data/processed/bundesliga_2010_2026.csv"
    )

    teams = sorted(
        set(df["HomeTeam"].dropna().unique())
        | set(df["AwayTeam"].dropna().unique())
    )

    return teams


teams = load_teams()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #9aa4b2;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">⚽ Bundesliga Match Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning • Elo Rating • Recent Form'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TEAM SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">🏟️ Spiel auswählen</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
    "Heimteam",
    teams,
    key="home_team"
)

with col2:
    away_team = st.selectbox(
    "Auswärtsteam",
    teams,
    key="away_team"
)


st.write("")


predict = st.button(
    "🔮 Vorhersage erstellen",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    if home_team == away_team:

        st.error(
            "Heimteam und Auswärtsteam müssen unterschiedlich sein."
        )

    else:

        try:

            # ------------------------------------------------
            # CREATE FEATURES
            # ------------------------------------------------

            features = create_match_features(
                home_team,
                away_team
            )

            # Ensure correct feature order
            features = features[feature_columns]


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            probabilities = model.predict_proba(features)[0]
            
            # ------------------------------------------------
            # HANDLE MODEL LABELS
            # ------------------------------------------------

            if set(model.classes_) == {"A", "D", "H"}:

                class_probabilities = dict(
                    zip(
                        model.classes_,
                        probabilities
                    )
                )

            elif set(model.classes_) == {0, 1, 2}:

                inverse_mapping = {
                    0: "A",
                    1: "D",
                    2: "H"
                }

                class_probabilities = {
                    inverse_mapping[int(cls)]: probability
                    for cls, probability in zip(
                        model.classes_,
                        probabilities
                    )
                }

            else:

                raise ValueError(
                    f"Unbekannte Modellklassen: "
                    f"{model.classes_}"
                )


            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            home_probability = class_probabilities["H"]
            draw_probability = class_probabilities["D"]
            away_probability = class_probabilities["A"]


            # ------------------------------------------------
            # PREDICTION SECTION
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">📊 Vorhersage</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # RESULT CARDS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="🏠 Heimsieg",
                    value=f"{home_probability:.1%}"
                )

            with col2:
                st.metric(
                    label="🤝 Unentschieden",
                    value=f"{draw_probability:.1%}"
                )

            with col3:
                st.metric(
                    label="✈️ Auswärtssieg",
                    value=f"{away_probability:.1%}"
                )


            # ------------------------------------------------
            # PROBABILITY BARS
            # ------------------------------------------------

            st.markdown("### Wahrscheinlichkeitsverteilung")

            st.progress(
                float(home_probability),
                text=f"Heimsieg — {home_probability:.1%}"
            )

            st.progress(
                float(draw_probability),
                text=f"Unentschieden — {draw_probability:.1%}"
            )

            st.progress(
                float(away_probability),
                text=f"Auswärtssieg — {away_probability:.1%}"
            )


            # ------------------------------------------------
            # MOST LIKELY RESULT
            # ------------------------------------------------

            predicted_result = max(
                class_probabilities,
                key=class_probabilities.get
            )

            result_names = {
                "H": "Heimsieg",
                "D": "Unentschieden",
                "A": "Auswärtssieg"
            }

            st.success(
                f"🔮 Das Modell prognostiziert: "
                f"**{result_names[predicted_result]}**"
            )


            # ------------------------------------------------
            # TEAM COMPARISON
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">📈 Teamvergleich</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)


            # ------------------------------------------------
            # HOME TEAM
            # ------------------------------------------------

            with col1:

                st.subheader(f"🏠 {home_team}")

                home_elo = features.iloc[0]["home_elo"]
                home_points = features.iloc[0]["home_points_avg"]
                home_goals = features.iloc[0]["home_goals_avg"]
                home_conceded = features.iloc[0]["home_conceded_avg"]

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Elo Rating",
                        f"{home_elo:.0f}"
                    )

                with metric_col2:
                    st.metric(
                        "Punkte / Spiel",
                        f"{home_points:.2f}"
                    )

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Tore / Spiel",
                        f"{home_goals:.2f}"
                    )

                with metric_col2:
                    st.metric(
                        "Gegentore / Spiel",
                        f"{home_conceded:.2f}"
                    )


            # ------------------------------------------------
            # AWAY TEAM
            # ------------------------------------------------

            with col2:

                st.subheader(f"✈️ {away_team}")

                away_elo = features.iloc[0]["away_elo"]
                away_points = features.iloc[0]["away_points_avg"]
                away_goals = features.iloc[0]["away_goals_avg"]
                away_conceded = features.iloc[0]["away_conceded_avg"]

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Elo Rating",
                        f"{away_elo:.0f}"
                    )

                with metric_col2:
                    st.metric(
                        "Punkte / Spiel",
                        f"{away_points:.2f}"
                    )

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Tore / Spiel",
                        f"{away_goals:.2f}"
                    )

                with metric_col2:
                    st.metric(
                        "Gegentore / Spiel",
                        f"{away_conceded:.2f}"
                    )


            # ------------------------------------------------
            # ELO DIFFERENCE
            # ------------------------------------------------

            st.markdown("### ⚖️ Elo-Vergleich")

            elo_difference = features.iloc[0]["elo_difference"]

            if elo_difference > 0:

                st.info(
                    f"{home_team} hat ein um "
                    f"**{abs(elo_difference):.0f} Elo-Punkte** "
                    f"höheres Rating."
                )

            elif elo_difference < 0:

                st.info(
                    f"{away_team} hat ein um "
                    f"**{abs(elo_difference):.0f} Elo-Punkte** "
                    f"höheres Rating."
                )

            else:

                st.info(
                    "Beide Teams haben dasselbe Elo-Rating."
                )


        except Exception as e:

            st.error(
                "Bei der Erstellung der Vorhersage ist ein Fehler aufgetreten."
            )

            st.exception(e)