import pandas as pd
from pathlib import Path
from urllib.request import urlopen


# --------------------------------------------------
# Einstellungen
# --------------------------------------------------

START_SEASON = 2010
END_SEASON = 2025

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/D1.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Daten herunterladen
# --------------------------------------------------

all_data = []

for year in range(START_SEASON, END_SEASON + 1):

    # Beispiel:
    # 2010 -> "1011"
    # 2011 -> "1112"
    season_code = f"{str(year)[-2:]}{str(year + 1)[-2:]}"

    url = BASE_URL.format(season=season_code)

    raw_file = RAW_DIR / f"{season_code}.csv"

    print(f"\nLade Saison {year}/{str(year + 1)[-2:]}...")
    print(url)

    try:
        # CSV herunterladen und direkt mit Pandas einlesen
        df = pd.read_csv(url)

        # Saison als zusätzliche Spalte speichern
        df["Season"] = f"{year}/{str(year + 1)[-2:]}"

        # Originaldatei lokal speichern
        df.to_csv(raw_file, index=False)

        all_data.append(df)

        print(f"  ✓ {len(df)} Spiele geladen")

    except Exception as e:
        print(f"  ✗ Fehler: {e}")


# --------------------------------------------------
# Alle Saisons zusammenführen
# --------------------------------------------------

if all_data:

    combined = pd.concat(all_data, ignore_index=True)

    output_file = PROCESSED_DIR / "bundesliga_2010_2026.csv"

    combined.to_csv(output_file, index=False)

    print("\n----------------------------------------")
    print("Fertig!")
    print("----------------------------------------")
    print(f"Saisons: {len(all_data)}")
    print(f"Spiele:  {len(combined)}")
    print(f"Spalten: {len(combined.columns)}")
    print(f"Datei:   {output_file}")

else:
    print("\nKeine Daten konnten geladen werden.")