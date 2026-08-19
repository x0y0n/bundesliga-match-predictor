# Bundesliga Match Predictor ⚽

Ein Machine-Learning-Projekt zur Vorhersage von Bundesliga-Spielausgängen auf Basis historischer Spieldaten, aktueller Teamform und Elo-Ratings.

Das Projekt untersucht, wie gut sich die Ergebnisse von Bundesliga-Spielen mit Machine-Learning-Modellen vorhersagen lassen. Besonderer Fokus liegt auf einer **zeitlich korrekten Evaluation**, bei der für jede Vorhersage ausschließlich Informationen verwendet werden, die vor dem jeweiligen Spiel verfügbar waren.

## Ziel des Projekts

Ziel ist es, Bundesliga-Spielausgänge mithilfe historischer Daten vorherzusagen und verschiedene Machine-Learning-Modelle miteinander zu vergleichen.

Das Modell unterscheidet zwischen drei möglichen Ergebnissen:

* `H` – Heimsieg
* `D` – Unentschieden
* `A` – Auswärtssieg

Zusätzlich soll untersucht werden, wie gut sich die Modelle auf zukünftige, bisher ungesehene Spielzeiten übertragen lassen.

## Datensatz

Für das Projekt werden historische Bundesliga-Spieldaten von **2010/11 bis 2025/26** verwendet.

Der Datensatz umfasst:

* **4.896 Spiele**
* Heim- und Auswärtsteams
* erzielte Tore
* kassierte Tore
* Spielergebnisse
* historische Wettquoten
* weitere verfügbare Spieldaten

Die historischen Spieldaten stammen von [Football-Data.co.uk](https://www.football-data.co.uk/).

## Feature Engineering

Für die Vorhersage werden ausschließlich Informationen verwendet, die vor dem jeweiligen Spiel bekannt waren.

### Aktuelle Teamform

Für jedes Team werden Statistiken aus den letzten fünf Spielen berechnet:

* durchschnittliche Punkte
* durchschnittlich erzielte Tore
* durchschnittlich kassierte Tore
* Anzahl der Siege
* Anzahl der Unentschieden
* Anzahl der Niederlagen

### Heim- und Auswärtsform

Zusätzlich werden die bisherigen Leistungen unter Berücksichtigung des Spielorts betrachtet:

* letzte Heimspiele des Heimteams
* letzte Auswärtsspiele des Auswärtsteams

### Elo-Rating

Für jedes Team wird ein chronologisches Elo-Rating berechnet.

Alle Teams starten mit einem Rating von `1500`. Nach jedem Spiel wird das Rating entsprechend dem tatsächlichen Ergebnis aktualisiert.

Für die Vorhersage wird unter anderem die Differenz der Ratings verwendet:

```text
elo_difference = home_elo - away_elo
```

Dabei wird immer das Elo-Rating **vor dem aktuellen Spiel** verwendet.

### Differenz-Features

Zusätzlich werden direkte Unterschiede zwischen Heim- und Auswärtsteam berechnet, beispielsweise:

* Punktedifferenz
* Tordifferenz
* Differenz der kassierten Tore
* Differenz bei der Anzahl der Siege

## Verwendete Machine-Learning-Modelle

### Baseline

Als einfache Referenz dient eine Strategie, die immer einen Heimsieg vorhersagt.

In den verwendeten Daten ergeben sich ungefähr:

```text
Heimsieg:        44,69 %
Unentschieden:   24,45 %
Auswärtssieg:    30,86 %
```

Eine Strategie, die immer `H` vorhersagt, erreicht damit eine Genauigkeit von etwa **44,69 %**.

### Logistic Regression

Als principales lineares Modell wird Logistic Regression eingesetzt.

Da die Features unterschiedliche Größenordnungen besitzen, werden sie vor dem Training standardisiert.

### XGBoost

Zusätzlich wird XGBoost als nichtlineares Modell getestet.

Die Hyperparameter werden mithilfe einer zeitbasierten Validierung optimiert.

## Evaluationsmethode

Ein zufälliger Train/Test-Split wäre für dieses Problem ungeeignet, da Fußballspiele eine zeitliche Reihenfolge besitzen.

Stattdessen verwendet das Projekt eine **Walk-Forward Evaluation**.

Dabei wird das Modell wiederholt mit vergangenen Daten trainiert und anschließend auf einer späteren, bisher ungesehenen Saison getestet:

```text
Vergangenheit → Zukunft

2010–2017 → 2018
2010–2018 → 2019
2010–2019 → 2020
...
```

Dadurch wird die Situation realistischer abgebildet, in der ein Modell ausschließlich aus vergangenen Spielen lernt und anschließend zukünftige Spiele vorhersagt.

## Ergebnisse

Die Walk-Forward-Evaluation liefert folgende durchschnittliche Ergebnisse:

| Modell                        | Durchschnittliche Accuracy | Durchschnittlicher Log Loss |
| ----------------------------- | -------------------------: | --------------------------: |
| **Logistic Regression + Elo** |                **51,27 %** |                  **0,9953** |
| XGBoost (getunt)              |                    50,86 % |                      1,0041 |

Die einfache Baseline „immer Heimsieg“ erreicht etwa **44,69 % Accuracy**.

### Interpretation

Das derzeit beste getestete Modell ist **Logistic Regression mit den entwickelten Fußball-Features und Elo-Rating**.

Gegenüber der einfachen Heimsieg-Baseline verbessert sich die durchschnittliche Accuracy um etwa **6,6 Prozentpunkte**.

Die Ergebnisse zeigen, dass bereits relativ einfache, fußballspezifische Features einen messbaren Beitrag zur Vorhersage leisten.

Die Ergebnisse bedeuten allerdings **nicht**, dass daraus automatisch eine profitable Wettstrategie folgt. Vorhersagegenauigkeit und finanzielle Performance sind zwei unterschiedliche Fragestellungen.

## Projektstruktur

```text
bundesliga-match-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_walk_forward_evaluation.ipynb
│
├── src/
│   └── download_data.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Verwendete Technologien

* Python
* Pandas
* NumPy
* scikit-learn
* XGBoost
* Matplotlib
* Jupyter Notebook
* Git / GitHub

## Ausführen des Projekts

Repository klonen und virtuelle Umgebung erstellen:

```bash
python -m venv .venv
```

Unter Windows aktivieren:

```powershell
.\.venv\Scripts\Activate.ps1
```

Benötigte Pakete installieren:

```powershell
python -m pip install -r requirements.txt
```

Die historischen Daten können anschließend mit folgendem Skript vorbereitet werden:

```powershell
python src/download_data.py
```

Die Notebooks werden anschließend in dieser Reihenfolge ausgeführt:

```text
01_data_exploration.ipynb
        ↓
02_feature_engineering.ipynb
        ↓
03_model_training.ipynb
        ↓
04_walk_forward_evaluation.ipynb
```

## Einschränkungen

Die aktuelle Version verwendet bewusst eine relativ kleine und interpretierbare Auswahl an Features.

Mögliche Einschränkungen sind:

* keine Spielerdaten
* keine Informationen zu Verletzungen oder Sperren
* keine Aufstellungen
* keine Expected-Goals-Daten
* keine detaillierten taktischen Informationen
* einfaches Elo-Modell
* Unentschieden sind besonders schwierig vorherzusagen

Das Projekt ist daher als **Portfolio- und Lernprojekt** gedacht und nicht als produktionsreifes Prognosesystem.

## Mögliche Erweiterungen

Das Projekt könnte später beispielsweise erweitert werden um:

* Expected Goals (`xG`)
* Spieler- und Kaderdaten
* Verletzungen und Sperren
* komplexere Elo-Modelle
* Probability Calibration
* weitere Machine-Learning-Modelle
* Vergleich mit Buchmacherwahrscheinlichkeiten
* Backtesting
* interaktives Dashboard
* weitere Ligen und Wettbewerbe

## Fazit

Das Projekt zeigt eine vollständige, zeitbewusste Machine-Learning-Pipeline zur Vorhersage von Bundesliga-Spielausgängen:

```text
Historische Daten
       ↓
Datenanalyse
       ↓
Feature Engineering
       ↓
Teamform + Heim-/Auswärtsform + Elo
       ↓
Machine Learning
       ↓
Walk-Forward Evaluation
```

Die bisherigen Ergebnisse zeigen, dass ein relativ einfaches Logistic-Regression-Modell mit fußballspezifischen Features eine naive Heimsieg-Baseline auf zuvor ungesehenen zukünftigen Spielzeiten übertreffen kann.

---

**Autor:** Oyon Rahman
**Projekt:** Bundesliga Match Predictor
**Sprache:** Python
