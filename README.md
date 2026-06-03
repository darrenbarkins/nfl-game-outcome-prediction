# Predicting NFL Game Outcomes

A binary classification project predicting whether the home team will win an NFL regular season game, built as part of the Advanced Machine Learning for Business Analytics (MSBA) program.

## Project Overview

This project uses NFL game data from 2002 through 2023 to predict home team win/loss outcomes. Two versions of each model were built, one using only historical performance features, and one that also includes Vegas betting lines, allowing for a direct comparison of what public data alone can predict versus what the betting market has already priced in.

**Primary benchmark:** Vegas spread direction accuracy (~66.3%)

## Repository Structure

```
nfl-game-outcome-prediction/
│
├── NFL_Game_Outcome_Prediction.ipynb   # Main notebook (EDA through modeling)
└── README.md
```

## Notebook Structure

| Section | Description |
|---|---|
| Cells 1–11 | Data loading, cleaning, and exploratory analysis |
| Cell 12 | Feature engineering |
| Cell 13 | Train/test split and preprocessing |
| Cell 14 | Logistic Regression (baseline) |
| Cell 15 | Random Forest with hyperparameter tuning |
| Cell 16 | XGBoost with hyperparameter tuning |
| Cell 17 | Neural Network (Keras / TensorFlow) |
| Cell 18 | Soft Voting Ensemble |
| Cell 19 | Final model comparison and key findings |

## Data

- **Source:** nflreadpy Python library
- **Scope:** NFL regular season games, 2002–2023
- **Total games:** 5,679
- **Train set:** Seasons 2002–2019 (4,608 games)
- **Test set:** Seasons 2020–2023 (1,071 games)
- **Split method:** Chronological by season (no random shuffle) to prevent data leakage

## Features

**Engineered features (no Vegas):**
- Rolling win percentage — last 4 games (home and away)
- Rolling point differential — last 4 games (home and away)
- Win percentage advantage (home minus away)
- Rest advantage (home rest days minus away rest days)
- Indoor game flag (dome or closed roof)
- Season segment (early / mid / late)

**Vegas features:**
- spread_line — opening Vegas point spread
- total_line — opening over/under total

## Models and Results

| Model | Accuracy (No Vegas) | Accuracy (With Vegas) | AUC-ROC (With Vegas) |
|---|---|---|---|
| Logistic Regression | 59.20% | 65.83% | 0.7203 |
| Random Forest | 59.29% | 65.17% | 0.7031 |
| XGBoost | 59.10% | 65.17% | 0.7156 |
| Neural Network | 59.20% | 64.99% | 0.7043 |
| Ensemble (Soft Voting) | 59.66% | 65.64% | 0.7146 |
| **Vegas Benchmark** | — | ~66.3% | — |

**Best model:** Logistic Regression with Vegas features (65.83% accuracy, AUC 0.7203)

## Key Findings

- Vegas betting lines are the dominant predictive signal, every model improved by ~6 percentage points when spread and total were added
- Logistic Regression outperforms all more complex models, suggesting the feature-outcome relationship is largely linear
- The soft voting ensemble achieves the best log loss (0.6184) but does not outperform Logistic Regression on accuracy or AUC
- The best model falls just 0.47 percentage points short of the Vegas benchmark

## Requirements

```
pandas
numpy
scikit-learn
xgboost
tensorflow
keras
nfl-data-py
pyarrow
optuna
matplotlib
```

Install all dependencies:
```
pip install pandas numpy scikit-learn xgboost tensorflow keras nfl-data-py pyarrow optuna matplotlib
```

## How to Run

1. Clone the repository
2. Install dependencies listed above
3. Open `NFL_Game_Outcome_Prediction.ipynb` in Jupyter or VS Code
4. Run all cells from top to bottom

## Author

Darren Barkins — github.com/darrenbarkins
