# World Cup Predictor

A machine learning model that predicts FIFA World Cup match outcomes with **76.17% accuracy**.

## What This Does

Analyzes 964 World Cup matches (1930-2022) to predict winners using:
- Team strength and recent form
- Head-to-head history
- Host nation advantage

Compares two algorithms: **Random Forest (76.17%)** vs **Logistic Regression (72.54%)**

## How to Run

### Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the predictor:
```bash
python world_cup.py
```

## What You Get

### Console Output:
- Top 10 teams by recent form
- Most important features affecting match outcomes
- Model accuracy and performance metrics (F1-Score, Precision, Recall)
- Comparison between Random
