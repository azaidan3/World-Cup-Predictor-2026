# World-Cup-Predictor-2026 
Machine learning model predicting FIFA World Cup match outcomes with 76% accuracy using Python, scikit-learn, and historical data from 1930-2022

# World Cup Predictor

A powerful machine learning model that predicts FIFA World Cup match outcomes with **76.17% accuracy**!

## Overview

This project utilizes historical World Cup data (1930-2022) to train a Random Forest classifier that predicts match winners. The model analyzes team performance, recent form, head-to-head records, and home advantage to make intelligent predictions.

## Key Features

- **Team Strength Analysis**: Calculates win rates, goals per game, and defensive stats for each team
- **Recent Form Weighting**: Uses exponential decay to prioritize recent performances (2022 matches weighted more heavily than 1930s data)
- **Head-to-Head History**: Analyzes how specific teams have performed against each other historically
- **Host Nation Advantage**: Detects and factors in the significant boost host countries receive
- **High Accuracy**: Achieves 76.17% prediction accuracy on test data

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 76.17% |
| **Training Matches** | 771 |
| **Test Matches** | 193 |
| **Correct Predictions** | 147/193 |


## 🛠️ Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning (Random Forest, LabelEncoder, train_test_split)

## Dataset

- **File**: `matches_1930_2022.csv`
- **Records**: 964 World Cup matches
- **Time Period**: 1930-2022
- **Features**: Teams, scores, years, rounds, hosts, and more

## How to Use

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/world-cup-predictor.git
cd world-cup-predictor

# Install dependencies
pip install pandas numpy scikit-learn
```

### Run the Model
```bash
python world_cup.py
```

### Output

The model will display:
- Top 10 teams by recent form
- Training progress
- Final accuracy and performance metrics

## How It Works

### 1. **Data Loading & Cleaning**
- Loads historical World Cup match data
- Extracts relevant features (teams, scores, years)
- Creates "winner" column (home/away/draw)

### 2. **Feature Engineering**
- **Team Statistics**: Win rate, goals per game, goals conceded
- **Recent Form**: Weighted performance (recent years matter more)
- **Head-to-Head**: Historical matchup results between specific teams
- **Host Advantage**: Binary flag for home nation status

### 3. **Model Training**
- Algorithm: Random Forest Classifier (100 trees)
- Split: 80% training, 20% testing
- Features: 14 engineered variables

### 4. **Prediction**
- Model predicts: "home", "away", or "draw."
- Based on learned patterns from 771 training matches

## Future Improvements

- [ ] Add tournament stage context (Final vs Group Stage)
- [ ] Implement player-level statistics
- [ ] Try other algorithms (XGBoost, Neural Networks)
- [ ] Build an interactive web interface for predictions
- [ ] Add FIFA rankings as a feature
- [ ] Create visualizations (confusion matrix, feature importance)

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## License

This project is open source and available under the MIT License.

## Author

**Ahmed** - Built as a machine learning learning project

## Acknowledgments

- Historical World Cup data from FIFA archives
- Inspiration from sports analytics and predictive modeling
- scikit-learn documentation and community

---

⭐ **Star this repo if you found it useful!**
