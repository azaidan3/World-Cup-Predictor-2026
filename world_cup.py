from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the data
df = pd.read_csv('matches_1930_2022.csv')

# Step 2: Keep only useful columns
columns_to_keep = ['home_team', 'away_team', 'home_score', 'away_score',
                   'Year', 'Round', 'Host']
df_clean = df[columns_to_keep].copy()

# Step 3: Create winner column


def determine_winner(row):
    if row['home_score'] > row['away_score']:
        return 'home'
    elif row['away_score'] > row['home_score']:
        return 'away'
    else:
        return 'draw'


df_clean['winner'] = df_clean.apply(determine_winner, axis=1)

# ============================================================================
# STEP 4: CREATE TEAM STRENGTH FEATURES WITH RECENT FORM
# ============================================================================

print("Building World Cup Predictor...")
print("="*80)


def calculate_team_stats_with_recency(df, current_year=2022):
    team_stats = {}

    # Get all unique teams
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

    for team in all_teams:
        # Get all matches for this team
        home_matches = df[df['home_team'] == team].copy()
        away_matches = df[df['away_team'] == team].copy()

        # Calculate recency weights (exponential decay)
        home_matches['weight'] = home_matches['Year'].apply(
            lambda year: np.exp(-0.05 * (current_year - year))
        )
        away_matches['weight'] = away_matches['Year'].apply(
            lambda year: np.exp(-0.05 * (current_year - year))
        )

        # Count weighted wins
        home_wins = ((home_matches['winner'] == 'home')
                     * home_matches['weight']).sum()
        away_wins = ((away_matches['winner'] == 'away')
                     * away_matches['weight']).sum()
        total_weighted_wins = home_wins + away_wins

        # Total weighted matches
        total_weighted_matches = home_matches['weight'].sum(
        ) + away_matches['weight'].sum()

        # Calculate weighted win rate
        win_rate = total_weighted_wins / \
            total_weighted_matches if total_weighted_matches > 0 else 0

        # Calculate weighted goals scored
        home_goals_scored = (
            home_matches['home_score'] * home_matches['weight']).sum()
        away_goals_scored = (
            away_matches['away_score'] * away_matches['weight']).sum()
        total_goals_scored = home_goals_scored + away_goals_scored
        goals_per_game = total_goals_scored / \
            total_weighted_matches if total_weighted_matches > 0 else 0

        # Calculate weighted goals conceded
        home_goals_conceded = (
            home_matches['away_score'] * home_matches['weight']).sum()
        away_goals_conceded = (
            away_matches['home_score'] * away_matches['weight']).sum()
        total_goals_conceded = home_goals_conceded + away_goals_conceded
        goals_conceded_per_game = total_goals_conceded / \
            total_weighted_matches if total_weighted_matches > 0 else 0

        # Calculate recent form (last 5 years)
        recent_home = home_matches[home_matches['Year'] >= current_year - 5]
        recent_away = away_matches[away_matches['Year'] >= current_year - 5]
        recent_wins = len(recent_home[recent_home['winner'] == 'home']) + \
            len(recent_away[recent_away['winner'] == 'away'])
        recent_matches = len(recent_home) + len(recent_away)
        recent_form = recent_wins / recent_matches if recent_matches > 0 else win_rate

        team_stats[team] = {
            'win_rate': win_rate,
            'goals_per_game': goals_per_game,
            'goals_conceded_per_game': goals_conceded_per_game,
            'recent_form': recent_form,
            'total_matches': len(home_matches) + len(away_matches)
        }

    return team_stats

# ============================================================================
# NEW: CALCULATE HEAD-TO-HEAD HISTORY
# ============================================================================


def calculate_head_to_head(df):
    """Calculate head-to-head win rates for each team matchup"""
    h2h_stats = {}

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Create a unique key for this matchup (sorted to handle both directions)
        matchup_key = tuple(sorted([home, away]))

        if matchup_key not in h2h_stats:
            h2h_stats[matchup_key] = {
                'team1': matchup_key[0],
                'team2': matchup_key[1],
                'team1_wins': 0,
                'team2_wins': 0,
                'draws': 0,
                'total': 0
            }

    # Now count the results
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        winner = row['winner']

        matchup_key = tuple(sorted([home, away]))
        h2h_stats[matchup_key]['total'] += 1

        if winner == 'draw':
            h2h_stats[matchup_key]['draws'] += 1
        elif winner == 'home':
            if home == matchup_key[0]:
                h2h_stats[matchup_key]['team1_wins'] += 1
            else:
                h2h_stats[matchup_key]['team2_wins'] += 1
        else:  # away wins
            if away == matchup_key[0]:
                h2h_stats[matchup_key]['team1_wins'] += 1
            else:
                h2h_stats[matchup_key]['team2_wins'] += 1

    return h2h_stats


def get_h2h_advantage(home_team, away_team, h2h_stats):
    """Get head-to-head advantage for home team (-1 to 1 scale)"""
    matchup_key = tuple(sorted([home_team, away_team]))

    if matchup_key not in h2h_stats:
        return 0.0  # No history

    stats = h2h_stats[matchup_key]

    if stats['total'] == 0:
        return 0.0

    # Calculate which team is team1 and team2
    if home_team == stats['team1']:
        home_wins = stats['team1_wins']
        away_wins = stats['team2_wins']
    else:
        home_wins = stats['team2_wins']
        away_wins = stats['team1_wins']

    # Return advantage score: positive favors home, negative favors away
    total = stats['total']
    return (home_wins - away_wins) / total

# ============================================================================
# NEW: HOME ADVANTAGE (Host Nation Boost)
# ============================================================================


def is_host_nation(team, host, year):
    """Check if team is the host nation"""
    # Handle multiple hosts (e.g., "South Korea / Japan")
    if pd.isna(host):
        return 0

    hosts = [h.strip() for h in str(host).split('/')]
    return 1 if team in hosts else 0


# Calculate all stats
team_stats = calculate_team_stats_with_recency(df_clean, current_year=2022)
h2h_stats = calculate_head_to_head(df_clean)

print("\n🏆 Top 10 Teams by Recent Form:")
sorted_teams = sorted(team_stats.items(),
                      key=lambda x: x[1]['recent_form'], reverse=True)
for i, (team, stats) in enumerate(sorted_teams[:10], 1):
    print(f"{i}. {team}: {stats['recent_form']*100:.1f}% recent form, "
          f"{stats['goals_per_game']:.2f} goals/game")

# Add team strength features
df_clean['home_win_rate'] = df_clean['home_team'].map(
    lambda x: team_stats[x]['win_rate'])
df_clean['away_win_rate'] = df_clean['away_team'].map(
    lambda x: team_stats[x]['win_rate'])
df_clean['home_goals_per_game'] = df_clean['home_team'].map(
    lambda x: team_stats[x]['goals_per_game'])
df_clean['away_goals_per_game'] = df_clean['away_team'].map(
    lambda x: team_stats[x]['goals_per_game'])
df_clean['home_goals_conceded'] = df_clean['home_team'].map(
    lambda x: team_stats[x]['goals_conceded_per_game'])
df_clean['away_goals_conceded'] = df_clean['away_team'].map(
    lambda x: team_stats[x]['goals_conceded_per_game'])
df_clean['home_recent_form'] = df_clean['home_team'].map(
    lambda x: team_stats[x]['recent_form'])
df_clean['away_recent_form'] = df_clean['away_team'].map(
    lambda x: team_stats[x]['recent_form'])

# Add NEW features: head-to-head and host advantage
df_clean['h2h_advantage'] = df_clean.apply(
    lambda row: get_h2h_advantage(row['home_team'], row['away_team'], h2h_stats), axis=1)
df_clean['home_is_host'] = df_clean.apply(
    lambda row: is_host_nation(row['home_team'], row['Host'], row['Year']), axis=1)
df_clean['away_is_host'] = df_clean.apply(
    lambda row: is_host_nation(row['away_team'], row['Host'], row['Year']), axis=1)

# ============================================================================
# ENCODE TEAM NAMES
# ============================================================================

le_home = LabelEncoder()
le_away = LabelEncoder()

df_clean['home_team_encoded'] = le_home.fit_transform(df_clean['home_team'])
df_clean['away_team_encoded'] = le_away.fit_transform(df_clean['away_team'])

# ============================================================================
# TRAIN MODEL WITH ALL FEATURES
# ============================================================================

print("\n" + "="*80)
print("🤖 Training Machine Learning Model with ALL Features...")
print("="*80)

# Use ALL features for prediction
X_improved = df_clean[['home_team_encoded', 'away_team_encoded', 'Year',
                       'home_win_rate', 'away_win_rate',
                       'home_goals_per_game', 'away_goals_per_game',
                       'home_goals_conceded', 'away_goals_conceded',
                       'home_recent_form', 'away_recent_form',
                       'h2h_advantage', 'home_is_host', 'away_is_host']]

y = df_clean['winner']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_improved, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} matches")
print(f"Testing on {len(X_test)} matches")

# Train model
model_improved = RandomForestClassifier(n_estimators=100, random_state=42)
model_improved.fit(X_train, y_train)

# Make predictions
y_pred_improved = model_improved.predict(X_test)

# Calculate accuracy
accuracy_improved = accuracy_score(y_test, y_pred_improved)

print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE")
print("="*80)
print(f" Accuracy: {accuracy_improved*100:.2f}%")
print(
    f" predicted: {sum(y_pred_improved == y_test)} out of {len(y_test)} matches")
print("="*80)
