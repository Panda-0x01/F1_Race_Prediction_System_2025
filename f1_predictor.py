import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from difflib import get_close_matches
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ===========================
# CONFIGURATION
# ===========================

OPENF1_BASE_URL = "https://api.openf1.org/v1"
YEARS = [2023, 2024]

# 2025 F1 Calendar (First 5 races)
RACES_2025 = [
    {"round": 1, "name": "Bahrain Grand Prix", "circuit": "Bahrain International Circuit", "country": "Bahrain", "continent": "Asia"},
    {"round": 2, "name": "Saudi Arabian Grand Prix", "circuit": "Jeddah Corniche Circuit", "country": "Saudi Arabia", "continent": "Asia"},
    {"round": 3, "name": "Australian Grand Prix", "circuit": "Albert Park Circuit", "country": "Australia", "continent": "Oceania"},
    {"round": 4, "name": "Japanese Grand Prix", "circuit": "Suzuka Circuit", "country": "Japan", "continent": "Asia"},
    {"round": 5, "name": "Miami Grand Prix", "circuit": "Miami International Autodrome", "country": "USA", "continent": "North America"}
]

# 2025 CONFIRMED DRIVER LINEUPS
DRIVERS_2025 = {
    "Red Bull Racing": ["Max Verstappen", "Liam Lawson"],
    "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
    "McLaren": ["Lando Norris", "Oscar Piastri"],
    "Mercedes": ["George Russell", "Andrea Kimi Antonelli"],
    "Aston Martin": ["Fernando Alonso", "Lance Stroll"],
    "Alpine": ["Pierre Gasly", "Jack Doohan"],
    "Haas F1 Team": ["Oliver Bearman", "Esteban Ocon"],
    "RB": ["Yuki Tsunoda", "Isack Hadjar"],
    "Sauber": ["Nico Hulkenberg", "Gabriel Bortoleto"],
    "Williams": ["Carlos Sainz", "Alex Albon"]
}

ACTIVE_DRIVERS_2025 = [driver for team in DRIVERS_2025.values() for driver in team]

# Driver Images
DRIVER_IMAGES = {
    "Max Verstappen": "https://www.formula1.com/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/2col/image.png",
    "Lando Norris": "https://www.formula1.com/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/2col/image.png",
    "Charles Leclerc": "https://www.formula1.com/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png.transform/2col/image.png",
    "Oscar Piastri": "https://www.formula1.com/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png.transform/2col/image.png",
    "Carlos Sainz": "https://www.formula1.com/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/2col/image.png",
    "George Russell": "https://www.formula1.com/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png.transform/2col/image.png",
    "Lewis Hamilton": "https://www.formula1.com/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/2col/image.png",
    "Fernando Alonso": "https://www.formula1.com/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png.transform/2col/image.png",
    "Lance Stroll": "https://www.formula1.com/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png.transform/2col/image.png",
    "Pierre Gasly": "https://www.formula1.com/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png.transform/2col/image.png",
    "Esteban Ocon": "https://www.formula1.com/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png.transform/2col/image.png",
    "Yuki Tsunoda": "https://www.formula1.com/content/dam/fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png.transform/2col/image.png",
    "Alex Albon": "https://www.formula1.com/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png.transform/2col/image.png",
    "Nico Hulkenberg": "https://www.formula1.com/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png.transform/2col/image.png",
}

TEAM_IMAGES = {
    "Red Bull Racing": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/red%20bull.png",
    "McLaren": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/mclaren.png",
    "Ferrari": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/ferrari.png",
    "Mercedes": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/mercedes.png",
    "Aston Martin": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/aston%20martin.png",
    "Alpine": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/alpine.png",
    "Williams": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/williams.png",
    "Haas F1 Team": "https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/team%20logos/haas.png",
}

# ===========================
# OPENF1 API FUNCTIONS
# ===========================

def get_continent(country):
    """Map country to continent"""
    continent_map = {
        'Bahrain': 'Asia', 'Saudi Arabia': 'Asia', 'Australia': 'Oceania',
        'Italy': 'Europe', 'USA': 'North America', 'United States': 'North America',
        'Spain': 'Europe', 'Monaco': 'Europe', 'Azerbaijan': 'Asia',
        'Canada': 'North America', 'United Kingdom': 'Europe', 'Austria': 'Europe',
        'France': 'Europe', 'Hungary': 'Europe', 'Belgium': 'Europe',
        'Netherlands': 'Europe', 'Singapore': 'Asia', 'Japan': 'Asia',
        'Qatar': 'Asia', 'Mexico': 'North America', 'Brazil': 'South America',
        'United Arab Emirates': 'Asia', 'UAE': 'Asia', 'China': 'Asia'
    }
    return continent_map.get(country, 'Europe')

def fetch_sessions_from_openf1(year):
    """Fetch all race sessions for a given year"""
    try:
        url = f"{OPENF1_BASE_URL}/sessions?year={year}&session_type=Race"
        print(f"    Fetching: {url}")
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            sessions = response.json()
            print(f"    → Found {len(sessions)} race sessions")
            return sessions
        else:
            print(f"    → HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"    → Error: {e}")
        return []

def fetch_results_from_openf1(session_key):
    """Fetch race results (final positions) for a session"""
    try:
        url_results = f"{OPENF1_BASE_URL}/results?session_key={session_key}"
        response = requests.get(url_results, timeout=10)
        
        if response.status_code == 200:
            results_data = response.json()
            if results_data:
                results = []
                for result in results_data:
                    if result.get('position') and result.get('position') <= 10:
                        results.append({
                            'driver_number': result.get('driver_number'),
                            'full_name': result.get('full_name', 'Unknown'),
                            'team_name': result.get('team_name', 'Unknown'),
                            'position': int(result.get('position'))
                        })
                
                if results:
                    results = sorted(results, key=lambda x: x['position'])
                    return results[:10]
        
        url_drivers = f"{OPENF1_BASE_URL}/drivers?session_key={session_key}"
        response = requests.get(url_drivers, timeout=10)
        
        if response.status_code != 200:
            return []
        
        drivers = response.json()
        
        url_positions = f"{OPENF1_BASE_URL}/position?session_key={session_key}"
        response_pos = requests.get(url_positions, timeout=10)
        
        if response_pos.status_code != 200:
            return []
        
        positions = response_pos.json()
        
        if not positions:
            return []
        
        df_pos = pd.DataFrame(positions)
        final_positions = df_pos.sort_values('date').groupby('driver_number').last()
        
        results = []
        for driver in drivers:
            driver_num = driver.get('driver_number')
            if driver_num and driver_num in final_positions.index:
                position = final_positions.loc[driver_num, 'position']
                if position and position <= 20:
                    results.append({
                        'driver_number': driver_num,
                        'full_name': driver.get('full_name', 'Unknown'),
                        'team_name': driver.get('team_name', 'Unknown'),
                        'position': int(position)
                    })
        
        results = sorted(results, key=lambda x: x['position'])
        return results[:10]
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def fetch_all_data_from_openf1():
    """Fetch all 2023-2024 race data from OpenF1 API and display summaries nicely"""
    print("\n🏁 Fetching REAL F1 data from OpenF1 API (2023-2024)...")
    print("=" * 60)

    all_data = []
    total_races = 0

    for year in YEARS:
        print(f"\n📅 Year: {year}")
        sessions = fetch_sessions_from_openf1(year)

        if not sessions:
            print(f"  ❌ No sessions found for {year}")
            continue

        year_races = 0

        for session in sessions:
            session_key = session.get('session_key')
            race_name = session.get('meeting_official_name', 'Unknown GP')
            circuit = session.get('circuit_short_name', session.get('circuit', 'Unknown'))
            country = session.get('country_name', 'Unknown')
            date = session.get('date_start', '')[:10] if session.get('date_start') else ''
            location = session.get('location', 'Unknown')

            # skip future races
            if date:
                try:
                    race_date = datetime.strptime(date, '%Y-%m-%d')
                    if race_date > datetime.now():
                        print(f"  ⏭️  {race_name} - Future race, skipping")
                        continue
                except Exception:
                    pass

            print(f"  → {race_name} ({location})...", end=" ", flush=True)

            results = fetch_results_from_openf1(session_key)

            if not results:
                print("No results")
                continue

            year_races += 1
            continent = get_continent(country)

            winner = results[0]['full_name'] if results else "N/A"
            print(f"Winner: {winner}")

            for result in results:
                all_data.append({
                    'Year': year,
                    'Round': year_races,
                    'RaceName': race_name,
                    'Circuit': circuit,
                    'Country': country,
                    'Continent': continent,
                    'Date': date,
                    'Driver': result['full_name'],
                    'Team': result['team_name'],
                    'Position': int(result['position']),
                    'IsWinner': 1 if int(result['position']) == 1 else 0
                })

            time.sleep(0.5)

        total_races += year_races
        print(f"\n  ✅ {year}: {year_races} races collected")

    print("\n" + "=" * 60)

    if total_races == 0 or not all_data:
        print("❌ Failed to fetch data from OpenF1 API")
        return None

    df = pd.DataFrame(all_data)

    # Ensure correct dtypes
    df['Year'] = df['Year'].astype(int)
    df['Round'] = df['Round'].astype(int)
    df['Position'] = pd.to_numeric(df['Position'], errors='coerce').fillna(99).astype(int)
    df['IsWinner'] = df['IsWinner'].astype(int)

    # Add RacePoints for simple scoring (only for positions 1-10)
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df['RacePoints'] = df['Position'].map(points_map).fillna(0).astype(int)

    # Overall summaries
    print(f"✅ Total records: {len(df)}")
    print(f"✅ Total races: {total_races}")
    print(f"✅ Unique drivers: {df['Driver'].nunique()}")
    print(f"✅ Unique teams: {df['Team'].nunique()}")
    print("-" * 60)

    # Top drivers summary (by points)
    driver_summary = df.groupby('Driver').agg(
        Wins=('IsWinner', 'sum'),
        Podiums=('Position', lambda s: (s <= 3).sum()),
        Points=('RacePoints', 'sum'),
        AvgPos=('Position', 'mean'),
        Appearances=('Driver', 'count')
    ).reset_index().sort_values('Points', ascending=False)

    print("\n🏆 Top Drivers (by estimated points):")
    print(driver_summary.head(15).to_string(index=False, justify='left', formatters={
        'Points': '{:d}'.format,
        'Wins': '{:d}'.format,
        'Podiums': '{:d}'.format,
        'AvgPos': lambda v: f"{v:.2f}",
        'Appearances': '{:d}'.format
    }))

    # Top teams summary
    team_summary = df.groupby('Team').agg(
        Wins=('IsWinner', 'sum'),
        Points=('RacePoints', 'sum'),
        Appearances=('Team', 'count')
    ).reset_index().sort_values('Points', ascending=False)

    print("\n🏎️ Top Teams (by estimated points):")
    print(team_summary.head(12).to_string(index=False, justify='left', formatters={
        'Points': '{:d}'.format,
        'Wins': '{:d}'.format,
        'Appearances': '{:d}'.format
    }))

    # Per-year quick stats
    print("\n📅 Per-Year Summary:")
    for year in sorted(df['Year'].unique()):
        df_y = df[df['Year'] == year]
        races_y = df_y['Round'].nunique()
        top_winner = df_y[df_y['IsWinner'] == 1].groupby('Driver').size().sort_values(ascending=False)
        top_driver = top_winner.index[0] if len(top_winner) > 0 else "N/A"
        print(f"  • {year}: {races_y} races, top winner: {top_driver}")

    # Show 2024 winners in order of Round
    if 2024 in df['Year'].values:
        print("\n🏁 REAL 2024 Race Winners:")
        winners_2024 = df[(df['Year'] == 2024) & (df['IsWinner'] == 1)].sort_values('Round')
        for _, row in winners_2024.iterrows():
            print(f"  Round {row['Round']:2d}: {row['RaceName']} - {row['Driver']} ({row['Team']})")

    # Sample data preview
    print("\n🔎 Sample records:")
    preview_cols = ['Year', 'Round', 'Date', 'RaceName', 'Circuit', 'Country', 'Driver', 'Team', 'Position', 'IsWinner']
    print(df[preview_cols].head(12).to_string(index=False))

    print("\n" + "=" * 60)
    return df

# ===========================
# FEATURE ENGINEERING
# ===========================

def calculate_driver_stats(df):
    """Calculate comprehensive driver statistics"""
    df = df.sort_values(['Year', 'Round']).reset_index(drop=True)
    
    df['DriverWins'] = 0
    df['DriverPodiums'] = 0
    df['DriverPoints'] = 0
    df['DriverAvgPosition'] = 0
    df['TeamWins'] = 0
    
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year].copy()
        
        driver_wins = year_data[year_data['IsWinner'] == 1].groupby('Driver').size()
        driver_podiums = year_data[year_data['Position'] <= 3].groupby('Driver').size()
        driver_avg_pos = year_data.groupby('Driver')['Position'].mean()
        
        year_data['RacePoints'] = year_data['Position'].map(points_map).fillna(0)
        driver_points = year_data.groupby('Driver')['RacePoints'].sum()
        
        team_wins = year_data[year_data['IsWinner'] == 1].groupby('Team').size()
        
        mask = df['Year'] == year
        df.loc[mask, 'DriverWins'] = df.loc[mask, 'Driver'].map(driver_wins).fillna(0)
        df.loc[mask, 'DriverPodiums'] = df.loc[mask, 'Driver'].map(driver_podiums).fillna(0)
        df.loc[mask, 'DriverPoints'] = df.loc[mask, 'Driver'].map(driver_points).fillna(0)
        df.loc[mask, 'DriverAvgPosition'] = df.loc[mask, 'Driver'].map(driver_avg_pos).fillna(10)
        df.loc[mask, 'TeamWins'] = df.loc[mask, 'Team'].map(team_wins).fillna(0)
    
    return df

def prepare_features(df):
    """Encode features for ML"""
    print("\n🔧 Engineering features...")
    
    df = calculate_driver_stats(df)
    
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()
    le_continent = LabelEncoder()
    
    df['Driver_Encoded'] = le_driver.fit_transform(df['Driver'])
    df['Team_Encoded'] = le_team.fit_transform(df['Team'])
    df['Circuit_Encoded'] = le_circuit.fit_transform(df['Circuit'])
    df['Continent_Encoded'] = le_continent.fit_transform(df['Continent'])
    
    print(f"  ✓ Features encoded")
    print(f"  ✓ Drivers: {df['Driver'].nunique()}")
    print(f"  ✓ Teams: {df['Team'].nunique()}")
    print(f"  ✓ Circuits: {df['Circuit'].nunique()}")
    
    return df, le_driver, le_team, le_circuit, le_continent

# ===========================
# MODEL TRAINING
# ===========================

def train_model(df):
    """Train Random Forest"""
    print("\n🤖 Training Random Forest Classifier...")
    
    features = ['Circuit_Encoded', 'Team_Encoded', 'Continent_Encoded', 
                'Round', 'DriverWins', 'DriverPodiums', 'DriverPoints', 
                'DriverAvgPosition', 'TeamWins']
    
    X = df[features]
    y = df['Driver_Encoded']
    
    class_counts = pd.Series(y).value_counts()
    min_samples = class_counts.min()
    
    print(f"  ℹ️  Total samples: {len(y)}")
    print(f"  ℹ️  Unique drivers: {len(class_counts)}")
    print(f"  ℹ️  Min samples per driver: {min_samples}")
    
    if min_samples < 2:
        print(f"  ⚠️  Some drivers have only 1 sample. Filtering for robust training...")
        valid_classes = class_counts[class_counts >= 2].index
        mask = y.isin(valid_classes)
        X = X[mask]
        y = y[mask]
        print(f"  ✓ Filtered to {len(y)} samples with {len(valid_classes)} drivers")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("  ✓ Using stratified train-test split")
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("  ✓ Using regular train-test split")
    
    model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=20, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"  ✓ Training Accuracy: {train_acc:.2%}")
    print(f"  ✓ Testing Accuracy: {test_acc:.2%}")
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n  📊 Top Feature Importance:")
    for _, row in importance.head(5).iterrows():
        print(f"     {row['Feature']}: {row['Importance']:.4f}")
    
    return model

# ===========================
# 2025 PREDICTION
# ===========================

def get_2025_team_for_driver(driver):
    """Get 2025 team for a driver"""
    for team, drivers in DRIVERS_2025.items():
        if driver in drivers:
            return team
    return None

def predict_2025_races(model, df, le_driver, le_team, le_circuit, le_continent):
    """Predict 2025 winners"""
    print("\n🔮 Predicting 2025 F1 Race Winners...")
    print("=" * 60)
    
    predictions = []
    
    df_2024 = df[df['Year'] == 2024]
    
    if len(df_2024) == 0:
        print("❌ No 2024 data available!")
        return pd.DataFrame()
    
    driver_stats = {}
    for driver in df_2024['Driver'].unique():
        driver_data = df_2024[df_2024['Driver'] == driver]
        driver_stats[driver] = {
            'wins': driver_data['IsWinner'].sum(),
            'podiums': (driver_data['Position'] <= 3).sum(),
            'avg_position': driver_data['Position'].mean(),
            'points': driver_data['Position'].map({1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
                                                     6: 8, 7: 6, 8: 4, 9: 2, 10: 1}).fillna(0).sum(),
            'team': driver_data.iloc[-1]['Team']
        }
    
    # Nicely display 2024 championship summary and pick candidate drivers for 2025
    print("\n🏆 2024 Championship Summary (Top Drivers):")
    print("-" * 60)

    # Sort drivers by points (descending)
    sorted_drivers = sorted(driver_stats.items(), key=lambda x: x[1]['points'], reverse=True)

    # Build a tidy DataFrame for pretty console output
    rows = []
    for rank, (driver, stats) in enumerate(sorted_drivers, start=1):
        rows.append({
            "Rank": rank,
            "Driver": driver,
            "Team": stats.get('team', 'Unknown'),
            "Wins": int(stats.get('wins', 0)),
            "Podiums": int(stats.get('podiums', 0)),
            "Points": float(stats.get('points', 0.0)),
            "AvgPos": float(stats.get('avg_position', 999.0))
        })

    df_top = pd.DataFrame(rows)
    # Friendly formatting
    df_top['Points'] = df_top['Points'].round(0).astype(int)
    df_top['AvgPos'] = df_top['AvgPos'].round(2)

    # Print top 10 overall nicely
    print(df_top.head(10).to_string(index=False))

    # Filter for drivers that are active in 2025 and prepare candidate list
    active_candidates = [(d, s) for d, s in sorted_drivers if d in ACTIVE_DRIVERS_2025]

    if not active_candidates:
        print("\n⚠️  No 2025 drivers found in 2024 data. Falling back to top performers from 2024.")
        candidate_drivers = sorted_drivers[:10]
    else:
        # Keep the active candidates ordered by 2024 points (already sorted)
        candidate_drivers = active_candidates
        # Print active candidate summary
        print("\n✅ 2025 Active Drivers (from 2024 data):")
        print("-" * 60)
        rows_active = []
        for rank, (driver, stats) in enumerate(candidate_drivers, start=1):
            rows_active.append({
                "Rank": rank,
                "Driver": driver,
                "Team": stats.get('team', 'Unknown'),
                "Wins": int(stats.get('wins', 0)),
                "Points": int(round(stats.get('points', 0))),
                "AvgPos": round(stats.get('avg_position', 0), 2)
            })
        df_active = pd.DataFrame(rows_active)
        print(df_active.to_string(index=False))

    # Ensure candidate_drivers is a list of tuples for downstream code
    candidate_drivers = list(candidate_drivers)
    
    print("\n2025 Race Predictions:")
    print("-" * 60)
    
    for race in RACES_2025:
        try:
            circuit_enc = le_circuit.transform([race['circuit']])[0]
        except:
            matches = get_close_matches(race['circuit'], le_circuit.classes_, n=1, cutoff=0.3)
            circuit_enc = le_circuit.transform([matches[0]])[0] if matches else 0
        
        try:
            continent_enc = le_continent.transform([race['continent']])[0]
        except:
            continent_enc = 0
        
        predictions_proba = []
        
        for driver, stats in candidate_drivers[:10]:
            team_2025 = get_2025_team_for_driver(driver)
            if not team_2025:
                continue
            
            try:
                team_enc = le_team.transform([team_2025])[0]
            except:
                try:
                    team_enc = le_team.transform([stats['team']])[0]
                except:
                    continue
            
            try:
                driver_enc = le_driver.transform([driver])[0]
            except:
                continue
            
            features = np.array([[
                circuit_enc, team_enc, continent_enc, race['round'],
                stats['wins'], stats['podiums'], stats['points'],
                stats['avg_position'], stats['wins']
            ]])
            
            proba = model.predict_proba(features)[0]
            driver_proba = proba[driver_enc] if driver_enc < len(proba) else 0
            
            predictions_proba.append({
                'driver': driver,
                'team': team_2025,
                'probability': driver_proba,
                'stats': stats
            })
        
        if predictions_proba:
            predictions_proba.sort(key=lambda x: x['probability'], reverse=True)
            winner = predictions_proba[0]
            
            predictions.append({
                'Round': race['round'],
                'Race Name': race['name'],
                'Circuit': race['circuit'],
                'Country': race['country'],
                'Predicted Winner': winner['driver'],
                'Predicted Team': winner['team'],
                'Confidence': f"{winner['probability']:.1%}",
                'Driver Image URL': get_driver_image(winner['driver']),
                'Team Image URL': get_team_image(winner['team'])
            })
            
            print(f"Round {race['round']}: {race['name']}")
            print(f"  🏆 Winner: {winner['driver']} ({winner['team']}) - {winner['probability']:.1%}")
            print(f"  📊 2024 Stats: {winner['stats']['wins']:.0f} wins, {winner['stats']['points']:.0f} pts")
            print()
        else:
            predictions.append({
                'Round': race['round'],
                'Race Name': race['name'],
                'Circuit': race['circuit'],
                'Country': race['country'],
                'Predicted Winner': "Max Verstappen",
                'Predicted Team': "Red Bull Racing",
                'Confidence': "N/A",
                'Driver Image URL': get_driver_image("Max Verstappen"),
                'Team Image URL': get_team_image("Red Bull Racing")
            })
    
    return pd.DataFrame(predictions)

def get_driver_image(name):
    """Get driver image"""
    if name in DRIVER_IMAGES:
        return DRIVER_IMAGES[name]
    matches = get_close_matches(name, DRIVER_IMAGES.keys(), n=1, cutoff=0.6)
    return DRIVER_IMAGES[matches[0]] if matches else "https://via.placeholder.com/150x150?text=Driver"

def get_team_image(name):
    """Get team image"""
    if name in TEAM_IMAGES:
        return TEAM_IMAGES[name]
    matches = get_close_matches(name, TEAM_IMAGES.keys(), n=1, cutoff=0.6)
    return TEAM_IMAGES[matches[0]] if matches else "https://via.placeholder.com/150x150?text=Team"

# ===========================
# EXPORT
# ===========================

def export_predictions(df, filename="f1_2025_predictions_neonet.csv"):
    """Export to CSV"""
    df.to_csv(filename, index=False)
    print(f"\n💾 CSV saved: {filename}")
    return filename

def generate_html_report(df, filename="f1_predictions_2025_neonet.html"):
    """Generate beautiful HTML report with NEONET-style ASCII logo"""
    
    ascii_logo = """
 ████████ ███████ ███████ ██    ██ ██   ██ ██   ██ ██       █████       ██ 
 ██       ██   ██ ██   ██ ███  ███ ██   ██ ██   ██ ██      ██   ██     ███ 
 █████    ██   ██ ██████  ████████ ██   ██ ██   ██ ██      ███████      ██ 
 ██       ██   ██ ██   ██ ██ ██ ██ ██   ██ ██   ██ ██      ██   ██      ██ 
 ██       ███████ ██   ██ ██    ██  ██ ██   ██ ██  ███████ ██   ██      ██ 
    """
    
    css_styles = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Arial', 'Helvetica', sans-serif;
            background: rgb(154, 210, 255);
            background-attachment: fixed;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 30px;
            background: #1a1a1a;
            border-radius: 30px;
            border: 5px solid #45d4ff;
            box-shadow: 0 0 30px rgba(34, 222, 255, 0.5), 12px 12px 0px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '🏁';
            position: absolute;
            font-size: 120px;
            opacity: 0.1;
            top: -20px;
            right: 50px;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }
        
        .ascii-logo {
            font-family: 'Courier New', monospace;
            font-size: 10px;
            line-height: 1.1;
            color: #ffffff;
            white-space: pre;
            margin-bottom: 20px;
            font-weight: bold;
            text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
            letter-spacing: 0px;
        }
        
        h1 {
            color: #ffffff;
            font-size: 3.5em;
            text-shadow: 0 0 20px rgba(0, 0, 0, 0.8), 4px 4px 0px rgba(0,0,0,0.3);
            margin-bottom: 15px;
            font-weight: 900;
            letter-spacing: 3px;
            animation: glow 2s ease-in-out infinite;
        }
        
        @keyframes glow {
            0%, 100% { text-shadow: 0 0 20px rgba(0, 0, 0, 0.8), 4px 4px 0px rgba(0,0,0,0.3); }
            50% { text-shadow: 0 0 30px rgb(0, 0, 0), 4px 4px 0px rgba(0,0,0,0.3); }
        }
        
        .subtitle { color: #fff; font-size: 1.4em; margin-bottom: 20px; font-weight: 600; }
        .badge-container { display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 20px; }
        .badge { background: linear-gradient(135deg, #ffffff, #ffffff); padding: 12px 25px; border-radius: 25px; font-weight: bold; color: #000; border: 3px solid #000; box-shadow: 4px 4px 0px rgba(0,0,0,0.2); transition: all 0.3s ease; font-size: 0.95em; }
        .badge:hover { transform: translateY(-3px); box-shadow: 6px 6px 0px rgba(0,0,0,0.3); }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .stat-card { background: #FFF; border-radius: 20px; padding: 25px; border: 4px solid #000; box-shadow: 6px 6px 0px rgba(0,0,0,0.2); text-align: center; transition: all 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px) rotate(-2deg); box-shadow: 8px 8px 0px rgba(0,0,0,0.3); }
        .stat-emoji { font-size: 3em; margin-bottom: 10px; }
        .stat-value { font-size: 2.5em; font-weight: 900; color: #e10600; margin-bottom: 5px; }
        .stat-label { font-size: 1.1em; color: #666; font-weight: 600; }
        .race-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin-top: 30px; }
        .race-card { background: #FFF; border-radius: 25px; padding: 30px; border: 5px solid #000; box-shadow: 8px 8px 0px rgba(0,0,0,0.2); transition: all 0.3s ease; position: relative; overflow: visible; }
        .race-card:nth-child(odd) { background: linear-gradient(135deg, #FFE5E5 0%, #FFF0E5 100%); }
        .race-card:nth-child(even) { background: linear-gradient(135deg, #E5F0FF 0%, #F0E5FF 100%); }
        .race-card:hover { transform: translateY(-8px) rotate(-1deg); box-shadow: 12px 12px 0px rgba(0,0,0,0.3); }
        .round-badge { position: absolute; top: -15px; right: 20px; background: linear-gradient(135deg, #FF6B6B, #FF8E53); width: 70px; height: 70px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2em; font-weight: 900; border: 4px solid #000; box-shadow: 4px 4px 0px rgba(0,0,0,0.3); color: #fff; text-shadow: 2px 2px 0px rgba(0,0,0,0.3); }
        .race-name { font-size: 1.6em; font-weight: 900; margin-bottom: 8px; color: #000; padding-right: 80px; }
        .race-location { color: #666; font-size: 1.1em; margin-bottom: 20px; font-weight: 600; }
        .winner-section { display: flex; align-items: center; gap: 20px; margin-top: 20px; padding: 20px; background: rgba(255,255,255,0.7); border-radius: 20px; border: 3px solid #000; }
        .driver-img { width: 90px; height: 90px; object-fit: contain; background: #fff; border-radius: 15px; padding: 10px; border: 3px solid #000; }
        .winner-info { flex: 1; }
        .winner-label { color: #666; font-size: 0.85em; margin-bottom: 5px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
        .winner-name { font-size: 1.7em; font-weight: 900; color: #e10600; margin-bottom: 5px; text-shadow: 2px 2px 0px rgba(0,0,0,0.1); }
        .team-name { font-size: 1.1em; color: #333; font-weight: 600; }
        .confidence { font-size: 0.9em; color: #4CAF50; margin-top: 8px; font-weight: 700; background: rgba(76,175,80,0.2); display: inline-block; padding: 5px 12px; border-radius: 15px; border: 2px solid #4CAF50; }
        .team-logo { width: 70px; height: 70px; object-fit: contain; background: #fff; border-radius: 10px; padding: 8px; border: 3px solid #000; }
        .deco-star { position: absolute; font-size: 2em; opacity: 0.3; animation: twinkle 2s ease-in-out infinite; }
        @keyframes twinkle { 0%, 100% { opacity: 0.3; } 50% { opacity: 0.8; } }
        .footer { text-align: center; margin-top: 60px; padding: 40px; background: rgba(26, 26, 26, 0.95); border-radius: 25px; border: 4px solid #aaf2ff; box-shadow: 0 0 20px rgba(40, 176, 255, 0.3), 8px 8px 0px rgba(0,0,0,0.2); }
        .footer-emoji { font-size: 3em; margin-bottom: 15px; }
        .timestamp { font-size: 1.1em; margin-bottom: 10px; color: #ffffff; font-weight: 600; }
        .credit { font-size: 0.95em; color: #fff; font-weight: 500; }
        @media (max-width: 768px) {
            .race-grid { grid-template-columns: 1fr; }
            h1 { font-size: 2.5em; }
            .ascii-logo { font-size: 6px; }
            .badge-container { flex-direction: column; align-items: center; }
        }
    """
    
    races_html = ""
    race_emojis = ['✨', '⭐', '🌟', '💫', '🔥']
    
    for idx, row in df.iterrows():
        deco_emoji = race_emojis[idx % len(race_emojis)]
        races_html += f"""
            <div class="race-card">
                <span class="deco-star" style="top: 10px; left: 10px;">{deco_emoji}</span>
                <div class="round-badge">R{row['Round']}</div>
                <div class="race-name">{row['Race Name']}</div>
                <div class="race-location">📍 {row['Circuit']}, {row['Country']}</div>
                
                <div class="winner-section">
                    <img src="{row['Driver Image URL']}" alt="{row['Predicted Winner']}" class="driver-img" onerror="this.src='https://via.placeholder.com/90x90?text=Driver'"/>
                    <div class="winner-info">
                        <div class="winner-label">🏆 Predicted Winner</div>
                        <div class="winner-name">{row['Predicted Winner']}</div>
                        <div class="team-name">{row['Predicted Team']}</div>
                        <div class="confidence">✓ {row['Confidence']}</div>
                    </div>
                    <img src="{row['Team Image URL']}" alt="{row['Predicted Team']}" class="team-logo" onerror="this.src='https://via.placeholder.com/70x70?text=Team'"/>
                </div>
            </div>
"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>F1 2025 Race Predictions - NEONET Style</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="ascii-logo">{ascii_logo}</div>
            <h1>2025 SEASON PREDICTIONS</h1>
            <div class="subtitle">AI-Powered Race Winner Predictions</div>
            <div class="badge-container">
                <div class="badge">🤖 Machine Learning</div>
                <div class="badge">📊 Real 2024 Data</div>
                <div class="badge">✅ Official 2025 Lineups</div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-emoji">🏁</div>
                <div class="stat-value">{len(df)}</div>
                <div class="stat-label">Races Predicted</div>
            </div>
            <div class="stat-card">
                <div class="stat-emoji">🏆</div>
                <div class="stat-value">{df['Predicted Winner'].nunique()}</div>
                <div class="stat-label">Unique Winners</div>
            </div>
            <div class="stat-card">
                <div class="stat-emoji">🏎️</div>
                <div class="stat-value">{df['Predicted Team'].nunique()}</div>
                <div class="stat-label">Teams Competing</div>
            </div>
        </div>
        
        <div class="race-grid">
            {races_html}
        </div>
        
        <div class="footer">
            <div class="footer-emoji">🏁</div>
            <div class="credit">Random Forest ML Model | NEONET Style</div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📄 HTML saved: {filename}")
    return filename

# ===========================
# MAIN
# ===========================

def display_data_nicely(df):
    """Print a clean, readable summary of fetched race results."""
    if df is None or df.empty:
        print("❌ No data to display.")
        return

    # Columns to show in preview (only include those present)
    preview_cols = ['Year', 'Round', 'Date', 'RaceName', 'Circuit', 'Country', 'Driver', 'Team', 'Position', 'IsWinner']
    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\n📋 CLEAN DATA PREVIEW (first 12 rows)")
    print("-" * 60)
    try:
        print(df[preview_cols].head(12).to_string(index=False))
    except Exception:
        # Fallback if any formatting issues
        print(df[preview_cols].head(12))

    # Quick overall counts
    print("\n" + "-" * 60)
    print("📊 QUICK COUNTS")
    print("-" * 60)
    total_records = len(df)
    unique_drivers = df['Driver'].nunique() if 'Driver' in df.columns else 0
    unique_teams = df['Team'].nunique() if 'Team' in df.columns else 0
    total_races = df[['Year', 'Round']].drop_duplicates().shape[0] if {'Year', 'Round'}.issubset(df.columns) else "N/A"
    print(f"• Total records: {total_records}")
    print(f"• Total races (unique Year+Round): {total_races}")
    print(f"• Unique drivers: {unique_drivers}")
    print(f"• Unique teams: {unique_teams}")

    # Top drivers by points or wins
    print("\n" + "-" * 60)
    if 'RacePoints' in df.columns:
        drv = df.groupby('Driver').agg(
            Wins=('IsWinner', 'sum'),
            Points=('RacePoints', 'sum'),
            Appearances=('Driver', 'count')
        ).sort_values('Points', ascending=False).head(10)
        print("🏆 Top Drivers (by estimated points)")
        print(drv.to_string())
    else:
        drv = df[df.get('IsWinner', 0) == 1].groupby('Driver').size().sort_values(ascending=False).head(10)
        print("🏆 Top Winners (counts)")
        print(drv.to_string())

    # Top teams
    print("\n" + "-" * 60)
    if 'RacePoints' in df.columns:
        tm = df.groupby('Team').agg(
            Wins=('IsWinner', 'sum'),
            Points=('RacePoints', 'sum'),
            Appearances=('Team', 'count')
        ).sort_values('Points', ascending=False).head(12)
        print("🏎️ Top Teams (by estimated points)")
        print(tm.to_string())
    else:
        tm = df[df.get('IsWinner', 0) == 1].groupby('Team').size().sort_values(ascending=False).head(12)
        print("🏎️ Top Teams (wins)")
        print(tm.to_string())

    # Per-year quick summary (if Year exists)
    if 'Year' in df.columns:
        print("\n" + "-" * 60)
        print("📅 Per-Year Summary (races, top winner)")
        for year in sorted(df['Year'].unique()):
            df_y = df[df['Year'] == year]
            races_y = df_y['Round'].nunique() if 'Round' in df_y.columns else "N/A"
            winners = df_y[df_y.get('IsWinner', 0) == 1].groupby('Driver').size().sort_values(ascending=False)
            top_driver = winners.index[0] if len(winners) > 0 else "N/A"
            print(f"  • {year}: {races_y} races, top winner: {top_driver}")

    print("\n" + "-" * 60)

    print("\n" + "=" * 60)
    print("📋 CLEAN DATA SUMMARY")
    print("=" * 60)

    # Overall metrics
    total_races = df[['Year','Round']].drop_duplicates().shape[0]
    unique_drivers = df['Driver'].nunique()
    unique_teams = df['Team'].nunique()
    print(f"• Total records: {len(df)}")
    print(f"• Total races (unique Year+Round): {total_races}")
    print(f"• Unique drivers: {unique_drivers}")
    print(f"• Unique teams: {unique_teams}")
    print("-" * 60)

    # Per-year winners in round order
    for year in sorted(df['Year'].unique()):
        print(f"\n🏁 Winners — {year}")
        winners = df[(df['Year'] == year) & (df['IsWinner'] == 1)].copy()
        if winners.empty:
            print("  No winners recorded for this year.")
            continue
        winners = winners.sort_values('Round')[['Round','RaceName','Date','Circuit','Driver','Team']]
        print(winners.to_string(index=False))

    # Top winners overall
    print("\n" + "-" * 60)
    print("🏆 Top Winners (counts)")
    top_winners = df[df['IsWinner'] == 1].groupby('Driver').size().sort_values(ascending=False).head(10)
    print(top_winners.to_string())

    # Top teams by estimated points (RacePoints must exist if fetch_all_data_from_openf1 ran)
    if 'RacePoints' in df.columns:
        print("\n" + "-" * 60)
        print("🏎️ Top Teams (by estimated points)")
        team_points = df.groupby('Team')['RacePoints'].sum().sort_values(ascending=False).head(12)
        print(team_points.to_string())
    print("=" * 60)

def main():
    """Main pipeline"""
    print("\n" + "=" * 60)
    ascii_f1 = """
    ███████╗ ██████╗ ██████╗ ███╗   ███╗██╗   ██╗██╗      █████╗      ██╗
    ██╔════╝██╔═══██╗██╔══██╗████╗ ████║██║   ██║██║     ██╔══██╗    ███║
    █████╗  ██║   ██║██████╔╝██╔████╔██║██║   ██║██║     ███████║    ╚██║
    ██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██║   ██║██║     ██╔══██║     ██║
    ██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║╚██████╔╝███████╗██║  ██║     ██║
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝     ╚═╝
    
     ██████╗ ██████╗  █████╗ ███╗   ██╗██████╗     ██████╗ ██████╗ ██╗██╗  ██╗
    ██╔════╝ ██╔══██╗██╔══██╗████╗  ██║██╔══██╗    ██╔══██╗██╔══██╗██║╚██╗██╔╝
    ██║  ███╗██████╔╝███████║██╔██╗ ██║██║  ██║    ██████╔╝██████╔╝██║ ╚███╔╝ 
    ██║   ██║██╔══██╗██╔══██║██║╚██╗██║██║  ██║    ██╔═══╝ ██╔══██╗██║ ██╔██╗ 
    ╚██████╔╝██║  ██║██║  ██║██║ ╚████║██████╔╝    ██║     ██║  ██║██║██╔╝ ██╗
     ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝     ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
    
    ██╗    ██╗██╗███╗   ██╗███╗   ██╗███████╗██████╗ 
    ██║    ██║██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
    ██║ █╗ ██║██║██╔██╗ ██║██╔██╗ ██║█████╗  ██████╔╝
    ██║███╗██║██║██║╚██╗██║██║╚██╗██║██╔══╝  ██╔══██╗
    ╚███╔███╔╝██║██║ ╚████║██║ ╚████║███████╗██║  ██║
     ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
    
    ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
    ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║██║   ██║██╔██╗ ██║
    ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║██║   ██║██║╚██╗██║
    ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
    ╚═╝     ╚═════╝ ╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    """
    print(ascii_f1)
    print("🏁 FORMULA 1 RACE PREDICTOR - 2025 SEASON")
    print("📊 USING REAL 2023 & 2024 DATA FROM OPENF1 API")
    print("=" * 60)
    
    try:
        df_raw = fetch_all_data_from_openf1()

        if df_raw is None or len(df_raw) == 0:
            print("\n❌ Failed to fetch data from API!")
            return

        # <<< NEW: display raw data nicely >>>
        display_data_nicely(df_raw)
        # <<< END NEW >>>

        print("\n✅ DATA VERIFICATION:")
        print(f"  • Total race results: {len(df_raw)}")
        print(f"  • Years covered: {sorted(df_raw['Year'].unique())}")
        
        df_proc, le_driver, le_team, le_circuit, le_continent = prepare_features(df_raw)
        
        print("\n" + "=" * 60)
        print("🎓 TRAINING MODEL ON REAL HISTORICAL DATA")
        print("=" * 60)
        model = train_model(df_proc)
        
        print("\n" + "=" * 60)
        print("🔮 GENERATING 2025 PREDICTIONS")
        print("=" * 60)
        predictions = predict_2025_races(model, df_proc, le_driver, le_team, le_circuit, le_continent)
        
        if len(predictions) == 0:
            print("\n❌ Failed to generate predictions!")
            return
        
        csv_file = export_predictions(predictions)
        html_file = generate_html_report(predictions)
        
        print("\n" + "=" * 60)
        print("📊 PREDICTIONS SUMMARY")
        print("=" * 60)
        
        # Display predictions in tabular format
        print("\n🏁 2025 RACE PREDICTIONS:\n")
        for idx, row in predictions.iterrows():
            print(f"Round {row['Round']:2d} | {row['Race Name']:30s} | {row['Country']:15s}")
            print(f"{'':8s} ├─ 🏆 Winner: {row['Predicted Winner']:20s} ({row['Predicted Team']})")
            print(f"{'':8s} ├─ 📍 Circuit: {row['Circuit']}")
            print(f"{'':8s} └─ ✓ Confidence: {row['Confidence']}")
            print()
        
        # Summary statistics
        print("=" * 60)
        print("📈 SUMMARY STATISTICS:")
        print("=" * 60)
        winner_counts = predictions['Predicted Winner'].value_counts()
        print(f"\n🏆 Top Predicted Winners:")
        for driver, count in winner_counts.head(5).items():
            print(f"  • {driver}: {count} race(s)")
        
        team_counts = predictions['Predicted Team'].value_counts()
        print(f"\n🏎️ Team Performance:")
        for team, count in team_counts.items():
            print(f"  • {team}: {count} race(s)")
        
        print("\n" + "=" * 60)
        print("✅ PREDICTION COMPLETE!")
        print("=" * 60)
        print(f"📊 CSV File: {csv_file}")
        print(f"🌐 HTML File: {html_file}")
        print("\n📋 2025 PREDICTED WINNERS:")
        print(predictions[['Round', 'Race Name', 'Predicted Winner', 'Predicted Team', 'Confidence']].to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✅ All predictions based on REAL race data!")
        print("📡 Data source: OpenF1 API (api.openf1.org)")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()