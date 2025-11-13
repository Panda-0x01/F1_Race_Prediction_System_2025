# F1 Race Prediction System (2025)



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
     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝     ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
    
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
    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    

## 📖 Overview
The **F1 Race Prediction System** is a machine learning project that predicts the **winners of the upcoming Formula 1 season (2025)** based on **historical race data (2023–2024)**.  
It uses the **OpenF1 API** to fetch real-time data, builds an encoded dataset with driver and team stats, and applies a **Random Forest Classifier** to forecast race outcomes.  
The project automatically exports results in **CSV** and **interactive HTML report** formats.

---

## 🚀 Features

- 📡 **Live Data Fetching:** Automatically retrieves the latest Formula 1 results via the [OpenF1 API](https://api.openf1.org/).  
- 🧠 **Machine Learning Engine:** Trains a Random Forest model using multi-year race stats.  
- 🏁 **Prediction Engine:** Predicts top drivers for the 2025 season (first 5 rounds).  
- 📊 **Visualization Ready:** Outputs CSV and styled HTML reports for analysis.  
- 💾 **Dataset Generator:** Cleans, encodes, and merges race results for model training.  
- 🧩 **Extensible:** Easy to adapt for other seasons, models, or visualization tools.  

---

## 🧱 Project Structure

F1-Predictor/
│                 <br>
├── f1_predictor.py      <br>
├── f1_predictions.csv   <br>
├── f1_predictions.html  <br>
├── requirements.txt     <br>
└── README.md            <br>


---

## ⚙️ How It Works

1. **Data Collection**  
   - The system connects to the OpenF1 API to fetch data for **2023–2024**.
   - Extracts race name, circuit, driver, team, and performance stats.

2. **Feature Engineering**  
   - Label encoding of categorical features (`Driver`, `Team`, `Circuit`, etc.).
   - Derived features like `TeamWinsPrevSeason` and `DriverPerformance`.

3. **Model Training**  
   - A **Random Forest Classifier** is trained to predict future winners.
   - Automatically tunes hyperparameters and handles categorical data.

4. **Prediction**  
   - Predicts likely winners for the **2025 season (Round 1–5)**.
   - Outputs driver, team, and probability scores.

5. **Reporting**  
   - Exports results to:
     - `f1_predictions.csv` → Data view  
     - `f1_predictions.html` → Neon-styled dashboard  

---

## 🧩 Dependencies

Install all required dependencies with:

```bash
pip install -r requirements.txt

Required Packages:

pandas
numpy
scikit-learn
requests
matplotlib

🖥️ Usage
Run the Predictor

python f1_predictor.py

Output

Console: Displays training progress and top 2025 race predictions.

CSV: Stores raw prediction data in f1_predictions.csv.

HTML: Generates styled race report (f1_predictions.html) with tables and charts.

📊 Example Output

Top 5 Predicted Drivers (2025):

| Rank | Driver          | Team            | Win Probability |
| ---- | --------------- | --------------- | --------------- |
| 1    | Max Verstappen  | Red Bull Racing | 0.82            |
| 2    | Lando Norris    | McLaren         | 0.67            |
| 3    | Charles Leclerc | Ferrari         | 0.58            |
| 4    | Lewis Hamilton  | Mercedes        | 0.49            |
| 5    | George Russell  | Mercedes        | 0.46            |

🧠 Model Details

| Component              | Description                       |
| ---------------------- | --------------------------------- |
| **Model Type**         | RandomForestClassifier            |
| **Core Features**      | Year, Circuit, Team, Driver Stats |
| **Training Data**      | 2023–2024 seasons (OpenF1 API)    |
| **Predicted Target**   | Race Winner                       |
| **Performance Metric** | Accuracy & Probability Ranking    |

📁 API Reference

OpenF1 Endpoint Used:

https://api.openf1.org/v1/results

Parameters:

year: Fetch race data for specific years (2023, 2024).

round: Round number in the season.

driver_number, position, team_name: Used for filtering and aggregation.

🧑‍💻 Developer Guide

Modify prediction years or rounds in the code section:

for year in [2023, 2024]:  # Change years here
for round_num in range(1, 6):  # Change race rounds

Adjust model type or hyperparameters for experimentation.

You can integrate visualizations via Matplotlib or Plotly.

🌐 Future Enhancements

 Integrate Flask Dashboard for real-time race visualization.

 Add live API streaming for current season updates.

 Implement Neural Network for driver performance analysis.

 Add F1 car telemetry-based metrics (lap times, DRS usage).

 Deploy model via Docker or Streamlit Web App.

🏆 Credits

Developer: Drumil Nikhare (HAHAHAHAHA)

Model: Random Forest (Scikit-Learn)

Data Source: OpenF1 API

Tools Used: Python, Pandas, NumPy, Matplotlib, Scikit-Learn

📜 License

This project is released under the MIT License — free for personal and academic use.

Attribution to the original developer is appreciated. ❤️


