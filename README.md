# ⚡ Electric Vehicle (EV) Registration Forecasting

A modern, machine-learning-powered web dashboard built to predict the growth of Electric Vehicle registrations in Washington State. 

This project leverages Deep Learning to forecast future EV adoption rates and presents the insights through a bespoke **Swiss Modernism / Industrial Editorial** interface, complete with high-end micro-interactions.

---

## 📊 Data Source & Preprocessing

The data originates from the official [Washington State Department of Licensing (data.wa.gov)](https://data.wa.gov/Transportation/Electric-Vehicle-Population-Data/f6w7-q2d2/data_preview).

**How the data is handled:**
- **Raw Data:** The original public dataset contains over 300,000+ individual rows, where each row represents a single registered vehicle (including Make, Model, VIN, County, etc.).
- **Time-Series Aggregation:** To perform predictive modeling, the raw dataset was aggregated into monthly intervals. Instead of tracking individual cars, the dataset was compressed into a **Time-Series format** (1 row = 1 month), tracking the total cumulative count of EVs from January 2017 to May 2026 (113 chronological data points).

## 🧠 Technical Approach & Architecture

> **Note:** This project utilizes **Time-Series Forecasting (Regression)**, not Classification. We are predicting continuous numerical values over time, rather than categorizing data into discrete classes.

### 1. Machine Learning Model (LSTM)
The predictive engine is a **Long Short-Term Memory (LSTM)** Neural Network built with TensorFlow/Keras. LSTMs are exceptionally well-suited for this task because they have an internal "memory" state that can learn long-term sequential patterns and seasonal trends in the data.

**Model Specifications:**
- **Architecture:** 2-layer LSTM (64 units -> 32 units) followed by Dense layers.
- **Sequence Lookback:** 12 months (The model looks at the past year of data to predict the next month).
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Accuracy:** The model achieved an impressive **Mean Absolute Percentage Error (MAPE) of 5.45%** during testing.

### 2. Frontend & UI Engineering
The dashboard is built using **Streamlit**, but heavily customized to break away from standard template constraints.
- **Design System:** High-contrast Swiss Modernism / Brutalism (Off-white backgrounds, sharp borders, bold typography using *Space Grotesk* and *JetBrains Mono*).
- **Animations:** Custom HTML injection to utilize **Anime.js** for smooth staggered reveals and dynamic number counting (60 FPS).
- **Visualizations:** Interactive, bespoke **Plotly** charts with custom hover tooltips and dynamic 95% confidence bands.

---

## 🚀 How to Run Locally

Follow these steps to run the forecast dashboard on your local machine.

### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Mikail79/ev_prediction.git
cd ev_prediction
```

### 2. Install Dependencies
It is recommended to create a virtual environment first, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
Execute the following command to start the Streamlit server:
```bash
streamlit run app.py
```

### 4. View the App
Open your web browser and navigate to:
**http://localhost:8501**

---

## 📁 Repository Structure
- `app.py`: The main Streamlit dashboard application and UI logic.
- `train_model.py`: The data pipeline and TensorFlow/Keras script used to build and train the LSTM model.
- `ev_population.csv`: The aggregated monthly time-series dataset.
- `ev_model.h5`: The pre-trained LSTM neural network.
- `scaler.pkl`: Scikit-learn MinMax scaler used to normalize/denormalize data.
- `requirements.txt`: Python package dependencies.
