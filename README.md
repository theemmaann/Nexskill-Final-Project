


📈 Multi-Stock Market Analysis & AI Research
A practical end-to-end financial data analysis and machine learning project built with Python.
This repository analyzes historical stock-market data for Adobe (ADBE), Microsoft (MSFT), Oracle (ORCL), and Salesforce (CRM) and combines exploratory data analysis, technical indicators, statistical analysis, machine learning, anomaly detection, market-regime clustering, Monte Carlo simulation, and deep learning.

Note: This project is for educational and research purposes only. It is not financial advice or a trading recommendation.

🚀 Project Overview
The project follows a complete data-to-model workflow:

Raw Stock Data
      ↓
Data Cleaning & Validation
      ↓
Exploratory Data Analysis
      ↓
Return & Risk Analysis
      ↓
Technical Feature Engineering
      ↓
Statistical Analysis
      ↓
Machine Learning
      ↓
Market Regime Detection
      ↓
Anomaly Detection
      ↓
Trading Strategy Evaluation
      ↓
Monte Carlo Simulation
      ↓
Deep Learning
      ↓
Results & Visualizations
The same research pipeline is applied across multiple technology companies so their historical market behavior can be studied using a consistent methodology.

🏢 Stocks Covered
Company	Ticker	Dataset Period
Adobe	ADBE	1986 – Dec 2024
Microsoft	MSFT	Historical data
Oracle	ORCL	2019 – Nov 2024
Salesforce	CRM	2004 – Dec 2024
The historical datasets are sourced from Yahoo Finance.

🔍 What This Project Does
1. Data Loading & Cleaning
Each stock dataset is loaded using Pandas and processed before analysis.

The pipeline checks for:

Missing values

Duplicate rows

Duplicate dates

Invalid OHLC relationships

Non-positive prices

Negative trading volume

Invalid and incomplete records are removed before further analysis.

2. Exploratory Data Analysis
The project examines:

Open price

High price

Low price

Close price

Adjusted close

Trading volume

Daily returns

Return distributions

Correlations

Monthly return behavior

Weekday return behavior

Summary statistics are also generated for the cleaned datasets.

📊 Financial & Statistical Analysis
The project calculates several important market statistics.

Returns
Daily returns

Log returns

Cumulative returns

Risk Metrics
Annualized volatility

Maximum drawdown

Skewness

Kurtosis

CAGR

Technical Features
The feature-engineering pipeline includes:

Moving averages: MA20, MA50, MA100, MA200

5-day momentum

20-day momentum

Rolling volatility

High-low range

Opening gap

Volume ratio

RSI-14

MACD

MACD signal

Bollinger Bands

ATR-14

These features are later used for statistical analysis and machine learning.

🤖 Machine Learning
The project uses both classification and regression approaches.

Classification
The target is next-day market direction:

1 → Next-day return is positive
0 → Next-day return is not positive
Models implemented:

Logistic Regression

Random Forest Classifier

A simple majority-class baseline is also calculated for comparison.

Classification Evaluation
Accuracy

Confusion Matrix

Regression
The project predicts next-day return as a continuous value.

Models implemented:

Linear Regression

Random Forest Regressor

The models are compared against a mean-return baseline.

Regression Evaluation
RMSE

R² Score

🧩 Feature Importance
Random Forest classification is used to estimate the relative importance of engineered market features.

The project saves feature-importance results as CSV files for further inspection.

🧭 Market Regime Detection
The project explores different market conditions using unsupervised learning.

Standardization
Market features are standardized using:

StandardScaler
PCA
Principal Component Analysis is used to reduce the feature space to two dimensions for visualization.

K-Means
K-Means clustering identifies four market regimes based on features such as:

Daily return

Rolling volatility

Momentum

Volume ratio

RSI

The resulting regimes can be inspected through their average characteristics and frequency.

🚨 Anomaly Detection
The project uses:

Isolation Forest

to identify unusual market observations.

The anomaly pipeline:

Market Features
      ↓
StandardScaler
      ↓
Isolation Forest
      ↓
Anomaly Flags
      ↓
Anomalous Dates & Returns
The detected anomaly dates and returns are exported to CSV files.

📈 Strategy Analysis
The Random Forest classification model is also used to create a simple directional signal.

A probability threshold is applied to determine whether the model produces a positive-return signal.

The resulting strategy is compared with a basic Buy & Hold approach.

Strategy Metrics
Sharpe Ratio

Sortino Ratio

CAGR

Maximum Drawdown

Calmar Ratio

The comparison is intended as an experimental research exercise rather than a production trading system.

🎲 Monte Carlo Simulation
The project performs a Monte Carlo simulation of future price paths.

Configuration includes:

10,000 simulations

252 simulated trading days

Historical mean daily return

Historical daily-return standard deviation

The simulation produces:

Simulated price paths

Ending-price distribution

Mean simulated ending price

5th percentile

95th percentile

The generated results are useful for understanding the range of possible outcomes under the simulation assumptions.

🧠 Deep Learning
The project also experiments with neural-network models for next-day return prediction.

Dense Neural Network
A TensorFlow/Keras feed-forward neural network is used with:

Input
 ↓
Dense(32, ReLU)
 ↓
Dense(16, ReLU)
 ↓
Dense(1)
The input features are scaled using MinMaxScaler.

LSTM
A Long Short-Term Memory network is used to experiment with sequential market data.

The workflow creates sequences of 20 time steps and feeds them into an LSTM model.

Architecture:

20-step sequence
      ↓
LSTM(32)
      ↓
Dense(16, ReLU)
      ↓
Dense(1)
The LSTM and dense neural network are evaluated using:

RMSE

R²

Their results are compared with traditional machine-learning models and a baseline.

📉 Visualizations
The analysis generates combined research charts including:

Stock closing price over time

Daily-return distribution

Correlation heatmap

Trading volume

PCA market-regime visualization

Strategy vs Buy & Hold

Monte Carlo simulated paths

Monte Carlo ending-price distribution

Charts are stored in the corresponding output directories.

📁 Repository Structure
Final Assesment/
│
├── Adobe_Stock/
│   ├── ADBE.py
│   ├── Adobe.csv
│   ├── ReadMe.txt
│   └── output/
│       ├── ADBE_all_charts.png
│       ├── ADBE_anomaly_days.csv
│       ├── ADBE_feature_importance.csv
│       ├── ADBE_month_avg_return.csv
│       ├── ADBE_summary_stats.csv
│       └── ADBE_weekday_avg_return.csv
│
├── Microsoft_Stock/
│   ├── code.py
│   ├── Microsoft_stock_history.csv
│   ├── Microsoft_stock_info.csv
│   ├── Microsoft_stock_action.csv
│   ├── Microsoft_stock_dividends.csv
│   ├── Microsoft_stock_spilts.csv
│   ├── ReadMe.txt
│   └── output/
│
├── Oracle_Stock/
│   ├── code.py
│   ├── oracle.csv
│   ├── ReadMe.txt
│   └── output/
│
├── Salesforce_Stock/
│   ├── code.py
│   ├── Salesforce (CRM) From 2004 To Dec-2024.csv
│   ├── ReadMe.txt
│   └── output/
│
└── output/
    └── generated analysis results
🛠️ Technologies Used
Programming
Python

Data Analysis
NumPy

Pandas

Visualization
Matplotlib

Seaborn

Statistics
SciPy

Machine Learning
Scikit-learn

Deep Learning
TensorFlow

Keras

⚙️ Installation
Make sure Python is installed, then install the required libraries:

pip install -r requirements.txt
▶️ Running the Project
Run the scripts from the repository root so that the relative dataset paths work correctly.

Adobe
python Adobe_Stock/ADBE.py
Microsoft
python Microsoft_Stock/code.py
Oracle
python Oracle_Stock/code.py
Salesforce
python Salesforce_Stock/code.py
The scripts create or update analysis outputs in the relevant output directory.

📦 Output Files
The project generates CSV files containing:

Summary statistics

Feature importance

Monthly average returns

Weekday average returns

Detected anomaly days

It also generates PNG files containing the main research visualizations.

🧪 Methodology
The project uses a chronological 80/20 train-test split for the main supervised-learning experiments rather than randomly shuffling the time series.

This helps preserve the temporal order of historical observations.

The general methodology is:

Historical Data
      ↓
Clean Data
      ↓
Create Returns
      ↓
Engineer Features
      ↓
Create Next-Day Target
      ↓
Chronological Train/Test Split
      ↓
Train Models
      ↓
Evaluate Models
      ↓
Compare With Baselines
⚠️ Limitations
This project is an educational financial-data research exercise and should not be treated as a production-grade trading system.

Important limitations include:

Historical market behavior does not guarantee future results.

The Monte Carlo simulation relies on simplified return assumptions.

Financial markets contain factors that historical price features cannot fully capture.

Transaction costs, slippage, liquidity, and taxes are not modeled.

The strategy experiment is not a live trading system.

More rigorous walk-forward validation would be appropriate for production research.

Some models and preprocessing steps could be further optimized to reduce potential time-series leakage.

📌 Future Improvements
Potential improvements include:

Walk-forward validation

Time-series cross-validation

Hyperparameter optimization

More robust feature selection

Transaction-cost modeling

Portfolio-level analysis

More advanced forecasting models

Transformer-based time-series models

Better uncertainty estimation

Experiment tracking

Model persistence and deployment

Interactive dashboards

Automated data updates

🎯 Learning Objectives
This project demonstrates practical experience with:

Financial time-series data

Data cleaning

Exploratory data analysis

Feature engineering

Statistical analysis

Technical indicators

Supervised machine learning

Unsupervised learning

Classification

Regression

Dimensionality reduction

Clustering

Anomaly detection

Strategy evaluation

Monte Carlo simulation

Neural networks

LSTM networks

Data visualization

End-to-end analytical workflows

👩‍💻 Author
Noor Ul Eman

Python & AI/ML enthusiast exploring Data Science, Machine Learning, Deep Learning, and practical AI applications through hands-on projects.

⭐ If you find this project useful
Feel free to explore the code, datasets, generated results, and experiments.

Built with Python, curiosity, and a lot of experimentation.
