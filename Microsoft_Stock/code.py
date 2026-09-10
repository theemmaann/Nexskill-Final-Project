import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv('Microsoft_Stock/Microsoft_stock_history.csv', delimiter=',')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print("rows:", len(df), "cols:", list(df.columns))
print(df.head(3))
print(df.tail(3))

missing_values = df.isna().sum().sum()
duplicate_rows = df.duplicated().sum()
duplicate_dates = df["Date"].duplicated().sum()
high_ok = df["High"] >= df[["Open", "Close", "Low"]].max(axis=1)
low_ok = df["Low"] <= df[["Open", "Close", "High"]].min(axis=1)
ohlc_violations = (~(high_ok & low_ok)).sum()
nonpositive_prices = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
negative_volume = (df["Volume"] < 0).sum()
print("missing:", missing_values, "dupes:", duplicate_rows, "dupe dates:", duplicate_dates)
print("ohlc violations:", ohlc_violations, "bad price rows:", nonpositive_prices, "neg vol:", negative_volume)

df = df.drop_duplicates().drop_duplicates(subset="Date")
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
df = df[df["Volume"] >= 0]
df = df.sort_values("Date").reset_index(drop=True)
print("rows after cleaning:", len(df))

print(df[["Open", "High", "Low", "Close", "Volume"]].describe())

df["Daily_Return"] = df["Close"].pct_change()
annual_volatility = df["Daily_Return"].std() * np.sqrt(252)
start_price = df["Close"].iloc[0]
end_price = df["Close"].iloc[-1]
n_years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
cagr = (end_price / start_price) ** (1 / n_years) - 1
cumulative = (1 + df["Daily_Return"].fillna(0)).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
print("cagr:", round(cagr * 100, 2), "vol:", round(annual_volatility * 100, 2), "max dd:", round(max_drawdown * 100, 2))
print("skew:", round(df["Daily_Return"].skew(), 3), "kurtosis:", round(df["Daily_Return"].kurtosis(), 3))

df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["MA100"] = df["Close"].rolling(100).mean()
df["MA200"] = df["Close"].rolling(200).mean()
df["Momentum_5"] = df["Close"] / df["Close"].shift(5)
df["Momentum_20"] = df["Close"] / df["Close"].shift(20)
df["Vol_5"] = df["Daily_Return"].rolling(5).std()
df["Vol_10"] = df["Daily_Return"].rolling(10).std()
df["Vol_20"] = df["Daily_Return"].rolling(20).std()
df["Vol_60"] = df["Daily_Return"].rolling(60).std()
df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
df["Gap"] = df["Open"] - df["Close"].shift(1)
df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df["RSI_14"] = 100 - (100 / (1 + rs))

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["BB_Mid"] = df["MA20"]
df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()

prev_close = df["Close"].shift(1)
tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
df["ATR_14"] = tr.rolling(14).mean()

print(df.tail(3))

df["Weekday"] = df["Date"].dt.day_name()
weekday_avg = df.groupby("Weekday")["Daily_Return"].mean().sort_values(ascending=False)
print(weekday_avg)

df["Month"] = df["Date"].dt.month
month_avg = df.groupby("Month")["Daily_Return"].mean()
print(month_avg)

df["Next_Day_Return"] = df["Daily_Return"].shift(-1)
feature_cols = ["Momentum_5", "Momentum_20", "Vol_20", "RSI_14", "MACD", "Volume_Ratio", "HL_Range"]
for col in feature_cols:
    valid = df[[col, "Next_Day_Return"]].dropna()
    r, p = stats.pearsonr(valid[col], valid["Next_Day_Return"])
    print(col, round(r, 4), round(p, 4))

model_features = ["Momentum_5", "Momentum_20", "Vol_20", "RSI_14", "MACD", "Volume_Ratio", "HL_Range", "Gap", "ATR_14"]
df["Target_Direction"] = (df["Next_Day_Return"] > 0).astype(int)
model_df = df[model_features + ["Next_Day_Return", "Target_Direction"]].dropna()

X = model_df[model_features]
y_class = model_df["Target_Direction"]
y_reg = model_df["Next_Day_Return"]
split_point = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_class_train, y_class_test = y_class.iloc[:split_point], y_class.iloc[split_point:]
y_reg_train, y_reg_test = y_reg.iloc[:split_point], y_reg.iloc[split_point:]
print("train:", len(X_train), "test:", len(X_test))

log_model = LogisticRegression(max_iter=1000).fit(X_train, y_class_train)
log_preds = log_model.predict(X_test)

rf_class_model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_class_train)
rf_class_preds = rf_class_model.predict(X_test)
baseline_class_preds = [y_class_train.mode()[0]] * len(y_class_test)

print("log acc:", round(accuracy_score(y_class_test, log_preds), 4))
print("rf acc:", round(accuracy_score(y_class_test, rf_class_preds), 4))
print("baseline acc:", round(accuracy_score(y_class_test, baseline_class_preds), 4))
print(confusion_matrix(y_class_test, rf_class_preds))

lin_model = LinearRegression().fit(X_train, y_reg_train)
lin_preds = lin_model.predict(X_test)
rf_reg_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_reg_train)
rf_reg_preds = rf_reg_model.predict(X_test)
baseline_reg_preds = [y_reg_train.mean()] * len(y_reg_test)

lin_rmse = np.sqrt(mean_squared_error(y_reg_test, lin_preds))
rf_rmse = np.sqrt(mean_squared_error(y_reg_test, rf_reg_preds))
baseline_rmse = np.sqrt(mean_squared_error(y_reg_test, baseline_reg_preds))
print("rmse - lin:", round(lin_rmse, 5), "rf:", round(rf_rmse, 5), "baseline:", round(baseline_rmse, 5))
print("r2 - lin:", round(r2_score(y_reg_test, lin_preds), 5), "rf:", round(r2_score(y_reg_test, rf_reg_preds), 5))

importance = pd.Series(rf_class_model.feature_importances_, index=model_features).sort_values(ascending=False)
print(importance)

regime_features = ["Daily_Return", "Vol_20", "Momentum_20", "Volume_Ratio", "RSI_14"]
regime_df = df[regime_features].dropna()
scaler = StandardScaler()
regime_scaled = scaler.fit_transform(regime_df)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(regime_scaled)
print("pca variance:", pca.explained_variance_ratio_)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
regime_labels = kmeans.fit_predict(regime_scaled)
regime_df = regime_df.copy()
regime_df["Regime"] = regime_labels
print(regime_df.groupby("Regime").mean())
print(regime_df["Regime"].value_counts())

anomaly_model = IsolationForest(contamination=0.02, random_state=42)
anomaly_flag = anomaly_model.fit_predict(regime_scaled)
regime_df["Is_Anomaly"] = anomaly_flag == -1
print("anomalies:", regime_df["Is_Anomaly"].sum())

anomaly_dates = df.loc[regime_df.index][regime_df["Is_Anomaly"]]["Date"]
anomaly_returns = df.loc[regime_df.index][regime_df["Is_Anomaly"]]["Daily_Return"]
anomaly_table = pd.DataFrame({"Date": anomaly_dates.values, "Return": anomaly_returns.values})
print(anomaly_table.sort_values("Return"))

actual_returns = y_reg_test.reset_index(drop=True)
rf_probs = rf_class_model.predict_proba(X_test)[:, 1]
signal = (rf_probs > 0.55).astype(int)
strategy_returns = actual_returns * signal
buy_hold_returns = actual_returns

def sharpe_ratio(r, n=252):
    return 0 if r.std() == 0 else (r.mean() / r.std()) * np.sqrt(n)

def sortino_ratio(r, n=252):
    down = r[r < 0]
    return 0 if down.std() == 0 else (r.mean() / down.std()) * np.sqrt(n)

def max_dd(r):
    cum = (1 + r.fillna(0)).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min()

def cagr_from(r, n=252):
    cum = (1 + r.fillna(0)).cumprod()
    yrs = len(r) / n
    return 0 if yrs == 0 or cum.iloc[-1] <= 0 else cum.iloc[-1] ** (1 / yrs) - 1

strat_sharpe = sharpe_ratio(strategy_returns)
strat_sortino = sortino_ratio(strategy_returns)
strat_dd = max_dd(strategy_returns)
strat_cagr = cagr_from(strategy_returns)
strat_calmar = strat_cagr / abs(strat_dd) if strat_dd != 0 else 0

bh_sharpe = sharpe_ratio(buy_hold_returns)
bh_sortino = sortino_ratio(buy_hold_returns)
bh_dd = max_dd(buy_hold_returns)
bh_cagr = cagr_from(buy_hold_returns)
bh_calmar = bh_cagr / abs(bh_dd) if bh_dd != 0 else 0

print("strategy sharpe:", round(strat_sharpe, 3), "cagr:", round(strat_cagr * 100, 2), "dd:", round(strat_dd * 100, 2))
print("buy&hold sharpe:", round(bh_sharpe, 3), "cagr:", round(bh_cagr * 100, 2), "dd:", round(bh_dd * 100, 2))

daily_mean = df["Daily_Return"].mean()
daily_std = df["Daily_Return"].std()
n_sims = 10000
n_days = 252
last_price = df["Close"].iloc[-1]
paths = np.zeros((n_days, n_sims))
paths[0] = last_price
rng = np.random.default_rng(42)
for d in range(1, n_days):
    paths[d] = paths[d - 1] * (1 + rng.normal(daily_mean, daily_std, n_sims))
ending_prices = paths[-1]
print("mc mean end price:", round(ending_prices.mean(), 2), "5th/95th pct:",
      round(np.percentile(ending_prices, 5), 2), round(np.percentile(ending_prices, 95), 2))

dl_df = df[model_features + ["Next_Day_Return"]].dropna()
dl_scaler = MinMaxScaler()
dl_X = dl_scaler.fit_transform(dl_df[model_features])
dl_y = dl_df["Next_Day_Return"].values
dl_split = int(len(dl_df) * 0.8)
dl_X_train, dl_X_test = dl_X[:dl_split], dl_X[dl_split:]
dl_y_train, dl_y_test = dl_y[:dl_split], dl_y[dl_split:]

dense_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(dl_X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])
dense_model.compile(optimizer="adam", loss="mse")
dense_model.fit(dl_X_train, dl_y_train, validation_split=0.1, epochs=30, batch_size=32, verbose=0)
dense_preds = dense_model.predict(dl_X_test, verbose=0).flatten()
dense_rmse = np.sqrt(mean_squared_error(dl_y_test, dense_preds))
dense_r2 = r2_score(dl_y_test, dense_preds)
print("dense rmse:", round(dense_rmse, 5))

seq_len = 20
def make_seqs(x, y, n):
    xs, ys = [], []
    for i in range(len(x) - n):
        xs.append(x[i:i+n])
        ys.append(y[i+n])
    return np.array(xs), np.array(ys)

lstm_X_train, lstm_y_train = make_seqs(dl_X_train, dl_y_train, seq_len)
lstm_X_test, lstm_y_test = make_seqs(dl_X_test, dl_y_test, seq_len)

lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_len, dl_X_train.shape[1])),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(lstm_X_train, lstm_y_train, validation_split=0.1, epochs=30, batch_size=32, verbose=0)
lstm_preds = lstm_model.predict(lstm_X_test, verbose=0).flatten()
lstm_rmse = np.sqrt(mean_squared_error(lstm_y_test, lstm_preds))
lstm_r2 = r2_score(lstm_y_test, lstm_preds)
print("lstm rmse:", round(lstm_rmse, 5))

print("rmse comparison - baseline:", round(baseline_rmse, 5), "lin:", round(lin_rmse, 5),
      "rf:", round(rf_rmse, 5), "dense:", round(dense_rmse, 5), "lstm:", round(lstm_rmse, 5))

summary_stats = pd.DataFrame([{
    "cagr_%": round(cagr * 100, 2), "volatility_%": round(annual_volatility * 100, 2),
    "max_drawdown_%": round(max_drawdown * 100, 2),
    "rf_classification_accuracy": round(accuracy_score(y_class_test, rf_class_preds), 4),
    "baseline_classification_accuracy": round(accuracy_score(y_class_test, baseline_class_preds), 4),
    "rf_regression_rmse": round(rf_rmse, 5), "dense_rmse": round(dense_rmse, 5), "lstm_rmse": round(lstm_rmse, 5),
    "baseline_rmse": round(baseline_rmse, 5),
    "strategy_sharpe": round(strat_sharpe, 3), "buy_hold_sharpe": round(bh_sharpe, 3),
    "strategy_cagr_%": round(strat_cagr * 100, 2), "buy_hold_cagr_%": round(bh_cagr * 100, 2),
    "monte_carlo_mean_ending_price": round(ending_prices.mean(), 2),
    "anomaly_days_found": int(regime_df["Is_Anomaly"].sum()),
}])
summary_stats.to_csv(f"{OUTPUT_DIR}/MSFT_summary_stats.csv", index=False)
weekday_avg.to_csv(f"{OUTPUT_DIR}/MSFT_weekday_avg_return.csv")
month_avg.to_csv(f"{OUTPUT_DIR}/MSFT_month_avg_return.csv")
importance.to_csv(f"{OUTPUT_DIR}/MSFT_feature_importance.csv")
anomaly_table.to_csv(f"{OUTPUT_DIR}/MSFT_anomaly_days.csv", index=False)

fig = plt.figure(figsize=(20, 22))
fig.suptitle("Microsoft (MSFT) - Full Research Summary", fontsize=18, fontweight="bold")

ax1 = fig.add_subplot(4, 2, 1)
ax1.plot(df["Date"], df["Close"], color="#1f77b4", linewidth=1)
ax1.set_title("Close Price Over Time")

ax2 = fig.add_subplot(4, 2, 2)
sns.histplot(df["Daily_Return"].dropna(), kde=True, ax=ax2, color="#ff7f0e")
ax2.set_title("Daily Return Distribution")

ax3 = fig.add_subplot(4, 2, 3)
corr = df[["Open", "High", "Low", "Close", "Volume", "Daily_Return"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
ax3.set_title("Correlation Heatmap")

ax4 = fig.add_subplot(4, 2, 4)
ax4.plot(df["Date"], df["Volume"], color="#2ca02c", linewidth=0.7)
ax4.set_title("Trading Volume Over Time")

ax5 = fig.add_subplot(4, 2, 5)
ax5.scatter(pca_result[:, 0], pca_result[:, 1], c=regime_labels, cmap="viridis", s=10)
ax5.set_title("Market Regimes (PCA + KMeans)")

ax6 = fig.add_subplot(4, 2, 6)
ax6.plot((1 + strategy_returns.fillna(0)).cumprod().values, label="Strategy")
ax6.plot((1 + buy_hold_returns.fillna(0)).cumprod().values, label="Buy and Hold")
ax6.set_title("Strategy vs Buy and Hold")
ax6.legend()

ax7 = fig.add_subplot(4, 2, 7)
ax7.plot(paths[:, :200], linewidth=0.5, alpha=0.5)
ax7.set_title("Monte Carlo Simulated Price Paths")

ax8 = fig.add_subplot(4, 2, 8)
ax8.hist(ending_prices, bins=50, color="#1f77b4")
ax8.axvline(last_price, color="red", linestyle="--", label="Current Price")
ax8.set_title("Monte Carlo Ending Price Distribution")
ax8.legend()

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUTPUT_DIR}/MSFT_all_charts.png", dpi=120)
plt.show()
print("done, saved to", OUTPUT_DIR)