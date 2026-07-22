import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

df = pd.read_csv('data/historical_data.csv', index_col='timestamp')

feature_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'SMA_20', 'SMA_50',
                'BB_upper', 'BB_middle', 'BB_lower', 'Volume']
X = df[feature_cols]
y = df['Target']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

joblib.dump(model, 'models/trading_model.pkl')

print(f"Model trained and saved as models/trading_model.pkl")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
