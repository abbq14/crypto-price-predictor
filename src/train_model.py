import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

FEATURE_COLS = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'SMA_20', 'SMA_50',
                'BB_upper', 'BB_middle', 'BB_lower', 'Volume']

# Fraction of the (chronologically ordered) dataset used for training. The
# remainder is held out for evaluation, and backtest.py imports this so the
# backtest scores the same unseen rows the metrics below are computed on.
TRAIN_SPLIT = 0.8


def main():
    df = pd.read_csv('data/historical_data.csv', index_col='timestamp')

    X = df[FEATURE_COLS]
    y = df['Target']

    # Split by time, not at random: shuffling would let the model train on
    # rows that come after the ones it is tested on.
    split_idx = int(len(df) * TRAIN_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    joblib.dump(model, 'models/trading_model.pkl')

    print(f"Model trained and saved as models/trading_model.pkl")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"\nBaseline (always predict the majority class): {baseline:.4f}")
    print(f"In-sample accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f} "
          f"(~1.0 is expected: an unpruned forest memorises\n"
          f" its training rows, which is why only the held-out metrics above are meaningful)")


if __name__ == '__main__':
    main()
