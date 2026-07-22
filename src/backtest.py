import pandas as pd
import joblib
import matplotlib.pyplot as plt

INITIAL_BALANCE = 1000.0
POSITION_SIZE = 0.2
FEE_RATE = 0.001

df = pd.read_csv('data/historical_data.csv', index_col='timestamp')
model = joblib.load('models/trading_model.pkl')

feature_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'SMA_20', 'SMA_50',
                'BB_upper', 'BB_middle', 'BB_lower', 'Volume']

df['Signal'] = model.predict(df[feature_cols])

balance = INITIAL_BALANCE
crypto = 0.0
equity_curve = []
total_trades = 0
winning_trades = 0
buy_price = None

for idx, row in df.iterrows():
    price = row['Close']

    if row['Signal'] == 1 and crypto == 0:
        invest = balance * POSITION_SIZE
        fee = invest * FEE_RATE
        crypto = (invest - fee) / price
        balance -= invest
        buy_price = price
        total_trades += 1

    elif row['Signal'] == 0 and crypto > 0:
        proceeds = crypto * price
        fee = proceeds * FEE_RATE
        balance += proceeds - fee
        if price > buy_price:
            winning_trades += 1
        crypto = 0.0
        buy_price = None

    equity = balance + crypto * price
    equity_curve.append(equity)

df['Equity'] = equity_curve

final_equity = df['Equity'].iloc[-1]
roi = ((final_equity - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

print(f"Starting Balance: $1000.00")
print(f"Final Equity:     ${final_equity:.2f}")
print(f"Total ROI:        {roi:.2f}%")
print(f"Total Trades:     {total_trades}")
print(f"Win Rate:         {win_rate:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Equity'], label='Portfolio Value', color='blue')
plt.title('Model Backtest Equity Curve (BTC/USDT)')
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('docs/equity_curve.png')
print("\ndocs/equity_curve.png saved successfully.")

print("\n=== HEAD (first 5 rows) ===")
print(df[['Close', 'Signal']].head().to_string())
print("\n=== TAIL (last 5 rows) ===")
print(df[['Close', 'Signal']].tail().to_string())
