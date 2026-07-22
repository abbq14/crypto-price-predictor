import pandas as pd
import joblib
import matplotlib.pyplot as plt

from train_model import FEATURE_COLS, TRAIN_SPLIT

INITIAL_BALANCE = 1000.0
POSITION_SIZE = 0.2
FEE_RATE = 0.001


def run_backtest(df, model):
    df['Signal'] = model.predict(df[FEATURE_COLS])

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
    return df, total_trades, winning_trades


def main():
    full_df = pd.read_csv('data/historical_data.csv', index_col='timestamp')
    model = joblib.load('models/trading_model.pkl')

    # Backtest only the rows the model was never trained on. An unpruned
    # RandomForest scores ~1.0 in-sample, so replaying the full file mostly
    # measures memorisation: it reports a ~96% win rate that collapses to
    # roughly coin-flip performance on unseen data. Anything above this split
    # point is the only honest estimate of how the strategy generalises.
    split_idx = int(len(full_df) * TRAIN_SPLIT)
    df = full_df.iloc[split_idx:].copy()

    df, total_trades, winning_trades = run_backtest(df, model)

    final_equity = df['Equity'].iloc[-1]
    roi = ((final_equity - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"Out-of-sample backtest: {len(df)} held-out candles "
          f"({df.index[0]} to {df.index[-1]})")
    print(f"Starting Balance: $1000.00")
    print(f"Final Equity:     ${final_equity:.2f}")
    print(f"Total ROI:        {roi:.2f}%")
    print(f"Total Trades:     {total_trades}")
    print(f"Win Rate:         {win_rate:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Equity'], label='Portfolio Value', color='blue')
    plt.title('Out-of-Sample Backtest Equity Curve (BTC/USDT, held-out 20%)')
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


if __name__ == '__main__':
    main()
