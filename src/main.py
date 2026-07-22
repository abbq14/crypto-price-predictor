import argparse
import joblib
import ccxt
import pandas as pd
import ta

def fetch_features(symbol, timeframe='1h', limit=100):
    """Fetch recent OHLCV data for `symbol` and compute the model's technical indicators."""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()

    return df


def predict(symbol, model_path='models/trading_model.pkl'):
    """Fetch the latest market data for `symbol` and return the model's direction prediction.

    Returns a dict with symbol, current_price, direction, confidence (probability of
    the predicted class), and the full indicator dataframe (for charting).
    """
    model = joblib.load(model_path)
    df = fetch_features(symbol)

    last_complete = df.iloc[-2]
    features = pd.DataFrame([{
        'RSI': last_complete['RSI'],
        'MACD': last_complete['MACD'],
        'MACD_signal': last_complete['MACD_signal'],
        'MACD_diff': last_complete['MACD_diff'],
        'SMA_20': last_complete['SMA_20'],
        'SMA_50': last_complete['SMA_50'],
        'BB_upper': last_complete['BB_upper'],
        'BB_middle': last_complete['BB_middle'],
        'BB_lower': last_complete['BB_lower'],
        'Volume': last_complete['Volume'],
    }])

    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = proba[pred]
    current_price = df['Close'].iloc[-1]
    direction = "UP" if pred == 1 else "DOWN"

    return {
        'symbol': symbol,
        'current_price': current_price,
        'direction': direction,
        'confidence': confidence,
        'df': df,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crypto Spot Trading Prediction Tool')
    subparsers = parser.add_subparsers(dest='command')
    predict_parser = subparsers.add_parser('predict', help='Predict price direction')
    predict_parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair (e.g., BTC/USDT)')
    args = parser.parse_args()
    if args.command == 'predict':
        try:
            result = predict(args.symbol)
            print(f"{'Symbol:':<20}{result['symbol']}")
            print(f"{'Current Price:':<20}{result['current_price']:.2f}")
            print(f"{'Prediction:':<20}{result['direction']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        parser.print_help()
