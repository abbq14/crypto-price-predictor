import os
import json
import asyncio
import websockets
import ccxt
from dotenv import load_dotenv
import joblib
import pandas as pd
import ta

load_dotenv()

API_KEY = os.getenv('BINANCE_TESTNET_API_KEY')
SECRET = os.getenv('BINANCE_TESTNET_SECRET')

SYMBOL = 'BTC/USDT'
TIMEFRAME = '5m'
WS_URL = 'wss://stream.binance.com:9443/ws/btcusdt@kline_5m'

MODEL_PATH = 'models/trading_model.pkl'
FEATURE_COLS = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'SMA_20', 'SMA_50',
                'BB_upper', 'BB_middle', 'BB_lower', 'Volume']


def calculate_features(df):
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
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    return df


def seed_data():
    exchange = ccxt.binance({'options': {'defaultType': 'spot'}})
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = calculate_features(df)
    return df


async def stream_loop(model, start_time):
    df = seed_data()
    in_position = False
    entry_price = 0.0
    current_tp = 0.0
    current_sl = 0.0

    print(f"Seeded with {len(df)} historical candles. Connecting to WebSocket...")

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                async for message in ws:
                    data = json.loads(message)
                    k = data['k']
                    close_price = float(k['c'])
                    high_price = float(k['h'])
                    low_price = float(k['l'])
                    volume = float(k['v'])
                    candle_closed = k['x']
                    candle_open_time = pd.to_datetime(k['t'], unit='ms')

                    status = "IN_POSITION" if in_position else "AWAITING"
                    print(f"Live Price: ${close_price:.2f} | Candle Closed: {candle_closed} | {status}")

                    if candle_open_time == df.index[-1]:
                        df.iloc[-1, df.columns.get_loc('High')] = max(df.iloc[-1]['High'], high_price)
                        df.iloc[-1, df.columns.get_loc('Low')] = min(df.iloc[-1]['Low'], low_price)
                        df.iloc[-1, df.columns.get_loc('Close')] = close_price
                        df.iloc[-1, df.columns.get_loc('Volume')] = volume

                    if in_position:
                        if close_price >= current_tp:
                            print(f"\n[EXECUTED SELL - TAKE PROFIT] Exit: ${close_price:.2f} | Profit: ${close_price - entry_price:.2f}")
                            in_position = False
                            entry_price = 0.0
                            current_tp = 0.0
                            current_sl = 0.0
                        elif close_price <= current_sl:
                            print(f"\n[EXECUTED SELL - STOP LOSS] Exit: ${close_price:.2f} | Loss: ${close_price - entry_price:.2f}")
                            in_position = False
                            entry_price = 0.0
                            current_tp = 0.0
                            current_sl = 0.0

                    if candle_closed:
                        df = calculate_features(df)
                        latest = df.iloc[-1]

                        features = pd.DataFrame([{c: latest[c] for c in FEATURE_COLS}])
                        signal = model.predict(features)[0]
                        atr = latest['ATR']
                        price = latest['Close']

                        direction = "UP" if signal == 1 else "DOWN"
                        print(f"\n[CANDLE CLOSED] Price: ${price:.2f} | Signal: {direction} | ATR: ${atr:.2f}")

                        if in_position:
                            print(f"   Skipping (already in position)")
                        elif signal == 1:
                            entry_price = price
                            current_tp = price + 1.5 * atr
                            current_sl = price - 1.0 * atr
                            in_position = True
                            print(f"[EXECUTED BUY] Entry: ${entry_price:.2f} | TP: ${current_tp:.2f} | SL: ${current_sl:.2f}")
                        else:
                            print(f"   No trade (signal DOWN)")

                    if asyncio.get_event_loop().time() - start_time >= 30:
                        print("\nValidation period complete (30s). Shutting down.")
                        return

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket disconnected. Reconnecting in 3 seconds...")
            await asyncio.sleep(3)


async def main():
    model = joblib.load(MODEL_PATH)
    start_time = asyncio.get_event_loop().time()
    await stream_loop(model, start_time)


if __name__ == '__main__':
    asyncio.run(main())
