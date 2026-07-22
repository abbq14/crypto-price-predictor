import ccxt
import pandas as pd
import ta

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=1000)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

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

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df.dropna(inplace=True)

df.to_csv('data/historical_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Feature columns: {[c for c in df.columns if c != 'Target']}")
print(f"Class distribution:\n{df['Target'].value_counts().to_string()}")
print(f"\nDate range: {df.index[0]} to {df.index[-1]}")
print("data/historical_data.csv saved successfully.")
