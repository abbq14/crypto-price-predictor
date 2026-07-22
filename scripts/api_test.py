import ccxt

exchange = ccxt.binance()
ticker = exchange.fetch_ticker('BTC/USDT')
print(f"Symbol: {ticker['symbol']}")
print(f"Last Price: {ticker['last']}")
print(f"24h High: {ticker['high']}")
print(f"24h Low: {ticker['low']}")
print(f"24h Volume: {ticker['quoteVolume']}")
print("API connection successful!")
