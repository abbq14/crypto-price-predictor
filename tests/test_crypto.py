import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))


class TestPredictFunction(unittest.TestCase):
    @patch("main.joblib")
    @patch("main.ccxt")
    @patch("main.pd")
    def test_predict_calls_model(self, mock_pd, mock_ccxt, mock_joblib):
        import pandas as pd
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_joblib.load.return_value = mock_model
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = [
            [1700000000000, 50000, 51000, 49000, 50500, 100]
        ] * 100
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_pd.DataFrame.return_value = mock_df
        fake_row = pd.Series({k: 0.5 for k in [
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'SMA_20', 'SMA_50', 'BB_upper', 'BB_middle', 'BB_lower', 'Volume'
        ]})
        mock_df.iloc.__getitem__.return_value = fake_row
        mock_df.__getitem__.return_value = fake_row
        mock_df.__contains__.return_value = True
        from main import predict
        predict("BTC/USDT")
        mock_joblib.load.assert_called_once_with('models/trading_model.pkl')
        mock_model.predict.assert_called_once()

    @patch("main.joblib")
    @patch("main.ccxt")
    @patch("main.pd")
    def test_predict_down(self, mock_pd, mock_ccxt, mock_joblib):
        import pandas as pd
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_joblib.load.return_value = mock_model
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        mock_exchange.fetch_ohlcv.return_value = [
            [1700000000000, 50000, 51000, 49000, 50500, 100]
        ] * 100
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_pd.DataFrame.return_value = mock_df
        fake_row = pd.Series({k: 0.5 for k in [
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'SMA_20', 'SMA_50', 'BB_upper', 'BB_middle', 'BB_lower', 'Volume'
        ]})
        mock_df.iloc.__getitem__.return_value = fake_row
        mock_df.__getitem__.return_value = fake_row
        mock_df.__contains__.return_value = True
        from main import predict
        result = predict("BTC/USDT")
        self.assertEqual(result['direction'], "DOWN")


class TestCalculateFeatures(unittest.TestCase):
    @patch("live_bot.ta")
    def test_calculate_features_returns_df(self, mock_ta):
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2026-01-01", periods=100, freq="h")
        df = pd.DataFrame({"Close": np.random.randn(100).cumsum() + 50000}, index=dates)
        mock_rsi = MagicMock()
        mock_rsi.rsi.return_value = pd.Series(np.random.rand(100))
        mock_ta.momentum.RSIIndicator.return_value = mock_rsi
        mock_macd = MagicMock()
        mock_macd.macd.return_value = pd.Series(np.random.rand(100))
        mock_macd.macd_signal.return_value = pd.Series(np.random.rand(100))
        mock_macd.macd_diff.return_value = pd.Series(np.random.rand(100))
        mock_ta.trend.MACD.return_value = mock_macd
        mock_sma20 = MagicMock()
        mock_sma20.sma_indicator.return_value = pd.Series(np.random.rand(100))
        mock_ta.trend.SMAIndicator.return_value = mock_sma20
        mock_bb = MagicMock()
        mock_bb.bollinger_hband.return_value = pd.Series(np.random.rand(100))
        mock_bb.bollinger_mavg.return_value = pd.Series(np.random.rand(100))
        mock_bb.bollinger_lband.return_value = pd.Series(np.random.rand(100))
        mock_ta.volatility.BollingerBands.return_value = mock_bb
        from live_bot import calculate_features
        result = calculate_features(df.copy())
        expected_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff',
                         'SMA_20', 'SMA_50', 'BB_upper', 'BB_middle', 'BB_lower']
        for col in expected_cols:
            self.assertIn(col, result.columns)


class TestFeatureColumns(unittest.TestCase):
    def test_live_bot_feature_cols(self):
        from live_bot import FEATURE_COLS
        expected = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff',
                    'SMA_20', 'SMA_50', 'BB_upper', 'BB_middle', 'BB_lower', 'Volume']
        self.assertEqual(set(FEATURE_COLS), set(expected))

    def test_main_uses_same_features(self):
        import inspect
        import main
        src = inspect.getsource(main.predict)
        expected_features = ['RSI', 'MACD', 'MACD_signal', 'MACD_diff',
                             'SMA_20', 'SMA_50', 'BB_upper', 'BB_middle', 'BB_lower', 'Volume']
        for f in expected_features:
            self.assertIn(f"'{f}'", src)


class TestLiveBotConfig(unittest.TestCase):
    def test_constants_defined(self):
        from live_bot import SYMBOL, TIMEFRAME, WS_URL, MODEL_PATH, FEATURE_COLS
        self.assertEqual(SYMBOL, 'BTC/USDT')
        self.assertEqual(TIMEFRAME, '5m')
        self.assertTrue(WS_URL.startswith('wss://'))
        self.assertEqual(MODEL_PATH, 'models/trading_model.pkl')
        self.assertEqual(len(FEATURE_COLS), 10)

    def test_env_vars_read(self):
        from live_bot import API_KEY, SECRET
        self.assertIsNotNone(API_KEY)


class TestBacktestConfig(unittest.TestCase):
    def test_constants(self):
        from backtest import INITIAL_BALANCE, POSITION_SIZE, FEE_RATE
        self.assertEqual(INITIAL_BALANCE, 1000.0)
        self.assertEqual(POSITION_SIZE, 0.2)
        self.assertEqual(FEE_RATE, 0.001)


if __name__ == '__main__':
    unittest.main()
