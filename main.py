from src.backtests.run_ml_backtest import run

if __name__ == "__main__":
    run(symbol="ETH", interval="1h", lookback_weeks=100)
