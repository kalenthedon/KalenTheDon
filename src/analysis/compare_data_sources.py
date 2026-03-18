import pandas as pd

from src.data.hyperliquid import load_or_fetch as load_hl
from src.data.binance_futures import load_or_fetch as load_binance


def compute_returns(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change().dropna()


def run_comparison(
    symbol: str = "ETH",
    interval: str = "1h",
    lookback_weeks: int = 28,
) -> None:
    print("\n=== Loading datasets ===")

    hl = load_hl(symbol, interval, lookback_weeks)
    bn = load_binance(symbol, interval, lookback_weeks)

    if hl.empty or bn.empty:
        raise RuntimeError("One of the datasets is empty.")

    print(f"Hyperliquid rows: {len(hl)} ({hl.index.min()} -> {hl.index.max()})")
    print(f"Binance rows:     {len(bn)} ({bn.index.min()} -> {bn.index.max()})")

    # Align overlap window
    start = max(hl.index.min(), bn.index.min())
    end = min(hl.index.max(), bn.index.max())

    hl = hl.loc[start:end]
    bn = bn.loc[start:end]

    print("\n=== Overlap window ===")
    print(f"{start} -> {end}")
    print(f"HL rows: {len(hl)}, Binance rows: {len(bn)}")

    # Ensure exact alignment
    merged = hl.join(bn, how="inner", lsuffix="_hl", rsuffix="_bn")

    print(f"\nAligned rows: {len(merged)}")

    if len(merged) < 1000:
        print("⚠️ Very small overlap — results may be noisy")

    # Returns
    hl_ret = compute_returns(merged[["Close_hl"]].rename(columns={"Close_hl": "Close"}))
    bn_ret = compute_returns(merged[["Close_bn"]].rename(columns={"Close_bn": "Close"}))

    aligned = pd.concat([hl_ret, bn_ret], axis=1).dropna()
    aligned.columns = ["hl_ret", "bn_ret"]

    print("\n=== Return comparison ===")

    corr = aligned["hl_ret"].corr(aligned["bn_ret"])
    mae = (aligned["hl_ret"] - aligned["bn_ret"]).abs().mean()

    print(f"Return correlation: {corr:.4f}")
    print(f"Mean abs return diff: {mae:.6f}")

    # Volatility comparison
    hl_vol = aligned["hl_ret"].std()
    bn_vol = aligned["bn_ret"].std()
    vol_ratio = hl_vol / bn_vol if bn_vol != 0 else float("nan")

    print("\n=== Volatility comparison ===")
    print(f"HL vol: {hl_vol:.6f}")
    print(f"BN vol: {bn_vol:.6f}")
    print(f"Vol ratio (HL/BN): {vol_ratio:.4f}")

    # Price comparison
    price_diff = (merged["Close_hl"] - merged["Close_bn"]).abs()
    rel_price_diff = price_diff / merged["Close_bn"]

    print("\n=== Price comparison ===")
    print(f"Mean abs price diff: {price_diff.mean():.4f}")
    print(f"Mean relative diff: {rel_price_diff.mean():.6f}")
    print(f"Max relative diff: {rel_price_diff.max():.6f}")

    # Missing alignment check
    expected_len = int((end - start) / pd.Timedelta(hours=1)) + 1
    alignment_ratio = len(merged) / expected_len

    print("\n=== Alignment check ===")
    print(f"Expected bars: {expected_len}")
    print(f"Actual aligned bars: {len(merged)}")
    print(f"Alignment ratio: {alignment_ratio:.4f}")

    print("\n=== Verdict guide ===")
    print("- Correlation > 0.95 → strong proxy")
    print("- Correlation 0.90–0.95 → usable with caution")
    print("- Correlation < 0.90 → weak proxy")
    print("- Rel diff < 0.001 → very tight")
    print("- Rel diff > 0.005 → meaningful divergence")


if __name__ == "__main__":
    run_comparison()