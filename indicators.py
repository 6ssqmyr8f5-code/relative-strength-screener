import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "slope_windows": [60, 120, 250],
    "breakout_windows": [60, 120, 250],
    "ma_windows": [20, 60],
    "below_zero_window": 250,
    "volume_ma_window": 20,
    "spread_zscore_window": 250,
    "rolling_low_window": 250,
}


def _params(config: dict | None = None) -> dict:
    params = DEFAULT_PARAMS.copy()
    if config:
        params.update(config)
    return params


def _int_windows(values, required: list[int] | None = None) -> list[int]:
    windows = {int(value) for value in values}
    if required:
        windows.update(required)
    return sorted(windows)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    用 numpy.polyfit 计算滚动斜率。
    """
    values = series.astype(float)
    x = np.arange(window)

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    return values.rolling(window).apply(_slope, raw=True)


def align_stock_index(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """
    按日期对齐个股和指数。
    """
    stock = stock_df.copy()
    index = index_df.copy()
    stock = stock.rename(columns={"close": "stock_close"})
    index = index.rename(columns={"close": "index_close"})

    stock_cols = ["date", "stock_close", "volume"]
    for col in ["open", "high", "low", "amount"]:
        if col in stock.columns:
            stock_cols.append(col)

    merged = pd.merge(
        stock[stock_cols],
        index[["date", "index_close"]],
        on="date",
        how="inner",
    )
    return merged.sort_values("date").reset_index(drop=True)


def add_difference_indicators(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    计算逐日差价图及相关指标。
    """
    if df.empty:
        raise ValueError("empty aligned data")

    df = df.copy()
    df["stock_index"] = df["stock_close"] / df["stock_close"].iloc[0] * 100
    df["market_index"] = df["index_close"] / df["index_close"].iloc[0] * 100
    df["spread"] = df["stock_index"] - df["market_index"]

    params = _params(config)
    slope_windows = _int_windows(params["slope_windows"], required=[60, 120, 250])
    breakout_windows = _int_windows(params["breakout_windows"], required=[60, 120, 250])
    ma_windows = _int_windows(params["ma_windows"], required=[20, 60])
    below_zero_window = int(params["below_zero_window"])
    volume_ma_window = int(params["volume_ma_window"])
    zscore_window = int(params["spread_zscore_window"])
    rolling_low_window = int(params["rolling_low_window"])

    for window in slope_windows:
        df[f"spread_slope_{window}"] = rolling_slope(df["spread"], window)

    for window in breakout_windows:
        df[f"spread_high_{window}"] = df["spread"].rolling(window).max().shift(1)
        df[f"price_high_{window}"] = df["stock_close"].rolling(window).max().shift(1)

    below_zero_windows = _int_windows([below_zero_window], required=[250])
    for window in below_zero_windows:
        df[f"below_zero_ratio_{window}"] = (df["spread"] < 0).rolling(window).mean()

    df[f"volume_ma{volume_ma_window}"] = df["volume"].rolling(volume_ma_window).mean()
    df["volume_ratio"] = df["volume"] / df[f"volume_ma{volume_ma_window}"]

    for window in ma_windows:
        df[f"market_ma{window}"] = df["index_close"].rolling(window).mean()
        df[f"stock_ma{window}"] = df["stock_close"].rolling(window).mean()
        df[f"spread_ma{window}"] = df["spread"].rolling(window).mean()

    zscore_windows = _int_windows([zscore_window], required=[250])
    for window in zscore_windows:
        df[f"spread_mean_{window}"] = df["spread"].rolling(window).mean()
        df[f"spread_std_{window}"] = df["spread"].rolling(window).std()
        df[f"spread_zscore_{window}"] = (
            (df["spread"] - df[f"spread_mean_{window}"]) / df[f"spread_std_{window}"]
        )

    low_windows = _int_windows([rolling_low_window], required=[250])
    for window in low_windows:
        df[f"rolling_{window}_low"] = df["stock_close"].rolling(window).min()
    return df


def get_breakout_level(latest: pd.Series, prefix: str, value_col: str) -> str:
    """
    判断突破级别。
    """
    level = "无"
    value = latest.get(value_col, np.nan)
    for window in [60, 120, 250]:
        high = latest.get(f"{prefix}_{window}", np.nan)
        if pd.notna(value) and pd.notna(high) and value > high:
            level = f"{window}日"
    return level
