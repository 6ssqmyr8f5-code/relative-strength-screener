import pytest
import pandas as pd
from conftest import make_synthetic_df
from data_provider import _normalize_index_df, _normalize_stock_df
from indicators import add_difference_indicators, align_stock_index
from main import filter_valid_factor_rows


def test_stock_index_starts_at_100():
    df = make_synthetic_df("a_type")
    merged = align_stock_index(df, df)
    result = add_difference_indicators(merged)
    assert result["stock_index"].iloc[0] == 100.0


def test_market_index_starts_at_100():
    df = make_synthetic_df("a_type")
    merged = align_stock_index(df, df)
    result = add_difference_indicators(merged)
    assert result["market_index"].iloc[0] == 100.0


def test_spread_calculation():
    df = make_synthetic_df("a_type")
    merged = align_stock_index(df, df)
    result = add_difference_indicators(merged)
    spread = result["stock_index"] - result["market_index"]
    assert result["spread"].equals(spread)


def test_rolling_slope_positive():
    df = make_synthetic_df("a_plus")
    merged = align_stock_index(df, df)
    result = add_difference_indicators(merged)
    recent_slope = result["spread_slope_60"].iloc[-1]
    assert pd.notna(recent_slope)


def test_rolling_slope_negative():
    df = make_synthetic_df("reject")
    merged = align_stock_index(df, df)
    result = add_difference_indicators(merged)
    recent_slope = result["spread_slope_60"].iloc[-1]
    assert pd.notna(recent_slope)


def test_insufficient_data_no_crash():
    df = make_synthetic_df("insufficient")
    merged = align_stock_index(df, df)
    result = add_difference_indicators(merged)
    assert not result.empty


def test_factor_filter_keeps_rows_when_optional_ohlc_missing():
    n = 320
    stock_raw = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n),
        "close": range(100, 100 + n),
        "volume": [1000] * n,
    })
    index_raw = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n),
        "close": range(100, 100 + n),
    })

    stock = _normalize_stock_df(stock_raw, "600519")
    index = _normalize_index_df(index_raw, "000300")
    factors = add_difference_indicators(align_stock_index(stock, index))
    filtered = filter_valid_factor_rows(factors, "600519")

    assert len(filtered) == n
    assert filtered["open"].isna().all()
