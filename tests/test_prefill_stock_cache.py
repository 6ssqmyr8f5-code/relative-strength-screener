import pandas as pd

from prefill_stock_cache import (
    cache_coverage,
    normalize_tencent_stock_df,
    stock_tx_symbol,
)
from data_provider import save_cache


def test_stock_tx_symbol_adds_market_prefix():
    assert stock_tx_symbol("600519") == "sh600519"
    assert stock_tx_symbol("000858") == "sz000858"
    assert stock_tx_symbol("300750") == "sz300750"
    assert stock_tx_symbol("831726") == "bj831726"
    assert stock_tx_symbol("920001") == "bj920001"


def test_normalize_tencent_stock_df_uses_amount_as_volume_when_needed():
    raw = pd.DataFrame(
        {
            "date": ["2026-05-25", "2026-05-26"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.6],
            "low": [9.8, 10.1],
            "close": [10.1, 10.4],
            "amount": [100000, 120000],
        }
    )

    result = normalize_tencent_stock_df(raw, "600519")

    assert result.columns.tolist() == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert result["volume"].tolist() == [100000, 120000]
    assert result["amount"].isna().all()


def test_normalize_tencent_stock_df_accepts_chinese_columns():
    raw = pd.DataFrame(
        {
            "日期": ["2026-05-25"],
            "开盘": [10.0],
            "最高": [10.5],
            "最低": [9.8],
            "收盘": [10.1],
            "成交量": [100000],
            "成交额": [1000000],
        }
    )

    result = normalize_tencent_stock_df(raw, "600519")

    assert result["volume"].iloc[0] == 100000
    assert result["amount"].iloc[0] == 1000000


def test_cache_coverage_detects_requested_range(tmp_path):
    path = tmp_path / "600519_qfq.csv"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=10),
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": range(10),
            "amount": [pd.NA] * 10,
        }
    )
    save_cache(df, str(path))

    covered = cache_coverage(str(path), "20220103", "20220108")
    uncovered = cache_coverage(str(path), "20211231", "20220108")

    assert covered["covers"] is True
    assert covered["rows"] == 10
    assert uncovered["covers"] is False
