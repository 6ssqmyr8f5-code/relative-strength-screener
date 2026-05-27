import pandas as pd
import pytest

from data_quality import check_price_data_quality, is_warning_fatal


def make_test_df(**kwargs) -> pd.DataFrame:
    base = {
        "date": pd.date_range("2022-01-01", periods=320),
        "close": [100 + i * 0.1 for i in range(320)],
        "volume": [1000000] * 320,
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def test_insufficient_rows_warning():
    df = make_test_df()[:50]
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "insufficient_rows" for w in warnings)


def test_non_positive_close_warning():
    df = make_test_df()
    df.loc[100, "close"] = 0
    df.loc[101, "close"] = -5
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "non_positive_close" for w in warnings)


def test_duplicate_date_warning():
    df = make_test_df()
    df.loc[50, "date"] = df.loc[49, "date"]
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "duplicate_date" for w in warnings)


def test_date_not_monotonic_warning():
    df = make_test_df()
    df.loc[50, "date"] = pd.Timestamp("2021-12-01")
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "date_not_monotonic" for w in warnings)


def test_negative_volume_warning():
    df = make_test_df()
    df.loc[50, "volume"] = -1000
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "negative_volume" for w in warnings)


def test_normal_data_no_fatal_warning():
    df = make_test_df()
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    fatal_warnings = [w for w in warnings if is_warning_fatal(w)]
    assert len(fatal_warnings) == 0


def test_consecutive_identical_prices_warning():
    df = make_test_df()
    for i in range(25):
        df.loc[100 + i, "close"] = 120.0
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "consecutive_identical_prices" for w in warnings)


def test_abnormal_price_change_warning():
    df = make_test_df()
    df.loc[100, "close"] = 200
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "abnormal_price_change" for w in warnings)


def test_stale_cache_warning():
    df = make_test_df()
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20231231",
        source="cache",
        min_bars=300,
    )
    assert any(w["warning_type"] == "stale_cache" for w in warnings)


def test_excessive_zero_volume_warning():
    df = make_test_df()
    for i in range(50):
        df.loc[i, "volume"] = 0
    warnings = check_price_data_quality(
        df=df,
        code="600519",
        name="贵州茅台",
        start="20220101",
        end="20260527",
        source="akshare",
        min_bars=300,
    )
    assert any(w["warning_type"] == "excessive_zero_volume" for w in warnings)


def test_is_warning_fatal():
    from data_quality import is_warning_fatal

    assert is_warning_fatal({"warning_type": "insufficient_rows"}) is True
    assert is_warning_fatal({"warning_type": "non_positive_close"}) is True
    assert is_warning_fatal({"warning_type": "duplicate_date"}) is True
    assert is_warning_fatal({"warning_type": "empty_data"}) is True
    assert is_warning_fatal({"warning_type": "stale_cache"}) is False
    assert is_warning_fatal({"warning_type": "negative_volume"}) is False
    assert is_warning_fatal({"warning_type": "date_not_monotonic"}) is False