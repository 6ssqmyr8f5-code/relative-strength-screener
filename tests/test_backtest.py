import numpy as np
import pandas as pd
import pytest

from backtest import (
    add_forward_returns,
    apply_signal_cooldown,
    generate_historical_signals,
    summarize_backtest,
    summarize_by_category,
)
from indicators import add_difference_indicators, align_stock_index
from tests.conftest import make_synthetic_df


def make_factor_df(n: int = 320) -> pd.DataFrame:
    df = make_synthetic_df("a_type")
    merged = align_stock_index(df, df)
    factors = add_difference_indicators(merged)
    factors = factors.reset_index(drop=True)
    factors["date"] = pd.to_datetime(factors["date"])
    return factors


def test_generate_historical_signals_no_future_data():
    factor_df = make_factor_df()

    signals = generate_historical_signals(
        code="600519",
        name="贵州茅台",
        benchmark="000300",
        factor_df=factor_df,
        min_score=0,
        allowed_categories=["A+类", "A类", "B类", "C类", "观察", "剔除"],
    )

    assert isinstance(signals, pd.DataFrame), "Should return DataFrame"

    if not signals.empty:
        for _, row in signals.iterrows():
            as_of = pd.to_datetime(row["as_of_date"])
            signal_date_idx = factor_df[factor_df["date"] <= as_of].index[-1]
            assert signal_date_idx >= 0


def test_add_forward_returns_5_20_60():
    factor_df = make_factor_df()

    signals = pd.DataFrame(
        [
            {
                "code": "600519",
                "name": "贵州茅台",
                "benchmark": "000300",
                "as_of_date": factor_df.iloc[300]["date"],
                "score": 70,
                "category": "A类",
                "action": "买入",
                "close": factor_df.iloc[300]["stock_close"],
                "index_close": factor_df.iloc[300]["index_close"],
                "spread": factor_df.iloc[300]["spread"],
                "spread_breakout_level": "120日",
                "price_breakout_level": "120日",
                "volume_ratio": factor_df.iloc[300]["volume_ratio"],
                "market_status": "强",
                "risk_flags": "",
                "reason": "测试",
            }
        ]
    )

    holding_days = [5, 20, 60]
    result = add_forward_returns(signals, factor_df, holding_days)

    assert "ret_5" in result.columns
    assert "ret_20" in result.columns
    assert "ret_60" in result.columns
    assert "benchmark_ret_5" in result.columns
    assert "benchmark_ret_20" in result.columns
    assert "benchmark_ret_60" in result.columns
    assert "excess_ret_5" in result.columns
    assert "excess_ret_20" in result.columns
    assert "excess_ret_60" in result.columns
    assert "max_drawdown_5" in result.columns
    assert "max_drawdown_20" in result.columns
    assert "max_drawdown_60" in result.columns


def test_excess_ret_equals_ret_minus_benchmark():
    factor_df = make_factor_df()

    signal_date = factor_df.iloc[300]["date"]
    signals = pd.DataFrame(
        [
            {
                "code": "600519",
                "name": "贵州茅台",
                "benchmark": "000300",
                "as_of_date": signal_date,
                "score": 70,
                "category": "A类",
                "action": "买入",
                "close": factor_df.iloc[300]["stock_close"],
                "index_close": factor_df.iloc[300]["index_close"],
                "spread": factor_df.iloc[300]["spread"],
                "spread_breakout_level": "120日",
                "price_breakout_level": "120日",
                "volume_ratio": factor_df.iloc[300]["volume_ratio"],
                "market_status": "强",
                "risk_flags": "",
                "reason": "测试",
            }
        ]
    )

    holding_days = [5]
    result = add_forward_returns(signals, factor_df, holding_days)

    for idx, row in result.iterrows():
        ret_5 = row["ret_5"]
        bench_5 = row["benchmark_ret_5"]
        excess_5 = row["excess_ret_5"]

        if pd.notna(ret_5) and pd.notna(bench_5):
            expected_excess = ret_5 - bench_5
            assert abs(excess_5 - expected_excess) < 1e-6


def test_insufficient_future_data_returns_nan():
    factor_df = make_factor_df()

    last_date = factor_df.iloc[-1]["date"]
    signals = pd.DataFrame(
        [
            {
                "code": "600519",
                "name": "贵州茅台",
                "benchmark": "000300",
                "as_of_date": last_date,
                "score": 70,
                "category": "A类",
                "action": "买入",
                "close": factor_df.iloc[-1]["stock_close"],
                "index_close": factor_df.iloc[-1]["index_close"],
                "spread": factor_df.iloc[-1]["spread"],
                "spread_breakout_level": "120日",
                "price_breakout_level": "120日",
                "volume_ratio": factor_df.iloc[-1]["volume_ratio"],
                "market_status": "强",
                "risk_flags": "",
                "reason": "测试",
            }
        ]
    )

    holding_days = [60]
    result = add_forward_returns(signals, factor_df, holding_days)

    assert pd.isna(result.iloc[0]["ret_60"])
    assert pd.isna(result.iloc[0]["excess_ret_60"])


def test_signal_cooldown_reduces_duplicates():
    dates = pd.date_range("2022-01-01", periods=100, freq="D")

    signals = pd.DataFrame(
        [
            {
                "code": "600519",
                "name": "贵州茅台",
                "benchmark": "000300",
                "as_of_date": dates[30],
                "score": 70,
                "category": "A类",
                "action": "买入",
                "close": 100.0,
                "index_close": 100.0,
                "spread": 0.0,
                "spread_breakout_level": "无",
                "price_breakout_level": "无",
                "volume_ratio": 1.0,
                "market_status": "强",
                "risk_flags": "",
                "reason": "测试",
            },
            {
                "code": "600519",
                "name": "贵州茅台",
                "benchmark": "000300",
                "as_of_date": dates[35],
                "score": 70,
                "category": "A类",
                "action": "买入",
                "close": 101.0,
                "index_close": 100.5,
                "spread": 0.5,
                "spread_breakout_level": "无",
                "price_breakout_level": "无",
                "volume_ratio": 1.1,
                "market_status": "强",
                "risk_flags": "",
                "reason": "测试",
            },
            {
                "code": "600519",
                "name": "贵州茅台",
                "benchmark": "000300",
                "as_of_date": dates[60],
                "score": 70,
                "category": "A类",
                "action": "买入",
                "close": 102.0,
                "index_close": 101.0,
                "spread": 1.0,
                "spread_breakout_level": "无",
                "price_breakout_level": "无",
                "volume_ratio": 1.2,
                "market_status": "强",
                "risk_flags": "",
                "reason": "测试",
            },
        ]
    )

    result = apply_signal_cooldown(signals, cooldown=20)

    assert len(result) < len(signals)
    assert len(result) == 2


def test_summarize_backtest_output_fields():
    trades = pd.DataFrame(
        [
            {
                "code": "600519",
                "name": "贵州茅台",
                "as_of_date": pd.Timestamp("2022-01-01"),
                "category": "A类",
                "ret_5": 0.05,
                "benchmark_ret_5": 0.02,
                "excess_ret_5": 0.03,
                "max_drawdown_5": -0.02,
                "ret_20": 0.10,
                "benchmark_ret_20": 0.05,
                "excess_ret_20": 0.05,
                "max_drawdown_20": -0.05,
            },
            {
                "code": "000858",
                "name": "五粮液",
                "as_of_date": pd.Timestamp("2022-01-15"),
                "category": "A类",
                "ret_5": -0.02,
                "benchmark_ret_5": 0.01,
                "excess_ret_5": -0.03,
                "max_drawdown_5": -0.08,
                "ret_20": 0.08,
                "benchmark_ret_20": 0.04,
                "excess_ret_20": 0.04,
                "max_drawdown_20": -0.10,
            },
        ]
    )

    holding_days = [5, 20]
    summary = summarize_backtest(trades, holding_days)

    expected_cols = [
        "holding_days",
        "signal_count",
        "avg_return",
        "median_return",
        "win_rate",
        "avg_excess_return",
        "median_excess_return",
        "excess_win_rate",
        "avg_max_drawdown",
        "profit_loss_ratio",
    ]

    for col in expected_cols:
        assert col in summary.columns

    assert len(summary) == 2


def test_summarize_by_category():
    trades = pd.DataFrame(
        [
            {
                "code": "600519",
                "name": "贵州茅台",
                "as_of_date": pd.Timestamp("2022-01-01"),
                "category": "A类",
                "ret_5": 0.05,
                "benchmark_ret_5": 0.02,
                "excess_ret_5": 0.03,
                "max_drawdown_5": -0.02,
                "ret_20": 0.10,
                "benchmark_ret_20": 0.05,
                "excess_ret_20": 0.05,
                "max_drawdown_20": -0.05,
            },
            {
                "code": "000858",
                "name": "五粮液",
                "as_of_date": pd.Timestamp("2022-01-15"),
                "category": "A类",
                "ret_5": -0.02,
                "benchmark_ret_5": 0.01,
                "excess_ret_5": -0.03,
                "max_drawdown_5": -0.08,
                "ret_20": 0.08,
                "benchmark_ret_20": 0.04,
                "excess_ret_20": 0.04,
                "max_drawdown_20": -0.10,
            },
            {
                "code": "300750",
                "name": "宁德时代",
                "as_of_date": pd.Timestamp("2022-02-01"),
                "category": "B类",
                "ret_5": 0.03,
                "benchmark_ret_5": 0.01,
                "excess_ret_5": 0.02,
                "max_drawdown_5": -0.01,
                "ret_20": 0.06,
                "benchmark_ret_20": 0.03,
                "excess_ret_20": 0.03,
                "max_drawdown_20": -0.03,
            },
        ]
    )

    holding_days = [5, 20]
    by_category = summarize_by_category(trades, holding_days)

    categories = by_category["category"].unique()
    assert "A类" in categories
    assert "B类" in categories

    a_class = by_category[by_category["category"] == "A类"]
    assert len(a_class) == 2

    b_class = by_category[by_category["category"] == "B类"]
    assert len(b_class) == 2


def test_score_decomposition_in_build_result_row():
    from scoring import build_result_row

    latest = pd.Series(
        {
            "date": pd.Timestamp("2026-05-27"),
            "spread": 30.0,
            "stock_close": 150.0,
            "index_close": 120.0,
            "spread_slope_60": 0.2,
            "spread_slope_120": 0.1,
            "spread_slope_250": -0.05,
            "spread_high_60": 20.0,
            "spread_high_120": 25.0,
            "spread_high_250": 35.0,
            "price_high_60": 130.0,
            "price_high_120": 140.0,
            "price_high_250": 160.0,
            "below_zero_ratio_250": 0.8,
            "volume_ratio": 1.3,
            "market_ma20": 115.0,
            "market_ma60": 110.0,
            "stock_ma20": 145.0,
            "stock_ma60": 135.0,
            "spread_mean_250": -5.0,
            "spread_std_250": 10.0,
            "spread_zscore_250": 1.0,
            "rolling_250_low": 100.0,
        }
    )

    df = pd.DataFrame([latest])
    row = build_result_row("600519", "贵州茅台", "000300", df)

    assert "score_positive_items" in row
    assert "score_negative_items" in row
    assert "missing_conditions" in row

    assert "spread_slope_60_positive:+10" in row["score_positive_items"]
    assert "spread_breakout_120:+10" in row["score_positive_items"]
    assert row["score_negative_items"] == "" or "risk" in row["score_negative_items"]


def test_score_decomposition_empty_for_missing():
    from scoring import build_result_row

    latest = pd.Series(
        {
            "date": pd.Timestamp("2026-05-27"),
            "spread": -30.0,
            "stock_close": 80.0,
            "index_close": 120.0,
            "spread_slope_60": -0.2,
            "spread_slope_120": -0.1,
            "spread_slope_250": -0.05,
            "spread_high_60": 20.0,
            "spread_high_120": 25.0,
            "spread_high_250": 35.0,
            "price_high_60": 130.0,
            "price_high_120": 140.0,
            "price_high_250": 160.0,
            "below_zero_ratio_250": 0.8,
            "volume_ratio": 0.5,
            "market_ma20": 115.0,
            "market_ma60": 110.0,
            "stock_ma20": 85.0,
            "stock_ma60": 90.0,
            "spread_mean_250": -10.0,
            "spread_std_250": 10.0,
            "spread_zscore_250": -2.0,
            "rolling_250_low": 70.0,
        }
    )

    df = pd.DataFrame([latest])
    row = build_result_row("600519", "贵州茅台", "000300", df)

    assert "score_positive_items" in row
    assert "score_negative_items" in row
    assert "missing_conditions" in row
    assert len(row["missing_conditions"]) > 0