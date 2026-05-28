import pandas as pd

from scoring import build_result_row, classify_latest, score_latest


def make_latest(**overrides) -> pd.Series:
    data = {
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
        "spread_ma60": 15.0,
        "spread_mean_250": -5.0,
        "spread_std_250": 10.0,
        "spread_zscore_250": 1.0,
        "rolling_250_low": 100.0,
    }
    data.update(overrides)
    return pd.Series(data)


def classify(latest: pd.Series) -> str:
    scored = score_latest(latest, {})
    category, _ = classify_latest(latest, scored["score"], {}, 60)
    return category


def test_a_type_recognized():
    latest = make_latest()
    assert classify(latest) == "A类"


def test_a_plus_recognized():
    latest = make_latest(
        spread_high_250=28.0,
        price_high_250=145.0,
    )
    assert classify(latest) == "A+类"


def test_b_type_recognized():
    latest = make_latest(
        spread=25.0,
        below_zero_ratio_250=0.2,
        spread_slope_250=0.1,
        spread_high_120=40.0,
        spread_high_250=50.0,
        spread_mean_250=0.0,
        spread_std_250=10.0,
        price_high_250=160.0,
    )
    assert classify(latest) == "B类"


def test_c_type_recognized():
    latest = make_latest(
        spread=10.0,
        below_zero_ratio_250=0.2,
        spread_slope_250=0.1,
        spread_high_120=40.0,
        spread_high_250=50.0,
        price_high_120=160.0,
        price_high_250=170.0,
        spread_mean_250=0.0,
        spread_std_250=10.0,
    )
    assert classify(latest) == "C类"


def test_c_hot_recognized():
    latest = make_latest(
        spread=10.0,
        below_zero_ratio_250=0.2,
        spread_slope_250=0.1,
        spread_high_120=40.0,
        spread_high_250=50.0,
        price_high_120=160.0,
        price_high_250=170.0,
        spread_mean_250=0.0,
        spread_std_250=10.0,
        spread_zscore_250=4.0,
    )
    assert classify(latest) == "C类-过热观察"
    scored = score_latest(latest, {})
    assert "相对强度过热" in scored["risk_flags"]


def test_strong_continuation_gets_candidate_score_without_breakout():
    latest = make_latest(
        spread=18.0,
        spread_slope_250=0.1,
        spread_high_60=25.0,
        spread_high_120=30.0,
        spread_high_250=40.0,
        price_high_60=160.0,
        price_high_120=170.0,
        price_high_250=180.0,
        below_zero_ratio_250=0.1,
        volume_ratio=1.25,
        spread_mean_250=12.0,
        spread_std_250=8.0,
        spread_zscore_250=0.75,
        rolling_250_low=120.0,
    )
    scored = score_latest(latest, {})
    assert classify(latest) == "C类"
    assert scored["score"] >= 60


def test_reject_score_below_threshold():
    latest = make_latest(
        spread=-20.0,
        stock_close=90.0,
        index_close=90.0,
        spread_slope_60=-0.2,
        spread_slope_120=-0.1,
        spread_slope_250=-0.05,
        spread_high_60=10.0,
        spread_high_120=20.0,
        spread_high_250=30.0,
        price_high_60=120.0,
        price_high_120=130.0,
        price_high_250=140.0,
        below_zero_ratio_250=1.0,
        volume_ratio=0.8,
        market_ma20=100.0,
        market_ma60=110.0,
        stock_ma20=100.0,
        stock_ma60=110.0,
        spread_mean_250=-10.0,
        spread_std_250=8.0,
        spread_zscore_250=-1.0,
    )
    scored = score_latest(latest, {})
    category, _ = classify_latest(latest, scored["score"], {}, 60)

    assert scored["score"] < 60
    assert category == "剔除"


def test_score_range():
    latest = make_latest(
        spread_high_250=28.0,
        price_high_250=145.0,
    )
    scored = score_latest(latest, {})
    assert 0 <= scored["score"] <= 100


def test_build_result_row_contains_breakout_levels():
    latest = make_latest(
        spread_high_250=28.0,
        price_high_250=145.0,
    )
    row = build_result_row("600519", "贵州茅台", "000300", pd.DataFrame([latest]))

    assert row["category"] == "A+类"
    assert row["spread_breakout_level"] == "250日"
    assert row["price_breakout_level"] == "250日"


def test_score_weights_configurable():
    from scoring import load_score_weights, score_latest

    latest = make_latest()

    default_weights = load_score_weights(None)
    result_default = score_latest(latest, None, default_weights)
    default_score = result_default["score"]

    modified_weights = default_weights.copy()
    modified_weights["spread_breakout_120"] = 0

    result_modified = score_latest(latest, None, modified_weights)
    modified_score = result_modified["score"]

    assert modified_score < default_score, "修改权重后分数应该变化"
    assert "spread_breakout_120:+0" in result_modified["score_positive_items"]


def test_missing_score_weights_key_does_not_crash():
    from scoring import load_score_weights

    partial_weights = {"spread_slope_60_positive": 15}
    weights = load_score_weights({"score": {"weights": partial_weights}})

    assert weights["spread_slope_60_positive"] == 15
    assert "spread_breakout_120" in weights
    assert weights["spread_breakout_120"] == 10
