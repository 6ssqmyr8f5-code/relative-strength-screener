import numpy as np
import pandas as pd

from indicators import get_breakout_level


CATEGORY_RANK = {
    "A+类": 1,
    "A类": 2,
    "B类": 3,
    "C类": 4,
    "C类-过热观察": 4,
    "观察": 5,
    "剔除": 6,
}

DEFAULT_CLASSIFICATION = {
    "below_zero_ratio_threshold": 0.7,
    "b_mean_abs_threshold": 20,
    "b_breakout_zscore": 1.0,
    "c_below_zero_ratio_max": 0.4,
    "c_hot_zscore": 3,
    "c_hot_price_multiple": 2.0,
}

DEFAULT_SCORE_WEIGHTS = {
    "spread_above_zero": 8,
    "spread_slope_60_positive": 10,
    "spread_slope_120_positive": 10,
    "spread_slope_250_positive": 8,
    "long_weak_short_turn_positive": 10,
    "spread_above_ma60": 6,
    "spread_breakout_60": 8,
    "spread_breakout_120": 10,
    "spread_breakout_250": 7,
    "stock_above_ma20": 6,
    "stock_ma20_above_ma60": 6,
    "price_breakout_60": 6,
    "price_breakout_120": 7,
    "price_breakout_250": 7,
    "market_above_ma60": 5,
    "market_ma20_above_ma60": 5,
    "volume_ratio_above_1_2": 5,
    "volume_ratio_above_1_5": 5,
    "risk_price_double_from_250_low": -5,
    "risk_spread_zscore_hot": -5,
    "risk_below_stock_ma20": -5,
    "risk_market_below_ma60": -10,
}


def load_score_weights(config: dict | None = None) -> dict:
    """从 config 中读取 score_weights，缺失字段用默认值兜底。"""
    weights = DEFAULT_SCORE_WEIGHTS.copy()
    if config and "score" in config and "weights" in config["score"]:
        weights.update(config["score"]["weights"])
    return weights


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _gt(left, right) -> bool:
    left = safe_float(left)
    right = safe_float(right)
    return pd.notna(left) and pd.notna(right) and left > right


def _lt(left, right) -> bool:
    left = safe_float(left)
    right = safe_float(right)
    return pd.notna(left) and pd.notna(right) and left < right


def classification_params(config: dict | None = None) -> dict:
    params = DEFAULT_CLASSIFICATION.copy()
    if config:
        params.update(config)
    return params


def score_latest(
    latest: pd.Series,
    classification_config: dict | None = None,
    score_weights: dict | None = None,
) -> dict:
    """
    对单只股票最新一天打分。
    """
    params = classification_params(classification_config)
    weights = load_score_weights({"score": {"weights": score_weights}} if score_weights else None)

    score = 0
    risk_flags = []
    reasons = []
    score_positive_items = []
    score_negative_items = []
    missing_conditions = []

    spread = safe_float(latest["spread"])
    close = safe_float(latest["stock_close"])
    index_close = safe_float(latest["index_close"])

    if _gt(spread, 0):
        score += weights["spread_above_zero"]
        reasons.append("差价位于零轴上方")
        score_positive_items.append(f"spread_above_zero:+{weights['spread_above_zero']}")
    else:
        missing_conditions.append("差价未站上零轴")

    if _gt(latest["spread_slope_60"], 0):
        score += weights["spread_slope_60_positive"]
        reasons.append("60日差价斜率转正")
        score_positive_items.append(f"spread_slope_60_positive:+{weights['spread_slope_60_positive']}")
    else:
        missing_conditions.append("60日差价斜率未转正")

    if _gt(latest["spread_slope_120"], 0):
        score += weights["spread_slope_120_positive"]
        reasons.append("120日差价斜率转正")
        score_positive_items.append(f"spread_slope_120_positive:+{weights['spread_slope_120_positive']}")
    else:
        missing_conditions.append("120日差价斜率未转正")

    if _gt(latest["spread_slope_250"], 0):
        score += weights["spread_slope_250_positive"]
        reasons.append("250日差价斜率为正")
        score_positive_items.append(f"spread_slope_250_positive:+{weights['spread_slope_250_positive']}")
    else:
        missing_conditions.append("250日差价斜率未转正")

    if _lt(latest["spread_slope_250"], 0) and _gt(latest["spread_slope_60"], 0):
        score += weights["long_weak_short_turn_positive"]
        reasons.append("长期弱势但短期转强")
        score_positive_items.append(f"long_weak_short_turn_positive:+{weights['long_weak_short_turn_positive']}")

    if _gt(spread, latest.get("spread_ma60")):
        score += weights["spread_above_ma60"]
        reasons.append("差价站上60日均线")
        score_positive_items.append(f"spread_above_ma60:+{weights['spread_above_ma60']}")
    else:
        missing_conditions.append("差价未站上60日均线")

    if _gt(spread, latest["spread_high_60"]):
        score += weights["spread_breakout_60"]
        reasons.append("差价突破60日高点")
        score_positive_items.append(f"spread_breakout_60:+{weights['spread_breakout_60']}")
    else:
        missing_conditions.append("未突破60日差价高点")

    if _gt(spread, latest["spread_high_120"]):
        score += weights["spread_breakout_120"]
        reasons.append("差价突破120日高点")
        score_positive_items.append(f"spread_breakout_120:+{weights['spread_breakout_120']}")
    else:
        missing_conditions.append("未突破120日差价高点")

    if _gt(spread, latest["spread_high_250"]):
        score += weights["spread_breakout_250"]
        reasons.append("差价突破250日高点")
        score_positive_items.append(f"spread_breakout_250:+{weights['spread_breakout_250']}")
    else:
        missing_conditions.append("未突破250日差价高点")

    if _gt(close, latest["stock_ma20"]):
        score += weights["stock_above_ma20"]
        reasons.append("股价站上20日线")
        score_positive_items.append(f"stock_above_ma20:+{weights['stock_above_ma20']}")
    else:
        missing_conditions.append("股价未站上20日线")

    if _gt(latest["stock_ma20"], latest["stock_ma60"]):
        score += weights["stock_ma20_above_ma60"]
        reasons.append("个股20日线高于60日线")
        score_positive_items.append(f"stock_ma20_above_ma60:+{weights['stock_ma20_above_ma60']}")
    else:
        missing_conditions.append("个股20日线未高于60日线")

    if _gt(close, latest["price_high_60"]):
        score += weights["price_breakout_60"]
        reasons.append("股价突破60日高点")
        score_positive_items.append(f"price_breakout_60:+{weights['price_breakout_60']}")
    else:
        missing_conditions.append("未突破60日价格高点")

    if _gt(close, latest["price_high_120"]):
        score += weights["price_breakout_120"]
        reasons.append("股价突破120日高点")
        score_positive_items.append(f"price_breakout_120:+{weights['price_breakout_120']}")
    else:
        missing_conditions.append("未突破120日价格高点")

    if _gt(close, latest["price_high_250"]):
        score += weights["price_breakout_250"]
        reasons.append("股价突破250日高点")
        score_positive_items.append(f"price_breakout_250:+{weights['price_breakout_250']}")
    else:
        missing_conditions.append("未突破250日价格高点")

    if _gt(index_close, latest["market_ma60"]):
        score += weights["market_above_ma60"]
        reasons.append("指数站上60日线")
        score_positive_items.append(f"market_above_ma60:+{weights['market_above_ma60']}")
    else:
        missing_conditions.append("指数未站上60日线")

    if _gt(latest["market_ma20"], latest["market_ma60"]):
        score += weights["market_ma20_above_ma60"]
        reasons.append("指数20日线高于60日线")
        score_positive_items.append(f"market_ma20_above_ma60:+{weights['market_ma20_above_ma60']}")
    else:
        missing_conditions.append("指数20日线未高于60日线")

    volume_ratio = safe_float(latest["volume_ratio"])
    if _gt(volume_ratio, 1.2):
        score += weights["volume_ratio_above_1_2"]
        reasons.append("成交量超过20日均量1.2倍")
        score_positive_items.append(f"volume_ratio_above_1_2:+{weights['volume_ratio_above_1_2']}")
    if _gt(volume_ratio, 1.5):
        score += weights["volume_ratio_above_1_5"]
        reasons.append("成交量超过20日均量1.5倍")
        score_positive_items.append(f"volume_ratio_above_1_5:+{weights['volume_ratio_above_1_5']}")
    if pd.isna(volume_ratio) or volume_ratio < 1.2:
        missing_conditions.append("成交量未达到1.2倍")

    rolling_low = safe_float(latest["rolling_250_low"])
    if (
        pd.notna(rolling_low)
        and rolling_low > 0
        and close / rolling_low > params["c_hot_price_multiple"]
    ):
        score += weights["risk_price_double_from_250_low"]
        risk_flags.append("一年内涨幅过大")
        score_negative_items.append(f"risk_price_double_from_250_low:{weights['risk_price_double_from_250_low']}")

    if _gt(latest["spread_zscore_250"], params["c_hot_zscore"]):
        score += weights["risk_spread_zscore_hot"]
        risk_flags.append("相对强度过热")
        score_negative_items.append(f"risk_spread_zscore_hot:{weights['risk_spread_zscore_hot']}")

    if _lt(close, latest["stock_ma20"]):
        score += weights["risk_below_stock_ma20"]
        risk_flags.append("跌破20日线")
        score_negative_items.append(f"risk_below_stock_ma20:{weights['risk_below_stock_ma20']}")

    if _lt(index_close, latest["market_ma60"]):
        score += weights["risk_market_below_ma60"]
        risk_flags.append("大盘弱势")
        score_negative_items.append(f"risk_market_below_ma60:{weights['risk_market_below_ma60']}")

    score = max(0, min(100, int(round(score))))
    return {
        "score": score,
        "risk_flags": risk_flags,
        "reason": reasons,
        "score_positive_items": score_positive_items,
        "score_negative_items": score_negative_items,
        "missing_conditions": missing_conditions,
    }


def classify_latest(
    latest: pd.Series,
    score: int,
    classification_config: dict | None = None,
    observe_score_threshold: int = 60,
) -> tuple[str, str]:
    """
    分类并给出动作。
    """
    params = classification_params(classification_config)
    spread = safe_float(latest["spread"])
    close = safe_float(latest["stock_close"])
    index_close = safe_float(latest["index_close"])
    below_zero_ratio = safe_float(latest["below_zero_ratio_250"])
    volume_ratio = safe_float(latest["volume_ratio"])
    spread_std = safe_float(latest["spread_std_250"])
    spread_mean = safe_float(latest["spread_mean_250"])
    spread_zscore = safe_float(latest.get("spread_zscore_250"))
    rolling_low = safe_float(latest["rolling_250_low"])
    market_ok = _gt(index_close, latest["market_ma60"])

    a_type = (
        _gt(below_zero_ratio, params["below_zero_ratio_threshold"])
        and _lt(latest["spread_slope_250"], 0)
        and _gt(latest["spread_slope_60"], 0)
        and _gt(spread, latest["spread_high_120"])
        and _gt(close, latest["price_high_120"])
        and market_ok
    )
    a_plus = (
        a_type
        and _gt(spread, latest["spread_high_250"])
        and _gt(close, latest["price_high_250"])
        and pd.notna(volume_ratio)
        and volume_ratio >= 1.2
    )
    sync_breakout = (
        pd.notna(spread_std)
        and spread_std > 0
        and pd.notna(spread_mean)
        and spread > spread_mean + params["b_breakout_zscore"] * spread_std
    )
    b_type = (
        pd.notna(spread_std)
        and spread_std > 0
        and pd.notna(spread_mean)
        and abs(spread_mean) <= params["b_mean_abs_threshold"]
        and _gt(latest["spread_slope_60"], 0)
        and _gt(latest["spread_slope_120"], 0)
        and sync_breakout
        and (_gt(spread, latest["spread_high_60"]) or _gt(spread, latest["spread_high_120"]))
        and (_gt(close, latest["stock_ma20"]) or _gt(close, latest["price_high_60"]))
        and market_ok
    )
    c_type = (
        _gt(spread, 0)
        and _gt(latest["spread_slope_120"], 0)
        and _gt(latest["spread_slope_250"], 0)
        and (
            pd.isna(below_zero_ratio)
            or below_zero_ratio <= params["c_below_zero_ratio_max"]
        )
        and _gt(close, latest["stock_ma20"])
        and market_ok
    )
    hot = _gt(spread_zscore, params["c_hot_zscore"]) or (
        pd.notna(rolling_low)
        and rolling_low > 0
        and close / rolling_low > params["c_hot_price_multiple"]
    )

    if a_plus:
        return "A+类", "重点关注，等待突破后回踩确认或小仓试错"
    if a_type:
        return "A类", "重点观察，属于长期弱势后转强"
    if b_type:
        return "B类", "同步转强，等待突破延续或回踩确认"
    if c_type:
        if hot:
            return "C类-过热观察", "强者恒强但已有过热风险，等待回踩或缩量整理"
        return "C类", "强者恒强，趋势健康时可持续跟踪"
    if score >= observe_score_threshold:
        return "观察", "信号不完整，等待价格或差价进一步确认"
    return "剔除", "不符合逐日差价图选股条件"


def build_result_row(
    code: str,
    name: str,
    benchmark: str,
    df: pd.DataFrame,
    classification_config: dict | None = None,
    observe_score_threshold: int = 60,
    score_weights: dict | None = None,
) -> dict:
    """
    生成单只股票的结果行。
    """
    latest = df.iloc[-1]
    scored = score_latest(latest, classification_config, score_weights)
    category, action = classify_latest(
        latest,
        scored["score"],
        classification_config,
        observe_score_threshold,
    )
    spread_breakout_level = get_breakout_level(latest, "spread_high", "spread")
    price_breakout_level = get_breakout_level(latest, "price_high", "stock_close")

    index_close = safe_float(latest["index_close"])
    market_ma60 = safe_float(latest["market_ma60"])
    market_ma20 = safe_float(latest["market_ma20"])
    if pd.notna(index_close) and pd.notna(market_ma60) and index_close > market_ma60:
        market_status = "强"
    elif pd.notna(index_close) and pd.notna(market_ma20) and index_close > market_ma20:
        market_status = "中性"
    else:
        market_status = "弱"

    return {
        "code": code,
        "name": name,
        "latest_date": latest["date"].strftime("%Y-%m-%d"),
        "benchmark": benchmark,
        "score": scored["score"],
        "category": category,
        "action": action,
        "close": round(safe_float(latest["stock_close"]), 3),
        "spread": round(safe_float(latest["spread"]), 3),
        "spread_slope_60": round(safe_float(latest["spread_slope_60"]), 5),
        "spread_slope_120": round(safe_float(latest["spread_slope_120"]), 5),
        "spread_slope_250": round(safe_float(latest["spread_slope_250"]), 5),
        "below_zero_ratio_250": round(safe_float(latest["below_zero_ratio_250"]), 3),
        "spread_breakout_level": spread_breakout_level,
        "price_breakout_level": price_breakout_level,
        "volume_ratio": round(safe_float(latest["volume_ratio"]), 3),
        "market_status": market_status,
        "risk_flags": "、".join(scored["risk_flags"]) if scored["risk_flags"] else "",
        "reason": "；".join(scored["reason"]),
        "score_positive_items": "|".join(scored["score_positive_items"]) if scored["score_positive_items"] else "",
        "score_negative_items": "|".join(scored["score_negative_items"]) if scored["score_negative_items"] else "",
        "missing_conditions": "|".join(scored["missing_conditions"]) if scored["missing_conditions"] else "",
    }


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    按分类优先级和得分排序。
    """
    if df.empty:
        return df
    df = df.copy()
    df["_rank"] = df["category"].map(CATEGORY_RANK).fillna(99)
    df = df.sort_values(["_rank", "score"], ascending=[True, False])
    return df.drop(columns=["_rank"]).reset_index(drop=True)
