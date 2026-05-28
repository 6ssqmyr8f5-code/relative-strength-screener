import pandas as pd
import numpy as np


def check_price_data_quality(
    df: pd.DataFrame,
    code: str,
    name: str,
    start: str,
    end: str,
    source: str,
    min_bars: int = 300,
) -> list[dict]:
    """
    检查个股或指数数据质量。
    返回 warning 列表。
    """
    warnings = []

    if df.empty:
        return [
            {
                "code": code,
                "name": name,
                "source": source,
                "stage": "data_quality",
                "date_range": f"{start}~{end}",
                "warning_type": "empty_data",
                "warning_message": "数据为空",
                "latest_date": None,
                "rows": 0,
            }
        ]

    rows = len(df)
    latest_date = df["date"].max()
    date_range = f"{start}~{end}"

    if rows < min_bars:
        warnings.append(
            {
                "code": code,
                "name": name,
                "source": source,
                "stage": "data_quality",
                "date_range": date_range,
                "warning_type": "insufficient_rows",
                "warning_message": f"数据行数不足，当前 {rows} 行，需要 {min_bars} 行",
                "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                "rows": rows,
            }
        )

    if "date" in df.columns:
        if df["date"].duplicated().any():
            duplicate_count = df["date"].duplicated().sum()
            warnings.append(
                {
                    "code": code,
                    "name": name,
                    "source": source,
                    "stage": "data_quality",
                    "date_range": date_range,
                    "warning_type": "duplicate_date",
                    "warning_message": f"日期重复：{duplicate_count} 条",
                    "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                    "rows": rows,
                }
            )

        if not df["date"].is_monotonic_increasing:
            warnings.append(
                {
                    "code": code,
                    "name": name,
                    "source": source,
                    "stage": "data_quality",
                    "date_range": date_range,
                    "warning_type": "date_not_monotonic",
                    "warning_message": "日期未按升序排列",
                    "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                    "rows": rows,
                }
            )

    close_columns = ["close", "stock_close"]
    for col in close_columns:
        if col not in df.columns:
            continue
        non_positive_close = (df[col] <= 0).sum()
        if non_positive_close > 0:
            warnings.append(
                {
                    "code": code,
                    "name": name,
                    "source": source,
                    "stage": "data_quality",
                    "date_range": date_range,
                    "warning_type": "non_positive_close",
                    "warning_message": f"{col} <= 0: {non_positive_close} 条",
                    "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                    "rows": rows,
                }
            )

    if "volume" in df.columns:
        negative_volume = (df["volume"] < 0).sum()
        if negative_volume > 0:
            warnings.append(
                {
                    "code": code,
                    "name": name,
                    "source": source,
                    "stage": "data_quality",
                    "date_range": date_range,
                    "warning_type": "negative_volume",
                    "warning_message": f"成交量 < 0：{negative_volume} 条",
                    "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                    "rows": rows,
                }
            )

        zero_volume = (df["volume"] == 0).sum()
        zero_volume_ratio = zero_volume / rows if rows > 0 else 0
        if zero_volume_ratio > 0.1:
            warnings.append(
                {
                    "code": code,
                    "name": name,
                    "source": source,
                    "stage": "data_quality",
                    "date_range": date_range,
                    "warning_type": "excessive_zero_volume",
                    "warning_message": f"成交量为0过多：{zero_volume} 条 ({zero_volume_ratio:.1%})",
                    "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                    "rows": rows,
                }
            )

    if ("close" in df.columns or "stock_close" in df.columns) and len(df) >= 20:
        close_col = "stock_close" if "stock_close" in df.columns else "close"
        close_series = df[close_col].values
        consecutive_equal = 0
        max_consecutive_equal = 0
        for i in range(1, len(close_series)):
            if close_series[i] == close_series[i - 1]:
                consecutive_equal += 1
                max_consecutive_equal = max(max_consecutive_equal, consecutive_equal)
            else:
                consecutive_equal = 0

        if max_consecutive_equal >= 20:
            warnings.append(
                {
                    "code": code,
                    "name": name,
                    "source": source,
                    "stage": "data_quality",
                    "date_range": date_range,
                    "warning_type": "consecutive_identical_prices",
                    "warning_message": f"连续 {max_consecutive_equal} 日价格不变",
                    "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                    "rows": rows,
                }
            )

    if ("close" in df.columns or "stock_close" in df.columns) and len(df) >= 2:
        close_col = "stock_close" if "stock_close" in df.columns else "close"
        close_series = df[close_col].dropna()
        if len(close_series) >= 2:
            returns = close_series.pct_change().abs()
            abnormal_days = (returns > 0.3).sum()
            if abnormal_days > 0:
                warnings.append(
                    {
                        "code": code,
                        "name": name,
                        "source": source,
                        "stage": "data_quality",
                        "date_range": date_range,
                        "warning_type": "abnormal_price_change",
                        "warning_message": f"单日涨跌幅异常（>30%）：{abnormal_days} 天",
                        "latest_date": str(latest_date) if pd.notna(latest_date) else None,
                        "rows": rows,
                    }
                )

    if str(source).startswith("cache") and pd.notna(latest_date):
        try:
            end_date = pd.to_datetime(end)
            days_diff = (end_date - latest_date).days
            if days_diff > 10:
                warnings.append(
                    {
                        "code": code,
                        "name": name,
                        "source": source,
                        "stage": "data_quality",
                        "date_range": date_range,
                        "warning_type": "stale_cache",
                        "warning_message": f"缓存最新日期早于 end 超过 10 天（{days_diff} 天）",
                        "latest_date": str(latest_date),
                        "rows": rows,
                    }
                )
        except Exception:
            pass

    return warnings


def is_warning_fatal(warning: dict) -> bool:
    """
    判断是否为严重问题，需要跳过。
    """
    fatal_types = [
        "insufficient_rows",
        "non_positive_close",
        "duplicate_date",
        "empty_data",
    ]
    return warning.get("warning_type") in fatal_types
