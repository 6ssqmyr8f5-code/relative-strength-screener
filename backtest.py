import argparse
import os
import time
import traceback
from typing import Any

import pandas as pd
import yaml

from data_provider import (
    get_index_daily,
    get_stock_daily,
    normalize_index_code,
    normalize_stock_code,
    read_stock_pool,
)
from indicators import add_difference_indicators, align_stock_index
from scoring import build_result_row, classify_latest, load_score_weights, score_latest
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐日差价图回测系统")
    parser.add_argument("--codes", type=str, default="", help="逗号分隔股票代码")
    parser.add_argument("--stock-pool", type=str, default="", help="股票池 CSV")
    parser.add_argument("--benchmark", type=str, default="", help="基准指数代码")
    parser.add_argument("--start", type=str, default="", help="开始日期")
    parser.add_argument("--end", type=str, default="", help="结束日期")
    parser.add_argument("--holding-days", type=str, default="5,20,60", help="持有天数")
    parser.add_argument("--min-score", type=int, default=60, help="最低分数")
    parser.add_argument(
        "--categories",
        type=str,
        default="A+类,A类,B类,C类",
        help="纳入的分类，逗号分隔",
    )
    parser.add_argument("--no-plot", action="store_true", help="不生成图片")
    parser.add_argument(
        "--signal-cooldown",
        type=int,
        default=20,
        help="同一股票信号冷却天数",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="忽略缓存",
    )
    return parser.parse_args()


def build_stock_pool(args: argparse.Namespace) -> pd.DataFrame:
    if args.stock_pool:
        return read_stock_pool(args.stock_pool)
    if args.codes:
        rows = []
        for code in args.codes.split(","):
            code = normalize_stock_code(code)
            rows.append({"code": code, "name": code})
        return pd.DataFrame(rows)
    raise ValueError("必须提供 --codes 或 --stock-pool")


def get_local_stock_path(config: dict, code: str) -> str:
    template = config["data"].get("local_stock_template", "")
    if template:
        return template.format(code=code)
    return ""


def get_local_index_path(config: dict, benchmark: str) -> str:
    template = config["data"].get("local_index_template", "")
    if template:
        return template.format(benchmark=benchmark)
    return ""


def safe_float(x: Any, default=float("nan")) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _gt(left: Any, right: Any) -> bool:
    left = safe_float(left)
    right = safe_float(right)
    return pd.notna(left) and pd.notna(right) and left > right


def _lt(left: Any, right: Any) -> bool:
    left = safe_float(left)
    right = safe_float(right)
    return pd.notna(left) and pd.notna(right) and left < right


def generate_historical_signals(
    code: str,
    name: str,
    benchmark: str,
    factor_df: pd.DataFrame,
    min_score: int,
    allowed_categories: list[str],
    classification_config: dict | None = None,
    score_weights: dict | None = None,
) -> pd.DataFrame:
    """
    对单只股票逐日生成历史信号。
    每一行代表某个 as_of_date 产生了一次候选信号。
    """
    signals = []
    start_idx = 300

    for i in range(start_idx, len(factor_df)):
        as_of_date = factor_df.iloc[i]["date"]
        latest = factor_df.iloc[: i + 1].iloc[-1]

        scored = score_latest(latest, classification_config, score_weights)
        category, action = classify_latest(latest, scored["score"], classification_config, min_score)

        if scored["score"] < min_score:
            continue
        if category not in allowed_categories:
            continue
        if category == "剔除":
            continue

        spread_breakout_level = "无"
        value = latest.get("spread", 0)
        for window in [60, 120, 250]:
            high = latest.get(f"spread_high_{window}", 0)
            if pd.notna(value) and pd.notna(high) and value > high:
                spread_breakout_level = f"{window}日"

        price_breakout_level = "无"
        value = latest.get("stock_close", 0)
        for window in [60, 120, 250]:
            high = latest.get(f"price_high_{window}", 0)
            if pd.notna(value) and pd.notna(high) and value > high:
                price_breakout_level = f"{window}日"

        index_close = safe_float(latest["index_close"])
        market_ma60 = safe_float(latest["market_ma60"])
        market_ma20 = safe_float(latest["market_ma20"])
        if pd.notna(index_close) and pd.notna(market_ma60) and index_close > market_ma60:
            market_status = "强"
        elif pd.notna(index_close) and pd.notna(market_ma20) and index_close > market_ma20:
            market_status = "中性"
        else:
            market_status = "弱"

        risk_flags = scored.get("risk_flags", [])
        risk_flags_str = "、".join(risk_flags) if risk_flags else ""

        reason = scored.get("reason", [])
        reason_str = "；".join(reason) if reason else ""

        signals.append(
            {
                "code": code,
                "name": name,
                "benchmark": benchmark,
                "as_of_date": as_of_date,
                "score": scored["score"],
                "category": category,
                "action": action,
                "close": round(safe_float(latest["stock_close"]), 3),
                "index_close": round(safe_float(latest["index_close"]), 3),
                "spread": round(safe_float(latest["spread"]), 3),
                "spread_breakout_level": spread_breakout_level,
                "price_breakout_level": price_breakout_level,
                "volume_ratio": round(safe_float(latest["volume_ratio"]), 3),
                "market_status": market_status,
                "risk_flags": risk_flags_str,
                "reason": reason_str,
            }
        )

    return pd.DataFrame(signals)


def add_forward_returns(
    signals_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    holding_days: list[int],
) -> pd.DataFrame:
    """
    给每条历史信号增加未来 N 日收益、基准收益、超额收益、最大回撤。
    """
    if signals_df.empty:
        return signals_df

    factor_df = factor_df.set_index("date")
    result = signals_df.copy()

    for days in holding_days:
        ret_col = f"ret_{days}"
        bench_col = f"benchmark_ret_{days}"
        excess_col = f"excess_ret_{days}"
        dd_col = f"max_drawdown_{days}"

        result[ret_col] = float("nan")
        result[bench_col] = float("nan")
        result[excess_col] = float("nan")
        result[dd_col] = float("nan")

    for idx, row in result.iterrows():
        as_of_date = pd.to_datetime(row["as_of_date"])
        signal_close = row["close"]
        signal_index_close = row["index_close"]

        future_dates = []
        for i in range(len(factor_df)):
            d = factor_df.index[i]
            if d > as_of_date:
                future_dates.append(d)
            if len(future_dates) >= max(holding_days):
                break

        for days in holding_days:
            if len(future_dates) < days:
                continue

            future_date = future_dates[days - 1]
            if future_date not in factor_df.index:
                continue

            future_row = factor_df.loc[future_date]
            future_stock_close = safe_float(future_row.get("stock_close"))
            future_index_close = safe_float(future_row.get("index_close"))

            if pd.isna(future_stock_close) or pd.isna(signal_close) or signal_close == 0:
                continue
            if pd.isna(future_index_close) or pd.isna(signal_index_close) or signal_index_close == 0:
                continue

            ret = future_stock_close / signal_close - 1
            benchmark_ret = future_index_close / signal_index_close - 1
            excess_ret = ret - benchmark_ret

            start_idx = list(factor_df.index).index(as_of_date)
            exit_idx = min(start_idx + days, len(factor_df) - 1)
            window_prices = factor_df["stock_close"].iloc[start_idx : exit_idx + 1]

            if len(window_prices) > 0:
                running_max = window_prices.expanding().max()
                drawdowns = (window_prices - running_max) / running_max
                max_dd = drawdowns.min()
            else:
                max_dd = float("nan")

            result.at[idx, f"ret_{days}"] = ret
            result.at[idx, f"benchmark_ret_{days}"] = benchmark_ret
            result.at[idx, f"excess_ret_{days}"] = excess_ret
            result.at[idx, f"max_drawdown_{days}"] = max_dd

    return result


def summarize_backtest(
    trades: pd.DataFrame,
    holding_days: list[int],
) -> pd.DataFrame:
    """
    汇总全部信号的收益统计。
    """
    rows = []

    for days in holding_days:
        ret_col = f"ret_{days}"
        excess_col = f"excess_ret_{days}"
        dd_col = f"max_drawdown_{days}"

        if ret_col not in trades.columns:
            continue

        valid = trades.dropna(subset=[ret_col])
        if valid.empty:
            continue

        rets = valid[ret_col]
        excess_rets = valid[excess_col] if excess_col in valid.columns else pd.Series()
        dds = valid[dd_col] if dd_col in valid.columns else pd.Series()

        wins = rets[rets > 0]
        losses = rets[rets < 0]

        win_rate = len(wins) / len(rets) if len(rets) > 0 else 0

        if excess_rets.notna().any():
            excess_wins = excess_rets[excess_rets > 0]
            excess_win_rate = len(excess_wins) / excess_rets.notna().sum() if excess_rets.notna().sum() > 0 else 0
        else:
            excess_win_rate = 0

        avg_pl_ratio = float("nan")
        if len(losses) > 0 and losses.abs().mean() > 0:
            avg_pl_ratio = wins.mean() / losses.abs().mean() if len(wins) > 0 else float("nan")

        rows.append(
            {
                "holding_days": days,
                "signal_count": len(valid),
                "avg_return": round(rets.mean(), 4) if len(rets) > 0 else float("nan"),
                "median_return": round(rets.median(), 4) if len(rets) > 0 else float("nan"),
                "win_rate": round(win_rate, 4),
                "avg_excess_return": round(excess_rets.mean(), 4) if excess_rets.notna().any() else float("nan"),
                "median_excess_return": round(excess_rets.median(), 4) if excess_rets.notna().any() else float("nan"),
                "excess_win_rate": round(excess_win_rate, 4),
                "avg_max_drawdown": round(dds.mean(), 4) if dds.notna().any() else float("nan"),
                "profit_loss_ratio": round(avg_pl_ratio, 4) if pd.notna(avg_pl_ratio) else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_category(
    trades: pd.DataFrame,
    holding_days: list[int],
) -> pd.DataFrame:
    """
    按 category 分组统计 A+类 / A类 / B类 / C类 的表现。
    """
    rows = []

    for category in trades["category"].unique():
        category_trades = trades[trades["category"] == category]

        for days in holding_days:
            ret_col = f"ret_{days}"
            excess_col = f"excess_ret_{days}"
            dd_col = f"max_drawdown_{days}"

            if ret_col not in category_trades.columns:
                continue

            valid = category_trades.dropna(subset=[ret_col])
            if valid.empty:
                continue

            rets = valid[ret_col]
            excess_rets = valid[excess_col] if excess_col in valid.columns else pd.Series()
            dds = valid[dd_col] if dd_col in valid.columns else pd.Series()

            wins = rets[rets > 0]
            losses = rets[rets < 0]

            win_rate = len(wins) / len(rets) if len(rets) > 0 else 0

            if excess_rets.notna().any():
                excess_wins = excess_rets[excess_rets > 0]
                excess_win_rate = len(excess_wins) / excess_rets.notna().sum() if excess_rets.notna().sum() > 0 else 0
            else:
                excess_win_rate = 0

            avg_pl_ratio = float("nan")
            if len(losses) > 0 and losses.abs().mean() > 0:
                avg_pl_ratio = wins.mean() / losses.abs().mean() if len(wins) > 0 else float("nan")

            rows.append(
                {
                    "category": category,
                    "holding_days": days,
                    "signal_count": len(valid),
                    "avg_return": round(rets.mean(), 4) if len(rets) > 0 else float("nan"),
                    "median_return": round(rets.median(), 4) if len(rets) > 0 else float("nan"),
                    "win_rate": round(win_rate, 4),
                    "avg_excess_return": round(excess_rets.mean(), 4) if excess_rets.notna().any() else float("nan"),
                    "median_excess_return": round(excess_rets.median(), 4) if excess_rets.notna().any() else float("nan"),
                    "excess_win_rate": round(excess_win_rate, 4),
                    "avg_max_drawdown": round(dds.mean(), 4) if dds.notna().any() else float("nan"),
                    "profit_loss_ratio": round(avg_pl_ratio, 4) if pd.notna(avg_pl_ratio) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def apply_signal_cooldown(signals_df: pd.DataFrame, cooldown: int) -> pd.DataFrame:
    """
    应用信号冷却，避免同一股票连续产生重复信号。
    """
    if signals_df.empty or cooldown <= 0:
        return signals_df

    signals_df = signals_df.sort_values("as_of_date").reset_index(drop=True)
    result = []
    last_signal_date = {}

    for _, row in signals_df.iterrows():
        code = row["code"]
        as_of_date = row["as_of_date"]

        if code in last_signal_date:
            days_diff = (as_of_date - last_signal_date[code]).days
            if days_diff < cooldown:
                continue

        result.append(row.to_dict())
        last_signal_date[code] = as_of_date

    return pd.DataFrame(result)


def process_one_stock(
    code: str,
    name: str,
    benchmark: str,
    index_df: pd.DataFrame,
    start: str,
    end: str,
    adjust: str,
    min_bars: int,
    cache_config: dict,
    retry_config: dict,
    local_stock_path: str,
    force_refresh: bool,
    indicator_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df, source = get_stock_daily(
        code=code,
        start=start,
        end=end,
        adjust=adjust,
        cache_config=cache_config,
        retry_config=retry_config,
        local_path=local_stock_path,
        force_refresh=force_refresh,
    )
    merged = align_stock_index(df, index_df)
    if len(merged) < min_bars:
        raise ValueError(f"数据不足：{code}, bars={len(merged)}")

    factors = add_difference_indicators(merged, indicator_config)
    return factors, merged, source


def save_backtest_outputs(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    by_category: pd.DataFrame,
    output_dir: str,
    run_params: dict,
) -> tuple[str, str, str, str]:
    os.makedirs(output_dir, exist_ok=True)

    trades_path = os.path.join(output_dir, "backtest_trades.csv")
    summary_path = os.path.join(output_dir, "backtest_summary.csv")
    by_category_path = os.path.join(output_dir, "backtest_by_category.csv")
    summary_md_path = os.path.join(output_dir, "backtest_summary.md")

    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_category.to_csv(by_category_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 逐日差价图回测报告",
        "",
        "## 回测参数",
        "",
        f"- 股票池：{run_params.get('stock_pool', '-')}",
        f"- 基准指数：{run_params.get('benchmark', '-')}",
        f"- 区间：{run_params.get('start', '-')} ~ {run_params.get('end', '-')}",
        f"- 持有周期：{', '.join(map(str, run_params.get('holding_days', [])))}",
        f"- 最低分数：{run_params.get('min_score', '-')}",
        f"- 信号冷却：{run_params.get('signal_cooldown', '-')} 天",
        "",
        "## 总体表现",
        "",
        "| 持有天数 | 信号数 | 平均收益 | 胜率 | 平均超额收益 | 超额胜率 | 平均最大回撤 |",
        "|---|---|---|---|---|---|---|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['holding_days'])} | {int(row['signal_count'])} | "
            f"{row['avg_return']:.2%} | {row['win_rate']:.2%} | "
            f"{row['avg_excess_return']:.2%} | {row['excess_win_rate']:.2%} | "
            f"{row['avg_max_drawdown']:.2%} |"
        )

    lines.extend(["", "## 分类型表现", "", "| 分类 | 持有天数 | 信号数 | 平均收益 | 胜率 | 平均超额收益 | 超额胜率 | 平均最大回撤 |", "|---|---|---|---|---|---|---|---|"])

    for _, row in by_category.sort_values(["category", "holding_days"]).iterrows():
        lines.append(
            f"| {row['category']} | {int(row['holding_days'])} | {int(row['signal_count'])} | "
            f"{row['avg_return']:.2%} | {row['win_rate']:.2%} | "
            f"{row['avg_excess_return']:.2%} | {row['excess_win_rate']:.2%} | "
            f"{row['avg_max_drawdown']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            "- 如果 A+类 20日平均超额收益显著为正，说明强突破有统计优势。",
            "- 如果 C类收益高但最大回撤也高，说明强者恒强可能有效，但追高风险大。",
            "- 如果 B类信号数少，要谨慎判断，不要过度解读。",
        ]
    )

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return trades_path, summary_path, by_category_path, summary_md_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    benchmark = normalize_index_code(args.benchmark or config["benchmark"]["default"])
    start = args.start or config["data"]["default_start"]
    end = args.end or config["data"].get("default_end") or "20260527"
    adjust = config["data"].get("adjust", "qfq")
    min_bars = int(config["data"].get("min_bars", 300))

    holding_days = [int(x.strip()) for x in args.holding_days.split(",")]
    min_score = args.min_score
    allowed_categories = [x.strip() for x in args.categories.split(",")]
    signal_cooldown = args.signal_cooldown

    cache_config = config.get("cache", {})
    fallback_config = config.get("local_fallback", {})
    retry_config = config.get("retry", {})
    force_refresh = args.force_refresh

    backtest_config = config.get("backtest", {})
    backtest_output_dir = backtest_config.get("output_dir", "reports/backtest")

    classification_config = config.get("classification", {})
    indicator_config = config.get("params", {})
    score_weights = load_score_weights(config)

    ensure_dirs(backtest_output_dir)
    stock_pool = build_stock_pool(args)

    date_range = f"{start}~{end}"
    print(f"回测参数：")
    print(f"  基准指数：{benchmark}")
    print(f"  日期范围：{date_range}")
    print(f"  持有天数：{holding_days}")
    print(f"  最低分数：{min_score}")
    print(f"  分类过滤：{allowed_categories}")
    print(f"  信号冷却：{signal_cooldown} 天")
    print(f"  股票数量：{len(stock_pool)}")

    print(f"\n读取基准指数：{benchmark}")
    try:
        index_df, index_source = get_index_daily(
            benchmark=benchmark,
            start=start,
            end=end,
            cache_config=cache_config,
            fallback_config=fallback_config,
            retry_config=retry_config,
            local_path=get_local_index_path(config, benchmark),
            force_refresh=force_refresh,
        )
    except Exception as e:
        print(f"ERROR: 基准指数读取失败：{benchmark}: {e}")
        return

    print(f"基准指数数据源：{index_source}")

    all_signals = []
    errors = []

    for idx, row in stock_pool.iterrows():
        code = row["code"]
        name = row["name"]
        try:
            print(f"[{idx + 1}/{len(stock_pool)}] 处理 {code} {name}")
            factor_df, merged, source = process_one_stock(
                code=code,
                name=name,
                benchmark=benchmark,
                index_df=index_df,
                start=start,
                end=end,
                adjust=adjust,
                min_bars=min_bars,
                cache_config=cache_config,
                retry_config=retry_config,
                local_stock_path=get_local_stock_path(config, code),
                force_refresh=force_refresh,
                indicator_config=indicator_config,
            )

            signals = generate_historical_signals(
                code=code,
                name=name,
                benchmark=benchmark,
                factor_df=factor_df,
                min_score=min_score,
                allowed_categories=allowed_categories,
                classification_config=classification_config,
                score_weights=score_weights,
            )

            if not signals.empty:
                signals = apply_signal_cooldown(signals, signal_cooldown)
                signals = add_forward_returns(signals, factor_df, holding_days)
                all_signals.append(signals)

        except Exception as e:
            error_msg = f"{code} {name}: {e}"
            print("ERROR:", error_msg)
            errors.append(
                {
                    "code": code,
                    "name": name,
                    "source": "stock_fetch",
                    "stage": "backtest",
                    "date_range": date_range,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

        if idx < len(stock_pool) - 1:
            if source and "akshare" in source.lower():
                sleep_seconds = retry_config.get("retry_sleep_seconds", 2)
                time.sleep(sleep_seconds)

    if not all_signals:
        print("\n没有产生任何回测信号。")
        return

    all_trades = pd.concat(all_signals, ignore_index=True)

    summary = summarize_backtest(all_trades, holding_days)
    by_category = summarize_by_category(all_trades, holding_days)

    stock_pool_path = args.stock_pool or args.codes or "cli"

    run_params = {
        "stock_pool": stock_pool_path,
        "benchmark": benchmark,
        "start": start,
        "end": end,
        "holding_days": holding_days,
        "min_score": min_score,
        "signal_cooldown": signal_cooldown,
    }

    trades_path, summary_path, by_category_path, summary_md_path = save_backtest_outputs(
        trades=all_trades,
        summary=summary,
        by_category=by_category,
        output_dir=backtest_output_dir,
        run_params=run_params,
    )

    print("\n回测完成。")
    print(f"信号总数：{len(all_trades)}")
    print(f"交易记录：{trades_path}")
    print(f"总体统计：{summary_path}")
    print(f"分类型统计：{by_category_path}")
    print(f"摘要报告：{summary_md_path}")

    if errors:
        error_path = os.path.join(backtest_output_dir, "backtest_errors.csv")
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"错误日志：{error_path}")


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    main()