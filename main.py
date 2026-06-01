import argparse
import os
import time
import traceback

import pandas as pd

from data_provider import (
    get_index_daily,
    get_stock_daily,
    normalize_index_code,
    normalize_stock_code,
    read_stock_pool,
)
from data_quality import check_price_data_quality, is_warning_fatal
from indicators import add_difference_indicators, align_stock_index
from report import ensure_dirs, plot_stock_report, save_csv, save_summary
from scoring import build_result_row, load_score_weights, sort_results
from utils import load_config, resolve_template, should_throttle_after_source, today_str


RESULT_COLUMNS = [
    "code",
    "name",
    "latest_date",
    "benchmark",
    "score",
    "category",
    "action",
    "close",
    "spread",
    "spread_slope_60",
    "spread_slope_120",
    "spread_slope_250",
    "below_zero_ratio_250",
    "spread_breakout_level",
    "price_breakout_level",
    "volume_ratio",
    "market_status",
    "risk_flags",
    "reason",
    "score_positive_items",
    "score_negative_items",
    "missing_conditions",
    "data_source",
]

ERROR_COLUMNS = ["code", "name", "source", "stage", "date_range", "error", "traceback"]

DATA_WARNING_COLUMNS = [
    "code",
    "name",
    "source",
    "stage",
    "date_range",
    "warning_type",
    "warning_message",
    "latest_date",
    "rows",
]

REQUIRED_FACTOR_COLUMNS = [
    "date",
    "stock_close",
    "index_close",
    "volume",
    "stock_index",
    "market_index",
    "spread",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐日差价图选股系统")
    parser.add_argument("--codes", type=str, default="", help="逗号分隔股票代码，例如 600519,000858")
    parser.add_argument("--stock-pool", type=str, default="", help="股票池 CSV，包含 code,name")
    parser.add_argument("--benchmark", type=str, default="", help="基准指数代码，例如 000300")
    parser.add_argument("--start", type=str, default="", help="开始日期，例如 20220101")
    parser.add_argument("--end", type=str, default="", help="结束日期，例如 20260527")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-plot", action="store_true", help="不生成图片")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="忽略缓存，强制从 AKShare 获取最新数据并更新缓存",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑，跳过已处理的股票",
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
    return resolve_template(template, code=code) if template else ""


def get_local_index_path(config: dict, benchmark: str) -> str:
    template = config["data"].get("local_index_template", "")
    return resolve_template(template, benchmark=benchmark) if template else ""


def filter_valid_factor_rows(factors: pd.DataFrame, code: str = "") -> pd.DataFrame:
    missing_factor_columns = [
        column for column in REQUIRED_FACTOR_COLUMNS if column not in factors.columns
    ]
    if missing_factor_columns:
        stock_label = f"{code}, " if code else ""
        raise ValueError(f"指标缺少必要字段：{stock_label}columns={missing_factor_columns}")
    return factors.dropna(subset=REQUIRED_FACTOR_COLUMNS).reset_index(drop=True)


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
    classification_config: dict,
    observe_score_threshold: int,
    indicator_config: dict | None = None,
    score_weights: dict | None = None,
) -> tuple[dict, pd.DataFrame, str]:
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
    factors = filter_valid_factor_rows(factors, code)
    if factors.empty:
        raise ValueError(f"指标计算后无有效数据：{code}")

    result_row = build_result_row(
        code,
        name,
        benchmark,
        factors,
        classification_config,
        observe_score_threshold,
        score_weights,
    )
    return result_row, factors, source


def select_chart_items(
    candidates: pd.DataFrame,
    chart_data_by_code: dict[str, tuple[str, str, dict, pd.DataFrame]],
    max_images: int,
) -> list[tuple[str, str, dict, pd.DataFrame]]:
    if candidates.empty or max_images <= 0:
        return []

    selected = []
    for code in candidates["code"].astype(str):
        item = chart_data_by_code.get(code)
        if item is None:
            continue
        selected.append(item)
        if len(selected) >= max_images:
            break
    return selected


def save_outputs(
    all_results: pd.DataFrame,
    candidates: pd.DataFrame,
    errors: list[dict],
    data_warnings: list[dict],
    report_dir: str,
    run_params: dict | None = None,
) -> tuple[str, str, str, str, str]:
    all_results_path = os.path.join(report_dir, "all_results.csv")
    candidates_path = os.path.join(report_dir, "candidates.csv")
    summary_path = os.path.join(report_dir, "summary.md")
    error_path = os.path.join(report_dir, "errors.csv")
    warning_path = os.path.join(report_dir, "data_warnings.csv")

    if all_results.empty:
        all_results = pd.DataFrame(columns=RESULT_COLUMNS)
    if candidates.empty:
        candidates = pd.DataFrame(columns=RESULT_COLUMNS)

    save_csv(all_results, all_results_path)
    save_csv(candidates, candidates_path)
    save_csv(pd.DataFrame(errors, columns=ERROR_COLUMNS), error_path)
    save_csv(pd.DataFrame(data_warnings, columns=DATA_WARNING_COLUMNS), warning_path)
    save_summary(candidates, all_results, summary_path, run_params or {})
    return all_results_path, candidates_path, summary_path, error_path, warning_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    benchmark = normalize_index_code(args.benchmark or config["benchmark"]["default"])
    start = args.start or config["data"]["default_start"]
    end = args.end or config["data"].get("default_end") or today_str()
    adjust = config["data"].get("adjust", "qfq")
    min_bars = int(config["data"].get("min_bars", 300))
    recent_bars = int(config["data"].get("recent_bars", 500))
    classification_config = config.get("classification", {})
    report_dir = config["paths"]["report_dir"]
    image_dir = config["paths"]["image_dir"]
    min_candidate_score = int(config["score"]["min_candidate_score"])
    strong_candidate_score = int(config["score"].get("strong_candidate_score", 80))
    indicator_config = config.get("params", {})
    score_weights = load_score_weights(config)

    cache_config = config.get("cache", {})
    fallback_config = config.get("local_fallback", {})
    retry_config = config.get("retry", {})
    force_refresh = args.force_refresh
    resume = args.resume

    ensure_dirs(report_dir, image_dir)
    stock_pool = build_stock_pool(args)

    existing_results = {}
    existing_rows_df = pd.DataFrame(columns=RESULT_COLUMNS)
    if resume:
        all_results_path = os.path.join(report_dir, "all_results.csv")
        if os.path.exists(all_results_path):
            try:
                existing_df = pd.read_csv(all_results_path, dtype={"code": str})
                if not existing_df.empty and "code" in existing_df.columns and "latest_date" in existing_df.columns:
                    for _, row in existing_df.iterrows():
                        existing_results[row["code"]] = row["latest_date"]
                    existing_rows_df = existing_df
                    print(f"断点续跑：发现 {len(existing_results)} 条已有记录")
            except Exception:
                pass

    date_range = f"{start}~{end}"
    print(f"读取基准指数：{benchmark}")
    print(f"日期范围：{date_range}")
    print(f"强制刷新：{force_refresh}")
    print(f"断点续跑：{resume}")

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
        errors = [
            {
                "code": benchmark,
                "name": "benchmark",
                "source": "unknown",
                "stage": "index_fetch",
                "date_range": date_range,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        ]
        all_results_path, candidates_path, summary_path, error_path, warning_path = save_outputs(
            all_results=pd.DataFrame(columns=RESULT_COLUMNS),
            candidates=pd.DataFrame(columns=RESULT_COLUMNS),
            errors=errors,
            data_warnings=[],
            report_dir=report_dir,
        )
        print("")
        print("完成，但基准指数读取失败，未处理股票。")
        print(f"全量结果：{all_results_path}")
        print(f"候选结果：{candidates_path}")
        print(f"摘要报告：{summary_path}")
        print(f"错误日志：{error_path}")
        return

    print(f"基准指数数据源：{index_source}")

    index_warnings = check_price_data_quality(
        df=index_df,
        code=benchmark,
        name=benchmark,
        start=start,
        end=end,
        source=index_source,
        min_bars=min_bars,
    )
    for w in index_warnings:
        if is_warning_fatal(w):
            print(f"WARNING: 基准指数数据质量严重问题：{w['warning_message']}")
        else:
            print(f"  数据质量提示：{w['warning_message']}")

    all_rows = []
    chart_data_by_code = {}
    errors = []
    data_warnings = []

    for idx, row in stock_pool.iterrows():
        code = row["code"]
        name = row["name"]
        source = ""
        had_error = False

        if resume and code in existing_results:
            existing_date = existing_results[code]
            latest_index_date = index_df["date"].max().strftime("%Y-%m-%d") if not index_df.empty else ""
            if existing_date == latest_index_date:
                print(f"SKIP {code} already processed (latest_date={existing_date})")
                continue

        try:
            print(f"[{idx + 1}/{len(stock_pool)}] 处理 {code} {name}")
            result_row, factor_df, source = process_one_stock(
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
                classification_config=classification_config,
                observe_score_threshold=min_candidate_score,
                indicator_config=indicator_config,
                score_weights=score_weights,
            )
            result_row["data_source"] = source
            all_rows.append(result_row)

            stock_warnings = check_price_data_quality(
                df=factor_df,
                code=code,
                name=name,
                start=start,
                end=end,
                source=source,
                min_bars=min_bars,
            )
            for w in stock_warnings:
                if is_warning_fatal(w):
                    print(f"    数据质量严重问题：{w['warning_message']}")
                else:
                    print(f"    数据质量提示：{w['warning_message']}")
                data_warnings.append(w)

            if result_row["score"] >= min_candidate_score and result_row["category"] != "剔除":
                chart_data_by_code[code] = (code, name, result_row, factor_df)
        except Exception as e:
            had_error = True
            error_msg = f"{code} {name}: {e}"
            print("ERROR:", error_msg)
            errors.append(
                {
                    "code": code,
                    "name": name,
                    "source": "stock_fetch",
                    "stage": "stock_fetch",
                    "date_range": date_range,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        if idx < len(stock_pool) - 1:
            sleep_seconds = retry_config.get("retry_sleep_seconds", 2)
            if sleep_seconds > 0 and should_throttle_after_source(source, had_error):
                time.sleep(sleep_seconds)

    new_results = pd.DataFrame(all_rows)
    if resume and not existing_rows_df.empty:
        if not new_results.empty:
            old_codes_to_drop = set(new_results["code"].tolist())
            existing_rows_df = existing_rows_df[~existing_rows_df["code"].isin(old_codes_to_drop)]
        all_results = pd.concat([existing_rows_df, new_results], ignore_index=True)
    else:
        all_results = new_results

    if not all_results.empty:
        all_results = sort_results(all_results)

    if all_results.empty:
        candidates = pd.DataFrame()
    else:
        candidates = all_results[
            (all_results["score"] >= min_candidate_score)
            & (all_results["category"] != "剔除")
        ].copy()
        candidates["is_core_candidate"] = (
            (candidates["score"] >= strong_candidate_score)
            & (candidates["category"].isin(["A+类", "A类", "B类"]))
        )

    plot_enabled = config["plot"].get("enabled", True) and not args.no_plot
    max_images = int(config["plot"].get("max_images", 50))
    if plot_enabled:
        for code, name, result_row, factor_df in select_chart_items(
            candidates,
            chart_data_by_code,
            max_images,
        ):
            try:
                plot_stock_report(
                    code=code,
                    name=name,
                    category=result_row["category"],
                    score=result_row["score"],
                    df=factor_df,
                    image_dir=image_dir,
                    recent_bars=recent_bars,
                    spread_breakout_level=result_row.get("spread_breakout_level", "无"),
                    price_breakout_level=result_row.get("price_breakout_level", "无"),
                )
            except Exception as e:
                print(f"画图失败 {code}: {e}")

    run_params = {
        "benchmark": benchmark,
        "start": start,
        "end": end,
        "stock_count": len(stock_pool),
        "candidate_count": len(candidates),
    }
    all_results_path, candidates_path, summary_path, error_path, warning_path = save_outputs(
        all_results=all_results,
        candidates=candidates,
        errors=errors,
        data_warnings=data_warnings,
        report_dir=report_dir,
        run_params=run_params,
    )

    print("")
    print("完成。")
    print(f"全量结果：{all_results_path}")
    print(f"候选结果：{candidates_path}")
    print(f"摘要报告：{summary_path}")
    if errors:
        print(f"错误日志：{error_path}")
    if data_warnings:
        print(f"数据质量警告：{warning_path}")


if __name__ == "__main__":
    main()
