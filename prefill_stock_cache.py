import argparse
import os
import time
import traceback

import pandas as pd

from data_provider import (
    _stock_longterm_cache_path,
    normalize_stock_code,
    read_cache,
    read_stock_pool,
    save_cache,
)
from utils import ensure_parent_dir, load_config, today_str


CACHE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用腾讯日线批量预填个股长期缓存")
    parser.add_argument("--codes", type=str, default="", help="逗号分隔股票代码")
    parser.add_argument("--stock-pool", type=str, default="data/full_pool.csv", help="股票池 CSV，包含 code,name")
    parser.add_argument("--start", type=str, default="", help="开始日期，例如 20220101")
    parser.add_argument("--end", type=str, default="", help="结束日期，例如 20260527")
    parser.add_argument("--adjust", type=str, default="", help="复权类型，默认读取 config.yaml data.adjust")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--cache-dir", type=str, default="", help="缓存目录，默认读取 config.yaml cache.stock_cache_dir")
    parser.add_argument("--errors-path", type=str, default="", help="错误 CSV 输出路径")
    parser.add_argument("--audit-path", type=str, default="", help="缓存预填审计 CSV 输出路径")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 只股票，0 表示不限制")
    parser.add_argument("--retry-times", type=int, default=-1, help="失败重试次数，默认读取 config.yaml retry.akshare_retry_times")
    parser.add_argument("--sleep", type=float, default=-1, help="请求间隔秒数，默认读取 config.yaml retry.retry_sleep_seconds")
    parser.add_argument("--force", action="store_true", help="即使缓存覆盖区间也重新抓取")
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="保留系统代理；默认设置 NO_PROXY='*' 直连",
    )
    return parser.parse_args()


def stock_tx_symbol(code: str) -> str:
    code = normalize_stock_code(code)
    if code.startswith("6"):
        return f"sh{code}"
    if code.startswith(("0", "3")):
        return f"sz{code}"
    if code.startswith(("4", "8", "920")):
        return f"bj{code}"
    raise ValueError(f"unsupported stock code for Tencent prefix: {code}")


def build_stock_pool(args: argparse.Namespace) -> pd.DataFrame:
    if args.codes:
        rows = []
        for code in args.codes.split(","):
            if not code.strip():
                continue
            code = normalize_stock_code(code)
            rows.append({"code": code, "name": code})
        return pd.DataFrame(rows)
    return read_stock_pool(args.stock_pool)


def normalize_tencent_stock_df(df: pd.DataFrame, code: str) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "date": "date",
        "开盘": "open",
        "open": "open",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "收盘": "close",
        "close": "close",
        "成交量": "volume",
        "volume": "volume",
        "vol": "volume",
        "amount": "amount",
        "成交额": "amount",
        "turnover": "amount",
    }
    normalized_columns = {}
    for column in df.columns:
        key = str(column).strip()
        normalized_columns[column] = rename_map.get(key, rename_map.get(key.lower(), column))

    out = df.rename(columns=normalized_columns).copy()
    if "volume" not in out.columns and "amount" in out.columns:
        out["volume"] = out["amount"]
        out["amount"] = pd.NA
    if "amount" not in out.columns:
        out["amount"] = pd.NA

    for column in ["open", "high", "low"]:
        if column not in out.columns:
            out[column] = pd.NA

    missing = [column for column in ["date", "close", "volume"] if column not in out.columns]
    if missing:
        raise ValueError(f"Tencent stock data missing columns {missing}: {code}")

    out = out[CACHE_COLUMNS]
    out["date"] = pd.to_datetime(out["date"])
    for column in CACHE_COLUMNS:
        if column != "date":
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close", "volume"])
    if out.empty:
        raise ValueError(f"Tencent stock data empty after normalization: {code}")
    return out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def fetch_tencent_stock_daily(
    code: str,
    start: str,
    end: str,
    adjust: str,
    retry_times: int,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, str]:
    import akshare as ak

    symbol = stock_tx_symbol(code)
    last_error = None
    for attempt in range(retry_times + 1):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            if df is not None and not df.empty:
                return normalize_tencent_stock_df(df, code), symbol
            last_error = ValueError(f"empty Tencent data: {code} {symbol}")
        except Exception as e:
            last_error = e
        if attempt < retry_times and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    raise ValueError(f"Tencent fetch failed for {code} {symbol}: {last_error}")


def cache_coverage(path: str, start: str, end: str) -> dict:
    if not os.path.exists(path):
        return {"covers": False, "rows": 0, "latest_date": "", "start_date": ""}
    try:
        df = read_cache(path)
    except Exception:
        return {"covers": False, "rows": 0, "latest_date": "", "start_date": ""}
    if df.empty:
        return {"covers": False, "rows": 0, "latest_date": "", "start_date": ""}

    cached_start = df["date"].min()
    cached_end = df["date"].max()
    req_start = pd.to_datetime(start)
    req_end = pd.to_datetime(end)
    return {
        "covers": req_start >= cached_start and req_end <= cached_end,
        "rows": len(df),
        "latest_date": cached_end.strftime("%Y-%m-%d"),
        "start_date": cached_start.strftime("%Y-%m-%d"),
    }


def prefill_one_stock(
    code: str,
    name: str,
    start: str,
    end: str,
    adjust: str,
    cache_dir: str,
    retry_times: int,
    sleep_seconds: float,
    force: bool,
) -> dict:
    cache_path = _stock_longterm_cache_path(cache_dir, code, adjust)
    cached = cache_coverage(cache_path, start, end)
    if not force and cached["covers"]:
        return {
            "code": code,
            "name": name,
            "status": "skipped",
            "source": "cache",
            "symbol": stock_tx_symbol(code),
            "rows": cached["rows"],
            "start_date": cached["start_date"],
            "latest_date": cached["latest_date"],
            "cache_path": cache_path,
            "error": "",
            "traceback": "",
        }

    df, symbol = fetch_tencent_stock_daily(
        code=code,
        start=start,
        end=end,
        adjust=adjust,
        retry_times=retry_times,
        sleep_seconds=sleep_seconds,
    )
    save_cache(df, cache_path)
    return {
        "code": code,
        "name": name,
        "status": "fetched",
        "source": "tencent_tx",
        "symbol": symbol,
        "rows": len(df),
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "latest_date": df["date"].max().strftime("%Y-%m-%d"),
        "cache_path": cache_path,
        "error": "",
        "traceback": "",
    }


def save_run_outputs(rows: list[dict], audit_path: str, errors_path: str) -> None:
    ensure_parent_dir(audit_path)
    ensure_parent_dir(errors_path)
    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    errors = audit[audit["status"] == "error"] if not audit.empty else audit
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not args.use_system_proxy:
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

    start = args.start or config["data"]["default_start"]
    end = args.end or config["data"].get("default_end") or today_str()
    adjust = args.adjust or config["data"].get("adjust", "qfq")
    cache_dir = args.cache_dir or config.get("cache", {}).get("stock_cache_dir", "data/cache/stocks")
    report_dir = config["paths"]["report_dir"]
    retry_times = args.retry_times if args.retry_times >= 0 else int(config.get("retry", {}).get("akshare_retry_times", 2))
    sleep_seconds = args.sleep if args.sleep >= 0 else float(config.get("retry", {}).get("retry_sleep_seconds", 2))
    errors_path = args.errors_path or os.path.join(report_dir, "cache_fetch_errors.csv")
    audit_path = args.audit_path or os.path.join(report_dir, "cache_prefill_audit.csv")

    stock_pool = build_stock_pool(args)
    if args.limit > 0:
        stock_pool = stock_pool.head(args.limit)

    print(f"股票数：{len(stock_pool)}")
    print(f"日期范围：{start}~{end}")
    print(f"缓存目录：{cache_dir}")
    print(f"请求间隔：{sleep_seconds}s")

    rows = []
    for idx, row in stock_pool.iterrows():
        code = row["code"]
        name = row["name"]
        try:
            result = prefill_one_stock(
                code=code,
                name=name,
                start=start,
                end=end,
                adjust=adjust,
                cache_dir=cache_dir,
                retry_times=retry_times,
                sleep_seconds=sleep_seconds,
                force=args.force,
            )
            rows.append(result)
            print(
                f"[{idx + 1}/{len(stock_pool)}] {code} {name} "
                f"{result['status']} rows={result['rows']} latest={result['latest_date']}"
            )
        except Exception as e:
            result = {
                "code": code,
                "name": name,
                "status": "error",
                "source": "tencent_tx",
                "symbol": "",
                "rows": 0,
                "start_date": "",
                "latest_date": "",
                "cache_path": _stock_longterm_cache_path(cache_dir, code, adjust),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            rows.append(result)
            print(f"[{idx + 1}/{len(stock_pool)}] ERROR {code} {name}: {e}")

        if idx < len(stock_pool) - 1 and sleep_seconds > 0 and result["status"] != "skipped":
            time.sleep(sleep_seconds)

    save_run_outputs(rows, audit_path, errors_path)
    audit = pd.DataFrame(rows)
    fetched = int((audit["status"] == "fetched").sum()) if not audit.empty else 0
    skipped = int((audit["status"] == "skipped").sum()) if not audit.empty else 0
    failed = int((audit["status"] == "error").sum()) if not audit.empty else 0
    if audit.empty:
        latest_dates = pd.Series(dtype="datetime64[ns]")
    else:
        latest_dates = pd.to_datetime(
            audit.loc[audit["status"].isin(["fetched", "skipped"]), "latest_date"],
            errors="coerce",
        ).dropna()
    common_latest = latest_dates.min().strftime("%Y%m%d") if not latest_dates.empty else ""

    print("")
    print("缓存预填完成。")
    print(f"抓取：{fetched}，跳过：{skipped}，失败：{failed}")
    if common_latest:
        print(f"建议 main.py --end {common_latest}")
    print(f"审计文件：{audit_path}")
    if failed:
        print(f"错误文件：{errors_path}")


if __name__ == "__main__":
    main()
