import os
import time
import traceback
from dataclasses import dataclass

import pandas as pd


@dataclass
class DataFetchResult:
    data: pd.DataFrame
    source: str
    warning: str = ""


def normalize_stock_code(code: str) -> str:
    code = str(code).strip()
    upper_code = code.upper()
    for suffix in [".SH", ".SZ", ".BJ"]:
        if upper_code.endswith(suffix):
            code = code[: -len(suffix)]
            break
    return code.zfill(6)


def normalize_index_code(index_code: str) -> str:
    index_code = str(index_code).strip()
    upper_code = index_code.upper()
    for suffix in [".SH", ".SZ", ".BJ"]:
        if upper_code.endswith(suffix):
            index_code = index_code[: -len(suffix)]
            break
    return index_code.zfill(6)


def index_exchange_symbol(index_code: str) -> str:
    index_code = normalize_index_code(index_code)
    prefix = "sz" if index_code.startswith("399") else "sh"
    return f"{prefix}{index_code}"


def _filter_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != "date"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _normalize_stock_df(df: pd.DataFrame, code: str) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=rename_map).copy()
    required = ["date", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"stock data missing columns {missing}: {code}")

    for col in ["open", "high", "low", "amount"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    df["date"] = pd.to_datetime(df["date"])
    df = _coerce_numeric(df)
    df = df.dropna(subset=["date", "close", "volume"])
    return df.sort_values("date").reset_index(drop=True)


def _normalize_index_df(df: pd.DataFrame, index_code: str) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "收盘": "close",
        "date": "date",
        "close": "close",
    }
    df = df.rename(columns=rename_map).copy()
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"index data missing date/close columns: {index_code}, "
            f"columns={df.columns.tolist()}"
        )
    df = df[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    return df.sort_values("date").reset_index(drop=True)


def retry_call(func, retry_times: int, sleep_seconds: int, *args, **kwargs):
    last_error = None
    for attempt in range(retry_times + 1):
        try:
            result = func(*args, **kwargs)
            if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                return result
        except Exception as e:
            last_error = e
        if attempt < retry_times:
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    return None


def _retry_call(fn, *args, retries=3, delay=3, **kwargs):
    for attempt in range(retries):
        try:
            result = fn(*args, **kwargs)
            if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                return result
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    return None


def fetch_stock_daily_akshare(
    code: str,
    start: str,
    end: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    import akshare as ak

    code = normalize_stock_code(code)
    df = _retry_call(
        ak.stock_zh_a_hist,
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust=adjust,
    )
    if df is None or df.empty:
        raise ValueError(f"empty stock data: {code}")
    return _normalize_stock_df(df, code)


def fetch_index_daily_akshare(index_code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    index_code = normalize_index_code(index_code)
    last_error = None
    df = None

    df = _retry_call(
        ak.stock_zh_index_daily_em,
        symbol=index_exchange_symbol(index_code),
        start_date=start,
        end_date=end,
    )

    if df is None:
        last_error = Exception("stock_zh_index_daily_em failed after retries")
        df = _retry_call(
            ak.stock_zh_index_daily_em,
            symbol=index_code,
            start_date=start,
            end_date=end,
        )

    if df is None or df.empty:
        try:
            df = ak.stock_zh_index_daily(symbol=index_exchange_symbol(index_code))
        except Exception as e:
            last_error = e
            df = None

    if df is None or df.empty:
        raise ValueError(f"empty index data: {index_code}, error={last_error}")

    df = _normalize_index_df(df, index_code)
    return _filter_by_date(df, start, end)


def load_local_stock_csv(path: str, code: str, start: str, end: str) -> pd.DataFrame:
    df = load_local_csv(path)
    df = _normalize_stock_df(df, code)
    return _filter_by_date(df, start, end)


def load_local_index_csv(path: str, index_code: str, start: str, end: str) -> pd.DataFrame:
    df = load_local_csv(path)
    df = _normalize_index_df(df, index_code)
    return _filter_by_date(df, start, end)


def fetch_stock_daily(
    code: str,
    start: str,
    end: str,
    adjust: str,
    local_path: str = "",
) -> DataFetchResult:
    code = normalize_stock_code(code)
    ak_error = None
    try:
        return DataFetchResult(
            data=fetch_stock_daily_akshare(code, start, end, adjust=adjust),
            source="akshare",
        )
    except Exception as e:
        ak_error = e

    if local_path:
        try:
            return DataFetchResult(
                data=load_local_stock_csv(local_path, code, start, end),
                source=f"local_csv:{local_path}",
                warning=f"AKShare failed: {ak_error}",
            )
        except Exception as local_error:
            raise ValueError(
                f"stock data failed for {code}; akshare={ak_error}; "
                f"local_csv={local_error}"
            ) from local_error

    raise ValueError(f"stock data failed for {code}; akshare={ak_error}")


def fetch_index_daily(
    index_code: str,
    start: str,
    end: str,
    local_path: str = "",
) -> DataFetchResult:
    index_code = normalize_index_code(index_code)
    ak_error = None
    try:
        return DataFetchResult(
            data=fetch_index_daily_akshare(index_code, start, end),
            source="akshare",
        )
    except Exception as e:
        ak_error = e

    if local_path:
        try:
            return DataFetchResult(
                data=load_local_index_csv(local_path, index_code, start, end),
                source=f"local_csv:{local_path}",
                warning=f"AKShare failed: {ak_error}",
            )
        except Exception as local_error:
            raise ValueError(
                f"index data failed for {index_code}; akshare={ak_error}; "
                f"local_csv={local_error}"
            ) from local_error

    raise ValueError(f"index data failed for {index_code}; akshare={ak_error}")


def read_stock_pool(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    if "code" not in df.columns:
        raise ValueError("stock pool csv must contain column: code")
    if "name" not in df.columns:
        df["name"] = df["code"]
    df["code"] = df["code"].apply(normalize_stock_code)
    df["name"] = df["name"].fillna(df["code"]).astype(str)
    return df[["code", "name"]]


def load_local_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in df.columns:
        raise ValueError(f"local csv missing date column: {path}")
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def is_cache_valid(path: str, expire_days: int) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return False
    mtime = os.path.getmtime(path)
    age_days = (time.time() - mtime) / 86400
    return age_days <= expire_days


def read_cache(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def save_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _stock_cache_path(cache_dir: str, code: str, start: str, end: str, adjust: str) -> str:
    return os.path.join(cache_dir, f"{code}_{start}_{end}_{adjust}.csv")


def _stock_longterm_cache_path(cache_dir: str, code: str, adjust: str) -> str:
    return os.path.join(cache_dir, f"{code}_{adjust}.csv")


def _index_cache_path(cache_dir: str, benchmark: str, start: str, end: str) -> str:
    return os.path.join(cache_dir, f"{benchmark}_{start}_{end}.csv")


def _index_longterm_cache_path(cache_dir: str, benchmark: str) -> str:
    return os.path.join(cache_dir, f"{benchmark}.csv")


def get_stock_daily(
    code: str,
    start: str,
    end: str,
    adjust: str,
    cache_config: dict,
    retry_config: dict,
    local_path: str = "",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    code = normalize_stock_code(code)
    cache_enabled = cache_config.get("enabled", True)
    expire_days = cache_config.get("expire_days", 1)
    cache_dir = cache_config.get("stock_cache_dir", "data/cache/stocks")
    retry_times = retry_config.get("akshare_retry_times", 2)
    sleep_seconds = retry_config.get("retry_sleep_seconds", 2)

    longterm_cache_path = _stock_longterm_cache_path(cache_dir, code, adjust)

    if force_refresh:
        return _fetch_and_cache_stock(
            code=code,
            start=start,
            end=end,
            adjust=adjust,
            cache_path=longterm_cache_path,
            retry_times=retry_times,
            sleep_seconds=sleep_seconds,
            local_path=local_path,
        )

    if cache_enabled and os.path.exists(longterm_cache_path):
        cached_df = read_cache(longterm_cache_path)
        cached_start = cached_df["date"].min()
        cached_end = cached_df["date"].max()
        req_start = pd.to_datetime(start)
        req_end = pd.to_datetime(end)

        if req_start >= cached_start and req_end <= cached_end:
            filtered = cached_df[
                (cached_df["date"] >= req_start) & (cached_df["date"] <= req_end)
            ]
            print(f"  个股长期缓存命中（区间覆盖）：{code}")
            return filtered, "cache"
        if req_start < cached_start and req_end <= cached_end:
            filtered = cached_df[cached_df["date"] <= req_end]
            print(f"  个股长期缓存命中（可用区间）：{code}")
            return filtered, "cache"

        needs_update = False
        update_start = req_start
        update_end = req_end

        if req_end > cached_end:
            needs_update = True
            update_start = cached_end + pd.Timedelta(days=1)
            update_end = req_end

        if needs_update:
            return _update_stock_cache_incremental(
                code=code,
                existing_df=cached_df,
                update_start=update_start.strftime("%Y%m%d"),
                update_end=update_end.strftime("%Y%m%d"),
                adjust=adjust,
                cache_path=longterm_cache_path,
                retry_times=retry_times,
                sleep_seconds=sleep_seconds,
                local_path=local_path,
                req_start=req_start.strftime("%Y%m%d"),
                req_end=req_end.strftime("%Y%m%d"),
            )

    return _fetch_and_cache_stock(
        code=code,
        start=start,
        end=end,
        adjust=adjust,
        cache_path=longterm_cache_path,
        retry_times=retry_times,
        sleep_seconds=sleep_seconds,
        local_path=local_path,
    )


def _fetch_and_cache_stock(
    code: str,
    start: str,
    end: str,
    adjust: str,
    cache_path: str,
    retry_times: int,
    sleep_seconds: int,
    local_path: str,
) -> tuple[pd.DataFrame, str]:
    try:
        import akshare as ak
    except ImportError as import_error:
        if os.path.exists(cache_path):
            try:
                df = read_cache(cache_path)
                print(f"  akshare 不可用，使用过期缓存：{code}")
                return df, "cache_stale"
            except Exception:
                pass

        if local_path:
            try:
                df = load_local_stock_csv(local_path, code, start, end)
                if df.empty:
                    raise ValueError(
                        f"local fallback data empty after date filter: {code} "
                        f"start={start} end={end}"
                    )
                print(f"  个股使用本地兜底数据：{code}")
                return df, f"local_csv:{local_path}"
            except Exception as local_error:
                raise ValueError(
                    f"stock data failed for {code}; akshare={import_error}; "
                    f"local_csv={local_error}"
                ) from local_error
        raise ValueError(f"stock data failed for {code}; akshare unavailable: {import_error}")

    try:
        df = retry_call(
            ak.stock_zh_a_hist,
            retry_times,
            sleep_seconds,
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
        if df is None or df.empty:
            raise ValueError(f"empty stock data: {code}")
        df = _normalize_stock_df(df, code)
        save_cache(df, cache_path)
        print(f"  个股数据已缓存（长期）：{code}")
        return df, "akshare_force_refresh"
    except Exception as ak_error:
        if os.path.exists(cache_path):
            try:
                df = read_cache(cache_path)
                print(f"  个股AKShare失败，使用过期缓存：{code}")
                return df, "cache_stale"
            except Exception:
                pass

        if local_path:
            try:
                df = load_local_stock_csv(local_path, code, start, end)
                if df.empty:
                    raise ValueError(
                        f"local fallback data empty after date filter: {code} "
                        f"start={start} end={end}"
                    )
                print(f"  个股使用本地兜底数据：{code}")
                return df, f"local_csv:{local_path}"
            except Exception as local_error:
                raise ValueError(
                    f"stock data failed for {code}; akshare={ak_error}; "
                    f"local_csv={local_error}"
                ) from local_error
        raise


def _update_stock_cache_incremental(
    code: str,
    existing_df: pd.DataFrame,
    update_start: str,
    update_end: str,
    adjust: str,
    cache_path: str,
    retry_times: int,
    sleep_seconds: int,
    local_path: str,
    req_start: str,
    req_end: str,
) -> tuple[pd.DataFrame, str]:
    import akshare as ak

    try:
        new_df = retry_call(
            ak.stock_zh_a_hist,
            retry_times,
            sleep_seconds,
            symbol=code,
            period="daily",
            start_date=update_start,
            end_date=update_end,
            adjust=adjust,
        )
        if new_df is None or new_df.empty:
            raise ValueError(f"empty incremental data: {code}")
        new_df = _normalize_stock_df(new_df, code)

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        save_cache(combined, cache_path)
        print(f"  个股缓存增量更新：{code} ({update_start}~{update_end})")

        req_start_dt = pd.to_datetime(req_start)
        req_end_dt = pd.to_datetime(req_end)
        filtered = combined[
            (combined["date"] >= req_start_dt) & (combined["date"] <= req_end_dt)
        ]
        return filtered, "cache_incremental_update"
    except Exception as ak_error:
        req_start_dt = pd.to_datetime(req_start)
        req_end_dt = pd.to_datetime(req_end)
        filtered = existing_df[
            (existing_df["date"] >= req_start_dt) & (existing_df["date"] <= req_end_dt)
        ]
        if not filtered.empty:
            print(f"  个股增量更新失败，使用现有缓存：{code}")
            return filtered, "cache_incremental_fallback"
        if local_path:
            try:
                df = load_local_stock_csv(local_path, code, req_start, req_end)
                print(f"  个股使用本地兜底数据：{code}")
                return df, f"local_csv:{local_path}"
            except Exception:
                pass
        raise ValueError(f"stock data failed for {code}; akshare={ak_error}")


def get_index_daily(
    benchmark: str,
    start: str,
    end: str,
    cache_config: dict,
    fallback_config: dict,
    retry_config: dict,
    local_path: str = "",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    benchmark = normalize_index_code(benchmark)
    cache_enabled = cache_config.get("enabled", True)
    cache_dir = cache_config.get("index_cache_dir", "data/cache/index")
    retry_times = retry_config.get("akshare_retry_times", 2)
    sleep_seconds = retry_config.get("retry_sleep_seconds", 2)
    fallback_enabled = fallback_config.get("enabled", True)
    fallback_dir = fallback_config.get("index_dir", "data/index")

    longterm_cache_path = _index_longterm_cache_path(cache_dir, benchmark)

    if force_refresh:
        return _fetch_and_cache_index(
            benchmark=benchmark,
            start=start,
            end=end,
            cache_path=longterm_cache_path,
            retry_times=retry_times,
            sleep_seconds=sleep_seconds,
            fallback_enabled=fallback_enabled,
            fallback_dir=fallback_dir,
            local_path=local_path,
        )

    if cache_enabled and os.path.exists(longterm_cache_path):
        cached_df = read_cache(longterm_cache_path)
        cached_start = cached_df["date"].min()
        cached_end = cached_df["date"].max()
        req_start = pd.to_datetime(start)
        req_end = pd.to_datetime(end)

        if req_start >= cached_start and req_end <= cached_end:
            filtered = cached_df[
                (cached_df["date"] >= req_start) & (cached_df["date"] <= req_end)
            ]
            print(f"指数长期缓存命中（区间覆盖）：{benchmark}")
            return filtered, "cache"
        if req_start < cached_start and req_end <= cached_end:
            filtered = cached_df[cached_df["date"] <= req_end]
            print(f"指数长期缓存命中（可用区间）：{benchmark}")
            return filtered, "cache"

        needs_update = False
        update_start = req_start
        update_end = req_end

        if req_end > cached_end:
            needs_update = True
            update_start = cached_end + pd.Timedelta(days=1)
            update_end = req_end

        if needs_update:
            return _update_index_cache_incremental(
                benchmark=benchmark,
                existing_df=cached_df,
                update_start=update_start.strftime("%Y%m%d"),
                update_end=update_end.strftime("%Y%m%d"),
                cache_path=longterm_cache_path,
                retry_times=retry_times,
                sleep_seconds=sleep_seconds,
                fallback_enabled=fallback_enabled,
                fallback_dir=fallback_dir,
                local_path=local_path,
                req_start=req_start.strftime("%Y%m%d"),
                req_end=req_end.strftime("%Y%m%d"),
            )

    return _fetch_and_cache_index(
        benchmark=benchmark,
        start=start,
        end=end,
        cache_path=longterm_cache_path,
        retry_times=retry_times,
        sleep_seconds=sleep_seconds,
        fallback_enabled=fallback_enabled,
        fallback_dir=fallback_dir,
        local_path=local_path,
    )


def _fetch_and_cache_index(
    benchmark: str,
    start: str,
    end: str,
    cache_path: str,
    retry_times: int,
    sleep_seconds: int,
    fallback_enabled: bool,
    fallback_dir: str,
    local_path: str,
) -> tuple[pd.DataFrame, str]:
    import akshare as ak

    last_error = None
    df = None

    df = retry_call(
        ak.stock_zh_index_daily_em,
        retry_times,
        sleep_seconds,
        symbol=index_exchange_symbol(benchmark),
        start_date=start,
        end_date=end,
    )

    if df is None:
        last_error = Exception("stock_zh_index_daily_em with exchange prefix failed")
        try:
            df = retry_call(
                ak.stock_zh_index_daily_em,
                retry_times,
                sleep_seconds,
                symbol=benchmark,
                start_date=start,
                end_date=end,
            )
        except Exception as e2:
            last_error = e2
            df = None

    if df is None or df.empty:
        try:
            df = ak.stock_zh_index_daily(symbol=index_exchange_symbol(benchmark))
        except Exception as e3:
            last_error = e3
            df = None

    if df is None or df.empty:
        return _index_fallback(
            benchmark=benchmark,
            start=start,
            end=end,
            fallback_enabled=fallback_enabled,
            fallback_dir=fallback_dir,
            local_path=local_path,
            error=last_error,
        )

    df = _normalize_index_df(df, benchmark)
    df = _filter_by_date(df, start, end)
    save_cache(df, cache_path)
    print(f"指数数据已缓存（长期）：{benchmark}")
    return df, "akshare_force_refresh"


def _update_index_cache_incremental(
    benchmark: str,
    existing_df: pd.DataFrame,
    update_start: str,
    update_end: str,
    cache_path: str,
    retry_times: int,
    sleep_seconds: int,
    fallback_enabled: bool,
    fallback_dir: str,
    local_path: str,
    req_start: str,
    req_end: str,
) -> tuple[pd.DataFrame, str]:
    import akshare as ak

    try:
        last_error = None
        df = None

        df = retry_call(
            ak.stock_zh_index_daily_em,
            retry_times,
            sleep_seconds,
            symbol=index_exchange_symbol(benchmark),
            start_date=update_start,
            end_date=update_end,
        )

        if df is None:
            last_error = Exception("stock_zh_index_daily_em failed")
            try:
                df = retry_call(
                    ak.stock_zh_index_daily_em,
                    retry_times,
                    sleep_seconds,
                    symbol=benchmark,
                    start_date=update_start,
                    end_date=update_end,
                )
            except Exception as e2:
                last_error = e2
                df = None

        if df is None or df.empty:
            raise ValueError(f"empty incremental index data: {benchmark}")

        df = _normalize_index_df(df, benchmark)

        combined = pd.concat([existing_df, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        save_cache(combined, cache_path)
        print(f"指数缓存增量更新：{benchmark} ({update_start}~{update_end})")

        req_start_dt = pd.to_datetime(req_start)
        req_end_dt = pd.to_datetime(req_end)
        filtered = combined[
            (combined["date"] >= req_start_dt) & (combined["date"] <= req_end_dt)
        ]
        return filtered, "cache_incremental_update"
    except Exception as ak_error:
        req_start_dt = pd.to_datetime(req_start)
        req_end_dt = pd.to_datetime(req_end)
        filtered = existing_df[
            (existing_df["date"] >= req_start_dt) & (existing_df["date"] <= req_end_dt)
        ]
        if not filtered.empty:
            print(f"指数增量更新失败，使用现有缓存：{benchmark}")
            return filtered, "cache_incremental_fallback"
        return _index_fallback(
            benchmark=benchmark,
            start=req_start,
            end=req_end,
            fallback_enabled=fallback_enabled,
            fallback_dir=fallback_dir,
            local_path=local_path,
            error=ak_error,
        )


def _index_fallback(
    benchmark: str,
    start: str,
    end: str,
    fallback_enabled: bool,
    fallback_dir: str,
    local_path: str,
    error: Exception,
) -> tuple[pd.DataFrame, str]:
    if not fallback_enabled:
        raise ValueError(f"index data failed for {benchmark}; akshare={error}")

    fallback_paths = []
    if local_path:
        fallback_paths.append(("local_csv", local_path))
    fallback_paths.append(("local_fallback", os.path.join(fallback_dir, f"{benchmark}.csv")))

    seen_paths = set()
    fallback_errors = []
    for source, fallback_path in fallback_paths:
        if not fallback_path or fallback_path in seen_paths:
            continue
        seen_paths.add(fallback_path)
        if not os.path.exists(fallback_path):
            fallback_errors.append(f"{fallback_path}: not found")
            continue
        try:
            df = load_local_index_csv(fallback_path, benchmark, start, end)
            if df.empty:
                raise ValueError(
                    f"local fallback data empty after date filter: {benchmark} "
                    f"start={start} end={end}"
                )
            print(f"指数使用本地兜底数据：{benchmark}")
            return df, f"{source}:{fallback_path}"
        except Exception as e:
            fallback_errors.append(f"{fallback_path}: {e}")

    if fallback_errors:
        raise ValueError(
            f"index data failed for {benchmark}; local fallback errors: "
            + "; ".join(fallback_errors)
        )

    raise ValueError(f"index data failed for {benchmark}; all sources exhausted")
