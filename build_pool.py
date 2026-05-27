import time

import akshare as ak
import pandas as pd
import requests


MIN_MARKET_CAP_YUAN = 1e10


def fetch_eastmoney_spot() -> pd.DataFrame:
    for attempt in range(3):
        try:
            return ak.stock_zh_a_spot_em()
        except Exception as e:
            print(f"Eastmoney 尝试 {attempt + 1}/3 失败: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Eastmoney 获取股票列表失败")


def fetch_sina_spot_with_market_cap() -> pd.DataFrame:
    from akshare.stock.stock_zh_a_sina import (
        _get_zh_a_page_count,
        demjson,
        zh_sina_a_stock_payload,
        zh_sina_a_stock_url,
    )

    rows = []
    page_count = _get_zh_a_page_count()
    for page in range(1, page_count + 1):
        payload = zh_sina_a_stock_payload.copy()
        payload.update({"page": page})
        response = requests.get(zh_sina_a_stock_url, params=payload, timeout=15)
        response.raise_for_status()
        rows.extend(demjson.decode(response.text))
        if page % 20 == 0 or page == page_count:
            print(f"Sina 已获取 {page}/{page_count} 页，rows={len(rows)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Sina 获取股票列表为空")

    out = pd.DataFrame(
        {
            "code": df["symbol"].astype(str).str.extract(r"(\d{6})")[0],
            "name": df["name"].astype(str),
            "market_cap": pd.to_numeric(df["mktcap"], errors="coerce") * 10000,
        }
    )
    return out.dropna(subset=["code"]).reset_index(drop=True)


def normalize_eastmoney_spot(df: pd.DataFrame) -> pd.DataFrame:
    name_col = [c for c in df.columns if "名称" in c][0]
    code_col = [c for c in df.columns if "代码" in c][0]
    cap_col = [c for c in df.columns if "总市值" in c][0]

    out = df.rename(
        columns={code_col: "code", name_col: "name", cap_col: "market_cap"}
    ).copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    return out[["code", "name", "market_cap"]]


def build_pool(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].fillna(df["code"]).astype(str)
    df["name_clean"] = df["name"].str.replace(" ", "", regex=False)

    st_mask = df["name_clean"].str.contains(r"\*?ST|退市|退", na=False)
    print(f"ST/退市股: {st_mask.sum()} 只，剔除")
    df = df[~st_mask]

    cap_mask = df["market_cap"] >= MIN_MARKET_CAP_YUAN
    print(f"市值<100亿: {(~cap_mask).sum()} 只，剔除")
    df = df[cap_mask]

    return df.sort_values("code").reset_index(drop=True)


def main() -> None:
    try:
        source = "eastmoney"
        raw_df = normalize_eastmoney_spot(fetch_eastmoney_spot())
    except Exception as e:
        print(f"Eastmoney 不可用，改用 Sina 原始行情兜底: {e}")
        source = "sina"
        raw_df = fetch_sina_spot_with_market_cap()

    print(f"全A股共 {len(raw_df)} 只，source={source}")
    pool = build_pool(raw_df)

    pool[["code", "name"]].to_csv("data/full_pool.csv", index=False, encoding="utf-8-sig")
    pool[["code", "name", "market_cap"]].to_csv(
        "data/full_pool_with_market_cap.csv", index=False, encoding="utf-8-sig"
    )
    print(f"股票池: {len(pool)} 只 -> data/full_pool.csv")
    print("市值审计: data/full_pool_with_market_cap.csv")


if __name__ == "__main__":
    main()
