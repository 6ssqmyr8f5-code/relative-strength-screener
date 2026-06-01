# AGENTS.md

## Project

A股个股相对强度筛选工具（逐日差价图选股）。对比个股与基准指数的标准化走势，计算差价(spread)及其技术指标，打分分类并输出候选股。

## Commands

```bash
# Install
pip install -r requirements.txt

# Single stocks
python main.py --codes 600519,000858 --benchmark 000300 --start 20220101

# From stock pool CSV (columns: code, name)
python main.py --stock-pool pool.csv --benchmark 000300

# Prefill Tencent stock cache before full-market scans
NO_PROXY='*' no_proxy='*' python prefill_stock_cache.py --stock-pool data/full_pool.csv --start 20220101 --end YYYYMMDD

# Skip chart generation (faster for batch runs)
python main.py --codes 600519 --no-plot
```

## Architecture

| Module | Role |
|---|---|
| `main.py` | CLI entry point, orchestrates fetch → align → score → report |
| `data_provider.py` | AKShare (online) first, local CSV fallback (`local_stock_template` / `local_index_template` in config) |
| `indicators.py` | Price normalization (base=100), spread = stock_index - market_index, rolling slopes, breakouts |
| `scoring.py` | Point-based scoring (0-100) + classification: A+ > A > B > C > C类-过热观察 > 观察 > 剔除 |
| `report.py` | Matplotlib charts (price + spread panels), CSV/Summary output |
| `prefill_stock_cache.py` | Tencent daily data prefill for long-term stock cache before full-market scans |
| `config.yaml` | All tunable params: windows, thresholds, paths |

## Key facts

- **Data source**: AKShare (`stock_zh_a_hist` for stocks, `index_zh_a_hist` / `stock_zh_index_daily_em` for index) with local CSV fallback
- **Code normalization**: 6-digit codes, strips `.SH`/`.SZ`/`.BJ` suffixes automatically
- **Index codes**: 000300 (沪深300), 000001 (上证指数), 000905 (中证500), 000852 (中证1000), 399006 (创业板指), 000688 (科创50)
- **Outputs**: `{report_dir}/all_results.csv`, `candidates.csv`, `summary.md`, `errors.csv`, plus PNG charts in `{image_dir}/`
- **Config-driven**: All windows, thresholds, score cutoffs in `config.yaml` — edit there, not in code
- **Chinese fonts**: matplotlib configured for Chinese labels via `Arial Unicode MS` / `PingFang SC` etc.
- **Tests**: pytest tests live under `tests/`; no linting/CI configured

## Full-market data workflow

When running the full A-share universe, do **not** rely only on the default Eastmoney AKShare path. In this environment, `requests` may pick up a system proxy such as `127.0.0.1:7897`, and Eastmoney endpoints can fail with `ProxyError` / `RemoteDisconnected`. Use this workflow:

1. Run network-sensitive commands with direct requests:

```bash
NO_PROXY='*' no_proxy='*' <command>
```

2. Run `python build_pool.py` to refresh `data/full_pool.csv`. The script now tries Eastmoney first and automatically falls back to Sina raw A-share quotes when Eastmoney disconnects. Sina's `mktcap` unit is 万元, so the 100 亿 filter is `mktcap >= 1,000,000` before converting to yuan. Keep `data/full_pool_with_market_cap.csv` as the audit file. A normal current run is roughly: 全A 5523, ST/退市 255, non-ST and >=100 亿 2010.

3. Before the main scan, prefill `data/cache/stocks/{code}_qfq.csv` for every code in `data/full_pool.csv` using AKShare Tencent daily data. The script automatically uses `sh` for `6xxxxx`, `sz` for `0xxxxx` / `3xxxxx`, and `bj` for Beijing-market codes, stores Tencent's standalone `amount` field as project `volume`, skips already-covered caches, writes `reports/difference_chart/cache_prefill_audit.csv`, and prints the recommended latest common `--end` date:

```bash
NO_PROXY='*' no_proxy='*' python prefill_stock_cache.py --stock-pool data/full_pool.csv --start 20220101 --end YYYYMMDD
```

4. Run the main scan from cache. Set `--end` to the latest common stock trading date actually present in the cache (for example `20260526`), not necessarily today's date, so the runner does not attempt pointless incremental online fetches.

```bash
NO_PROXY='*' no_proxy='*' python main.py --stock-pool data/full_pool.csv --benchmark 000300 --start 20220101 --end YYYYMMDD
```

5. If doing a faster batch run first, add `--no-plot`; then rerun without `--no-plot` to generate candidate charts after the results look sane.

## Stock pool CSV format

```csv
code,name
600519,贵州茅台
000858,五粮液
```
