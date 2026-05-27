#!/usr/bin/env python3
"""Generate a professional HTML report from test results."""

import csv
import json
import re
import os
from collections import Counter, defaultdict

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))

all_results_path = os.path.join(REPORTS_DIR, "all_results.csv")
candidates_path = os.path.join(REPORTS_DIR, "candidates.csv")
errors_path = os.path.join(REPORTS_DIR, "errors.csv")
summary_path = os.path.join(REPORTS_DIR, "summary.md")


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_summary(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    stock_count = int(re.search(r"全量股票数：(\d+)", text).group(1))
    candidate_count = int(re.search(r"候选股票数：(\d+)", text).group(1))
    benchmark = re.search(r"benchmark: (\d+)", text).group(1)
    start = re.search(r"start: (\d+)", text).group(1)
    end = re.search(r"end: (\d+)", text).group(1)
    cat_counts = {}
    for m in re.finditer(r"- (\S+): (\d+)", text):
        cat_counts[m.group(1)] = int(m.group(2))
    avg_scores = {}
    for m in re.finditer(r"(\S+类):\s*平均得分=([\d.]+)", text):
        avg_scores[m.group(1)] = float(m.group(2))
    return {
        "stock_count": stock_count,
        "candidate_count": candidate_count,
        "benchmark": benchmark,
        "start": start,
        "end": end,
        "cat_counts": cat_counts,
        "avg_scores": avg_scores,
    }


def parse_score_detail(s):
    """Parse score_positive_items like 'spread_slope_60_positive:+10|...' into list."""
    if not s:
        return []
    items = []
    for part in s.split("|"):
        if ":" in part:
            name, score = part.rsplit(":", 1)
            items.append((name, int(score)))
    return items


def score_badge(score):
    if score >= 80:
        return "score-a"
    elif score >= 60:
        return "score-b"
    elif score >= 40:
        return "score-c"
    else:
        return "score-d"


def category_color(cat):
    colors = {
        "A+类": "#10b981",
        "A类": "#22c55e",
        "B类": "#3b82f6",
        "C类": "#f59e0b",
        "C类-过热观察": "#ef4444",
        "观察": "#8b5cf6",
        "剔除": "#6b7280",
    }
    return colors.get(cat, "#6b7280")


def risk_tags(risk_flags):
    if not risk_flags:
        return ""
    tags = risk_flags.split("、")
    html = ""
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        if "过热" in tag:
            cls = "risk-hot"
        elif "涨幅" in tag:
            cls = "risk-price"
        elif "跌破" in tag:
            cls = "risk-break"
        else:
            cls = "risk-default"
        html += f'<span class="risk-tag {cls}">{tag}</span>'
    return html


def main():
    all_results = read_csv(all_results_path)
    candidates = read_csv(candidates_path)
    errors = read_csv(errors_path)
    summary = parse_summary(summary_path)

    # Compute category distribution
    cat_dist = Counter(r["category"] for r in all_results)

    # Score distribution
    score_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    score_dist = {}
    for lo, hi in score_ranges:
        label = f"{lo}-{hi}"
        if hi == 100:
            label = f"{lo}+"
        if lo == 0:
            label = "<20"
        score_dist[label] = sum(1 for r in all_results if lo <= int(r["score"]) < (hi if hi < 100 else 200))

    # Latest date
    latest_date = all_results[0]["latest_date"] if all_results else ""

    # Market status distribution
    market_dist = Counter(r["market_status"] for r in all_results)

    # Average metrics by category
    cat_metrics = defaultdict(lambda: {"scores": [], "vol_ratios": [], "spreads": []})
    for r in all_results:
        cat = r["category"]
        cat_metrics[cat]["scores"].append(float(r["score"]))
        cat_metrics[cat]["vol_ratios"].append(float(r["volume_ratio"]))
        cat_metrics[cat]["spreads"].append(float(r["spread"]))

    cat_avg = {}
    for cat, vals in cat_metrics.items():
        cat_avg[cat] = {
            "avg_score": sum(vals["scores"]) / len(vals["scores"]),
            "avg_vol": sum(vals["vol_ratios"]) / len(vals["vol_ratios"]),
            "avg_spread": sum(vals["spreads"]) / len(vals["spreads"]),
            "count": len(vals["scores"]),
        }

    # Top candidates (core candidates from candidates)
    core_candidates = [c for c in candidates if c.get("is_core_candidate", "").strip().lower() == "true"]
    top_candidates = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)[:20]

    # Build score detail items for top candidates
    def score_summary(candidate):
        pos_items = parse_score_detail(candidate.get("score_positive_items", ""))
        neg_items = parse_score_detail(candidate.get("score_negative_items", ""))
        total = sum(v for _, v in pos_items) + sum(v for _, v in neg_items)
        return pos_items, neg_items, total

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>逐日差价图选股报告</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --card-hover: #273548;
    --border: #334155;
    --text: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent: #3b82f6;
    --accent-light: #60a5fa;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --purple: #8b5cf6;
    --gray: #6b7280;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; padding: 20px; }
  .container { max-width: 1400px; margin: 0 auto; }

  /* Header */
  .header { text-align: center; padding: 40px 20px; border-bottom: 1px solid var(--border); margin-bottom: 30px; }
  .header h1 { font-size: 32px; font-weight: 700; background: linear-gradient(135deg, var(--accent-light), var(--green)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header .subtitle { color: var(--text-secondary); margin-top: 8px; font-size: 14px; }
  .header .meta { display: flex; justify-content: center; gap: 30px; margin-top: 16px; flex-wrap: wrap; }
  .header .meta-item { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); font-size: 13px; }
  .header .meta-item svg { width: 16px; height: 16px; flex-shrink: 0; }

  /* Stats Grid */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 30px; }
  .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; transition: all .2s; }
  .stat-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .stat-card .stat-value { font-size: 28px; font-weight: 700; }
  .stat-card .stat-label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }
  .stat-card.green .stat-value { color: var(--green); }
  .stat-card.blue .stat-value { color: var(--accent-light); }
  .stat-card.yellow .stat-value { color: var(--yellow); }
  .stat-card.red .stat-value { color: var(--red); }
  .stat-card.purple .stat-value { color: var(--purple); }

  /* Category Distribution */
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }
  .section-title { font-size: 18px; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
  .section-title .count { color: var(--text-secondary); font-size: 14px; font-weight: 400; }

  .cat-chart { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; min-height: 200px; padding: 10px 0; }
  .cat-bar-wrapper { flex: 1; min-width: 70px; display: flex; flex-direction: column; align-items: center; }
  .cat-bar { width: 48px; border-radius: 6px 6px 0 0; transition: height .5s ease; min-height: 4px; position: relative; }
  .cat-bar .cat-count { position: absolute; top: -22px; left: 50%; transform: translateX(-50%); font-size: 13px; font-weight: 600; white-space: nowrap; }
  .cat-bar-label { margin-top: 8px; font-size: 11px; color: var(--text-secondary); text-align: center; }

  /* Table */
  .table-wrapper { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border); color: var(--text-secondary); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }
  td { padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:hover td { background: var(--card-hover); }
  .text-right { text-align: right; }
  .text-center { text-align: center; }

  .score-badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-weight: 700; font-size: 13px; min-width: 38px; text-align: center; }
  .score-a { background: rgba(16, 185, 129, .2); color: var(--green); }
  .score-b { background: rgba(59, 130, 246, .2); color: var(--accent-light); }
  .score-c { background: rgba(245, 158, 11, .2); color: var(--yellow); }
  .score-d { background: rgba(239, 68, 68, .2); color: var(--red); }

  .cat-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; white-space: nowrap; }

  .risk-tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin: 1px; }
  .risk-hot { background: rgba(239, 68, 68, .2); color: var(--red); }
  .risk-price { background: rgba(245, 158, 11, .2); color: var(--yellow); }
  .risk-break { background: rgba(139, 92, 246, .2); color: var(--purple); }
  .risk-default { background: rgba(107, 114, 128, .2); color: var(--text-secondary); }

  .action-text { font-size: 12px; color: var(--text-secondary); max-width: 200px; }

  .num { font-variant-numeric: tabular-nums; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }

  /* Score detail */
  .score-detail { font-size: 11px; color: var(--text-secondary); }
  .score-detail .pos { color: var(--green); }
  .score-detail .neg { color: var(--red); }

  /* Progress bar */
  .bar-bg { background: var(--border); border-radius: 4px; height: 6px; width: 100px; display: inline-block; vertical-align: middle; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .5s; }

  /* Market status */
  .market-strong { color: var(--green); }
  .market-weak { color: var(--red); }

  /* Summary grid */
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .summary-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .summary-card h3 { font-size: 14px; color: var(--text-secondary); margin-bottom: 12px; text-transform: uppercase; letter-spacing: .5px; }
  .summary-card .metric-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }

  /* Error section */
  .error-empty { color: var(--green); font-size: 14px; }

  @media (max-width: 768px) {
    .header .meta { gap: 12px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .cat-chart { gap: 8px; }
    .cat-bar { width: 32px; }
  }
</style>
</head>
<body>
<div class="container">
""")

    # Header
    html_parts.append(f"""
<div class="header">
  <h1>📊 逐日差价图选股报告</h1>
  <div class="subtitle">基于相对强弱分析的全市场扫描 · {latest_date}</div>
  <div class="meta">
    <span class="meta-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>{summary["end"]}</span>
    <span class="meta-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7V4h16v3"/><path d="M9 20h6"/><path d="M12 4v16"/></svg>基准指数: {summary["benchmark"]} (沪深300)</span>
    <span class="meta-item"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>数据区间: {summary["start"]} - {summary["end"]}</span>
  </div>
</div>
""")

    # Top stats
    candidate_cat_dist = Counter(r["category"] for r in candidates)
    error_count = len(errors) - 1 if len(errors) > 1 else 0
    avg_score_all = sum(float(r["score"]) for r in all_results) / len(all_results) if all_results else 0
    strong_count = sum(1 for r in all_results if r.get("market_status", "").strip() == "强")

    html_parts.append(f"""
<div class="stats-grid">
  <div class="stat-card blue">
    <div class="stat-value">{summary["stock_count"]}</div>
    <div class="stat-label">全量扫描</div>
  </div>
  <div class="stat-card green">
    <div class="stat-value">{summary["candidate_count"]}</div>
    <div class="stat-label">候选股票</div>
  </div>
  <div class="stat-card yellow">
    <div class="stat-value">{avg_score_all:.1f}</div>
    <div class="stat-label">平均得分</div>
  </div>
  <div class="stat-card purple">
    <div class="stat-value">{strong_count}</div>
    <div class="stat-label">强势股数</div>
  </div>
  <div class="stat-card red">
    <div class="stat-value">{error_count}</div>
    <div class="stat-label">错误数量</div>
  </div>
</div>
""")

    # Category Distribution Chart
    cat_order = ["A+类", "A类", "B类", "C类", "C类-过热观察", "观察", "剔除"]
    cat_colors = ["#10b981", "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#6b7280"]
    max_cat_count = max(cat_dist.values()) if cat_dist else 1
    bars_html = ""
    for cat, color in zip(cat_order, cat_colors):
        count = cat_dist.get(cat, 0)
        pct = count / summary["stock_count"] * 100 if summary["stock_count"] else 0
        height = max(4, count / max_cat_count * 160)
        bars_html += f"""
  <div class="cat-bar-wrapper">
    <div class="cat-bar" style="height:{height:.0f}px;background:{color}">
      <span class="cat-count">{count}</span>
    </div>
    <div class="cat-bar-label">{cat}</div>
    <div style="font-size:10px;color:var(--text-secondary)">{pct:.1f}%</div>
  </div>"""

    html_parts.append(f"""
<div class="section">
  <div class="section-title">📈 分类分布 <span class="count">({summary["stock_count"]} 只股票)</span></div>
  <div class="cat-chart">
    {bars_html}
  </div>
</div>
""")

    # Per-category average metrics
    html_parts.append("""
<div class="summary-grid">
  <div class="summary-card">
    <h3>🏆 候选股分类</h3>""")
    for cat, cnt in sorted(candidate_cat_dist.items(), key=lambda x: -x[1]):
        c = category_color(cat)
        html_parts.append(f'    <div class="metric-row"><span><span class="cat-badge" style="background:{c}22;color:{c}">{cat}</span></span><span>{cnt}</span></div>')
    html_parts.append("  </div>")

    # Average metrics by category
    html_parts.append("""
  <div class="summary-card">
    <h3>📊 分类平均指标</h3>""")
    for cat in cat_order:
        if cat in cat_avg and cat_avg[cat]["count"] > 0:
            m = cat_avg[cat]
            c = category_color(cat)
            html_parts.append(f'    <div class="metric-row"><span><span class="cat-badge" style="background:{c}22;color:{c}">{cat}</span></span><span>得分 {m["avg_score"]:.1f} · 量比 {m["avg_vol"]:.2f}</span></div>')
    html_parts.append("  </div>")

    # Market status
    html_parts.append("""
  <div class="summary-card">
    <h3>📌 市场状态分布</h3>""")
    for status, cnt in sorted(market_dist.items(), key=lambda x: -x[1]):
        cls = "market-strong" if status == "强" else ""
        pct = cnt / summary["stock_count"] * 100
        html_parts.append(f'    <div class="metric-row"><span class="{cls}">{status}</span><span>{cnt} ({pct:.1f}%)</span></div>')
    html_parts.append("""
  </div>
</div>
""")

    # Top Candidates Table
    html_parts.append(f"""
<div class="section">
  <div class="section-title">🏅 候选股 Top {len(candidates)}</div>
  <div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>代码</th>
        <th>名称</th>
        <th>分类</th>
        <th>得分</th>
        <th>最新价</th>
        <th>差价</th>
        <th>量比</th>
        <th>差价斜率60</th>
        <th>差价斜率120</th>
        <th>差价突破</th>
        <th>价格突破</th>
        <th>风险标记</th>
        <th>操作建议</th>
      </tr>
    </thead>
    <tbody>
""")
    for i, c in enumerate(candidates, 1):
        score = int(float(c["score"]))
        close = float(c["close"])
        spread = float(c["spread"])
        vol = float(c["volume_ratio"])
        cat = c["category"]
        cc = category_color(cat)
        sb = score_badge(score)
        risk_html = risk_tags(c.get("risk_flags", ""))
        action = c.get("action", "")
        action_short = action[:25] + "…" if len(action) > 25 else action

        spread_str = f'{spread:+.2f}' if spread >= 0 else f'{spread:.2f}'

        html_parts.append(f"""
      <tr>
        <td>{i}</td>
        <td><strong>{c["code"]}</strong></td>
        <td>{c["name"]}</td>
        <td><span class="cat-badge" style="background:{cc}22;color:{cc}">{cat}</span></td>
        <td><span class="score-badge {sb}">{score}</span></td>
        <td class="num">{close:.2f}</td>
        <td class="num {'positive' if spread >= 0 else 'negative'}">{spread_str}</td>
        <td class="num">{vol:.2f}</td>
        <td class="num">{float(c["spread_slope_60"]):+.3f}</td>
        <td class="num">{float(c["spread_slope_120"]):+.3f}</td>
        <td>{c.get("spread_breakout_level", "-")}</td>
        <td>{c.get("price_breakout_level", "-")}</td>
        <td>{risk_html or '-'}</td>
        <td class="action-text">{action_short}</td>
      </tr>""")

    html_parts.append("""
    </tbody>
  </table>
  </div>
</div>
""")

    # Core candidates detail
    if core_candidates:
        html_parts.append(f"""
<div class="section">
  <div class="section-title">⭐ 核心推荐 ({len(core_candidates)} 只)</div>
  <div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>代码</th>
        <th>名称</th>
        <th>分类</th>
        <th>得分</th>
        <th>得分明细</th>
        <th>操作建议</th>
      </tr>
    </thead>
    <tbody>
""")
        for c in core_candidates:
            score = int(float(c["score"]))
            cat = c["category"]
            cc = category_color(cat)
            sb = score_badge(score)
            pos_items, neg_items, total = score_summary(c)
            score_detail_html = ""
            for name, val in pos_items:
                display_name = name.replace("_", " ").replace("spread", "差价").replace("slope", "斜率").replace("breakout", "突破").replace("price", "股价").replace("market", "指数").replace("volume", "量").replace("ratio", "比").replace("above", "高于").replace("ma", "均线").replace("positive", "转正").replace("long weak short turn", "长期弱转强")[:25]
                score_detail_html += f'<div class="pos">+{val} {display_name}</div>'
            for name, val in neg_items:
                score_detail_html += f'<div class="neg">{val} {name}</div>'
            missing = c.get("missing_conditions", "")
            if missing:
                missing_short = missing[:40] + "…" if len(missing) > 40 else missing
                score_detail_html += f'<div style="color:var(--yellow);font-size:10px;margin-top:4px">⚠ {missing_short}</div>'
            action = c.get("action", "")

            html_parts.append(f"""
      <tr>
        <td><strong>{c["code"]}</strong></td>
        <td>{c["name"]}</td>
        <td><span class="cat-badge" style="background:{cc}22;color:{cc}">{cat}</span></td>
        <td><span class="score-badge {sb}">{score}</span></td>
        <td class="score-detail">{score_detail_html}</td>
        <td class="action-text">{action}</td>
      </tr>""")
        html_parts.append("""
    </tbody>
  </table>
  </div>
</div>
""")

    # All results summary (top 50)
    html_parts.append(f"""
<div class="section">
  <div class="section-title">📋 全量结果 <span class="count">({len(all_results)} 只, 显示前50)</span></div>
  <div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>代码</th>
        <th>名称</th>
        <th>分类</th>
        <th>得分</th>
        <th>收盘价</th>
        <th>差价</th>
        <th>量比</th>
        <th>市场</th>
        <th>风险</th>
        <th>操作建议</th>
      </tr>
    </thead>
    <tbody>
""")
    for i, r in enumerate(all_results[:50], 1):
        score = int(float(r["score"]))
        cat = r["category"]
        cc = category_color(cat)
        sb = score_badge(score)
        close = float(r["close"])
        spread = float(r["spread"])
        vol = float(r["volume_ratio"])
        risk_html = risk_tags(r.get("risk_flags", ""))
        market = r.get("market_status", "")
        action = r.get("action", "")
        action_short = action[:20] + "…" if len(action) > 20 else action
        spread_str = f'{spread:+.2f}' if spread >= 0 else f'{spread:.2f}'
        html_parts.append(f"""
      <tr>
        <td>{i}</td>
        <td><strong>{r["code"]}</strong></td>
        <td>{r["name"]}</td>
        <td><span class="cat-badge" style="background:{cc}22;color:{cc}">{cat}</span></td>
        <td><span class="score-badge {sb}">{score}</span></td>
        <td class="num">{close:.2f}</td>
        <td class="num {'positive' if spread >= 0 else 'negative'}">{spread_str}</td>
        <td class="num">{vol:.2f}</td>
        <td>{market}</td>
        <td>{risk_html or '-'}</td>
        <td class="action-text">{action_short}</td>
      </tr>""")
    html_parts.append("""
    </tbody>
  </table>
  </div>
</div>
""")

    # Errors
    error_data_rows = errors[1:] if len(errors) > 1 else []
    if error_data_rows:
        html_parts.append("""
<div class="section">
  <div class="section-title">⚠️ 错误记录</div>
  <div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>代码</th>
        <th>名称</th>
        <th>阶段</th>
        <th>错误信息</th>
      </tr>
    </thead>
    <tbody>
""")
        for e in error_data_rows:
            html_parts.append(f"""
      <tr>
        <td>{e.get("code", "-")}</td>
        <td>{e.get("name", "-")}</td>
        <td>{e.get("stage", "-")}</td>
        <td style="color:var(--red);font-size:12px">{e.get("error", "-")}</td>
      </tr>""")
        html_parts.append("""
    </tbody>
  </table>
  </div>
</div>
""")
    else:
        html_parts.append(f"""
<div class="section">
  <div class="section-title">✅ 错误记录</div>
  <div class="error-empty">无错误 — 全部 {summary["stock_count"]} 只股票处理成功</div>
</div>
""")

    # Footer
    html_parts.append(f"""
<div style="text-align:center;padding:30px 0;color:var(--text-secondary);font-size:12px;border-top:1px solid var(--border);margin-top:20px">
  报告生成于 {summary["end"]} · 全量扫描 {summary["stock_count"]} 只 · 候选 {summary["candidate_count"]} 只
</div>
</div>
</body>
</html>""")

    output_path = os.path.join(REPORTS_DIR, "report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"✅ Report generated: {output_path}")


if __name__ == "__main__":
    main()
