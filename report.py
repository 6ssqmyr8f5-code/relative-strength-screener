import os

import matplotlib.pyplot as plt
import pandas as pd


def setup_chinese_font() -> None:
    """
    macOS 常见中文字体兼容。
    """
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dirs(report_dir: str, image_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_summary(
    candidates: pd.DataFrame,
    all_results: pd.DataFrame,
    path: str,
    run_params: dict | None = None,
) -> None:
    lines = [
        "# 逐日差价图选股报告",
        "",
        f"全量股票数：{len(all_results)}",
        f"候选股票数：{len(candidates)}",
        "",
    ]

    if run_params:
        lines.extend([
            "## 本次扫描参数",
            "",
            f"- benchmark: {run_params.get('benchmark', '-')}",
            f"- start: {run_params.get('start', '-')}",
            f"- end: {run_params.get('end', '-')}",
            f"- stock_count: {run_params.get('stock_count', '-')}",
            f"- candidate_count: {run_params.get('candidate_count', '-')}",
            "",
        ])

    if candidates.empty:
        lines.append("没有符合条件的候选股。")
    else:
        lines.extend(["## 候选股分类统计", ""])
        counts = candidates["category"].value_counts()
        for category, count in counts.items():
            lines.append(f"- {category}: {count}")

        lines.extend(["", "## 各类平均指标", ""])
        for category in ["A+类", "A类", "B类", "C类", "C类-过热观察", "观察"]:
            subset = candidates[candidates["category"] == category]
            if not subset.empty:
                avg_score = subset["score"].mean()
                avg_vol_ratio = subset["volume_ratio"].mean()
                avg_below_zero = subset["below_zero_ratio_250"].mean()
                lines.append(
                    f"- {category}: 平均得分={avg_score:.1f}, "
                    f"平均成交量比率={avg_vol_ratio:.2f}, "
                    f"平均below_zero_ratio_250={avg_below_zero:.3f}"
                )

        lines.extend(["", "## Top 候选解释", ""])
        for idx, (_, row) in enumerate(candidates.head(10).iterrows(), 1):
            risk_flags = row["risk_flags"] or "无"
            spread_breakout = row.get("spread_breakout_level", "无")
            price_breakout = row.get("price_breakout_level", "无")
            vol_ratio = row.get("volume_ratio", 0)
            reason = row.get("reason", "") or "无"

            lines.extend([
                f"### {idx}. {row['code']} {row['name']} | {row['category']} | {row['score']}分",
                "",
                "核心理由：",
                f"- {reason}",
                "",
                "风险标记：",
                f"- {risk_flags}",
                "",
                "动作建议：",
                f"- {row.get('action', '-')}",
                "",
            ])

        lines.extend(["", "## Top 20", ""])
        for _, row in candidates.head(20).iterrows():
            risk_flags = row["risk_flags"] or "无"
            lines.append(
                f"- {row['code']} {row['name']} | {row['category']} | "
                f"得分 {row['score']} | {row['action']} | 风险：{risk_flags}"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_stock_report(
    code: str,
    name: str,
    category: str,
    score: int,
    df: pd.DataFrame,
    image_dir: str,
    recent_bars: int = 500,
    spread_breakout_level: str = "无",
    price_breakout_level: str = "无",
) -> str:
    """
    生成单只股票价格图 + 差价图。
    """
    setup_chinese_font()
    data = df.copy().tail(recent_bars)
    latest = data.iloc[-1]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(data["date"], data["stock_close"], label="收盘价", linewidth=1.4)
    axes[0].plot(data["date"], data["stock_ma20"], label="20日线", linewidth=1.1)
    axes[0].plot(data["date"], data["stock_ma60"], label="60日线", linewidth=1.1)
    axes[0].set_title(f"{code} {name} | {category} | 得分 {score}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    price_info = (
        f"最新:{latest.get('stock_close', 0):.2f}\n"
        f"MA20:{latest.get('stock_ma20', 0):.2f}\n"
        f"MA60:{latest.get('stock_ma60', 0):.2f}\n"
        f"120日高:{latest.get('price_high_120', 0):.2f}\n"
        f"250日高:{latest.get('price_high_250', 0):.2f}"
    )
    axes[0].text(
        0.98,
        0.95,
        price_info,
        transform=axes[0].transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    axes[1].plot(data["date"], data["spread"], label="逐日差价", linewidth=1.4)
    axes[1].plot(data["date"], data["spread_ma60"], label="差价60日均线", linewidth=1.1)
    axes[1].plot(data["date"], data["spread_high_120"], label="差价120日高点", linewidth=1.0)
    axes[1].plot(data["date"], data["spread_high_250"], label="差价250日高点", linewidth=1.0)
    axes[1].axhline(0, linestyle="--", linewidth=1, label="零线", color="black")
    axes[1].set_title("逐日差价图")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    vol_ratio = latest.get("volume_ratio", 0)
    spread_info = (
        f"分类:{category}\n"
        f"得分:{score}\n"
        f"最新spread:{latest.get('spread', 0):.2f}\n"
        f"差价突破:{spread_breakout_level}\n"
        f"价格突破:{price_breakout_level}\n"
        f"成交量比率:{vol_ratio:.2f}"
    )
    axes[1].text(
        0.98,
        0.95,
        spread_info,
        transform=axes[1].transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.5),
    )

    plt.tight_layout()
    filename = f"{code}_{name}_{category}_{score}.png".replace("/", "_")
    path = os.path.join(image_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path
