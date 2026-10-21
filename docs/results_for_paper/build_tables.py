#!/usr/bin/env python3
"""Build the two paper candidate tables from archived, machine-readable scores."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ("PSNR", "SSIM", "LPIPS")
RANK_MARKDOWN = {1: "🟥", 2: "🟧", 3: "🟨"}
RANK_LATEX = {1: "rankbest", 2: "ranksecond", 3: "rankthird"}


def metric_ranks(rows: list[dict], metric: str, key: str | None = None) -> dict[str, int]:
    """Return per-method rank for each metric; LPIPS is minimized."""
    value = (lambda row: row[metric]) if key is None else (lambda row: row[key])
    ordered = sorted(rows, key=value, reverse=metric != "LPIPS")
    return {row["method"]: rank for rank, row in enumerate(ordered, start=1)}


def markdown_metric(rank: int, value: str) -> str:
    marker = RANK_MARKDOWN.get(rank, "")
    return f"{marker} **{value}**" if rank == 1 else f"{marker} {value}" if marker else value


def latex_metric(value: str, rank: int) -> str:
    color = RANK_LATEX.get(rank)
    return f"\\cellcolor{{{color}}}{value}" if color else value


def latex_preamble_note() -> str:
    return (
        "% Preamble requirement: \\usepackage[table]{xcolor}\n"
        "\\definecolor{rankbest}{HTML}{F5C6C6}\n"
        "\\definecolor{ranksecond}{HTML}{FCE0CC}\n"
        "\\definecolor{rankthird}{HTML}{FFF9C4}\n"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def llff() -> None:
    source = ROOT / "output/llff/summary.csv"
    data = {row["method"]: row for row in read_csv(source)}
    items = [
        ("vanilla_3DGS", "3DGS", "3-view MVS", "10k"),
        ("FSGS", "FSGS", "3-view MVS", "10k"),
        ("CoR-GS", "CoR-GS", "3-view MVS", "10k"),
        ("DNGaussian", "DNGaussian", "3-view MVS", "10k"),
        ("DropoutGS", "DropoutGS", "3-view MVS", "10k"),
        ("Difix3D_60k", "Difix3D+", "3-view MVS", "30k+30k"),
        ("our_R4", "Ours", "3-view MVS", "10k"),
    ]
    rows = [dict(method=label, initialization=init, iterations=steps,
                 PSNR=float(data[key]["PSNR"]), SSIM=float(data[key]["SSIM"]),
                 LPIPS=float(data[key]["LPIPS"]), archive_key=key)
            for key, label, init, steps in items]
    write_csv(HERE / "llff_selected.csv", rows,
              ["method", "initialization", "iterations", "PSNR", "SSIM", "LPIPS", "archive_key"])
    ranks = {metric: metric_ranks(rows, metric) for metric in METRICS}
    body = "\n".join(
        f"| {r['method']} | " + " | ".join(
            markdown_metric(ranks[m][r["method"]], f"{r[m]:.4f}")
            for m in METRICS) + " |"
        for r in rows)
    (HERE / "LLFF_TABLE.md").write_text(
        "# LLFF：三视角统一 MVS 初始化\n\n"
        "Ours 对应历史归档键 our_R4；当前实现包含联合 RGB–Depth 先验学习与下游引导优化。\n\n"
        "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |\n"
        "|---|---:|---:|---:|\n" + body + "\n\n"
        "逐指标着色标识前三名：🟥 第一，🟧 第二，🟨 第三。正式论文 LaTeX 表使用对应底色。\n\n"
        "## 评测协议与表格读法\n\n"
        "八个 LLFF 场景，每景三张训练 RGB，共 43 张留出测试图。PSNR、SSIM 越高越好，LPIPS 越低越好；"
        "先平均同一场景的测试视角，再对八景等权平均，因此表中每行是场景宏平均。"
        "全部数值来自本地训练及同一测试协议的重新测评，并非原论文表格摘录。"
        "各方法使用同一三视角 MVS 点云初始化；除 Difix3D+ 的 30k 基础训练加 30k 修复续训外，其他方法均为 10k。"
        "统一起点和轮数仍不等于训练信息完全一致：各方法的损失、先验和可用相机位姿不同。"
        "Ours 与 Difix3D+ 使用留出相机**位姿**，没有把这些视角的 GT RGB 用作训练监督。\n\n"
        "## 结果分析\n\n"
        "与同起点、同 10k 的普通 3DGS 相比，Ours 的 PSNR 从 14.4799 提升至 19.7992 dB，"
        "SSIM 从 0.4818 提升至 0.6587，LPIPS 从 0.4155 降至 0.2856。"
        "这支持联合预测器参与稀疏视角优化有帮助；由于训练损失和先验也改变了，表格本身不能把全部增益归因于单个模块。\n\n"
        "与更强的稀疏视角方法相比，Ours 的 PSNR 与 DropoutGS 的 19.8039 dB 仅差 0.0047 dB；"
        "但 DropoutGS 的 SSIM 为 0.6797，高于 Ours 的 0.6587。"
        "FSGS 的 LPIPS 为 0.2700，优于 Ours 的 0.2856。"
        "因此主表适合说明 Ours 的 PSNR 有竞争力，不能声称在三项指标上全面领先；"
        "0.0047 dB 的均值差异也不宜解读为实质胜负。这里只有一次本地运行，未报告多随机种子方差。\n\n"
        "## 图像与深度的解释\n\n"
        "[RGB 与深度主图](figures/llff_rgb_depth_main.png)展示 trex 和 room 两个测试视角；"
        "[扩展图](figures/llff_rgb_depth_expanded.png)另含 CoR-GS 和 Difix3D+。"
        "DNGaussian 保留在指标表中，依选图方案不放入图。图中 Reference 列上行为 GT RGB，"
        "下行为从该 RGB 运行 Depth Anything V2 得到的**相对深度代理**，并非测量深度 GT。"
        "各方法的 Gaussian 相机深度在同一视角共用色标；LLFF 使用跨方法全图有效深度的 2%–98% 分位数。"
        "参考深度单独用 1%–99% 分位数归一化；暖色表示近处、冷色表示远处，"
        "参考列与 Gaussian 列的颜色数值不可直接比较。"
        "两景的局部现象、局限和逐视角色标见 [定性分析](QUALITATIVE_ANALYSIS.md)及"
        "[显示范围](depth_display_ranges.json)。\n\n"
        "数值源：[output/llff/summary.csv](../../output/llff/summary.csv)；"
        "逐场景渲染见 [output/llff/](../../output/llff/)；"
        "完整口径见 [LLFF 实验审计](../LLFF_3view_MVS_10k.md)。\n",
        encoding="utf-8")
    tex_rows = []
    for r in rows:
        vals = [latex_metric(f"{r[m]:.4f}", ranks[m][r["method"]]) for m in METRICS]
        tex_rows.append(" & ".join([r["method"], *vals]) + " \\\\")
    (HERE / "llff_table.tex").write_text(
        latex_preamble_note() + "\\begin{table}[t]\n\\centering\n"
        "\\caption{LLFF three-view comparison. Per metric, the best, second-best, and third-best results are shaded red, orange, and yellow. "
        "All methods use the same three-view MVS initialization and evaluation; all use 10k steps except Difix3D+ (30k base + 30k refinement). "
        "Training details and camera-pose information are described in the experiment protocol.}\n\\label{tab:llff_main}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\nMethod & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\\n"
        "\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8")


def dtu() -> None:
    data = {row["method"]: row for row in read_csv(ROOT / "output/dtu/summary.csv")}
    items = [
        ("vanilla_3DGS", "3DGS", "3-view MVS / 10k", "local MVS baseline"),
        ("CoR-GS", "CoR-GS", "3-view MVS / 10k", "author-style local run"),
        ("DNGaussian", "DNGaussian", "random ~1k / 6k", "author-style local run"),
        ("DropoutGS", "DropoutGS", "random ~1k / 6k", "author-style local run"),
        ("our_R4", "Ours", "3-view MVS / 10k", "local run; target poses known"),
    ]
    rows = []
    for key, label, setup, provenance in items:
        row = dict(method=label, setup=setup, provenance=provenance,
                   archive_key=key)
        for metric in ("PSNR", "SSIM", "LPIPS"):
            mean = float(data[key][f"{metric}_mean"])
            std = float(data[key][f"{metric}_scene_std"])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_scene_std"] = std
        rows.append(row)
    fields = ["method", "setup", "provenance", "archive_key"] + [
        f"{m}_{suffix}" for m in ("PSNR", "SSIM", "LPIPS")
        for suffix in ("mean", "scene_std")]
    write_csv(HERE / "dtu_selected.csv", rows, fields)
    ranks = {m: metric_ranks(rows, m, f"{m}_mean") for m in METRICS}
    def fmt(r: dict, metric: str, latex: bool = False) -> str:
        val = f"{r[f'{metric}_mean']:.3f} ± {r[f'{metric}_scene_std']:.3f}"
        rank = ranks[metric][r["method"]]
        return (latex_metric(val.replace("±", r"$\pm$"), rank) if latex
                else markdown_metric(rank, val))
    body = "\n".join(
        f"| {r['method']} | " + " | ".join(fmt(r, m) for m in METRICS) + " |"
        for r in rows)
    (HERE / "DTU_TABLE.md").write_text(
        "# DTU：各方法本地复跑的共同评测\n\n"
        "Ours 对应历史归档键 our_R4；当前实现包含联合 RGB–Depth 先验学习与下游引导优化。\n\n"
        "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |\n"
        "|---|---:|---:|---:|\n" + body + "\n\n"
        "逐指标着色标识前三名：🟥 第一，🟧 第二，🟨 第三。正式论文 LaTeX 表使用对应底色。\n\n"
        "## 评测协议与训练路线\n\n"
        "15 个 DTU scan，每景三张训练 RGB；所有方法都在同一组 25 个留出视角按 CoR-GS 前景 mask 口径评测。"
        "PSNR、SSIM 越高越好，LPIPS 越低越好。表中的 `±` 为**15 个 scan 之间的样本标准差**，"
        "不是多随机种子误差或均值置信区间。所有行均为本地复跑，**不是原论文直接摘录的数值**。\n\n"
        "本表统一的是测试口径，不是训练口径。3DGS、CoR-GS、Ours 使用三视角 MVS 初始化和 10k；"
        "CoR-GS、DNGaussian、DropoutGS 尽量沿各自作者的 DTU 训练路线，其中 DNGaussian、"
        "DropoutGS 使用随机点初始化和 6k，随机点范围由本地 49 相机 COLMAP 几何给定。"
        "Ours 使用 8 个留出相机位姿，其中 5 个与测试集相机重合，但没有使用这些视角的 GT RGB 训练。"
        "FSGS 缺少作者 DTU 原生训练入口，本地主表不列它；移植实验仍保存在"
        "[诊断归档](../../output/dtu/_diagnostics/README.md)中。\n\n"
        "## 结果分析\n\n"
        "Ours 的 PSNR 为 20.325 ± 2.259 dB，相对同 MVS/10k 起点的普通 3DGS（17.753 ± 2.726 dB）"
        "有约 2.57 dB 的场景宏平均提升；但 SSIM 从 0.827 降至 0.815，LPIPS 从 0.146 升至 0.162。"
        "这说明当前改进主要体现在 PSNR，不能称作所有图像质量指标均有提升。\n\n"
        "Ours 与 DropoutGS 的 PSNR 均值分别为 20.325 和 20.322 dB，相差约 0.003 dB；"
        "两者训练路线不同，这个差距不足以说明稳定优势。CoR-GS 的 SSIM 0.856 和 LPIPS 0.121"
        "均优于 Ours 的 0.815 和 0.162。跨 scan 标准差较大，又只有一次本地运行，"
        "不能把表中末位数字当成跨随机种子的显著性证据。\n\n"
        "## 图像与深度的解释\n\n"
        "[RGB 与深度主图](figures/dtu_rgb_depth_main.png)展示 scan30 与 scan21 的典型视角；"
        "这些局部选图用于观察物体边缘、窗框和表面结构，不替代 15 scan 的整体结果。"
        "图中 Reference 列上行为 GT RGB，下行为从该 RGB 运行 Depth Anything V2 得到的"
        "**相对深度代理**，并非测量深度 GT。Gaussian 相机深度在同一视角跨方法共用色标，"
        "色标由共同放大 ROI 中有效深度的 2%–98% 分位数确定，然后应用到完整深度图；"
        "参考列单独用 1%–99% 分位数归一化。暖色表示近处、冷色表示远处，"
        "参考列与 Gaussian 列的颜色数值不可直接比较，深度图也不能作为几何精度的定量评测。"
        "逐视角说明及色标数值见 [定性分析](QUALITATIVE_ANALYSIS.md)与"
        "[显示范围](depth_display_ranges.json)。\n\n"
        "数值源：[DTU 主表 CSV](../../output/dtu/summary.csv)；"
        "完整配置、信息条件与 FSGS 诊断见 [DTU 实验审计](../DTU_3view_author_configs.md)。\n",
        encoding="utf-8")
    tex_rows = [" & ".join([r["method"], *[fmt(r, m, True) for m in METRICS]]) + " \\\\"
                for r in rows]
    (HERE / "dtu_table.tex").write_text(
        latex_preamble_note() + "\\begin{table}[t]\n\\centering\n"
        "\\caption{DTU three-view comparison under the common 25-view foreground-mask evaluation. Values are mean $\\pm$ standard deviation across 15 scans. "
        "Per metric, the best, second-best, and third-best results are shaded red, orange, and yellow. Training configurations are described in the experiment protocol.}\n\\label{tab:dtu_main}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\nMethod & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\\n"
        "\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8")


if __name__ == "__main__":
    llff()
    dtu()
    print("Wrote LLFF_TABLE.md, DTU_TABLE.md, selected CSVs and LaTeX tables")
