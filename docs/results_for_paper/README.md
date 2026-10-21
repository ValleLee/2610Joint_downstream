# 论文候选结果素材

暂定研究题目为**面向稀疏视角三维重建的先验学习与引导优化研究**；当前联合预测器训练与下游优化两部分的设计、公式、实验和上游图像见[完整研究参考](../JARO_GS_RESEARCH_REFERENCE_20261022_045346.md)。表中 Ours 对应历史 our_R4 结果键；旧稿曾用 JARO-GS 暂名。

综合实验与写作参考：[EXPERIMENT_REFERENCE_20261022_040857.md](EXPERIMENT_REFERENCE_20261022_040857.md)（北京时间 2026-10-22；含数据集、方法、协议、两张主表、逐景统计、结果图、普遍性与局限分析）。

本目录放目前选定的两张主表和 RGB+depth 定性对比图。数值从归档 CSV 生成；RGB 来自同一测试相机的最终渲染，GT 是原工作站测试图的逐字节副本。各方法深度从最终 Gaussian PLY 在同一相机下渲染，并将 rasterizer 的透明度加权深度除以 alpha；拼图中 Reference 列上方是 RGB GT，下方是 **DA-V2 从该 RGB GT 预测的相对深度代理**。它们都不是测量深度 GT。`asset_manifest.json`、`depth_manifest.json` 和 `da_reference_manifest.json` 记录来源与 SHA-256。

## 定量表

| 数据集 | 阅读版 | LaTeX | 机器可读 |
|---|---|---|---|
| LLFF：统一三视角 MVS 初始化，Difix3D+ 为 60k | [LLFF_TABLE.md](LLFF_TABLE.md) | [llff_table.tex](llff_table.tex) | [llff_selected.csv](llff_selected.csv) |
| DTU：各方法本地复跑，统一 25-view mask 测评 | [DTU_TABLE.md](DTU_TABLE.md) | [dtu_table.tex](dtu_table.tex) | [dtu_selected.csv](dtu_selected.csv) |

论文表格只列方法与指标；每个指标的前三名分别以红、橙、黄底色标示。训练初始化、轮数与信息条件写在指标文档正文和[综合实验参考](EXPERIMENT_REFERENCE_20261022_040857.md)中。LaTeX 表片段需要在论文导言区加载 \usepackage[table]{xcolor}。

LLFF 的 FSGS 是共同 MVS/10k 对照，保留在主表。DTU 主表不列 FSGS：其本地三图 MVS/10k 移植结果见[诊断归档](../../output/dtu/_diagnostics/README.md)，但没有 FSGS 作者 DTU 原生训练配置。两份指标文档分别说明评测口径、训练条件、数值比较和图像解释。两张表都不能单凭一列最高值宣称跨协议算法优势。

## 图像与分析

| 图 | 内容 |
|---|---|
| [LLFF 五列 RGB+depth 主图](figures/llff_rgb_depth_main.png) / [PDF](figures/llff_rgb_depth_main.pdf) | trex、room；3DGS、FSGS、DropoutGS、Ours、Reference |
| [LLFF 七列扩展 RGB+depth 图](figures/llff_rgb_depth_expanded.png) / [PDF](figures/llff_rgb_depth_expanded.pdf) | 加入 CoR-GS、Difix3D+ 60k；按选图要求不含 DNGaussian，指标表仍保留 |
| [DTU 六列 RGB+depth 主图](figures/dtu_rgb_depth_main.png) / [PDF](figures/dtu_rgb_depth_main.pdf) | scan30 与 scan21；3DGS、CoR-GS、DNGaussian、DropoutGS、Ours、Reference |

逐视角解释、图注和局限见 [QUALITATIVE_ANALYSIS.md](QUALITATIVE_ANALYSIS.md)；LaTeX 插图模板见 [figure_includes.tex](figure_includes.tex)。每组 RGB/depth 行使用同一相机和同一 ROI；红框标出放大区域。高斯相机深度在每个视角跨方法使用同一范围：LLFF 按全图、DTU 按共享 ROI 的有效深度分位数确定。DA-V2 相对深度单独归一化；近处统一显示为暖色。范围与极性见 [depth_display_ranges.json](depth_display_ranges.json)。

## 重建素材

```bash
python docs/results_for_paper/build_tables.py
python docs/results_for_paper/build_figures.py
```

这些命令从仓库原始汇总 CSV、本目录已复制的 PNG 和 float32 深度 TIFF 重建表格与拼图，克隆后无需原工作站 GT 路径。若在原工作站要重新复制选定视角，再运行 `python docs/results_for_paper/build_figures.py --collect`；它要求原始 LLFF/DTU GT 路径仍存在。PNG、PDF、表格生成脚本与资产哈希均保存在本目录。

若要从本地归档 PLY 重新导出原始 float32 深度 TIFF，使用 `render_depth.py`。它读取本目录的 [selected_cameras.json](selected_cameras.json)，需本地最终 PLY、CUDA、`simple_knn` 和带 depth 输出的 confidence rasterizer。可在已配置 FSGS 扩展的 Python 环境下运行：

```bash
PYTHONPATH="$PWD/code/ours/reconstruction/fsgs_base:$PWD/code/compare/FSGS/llff_mvs10k/source/submodules/simple-knn:$PWD/code/compare/FSGS/llff_mvs10k/source/submodules/diff-gaussian-rasterization-confidence" \
  python docs/results_for_paper/render_depth.py
```

这些 PLY 不包含在拟上传 GitHub 的素材中，但 depth TIFF 已复制到 `assets/`，因此克隆后可直接重建拼图。完整训练复现说明见 [REPRODUCE.md](../REPRODUCE.md)。

若需对新选的 RGB GT 重算 DA-V2 代理深度，在安装 Depth Anything V2 依赖的环境中运行 `python docs/results_for_paper/make_da_reference.py --da-root <Depth-Anything-V2> --checkpoint <ViT-L checkpoint>`。当前四张输出与权重 SHA-256 已记录在 [da_reference_manifest.json](da_reference_manifest.json)；权重本身不上传 GitHub。
