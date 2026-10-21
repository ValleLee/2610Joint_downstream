# 面向稀疏视角三维重建的先验学习与引导优化研究

当前实现包含联合 RGB–Depth 预测器训练，以及冻结预测器引导的下游重建优化；研究题目采用较宽的“先验学习与引导优化”，便于后续调整预测器输出。背景、方法公式、采样与损失、上下游指标和图像见[完整研究参考（2026-10-22，北京时间）](docs/JARO_GS_RESEARCH_REFERENCE_20261022_045346.md)。历史运行键 our_R4 对应本文的下游结果；旧稿中的 JARO-GS 是曾用暂名。

这里保存论文使用的两组本地实验：[LLFF 统一 MVS 起点与统一评测](output/llff/README.md) 和 [DTU 各方法配置、统一 25-view mask 测评](output/dtu/README.md)。每个数据集都有主表 `summary.csv`、方法/场景目录、指标、RGB 渲染图及结果来源清单。

```text
code/
  compare/<方法>/llff_mvs10k/          # LLFF 原方法代码快照、运行配置
  compare/Difix3D_60k/llff_mvs_60k/  # LLFF Difix3D+ 两阶段 60k
  compare/<方法>/dtu_author/           # DTU 作者路线代码与配置
  compare/<方法>/dtu_diagnostic/       # 单独标注的 DTU 本地适配
  ours/predictor/                     # 冻结 RGB-D predictor 推理与配置
  ours/reconstruction/                # R4 重建依赖
  ours/{llff_mvs10k,dtu}/             # 我们的方法入口
  evaluation/{llff,dtu}/              # 指标实现
data/{llff,dtu}/                      # split、输入哈希、运行协议
output/
  llff/{summary.csv,README.md,<方法>/<场景>/}
  dtu/{summary.csv,README.md,<方法>/<scan>/,_diagnostics/}
```

每个方法/场景目录中，`renders/` 是纳入 Git 的最终 RGB 渲染，`metrics/` 含原始指标与统一的 `summary.json` 索引，`run_manifest.json` 与 `artifact_manifest.json` 记录运行和来源。LLFF 缺少原生运行清单的方法，其 `run_manifest.json` 已明确标为从归档配置生成。`point_cloud/`、`checkpoints/` 是工作站本地留存，上传 GitHub 时由 `.gitignore` 排除。`SOURCE_MANIFEST.json` 记录结果文件的原始路径、大小和 SHA-256；换机器时原始绝对路径仅供追溯。

为节省本机空间，大文件在原工作站以硬链接归档。它们是普通文件；Git 上传的是渲染 PNG 的实际内容，克隆后不依赖原工作站。打包后若原位修改原始运行文件，归档内容也会变化，`verify_release.py` 会通过哈希报错。

## 两张主表的范围

- **LLFF**：八景、每景三张训练 RGB、共 43 张留出图；同景统一 FSGS 三训练图 MVS 点云、seed 2024、同一套 PSNR / Graphdeco SSIM / VGG LPIPS 评测。原版 3DGS、FSGS、CoR-GS、DNGaussian、DropoutGS 和 R4 训练 10k 步；Difix3D+ 使用本地 **30k 基础训练 + 30k 修复训练**。各方法的损失及内部参数并不相同。R4 与 Difix3D+ 使用留出相机位姿，未使用对应测试 RGB。
- **DTU**：15 个 scan、每景三张训练 RGB、共同 25 张测试图及 mask。原版 3DGS、CoR-GS、R4 为 MVS / 10k；DNGaussian、DropoutGS 为其 DTU 作者路线的随机点 / 6k。两种设置不可互换，其他本地 MVS / 10k 适配集中放在 `_diagnostics/`。

论文数字在两份数据集 README 中注明训练信息差异；不能把主表解释为完全相同的先验或优化日程。DTU `±` 是 15 景间样本标准差。

原 `compare/` 的两份详细指标文档已归入 `docs/`：[LLFF 三视角对比](docs/LLFF_3view_MVS_10k.md)、[DTU 各方法配置对比](docs/DTU_3view_author_configs.md)。

我们方法的训练机制见[联合 RGB–Depth 预测器指导 3DGS 的方法说明](docs/JOINT_PREDICTOR_3DGS_METHOD.md)：真/伪视角采样、渐进位姿、伪标签生成与配对损失均按实际代码记录。

论文候选的两张数据表、LaTeX、真实渲染拼图和典型视角分析见 [docs/results_for_paper/](docs/results_for_paper/README.md)。

## 核验与复现

```bash
python scripts/verify_release.py
```

该命令复核已归档文件、渲染数量和指标汇总，无需 GPU。训练数据、冻结 predictor 权重、环境及运行入口见[复现说明](docs/REPRODUCE.md)和[代码与数据索引](docs/CODE_AND_DATA.md)。原始训练数据与权重不在 GitHub 包内；发布完整重训环境前仍需提供其合法下载途径。本包已包含最终 RGB 渲染 PNG，可以在取得对应 GT/mask 后重新计算像素指标。
