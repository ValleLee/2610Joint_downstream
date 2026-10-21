# LLFF 三视角：统一 MVS 起点对比（Difix3D+ 为 60k）

本页选取最早的 `1007_fsgs_mvs_baselines` 同起点批次及同协议的原版 3DGS、R4：**原版 3DGS、FSGS、CoR-GS、DNGaussian、DropoutGS 和 R4 为 10k；Difix3D+ 改用已经完成的 30k 基础训练 + 30k Difix 续训结果。**所有数字均为本地模型在共同测试集上的重评结果，不是从各论文摘录的成绩。Difix3D+ 的训练预算不同，表格用于展示完整方法结果，不用于等预算排名。

## 数据与评测协议

- LLFF 八景：fern、flower、fortress、horns、leaves、orchids、room、trex；每景三张训练 RGB，`images_4`，43 张 held-out 测试图。
- 所有行均从对应场景**同一份、仅由三张训练图生成的 FSGS MVS PLY**初始化，seed 2024。除 Difix3D+ 训练 60,000 步外，其余方法均为 10,000 步。相同点云和种子不代表各方法的损失、优化日程或额外先验相同。
- 每张测试图计算 RGB PSNR、Graphdeco SSIM、VGG LPIPS；先取场景内测试图均值，再对八景等权平均。PSNR/SSIM 越高越好，LPIPS 越低越好。原版 3DGS 的 SSIM 已用这套 Graphdeco 实现重算；43 张 GT 与原批次逐像素一致。
- 原始 10k 批次的 MVS 输入哈希、40 项训练任务及 43 张测试图的核验见[原始输入与初始化审计](../data/llff/protocol/actual_initialization_audit.json)；Difix3D+ 60k 的八景训练、输入哈希、视角映射及共同评测见[60k 运行审计](../code/compare/Difix3D_60k/llff_mvs_60k/config/run_audit.json)；3DGS 交叉重评数值见[同口径 JSON](../output/llff/source_metrics/crosscheck/vanilla_ours_common_4.json)。

机器可读主表：[summary.csv](../output/llff/summary.csv)。各方法最终渲染与指标位于 [LLFF 结果目录](../output/llff/) 的同名方法目录。

## 论文候选对比表

| 方法 | 训练实现与本地配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| 原版 3DGS | Graphdeco 原始优化和 SH 渲染；共同 MVS / 10k | 14.4799 | 0.4818 | 0.4155 |
| FSGS | FSGS 伪视角、增密；共同 MVS / 10k | 19.7404 | 0.6640 | **0.2700** |
| CoR-GS | 双高斯、co-pruning 与 co-regularization；共同 MVS / 10k | 19.5491 | 0.6730 | 0.2732 |
| DNGaussian | 本地 LLFF/MVS 参数，含深度及形状约束；共同 MVS / 10k | 19.3770 | 0.6595 | 0.2831 |
| DropoutGS | 本地 LLFF/MVS 参数，dropout 0.4、一致性权重 0.4；共同 MVS / 10k | **19.8039** | **0.6797** | 0.2708 |
| Difix3D+† | 作者两阶段日程的本地 MVS 适配：30k 基础训练 + 30k Difix 续训；共同 MVS / **60k** | 18.4940 | 0.6017 | 0.2842 |
| 我们的方法 R4‡ | 共同 MVS / 10k；深度和伪视角训练 | 19.7992 | 0.6587 | 0.2856 |

† **Difix3D+ 采用作者的 30k + 30k 两阶段训练日程，但这行仍是 LLFF 三视角、本地统一 MVS 起点的适配复跑，并非原论文数值。**前 30k 只用真实训练视角建立基础模型，后 30k 从该 checkpoint 续训并每 2k 步刷新目标测试相机位姿上的 Difix 伪视角；无深度损失。报告的是 60k 最终模型的原始 3D 渲染，不是 render-time 修复图。其 30k 基础模型中间指标为 17.2653 / 0.5480 / 0.3466；它尚未开启 Difix，不作为主表独立方法。Difix3D+ 与其他行的训练预算相差六倍，论文表注必须标出步数。

‡ R4 也使用 held-out/测试相机**位姿**生成伪视角目标，不使用对应测试 RGB。因此虽然 MVS 点云、训练 RGB、步数一致，R4 的可用位姿信息比只从训练视角采样伪视角的方法更多；表格须明确该输入差异，不能将差值写成单一算法模块带来的提升。

## 结果分析与论文写法

- R4 相对本地原版 3DGS 为 **+5.3193 dB PSNR、+0.1769 SSIM、−0.1300 LPIPS**。这是两个完整训练方案的差值，包含伪视角、深度与可用位姿等共同变化。
- 最高 PSNR 是 DropoutGS 的 19.8039，R4 为 19.7992，差 **0.0047 dB**；只有单次 seed，不能据此宣称两者有可重复的 PSNR 优劣。DropoutGS 的 SSIM 比 R4 高 0.0210，LPIPS 低 0.0148。
- FSGS 的 LPIPS 最低；R4 相对 FSGS 仅高 0.0588 dB PSNR，但 SSIM 低 0.0053、LPIPS 高 0.0156。论文正文宜写成“R4 在本地统一 MVS/10k 设定下达到约 19.80 dB，与最好的 PSNR 基线接近”，同时报告另外两项指标。
- 此表可以称为**统一三视角 MVS 初始化与统一评测的本地比较**。除 Difix3D+ 采用 60k 两阶段日程外，其余行均为 10k；R4 和 Difix3D+ 使用测试相机位姿。不能称作“所有方法同预算、完全同输入的算法排名”。Difix3D+ 60k 相对 R4 10k 的 PSNR 低 1.3052 dB，但步数和训练阶段不同，差值不表示等预算优劣。

## 数值与配置来源

| 内容 | 可追溯文件 |
|---|---|
| FSGS、CoR-GS、DNGaussian、DropoutGS、R4 的 10k 宏平均 | [原始统一 CSV](../output/llff/source_metrics/COMPARISON_WITH_1007FSGSMVS.csv)；[运行与初始化审计](../data/llff/protocol/actual_initialization_audit.json) |
| Difix3D+ 的 30k 中间点及 60k 最终点 | [60k 同口径指标 CSV](../output/llff/source_metrics/DIFIX3D_60K_METRICS.csv)；[60k 运行审计](../code/compare/Difix3D_60k/llff_mvs_60k/config/run_audit.json) |
| 原版 3DGS 重评、八景逐场景数据 | [3DGS/R4 同口径 JSON](../output/llff/source_metrics/crosscheck/vanilla_ours_common_4.json) |

