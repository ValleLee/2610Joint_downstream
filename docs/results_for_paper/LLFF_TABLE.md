# LLFF：三视角统一 MVS 初始化

Ours 对应历史归档键 our_R4；当前实现包含联合 RGB–Depth 先验学习与下游引导优化。

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 3DGS | 14.4799 | 0.4818 | 0.4155 |
| FSGS | 🟨 19.7404 | 🟨 0.6640 | 🟥 **0.2700** |
| CoR-GS | 19.5491 | 🟧 0.6730 | 🟨 0.2732 |
| DNGaussian | 19.3770 | 0.6595 | 0.2831 |
| DropoutGS | 🟥 **19.8039** | 🟥 **0.6797** | 🟧 0.2708 |
| Difix3D+ | 18.4940 | 0.6017 | 0.2842 |
| Ours | 🟧 19.7992 | 0.6587 | 0.2856 |

逐指标着色标识前三名：🟥 第一，🟧 第二，🟨 第三。正式论文 LaTeX 表使用对应底色。

## 评测协议与表格读法

八个 LLFF 场景，每景三张训练 RGB，共 43 张留出测试图。PSNR、SSIM 越高越好，LPIPS 越低越好；先平均同一场景的测试视角，再对八景等权平均，因此表中每行是场景宏平均。全部数值来自本地训练及同一测试协议的重新测评，并非原论文表格摘录。各方法使用同一三视角 MVS 点云初始化；除 Difix3D+ 的 30k 基础训练加 30k 修复续训外，其他方法均为 10k。统一起点和轮数仍不等于训练信息完全一致：各方法的损失、先验和可用相机位姿不同。Ours 与 Difix3D+ 使用留出相机**位姿**，没有把这些视角的 GT RGB 用作训练监督。

## 结果分析

与同起点、同 10k 的普通 3DGS 相比，Ours 的 PSNR 从 14.4799 提升至 19.7992 dB，SSIM 从 0.4818 提升至 0.6587，LPIPS 从 0.4155 降至 0.2856。这支持联合预测器参与稀疏视角优化有帮助；由于训练损失和先验也改变了，表格本身不能把全部增益归因于单个模块。

与更强的稀疏视角方法相比，Ours 的 PSNR 与 DropoutGS 的 19.8039 dB 仅差 0.0047 dB；但 DropoutGS 的 SSIM 为 0.6797，高于 Ours 的 0.6587。FSGS 的 LPIPS 为 0.2700，优于 Ours 的 0.2856。因此主表适合说明 Ours 的 PSNR 有竞争力，不能声称在三项指标上全面领先；0.0047 dB 的均值差异也不宜解读为实质胜负。这里只有一次本地运行，未报告多随机种子方差。

## 图像与深度的解释

[RGB 与深度主图](figures/llff_rgb_depth_main.png)展示 trex 和 room 两个测试视角；[扩展图](figures/llff_rgb_depth_expanded.png)另含 CoR-GS 和 Difix3D+。DNGaussian 保留在指标表中，依选图方案不放入图。图中 Reference 列上行为 GT RGB，下行为从该 RGB 运行 Depth Anything V2 得到的**相对深度代理**，并非测量深度 GT。各方法的 Gaussian 相机深度在同一视角共用色标；LLFF 使用跨方法全图有效深度的 2%–98% 分位数。参考深度单独用 1%–99% 分位数归一化；暖色表示近处、冷色表示远处，参考列与 Gaussian 列的颜色数值不可直接比较。两景的局部现象、局限和逐视角色标见 [定性分析](QUALITATIVE_ANALYSIS.md)及[显示范围](depth_display_ranges.json)。

数值源：[output/llff/summary.csv](../../output/llff/summary.csv)；逐场景渲染见 [output/llff/](../../output/llff/)；完整口径见 [LLFF 实验审计](../LLFF_3view_MVS_10k.md)。
