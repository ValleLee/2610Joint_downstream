# DTU 三视角：各方法自身配置的本地复跑对比

本页按当前论文选择，先列各研究在 DTU 上**实际完成且最接近其自身训练方式**的本地复跑，再单列已发表论文的 DTU 三视角数值供引用。第一张表统一测试视角、GT、mask 和计算公式；第二张表明确标出每行的来源。不同文献的初始化、训练预算和评测实现不能默认相同，文献值不能与本地值直接组成同协议排名。

## 共同数据与指标

- DTU 15 个 scan：8、21、30、31、34、38、40、41、45、55、63、82、103、110、114。每景训练 RGB 为 `rect_023/026/029` 三张，约 1/4 分辨率；每景评测同一组 25 张留出图，共 375 张。
- 沿用 CoR-GS 25-view mask 口径：PSNR 只在前景 mask 内计算，SSIM 和 VGG LPIPS 在 mask 外置白后计算。先平均各场景 25 张，再对 15 景等权平均。表中的 `±` 是 **15 景之间的样本标准差**，不是多随机种子标准差。
- 各方法用各自原生渲染器输出，再进入同一评测。DNGaussian / DropoutGS 的神经颜色渲染器不能由原版 3DGS 的 SH 渲染器替代。

机器可读主表：[summary.csv](summary.csv)。每个方法的 15 个 scan 位于同名方法目录；本地 MVS 适配结果在 [_diagnostics/](_diagnostics/)。

## 论文候选对比表

| 方法 | 本地训练起点与预算 | seed | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| 原版 3DGS¹ | 三训练图 MVS；Graphdeco 3DGS，10k | 2024 | 17.753 ± 2.726 | 0.827 ± 0.085 | 0.146 ± 0.063 |
| CoR-GS | 三训练图 MVS；作者 DTU 训练参数，10k | 2024 | 19.319 ± 2.945 | **0.856 ± 0.079** | **0.121 ± 0.057** |
| DNGaussian² | 作者 DTU `rand_pcd`，约 1000 点；6k | 0 | 18.660 ± 2.517 | 0.771 ± 0.139 | 0.182 ± 0.099 |
| DropoutGS² | 作者 DTU `rand_pcd`，约 1000 点；6k | 0 | 20.322 ± 2.338 | 0.833 ± 0.101 | 0.159 ± 0.082 |
| 我们的方法 R4³ | 三训练图 MVS；深度与伪视角训练，10k | 2024 | **20.325 ± 2.259** | 0.815 ± 0.087 | 0.162 ± 0.076 |

¹ 原版 3DGS 行采用 CoR-GS 稀疏 DTU 对照惯例的三视角 MVS / 10k；它不是“原版 3DGS 使用所有视图 SfM 初始化”的默认设定。CoR-GS 基本沿用作者 DTU 命令，调整了共享 seed 及数据路径。

² DNGaussian / DropoutGS 采用作者 DTU 路线的随机点初始化、6k 训练和各自原生损失。为兼容当前 PyTorch 图像 loader，DPT 深度由 16-bit 量化成 8-bit，保留逐图归一化值域；这是本地实现差异。**随机点采样盒由本地 49 相机 COLMAP `points3D.bin` 确定**，所以只有三张 RGB 进入训练，但几何范围并非严格由三训练视角独立得到。两方法各只有 seed 0 一次本地运行。

³ R4 还使用预训练 RGB-D 教师和 8 个留出相机位姿作为伪目标；每景其中 **5 个位姿与正式测试相机重合**，没有使用对应的测试 RGB。这是测试位姿可见的 transductive / pose-aware 设置。把 R4 与只用训练相机信息的方法并列时，正文和表注都必须说明这一信息条件；目前的高 PSNR 不能单独归因于重建模块。

## 已发表 DTU 三视角数字与本地对照

以下按 **PSNR / SSIM / LPIPS** 排列；“论文报告”指该行确实出现在所链接论文的 DTU 三视角表中，**不意味着该方法的原作者亲自运行或报告了它**。我们的 R4 尚无已发表论文数字，列入本表的是上方同一份本地结果。

| 方法及设定 | 数值性质与出处 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| 原版 3DGS | [CoR-GS 论文 Table 3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00139.pdf)的三视角 MVS 基线 | 17.65 | 0.816 | 0.146 |
| 原版 3DGS | [DNGaussian 论文 Table 1](https://arxiv.org/html/2403.06912)的普通 3DGS 基线；[DropoutGS Table 1](https://arxiv.org/html/2504.09491)再次引用 | 10.99 | 0.585 | 0.313 |
| 3DGS† | [DNGaussian 论文 Table 1](https://arxiv.org/html/2403.06912)：采用 DNGaussian 相同超参数与神经颜色渲染器的改造基线 | 14.74 | 0.672 | 0.249 |
| 原版 3DGS | [SparseGS 论文 Table 2](https://arxiv.org/html/2312.00206)报告的基线 | 14.18 | 0.628 | 0.301 |
| 原版 3DGS + 三图 MVS | **本地复跑**，与上方主表同一结果 | 17.753 | 0.827 | 0.146 |
| FSGS | [SparseGS 论文 Table 2](https://arxiv.org/html/2312.00206)报告的 FSGS 基线；**不是 FSGS 原论文的 DTU 成绩** | 18.36 | 0.707 | 0.232 |
| FSGS | [Binocular-Guided GS 论文 DTU 三视角表](https://arxiv.org/html/2410.18822)报告的另一组 FSGS 基线 | 17.34 | 0.818 | 0.169 |
| FSGS，三图 MVS / 10k | **本地 DTU 适配**，与下方备查表同一结果 | 17.460 | 0.830 | 0.155 |
| CoR-GS | [CoR-GS 原论文 Table 3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00139.pdf) | 19.21 | 0.853 | 0.119 |
| DNGaussian | [DNGaussian 原论文 Table 1](https://arxiv.org/html/2403.06912) | 18.91 | 0.790 | 0.176 |
| DropoutGS | [DropoutGS 原论文 Table 1](https://arxiv.org/html/2504.09491)，论文称其结果为三次实验平均 | 20.22 | 0.830 | 0.150 |
| 我们的方法 R4 | **本地复跑，待投稿**；与上方主表同一结果 | 20.325 | 0.815 | 0.162 |

[原版 3DGS 论文](https://arxiv.org/html/2308.04079)本身没有这一稀疏三视角 DTU 表，因此表中的 3DGS 文献数值来自后续研究在各自协议下实现的基线。[FSGS 原论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05583.pdf)也没有 DTU 定量表；上方两组 FSGS 数值分别来自 SparseGS 和 Binocular-Guided GS 的论文表，不能写成“FSGS 原论文报告”。[Difix3D+ 原论文](https://arxiv.org/html/2503.01774)没有 DTU 三视角数字。

同为“3DGS”的文献值从 10.99 到 17.65 dB，说明各论文的基线定义与实验条件需要逐项核对；上表不能据此推断同一配置存在这么大的随机波动。R4 与已发表值可以作为研究背景并列展示，但涉及 R4 的测试位姿和外部教师先验，以及各文献的初始化与指标实现差异，不能直接用这张表宣称严格同协议超越。

## 结果分析与可支持的表述

- R4 与 DropoutGS 的宏平均 PSNR 分别为 **20.3246** 与 **20.3219 dB**，只差 **0.0027 dB**。两者初始化、预算、种子和可用先验不同，且均为单次运行；不宜宣称 R4 在 DTU PSNR 上稳定领先。
- CoR-GS 的 SSIM **0.856** 和 LPIPS **0.121** 均为表中最好。R4 相对 CoR-GS 高约 **1.005 dB PSNR**，但 SSIM 低约 **0.042**、LPIPS 高约 **0.041**，三指标没有共同领先。
- R4 相对同 MVS / 10k 的原版 3DGS 为 **+2.571 dB PSNR、−0.0116 SSIM、+0.0157 LPIPS**。这是完整方法及额外位姿、教师先验的差值；不能写作单个损失项的增益。
- DNGaussian / DropoutGS 的随机/6k 分数不能和另存的 MVS/10k 适配分数互换。初始化、预算、seed 和部分训练日程均改变；本表展示的是最接近其 DTU 作者路线的已完成本地复跑。
- 论文可以报告这张表为“各方法自身配置下，统一 DTU 25-view mask 评测的本地复跑”。方法段需写明随机点的 49 相机几何范围、深度量化和 R4 测试位姿可见性。若论文主张“严格三图几何先验下的公平方法排名”，这张表不支持该主张，需要另做统一信息条件实验。

## 不纳入主表的已测结果

| 方法 | 已测本地结果：PSNR / SSIM / LPIPS | 不进入作者配置主表的原因 |
|---|---:|---|
| FSGS | 17.460 / 0.830 / 0.155 | 本地三视角 MVS / 10k DTU 移植；官方仓库没有独立 DTU 训练脚本，不能标作作者 DTU 原生配置。 |
| DNGaussian（MVS 适配） | 13.839 / 0.656 / 0.265 | 强制共同 MVS / 10k，偏离作者随机/6k DTU 路线。 |
| DropoutGS（MVS 适配） | 16.267 / 0.755 / 0.196 | 同上。 |
| Difix3D+（本地 10k） | 16.077 / 0.809 / 0.166 | 本地从 MVS 直接训练 10k 的改编；原路线为较长的基础训练后再修复，本地没有对应 DTU 作者配置的完整复跑。 |

这些数据保留作内部诊断或单独标注的补充实验，不能把它们当作上述方法在作者 DTU 设置下的成绩。DTU 主表暂不列 FSGS、Difix3D+ 的“作者配置”数字。

## 数值、实现与审计来源

| 内容 | 可追溯文件 |
|---|---|
| 原版 3DGS 与 R4 | [统一汇总及逐景 JSON](source_metrics/graphdeco/dtu_summary.json) |
| CoR-GS 及本地适配备查行 | [MVS/10k 汇总 JSON](source_metrics/mvs10k/summary_corgs25_mask.json)；[逐景 CSV](source_metrics/mvs10k/per_scene_corgs25_mask.csv) |
| DNGaussian / DropoutGS 作者路线复跑 | [随机/6k 汇总 JSON](source_metrics/official_random_6k/summary_corgs25_mask.json)；[逐景 CSV](source_metrics/official_random_6k/per_scene_corgs25_mask.csv)；[配置与兼容性记录](source_metrics/official_random_6k/复现实验说明_中文.md) |
| 训练输入、指标口径、测试位姿与选择理由 | [DTU 设置审计](source_metrics/audits/基线配置与指标审计_中文.md)；[测试位姿核查](source_metrics/audits/更正后结果与分析_中文.md) |

