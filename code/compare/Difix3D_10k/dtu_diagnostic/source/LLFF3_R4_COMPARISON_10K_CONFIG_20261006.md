# Difix3D+：LLFF 三视角、10k 预算的新对比配置与运行记录

日期：2026-10-06。本文保留实验开始前的**预注册配置**，并在末尾追加实际运行审计与结果。主实验记为 `Difix3D+-MVS-10k-5k5k-targetpose`。目的：在与 [R4 结果及协议](/data/lilinwei/3D/1006difix/output/r4_all_scenes/summary.md)相同的数据、MVS 初始化、3D 更新次数和公开测评器下，重测保留 Difix3D+ 原有监督方式的 10k 版本。此版本缩短了训练日程，不能称为论文原预算复现。

**执行状态（2026-10-07 07:09 Asia/Shanghai）：** 10k 八场景全部完成；首次 orchids 运行因隐藏的 `.ipynb_checkpoints` PNG 进入递归 `images_4` 文件清单而错配，作废并隔离。清理后的独立输入只含 manifest 对应的 25 张 canonical PNG，10k orchids 已用相同 seed/config 重跑。最终 172/172 个归档 GT 与映射 PNG 像素逐一一致。04:00 启动前按用户要求核对 GPU 计算进程归属；未中断其他 `lilinwei` 作业，GPU 6 排除。

**点云来源更正：** 下文沿用历史路径名所称的 “MVS 初始化” 实际为对全场景 COLMAP `points3D.bin` 按三张训练图的观测轨迹筛选，再写入 `3_views/dense/fused.ply`；它不是重新由三视角稠密 MVS 重建的点云。点坐标仍来自全场景 SfM。详见[数据准备代码](/data/lilinwei/3D/D23E/Aref/FSGS_DA/tools/prepare_do3_fsgs_data.py:163)与运行[点云来源说明](/data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/README.md)。

10k 宏平均：5k raw **14.7610 / 0.4924 / 0.4441**，10k raw **16.1640 / 0.5065 / 0.4040**，10k Difix post **16.0347 / 0.4882 / 0.3602**（PSNR / SSIM / LPIPS）。R4 对照及逐场景差值见 [`SUMMARY_AND_ANALYSIS.md`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/10k_targetpose/outputs/SUMMARY_AND_ANALYSIS.md>)。逐轮伪视角调用、alpha 覆盖率和按 seed 重放的真实/伪采样计数见 [`PSEUDO_GENERATION_AUDIT.json`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/10k_targetpose/outputs/PSEUDO_GENERATION_AUDIT.json>)；固定教师与逐场景输入 provenance 见 [`RUN_PROVENANCE.json`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/10k_targetpose/outputs/RUN_PROVENANCE.json>)。

## 先区分论文描述与公开 gsplat 代码

- [论文第 4.2 节](https://arxiv.org/html/2503.01774)描述：先用真实参考图训练 3D 场景；每约 1.5k 次迭代把参考相机逐步移向目标相机，渲染、用 Difix 修复并加入训练；还可在最终渲染时再经 Difix 修复。这里的“目标相机”提供**位姿**，其测试 RGB 不用作训练标签。
- [公开 gsplat 实现](https://github.com/nv-tlabs/Difix3D/blob/main/examples/gsplat/simple_trainer_difix3d.py)默认 `max_steps=60000`；默认 `fix_steps` 前面有 3k、6k，之后主要每 2k 一次。[官方 README](https://github.com/nv-tlabs/Difix3D#gsplat)用约 30k 的预训练 checkpoint 续训；现有本地“官方日程适配”实验据此用 30k 真图建模、30k Difix 续训，显式把刷新设在 30k、32k、…、60k。论文文字中的 1.5k 与此代码日程不完全相同，记录结果时应写清实际版本和参数。
- 公开 gsplat 默认 `depth_loss=False`。因此真实视角**无深度监督**，伪视角也**无深度监督**。仅使用共同 MVS 点云来初始化高斯，不等于训练时加深度损失。代码里可选的 `depth_lambda=0.01` 是稀疏点视差损失开关打开后才使用的配置；它不是 R4 的真实视角单目深度权重 `0.05`，也不是 R4 的伪深度权重 `0.01`。当前 Difix 伪数据载入逻辑没有准备该可选深度字段，因此主实验保持 `depth_loss=False`。
- 对真实/伪视角，共用 `L_rgb = 0.8 L1 + 0.2 (1-SSIM)`；每个训练步有真实数据时以概率 0.7 选真实视角、0.3 选**最新一轮**伪视角。选真实视角时整步损失再乘 `1.5`，选伪视角时再乘 `novel_data_lambda=0.3`。伪视角先用渲染 alpha 的 `>0.5` 区域遮罩 RGB，再计算损失。这里的 `0.3` 是**整个伪视角训练步**的系数，不能直接等同于 R4 伪 RGB 项的 `0.35`。
- 伪视角每次从当前位姿朝各目标相机位姿推进最多 `0.5` 个归一化场景距离；使用最近的真实训练图作参考。Difix 使用 `nvidia/difix_ref`、`prompt="remove degradation"`、`num_inference_steps=1`、`timesteps=[199]`、`guidance_scale=0.0`。最新一轮标签替代上一轮的采样来源；旧标签可以留盘审计，但**不进入累积训练池**。训练 Difix 教师时论文提到的 L2、LPIPS、Gram 损失不属于这里的 3DGS 续训损失。

## 主实验：10k 总更新，5k 建模 + 5k Difix

| 项目 | 固定配置 |
|---|---|
| 场景与划分 | LLFF 八场景；每场景相同 3 张训练图，八场景合计相同 43 张 held-out 图，复用 R4 和旧 Difix 的 `data_mapping`，核对文件名顺序 |
| 图像与相机 | `images_4`、1008×756、`data_factor=4`、`strict_no_undistort=true`；相机归一化和评测器与现有统一基线一致 |
| 几何初始化 | 每场景同一 `3_views/dense/fused.ply`；记录 SHA256、初始化高斯数和归一化后的点云范围 |
| 优化 | **连续** 10,000 次 3DGS 更新；前 5,000 次只用真实图；不在 5k 重建模型或重置优化器。其余 gsplat/adapter 学习率、densification、背景等配置固定为已有 60k 适配器默认值，并归档实际 `cfg.yml` |
| 伪视角刷新 | 完成第 `5000, 5200, 5400, …, 9400` 次更新后刷新，共 **23 轮**；刷新点之后的下一步才可能抽到新伪数据。这是每 200 更新一次，选择该间隔是为了与 R4 的刷新粒度一致，同时给首次伪数据前 5k 次几何建模；它比公开 gsplat 的绝对 2k 间隔更密集 |
| 伪视角目标 | 每轮覆盖该场景**全部 held-out 测试相机位姿**，共 43 个目标；从训练相机附近逐轮朝对应目标位姿前进，`distance=0.5`；**任何 held-out RGB 均不得在训练或伪标签生成时读取** |
| Difix 教师 | 原公开 `nvidia/difix_ref` 权重和单步参数；固定权重及哈希，所有场景一致 |
| 训练采样/损失 | 真实/伪抽样 0.7/0.3；真实步 `1.5 × (0.8L1+0.2L_SSIM)`；伪步 `0.3 × (0.8L1+0.2L_SSIM)`，伪 RGB 先应用 alpha 遮罩；`depth_loss=False`，伪数据仅取最新一轮 |
| 随机性 | **主实验固定 `seed=2024`，八个场景均使用此配置，并在每场景 `cfg.yml` 与日志中记录。** 当前官方及本地 60k 适配器写死 `42+local_rank`；运行前必须给新的独立 10k 适配器增加显式 seed 参数，并验证单卡运行时实际传给 Python、NumPy、PyTorch 的种子均为 2024。若仍为 42，该运行不属于本文定义的主实验。同一个种子编号不保证不同算法抽样轨迹相同 |
| 终点 | 10k 原始 3DGS 渲染为主指标；10k 渲染再经 Difix 修复为**单独的** `+post` 指标，不与 R4 原始渲染混作一个主比较 |

主配置预计生成 `23 × 43 = 989` 张修复伪标签，按场景分别为 horns 184、fern 69、flower 115、fortress 138、leaves 92、orchids 92、room 138、trex 161。R4 从 2,001 到 9,401 每 200 更新刷新 38 轮，累计 1,634 张标签；两组 3D 更新同为 10k，但伪监督开始时间、标签数、教师调用数、教师模型、是否用深度、是否累积旧标签均不同。**10k 同预算只指 3D 更新次数相同，不代表总 GPU 时间相同。**

### 配置实施与审计记录

1. 从现有 [`60k 适配器`](/data/lilinwei/3D/baselines/Difix3D/run/20261001_1225_llff3_images4_mvs_original60k/adapter/simple_trainer_difix3d_common.py)派生独立、版本化的 10k 运行文件；不要覆写旧 10k 或 60k 产物。使用官方朝 held-out 目标相机推进的 `shift_poses` 路径，**不要**使用旧 10k 的仅训练视角插值函数。单次连续训练设 `--max-steps 10000`，`--fix-steps 5000 5200 ... 9400`，`--seed 2024`，不传 `--disable-difix`；`--eval-steps 5000 10000`、`--save-steps 5000 10000`。`--seed` 是待增加的参数，不能直接传给当前写死 42 的适配器；实现后须检查日志中的实际种子。归档的 5k/10k checkpoint 必须对应完成相应更新后的模型。现有适配器的普通周期 checkpoint 与最终 post-update checkpoint 在步数命名上并不完全一致，需以日志和实际保存时机判定。
2. 逐场景检查 3 个训练图、held-out 图列表、相机坐标系和内参；检查各伪目标位姿到目标的距离在刷新过程中不增加，单次推进不超过 0.5（容许浮点误差）；首轮每场景修复图数等于其目标相机数，修复图和参考图都不得是 held-out RGB。公开代码在伪数据 parser 中将相机内参设成同一个 `camera_ids[0]`；先核实 LLFF 此协议下目标相机内参一致，若不一致应修正映射并在结果中披露。
3. 记录每场景每轮实际更新号、目标位姿数、图像数量、Difix 调用次数、真实/伪训练步数、伪图平均 alpha 有效比例、5k 建模指标、10k 原始指标、10k 后处理指标、训练时长/教师时长、GPU 和随机种子。检查 5k 建模是否明显退化，但**不按单场景指标临时改变固定日程**。
4. 最终只用既有 [`compute_unified_public_metrics.py`](/data/lilinwei/3D/baselines/compute_unified_public_metrics.py)在相同 GT、相同 43 张测试图上计算 PSNR/SSIM/LPIPS；先逐视角、再场景内平均、最后八场景等权平均。主表列出 `5k base raw`、`10k Difix raw`、`10k Difix +post` 和 R4 原始渲染；原始和后处理数值分栏，不按较优者择优填表。

## 解读边界

此前 [`10k strict 运行`](/data/lilinwei/3D/baselines/Difix3D/run/20261001_0135_llff3_images4_mvs10k_strict/run_scene.sh)只有 `5001/6501/8001/9501` 四次刷新，且伪相机仅在训练相机之间插值；它不能代替本文的朝目标相机渐进版本。此前 [`60k 运行`](/data/lilinwei/3D/baselines/Difix3D/run/20261001_1225_llff3_images4_mvs_original60k/run_scene.sh)已采用目标相机轨迹和 30k+30k 日程，可作为长预算参考，但其总更新数不能放入 10k 主表直接排名。即使新 10k 实验有改善，也需通过 5k 建模质量、最终 raw/+post 拆分和伪标签审计判断原因；一次单种子比较不足以证明方法本身优劣。


## 最终运行记录（2026-10-07）

- 输出根目录：[`run/20261007_recheck_pair/10k_targetpose/outputs/`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/10k_targetpose/outputs/>)；逐场景保存 5k/10k checkpoint、点云、原始渲染、10k Difix 后处理渲染、GT、配置、训练日志及 unified per-view JSON。大部分训练临时文件仍在 `/tmp/lilinwei/Difix3D_Recheck_20261007/scratch/10k/`。
- 八场景实际生成 **23 轮、989 张 Difix 伪标签**；训练数据分支按固定 `seed=2024` 和适配器随机分支重放为每场景 8,466 个真实步、1,534 个伪步。每轮 alpha 有效比例以 `alpha >= 128/255` 统计，逐轮逐场景值见审计 JSON。
- R4 与本次 10k 的 raw 宏平均差（本次 − R4）为 **−3.5740 PSNR / −0.1533 SSIM / +0.1184 LPIPS**。两者同为 seed 2024，但训练目标、修复教师、刷新次数、深度监督和标签池机制不同；不能将差异归结为 Difix 教师本身。
