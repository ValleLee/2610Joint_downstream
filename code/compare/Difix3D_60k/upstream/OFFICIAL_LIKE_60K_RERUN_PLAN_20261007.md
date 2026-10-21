# Difix3D+ 60k 复跑计划：仅改坐标归一化

**状态：已完成（2026-10-07 07:09 CST）。** 16/16 个场景阶段训练成功，统一指标已重算；canonical GT 像素对齐预检 172/172 通过。日期：2026-10-07。目标是检验旧 60k 结果偏低是否与坐标归一化有关。复跑与[旧 60k 运行](/data/lilinwei/3D/baselines/Difix3D/run/20261001_1225_llff3_images4_mvs_original60k/EXPERIMENT_20261001_1225.md)保持相同 LLFF 数据、3 张训练图、共同 MVS 点云、图像像素、Difix 权重、随机种子及 30k+30k 日程；主变量仅为将 `normalize_world_space: true` 改成[官方 gsplat README 示例](https://github.com/nv-tlabs/Difix3D#gsplat)使用的 `--no-normalize-world-space`。这是“官方参数下的 LLFF/MVS 适配复跑”，并非原论文数据集的逐项复现。

**点云来源更正：** 下文沿用历史路径名所称的 “MVS 点云” 实际为全场景 COLMAP `points3D.bin` 中被至少一张选定训练图观测到的点的子集，写入 `3_views/dense/fused.ply`；没有重新仅用三视角三角化或运行独立稠密 MVS。点坐标仍来自全场景 SfM，不能仅凭筛选就认定没有测试图像贡献几何。详见[数据准备代码](/data/lilinwei/3D/D23E/Aref/FSGS_DA/tools/prepare_do3_fsgs_data.py:163)与运行[点云来源说明](/data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/README.md)。

**执行记录：** 04:00 按用户要求检查 GPU 进程归属；没有中断或终止其他 `lilinwei` 实验，GPU 6 全程排除。首次 04:08 启动的 8 场景训练中，7 个场景通过训练与输出归档，orchids 的 strict GT 检查发现临时 `images_4` 混入 Jupyter checkpoint PNG，parser 错配图像；该次 orchids 结果作废。错误产物隔离于 [`incident_hidden_checkpoint_misalignment_20261007/`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/incident_hidden_checkpoint_misalignment_20261007/> )。原始数据目录已恢复，独立修复输入只含 25 张 SHA-256 与 manifest 一致的 canonical PNG；parser 级测试确认 holdout 像素匹配。干净数据下 10k 与 60k orchids 均已重跑并成功。最终公共 GT 对齐预检为 172/172；统一指标现已包括这两次有效重跑。60k 的实际 seed 42 与官方适配器一致；10k R4 配置用 seed 2024，两者不是同 seed 的严格配对实验。

本次最终 60k 宏平均：raw **15.7083 PSNR / 0.4734 SSIM / 0.3944 LPIPS**；Difix post **15.4053 / 0.4595 / 0.3808**。对应摘要和逐场景表见运行目录的 `60k_no_normalize/outputs/SUMMARY_AND_ANALYSIS.md`。

## 固定协议与明确例外

| 项目 | 新复跑配置 | 相对旧 60k / 官方示例 |
|---|---|---|
| 场景、划分、输入 | LLFF 八场景；沿用旧运行的各场景 `data_mapping/*.json`，每场景 3 张训练图，合计 43 张 held-out 测试图；`images_4` 原始 PNG，1008×756，`data_factor=4` | 与旧 60k 一致；这是为了共同基线协议必须保留的数据适配 |
| 初始化 | 各场景同一个 `3_views/dense/fused.ply`，保留 `init_type=sfm` 的 gsplat 参数化，但通过适配器以 MVS 点替换 parser 点云 | 与旧 60k 一致；相对于官方示例的起始点云是明确例外 |
| 世界坐标 | **`--no-normalize-world-space`**，相机和点云使用原 COLMAP 世界坐标；记录 parser transform 是否为单位矩阵、相机间距及 MVS 点范围 | **唯一计划改变的训练配置**：旧运行 `normalize_world_space: true`；与官方 README 命令一致 |
| 去畸变 | **`strict_no_undistort=true`，继续关闭额外去畸变** | 与旧 60k 一致、与官方 parser 默认去畸变不同。已检查八场景 `cameras.bin` 均为 `SIMPLE_RADIAL` 且径向畸变系数非零；若启用去畸变，图像/K/裁切区域会改变，不能直接与现有 `images_4` 测试 GT 做同像素统一指标。此例外由本次对比目标决定 |
| 训练与 Difix | 官方 gsplat 的 DefaultStrategy、RGB `0.8 L1 + 0.2 (1−SSIM)`、真实/伪步概率 `0.7/0.3`、真实步乘 `1.5`、伪步乘 `0.3`、仅最新轮伪图、`depth_loss=false`；`nvidia/difix_ref` 原权重，1 个扩散推理步 | 与旧 60k 及官方代码默认一致；不引入 R4 的单目深度、伪深度或累积池 |
| 随机性 | 沿用官方代码 `42+local_rank`；单卡单进程每场景实际种子 **42**，归档日志和配置 | 与旧 60k 一致，以隔离归一化这一变量；此次 60k 诊断**不改成 2024**。将来与 R4 对比时须注明种子差异 |
| 相机轨迹 | 每轮以当前伪相机朝每个 held-out 目标**位姿**前进，`distance=0.5`；参考 RGB 只取三张训练图。held-out RGB 仅用于评估 | 与旧 60k 及官方方法一致；固定数字 `0.5`，不按坐标尺度另行调参，才能检验归一化的影响 |
| 测评与保存 | 保留本地适配器的最终 post-update checkpoint、PLY、原始 RGB、Difix 后处理 RGB、depth/alpha 诊断图和统一指标输出 | 工程记录扩展；depth/alpha **不进入训练损失** |

## 精确运行日程

本次实际使用独立运行目录 [`run/20261007_recheck_pair/60k_no_normalize/`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/60k_no_normalize/>)，复制并版本化现有[60k 适配器](/data/lilinwei/3D/baselines/Difix3D/run/20261001_1225_llff3_images4_mvs_original60k/adapter/simple_trainer_difix3d_common.py)与启动脚本。旧运行的输出、checkpoint、指标和脚本均不覆盖。不要使用旧 30k checkpoint：它是在归一化坐标中训练的，不能和新原始坐标混用。

1. **预检（八场景，训练前）**：记录官方仓库 commit、适配器 SHA256、Difix 权重 revision/哈希、MVS PLY SHA256、训练/测试文件名、相机模型/畸变系数、图像分辨率。核对新 parser 的 `normalize_world_space=false`、`strict_no_undistort=true`、`depth_loss=false`；三个训练图与旧运行完全相同；MVS 点能投影到训练相机可见区域；测试 RGB 不进入训练 dataloader 或伪标签修复器。
2. **阶段 A，真视角基础重建**：每场景从头训练至完成 30,000 次更新。使用已有阶段 A 的 `--disable-difix --fix-steps 1000000`，以及 `--max-steps 30000 --eval-steps 30000 --save-steps 30000`。共同参数包括 `--no-normalize-world-space --strict-no-undistort --data-factor 4 --test-every 1 --strict-split-json ... --mvs-ply ...`。显式测试名单优先于 `test_every`；传 1 仅保持与 README 命令一致。另存 `cfg_base.yml`、日志、30k 的原始渲染与指标；报告完成 30k 后的基础质量。
3. **阶段 B，Difix 渐进续训**：按官方 README 的 checkpoint 语义从本次阶段 A 的 `ckpt_29999_rank0.pt` 接续到 60,000 次更新。使用同一输入、点云、相机及 no-normalize 参数，`--max-steps 60000 --eval-steps 60000 --save-steps 60000`，`--fix-steps 30000 32000 34000 … 60000`，不传 `--disable-difix`。按原代码运行 16 次修复；第 60k 次修复发生在最后一次参数更新后，因而前 15 批伪图可影响最终模型，最后一批仅留作审计。不要为“改善效果”改变频率、步长、loss 或教师。
4. **最终统一评价**：在与旧运行相同的 43 张 canonical `images_4` GT 上，分别计算 60k **原始 gsplat** 和 **Difix 后处理**的 PSNR/SSIM/LPIPS，使用同一[`compute_unified_public_metrics.py`](/data/lilinwei/3D/baselines/compute_unified_public_metrics.py)；每场景先平均，再八场景等权宏平均。逐视角验证渲染/GT 文件名、宽高与旧运行一致，保留完整 per-view JSON。后处理结果单独列，不择优填主表。

## 对照、判读和停止条件

- 主对照是旧 60k 同种子、同数据和同点云的结果：原始渲染 **15.5707 PSNR / 0.4619 SSIM / 0.3938 LPIPS**，后处理 **15.2887 / 0.4489 / 0.3844**。若新结果改变，可以把差异主要归因于 `normalize_world_space` **及其引起的场景尺度、初始化尺度和固定 0.5 位姿步长的联动变化**；一次运行仍不能给出各联动因素的独立贡献。
- 旧运行在 Difix 开始前的 30k 内部验证 PSNR 宏平均约 **15.0106**；horns **10.54**、room **11.11**、trex **10.71**。新运行必须记录同一时点；若这些场景在 30k 时仍低，先检查基础重建输入/相机/点云及本方法对三视角的适配，不能把问题都归到伪视角阶段。旧 30k 图像未完整归档，30k 内部指标只作阶段定位；60k 结论使用同一个公共测评器。
- 预检如发现训练图名单、像素大小、相机内参、点云坐标或实际布尔参数不符合上述配置，**先停止该场景并修正输入**，不可用偏离配置的结果填入主表。若训练出现 NaN、伪图数与测试位姿数不符、测试 RGB 被读取作训练监督，停止并保留日志。质量低本身不触发更改日程或按场景调参。
- 预计计算量参考旧运行的八场景累计约 **6.9 场景 GPU 小时**，不同机器和并行度下墙钟时间会变；归档每场景实际 GPU 时间、训练步数、修复轮数与随机种子。

**交付状态：** 双阶段脚本、最终和 base 配置、训练日志、输入/适配器/Difix 模型哈希、30k/60k checkpoint 与 PLY、逐轮伪视角及 alpha 审计、统一 per-view 指标 JSON、原始与 Difix 后处理渲染均已归档。30k/60k 原始和后处理宏平均与逐场景旧运行差值见 [`SUMMARY_AND_ANALYSIS.md`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/60k_no_normalize/outputs/SUMMARY_AND_ANALYSIS.md>)；stage-A checkpoint/PLY 哈希及最终伪视角批次见 [`BASE_STAGE_ARCHIVE_SHA256.json`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/60k_no_normalize/outputs/BASE_STAGE_ARCHIVE_SHA256.json>) 和各场景 `outputs/<scene>/novel/`。统一 canonical GT 像素预检 172/172 通过；各阶段状态 16/16 成功。

实际 60k raw 宏平均为 **15.7083 / 0.4734 / 0.3944**，Difix post **15.4053 / 0.4595 / 0.3808**（PSNR / SSIM / LPIPS）。相对旧 normalized 60k raw 改善 **+0.1376 / +0.0115 / +0.0006**；改善幅度很小，且坐标缩放会与固定 0.5 位姿步长联动。完整逐轮生成和采样重放审计见 [`PSEUDO_GENERATION_AUDIT.json`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/60k_no_normalize/outputs/PSEUDO_GENERATION_AUDIT.json>)；场景级输入、适配器、MVS 和权重 provenance 见 [`RUN_PROVENANCE.json`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/60k_no_normalize/outputs/RUN_PROVENANCE.json>) 与 [`DIFIX_MODEL_SHA256.json`](</data/lilinwei/3D/baselines/Difix3D/run/20261007_recheck_pair/60k_no_normalize/outputs/DIFIX_MODEL_SHA256.json>)。
