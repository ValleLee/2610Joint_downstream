# 复现说明

## 先核对归档结果

在仓库根目录运行：

```bash
python scripts/verify_release.py
```

脚本检查最终 RGB 渲染图、指标文件、来源 SHA-256、场景数量和两张主表的汇总。它不调用 GPU，也不重新训练。`output/<数据集>/<方法>/<场景>/artifact_manifest.json` 逐文件记录了来源与哈希。`point_cloud/` 和 `checkpoints/` 在原工作站保留，但 GitHub 版本不会包含它们；核验器允许这些本地文件缺席。

取得 GT 与 mask 后，可从归档 PNG 独立重算像素指标。LLFF 的 `--gt-root` 需包含 `<scene>/<测试图 stem>.png`；DTU 需包含 `<scan>/<rect 文件名>.png`。使用各方法原生代码依赖的 PyTorch 环境：

```bash
python code/evaluation/reevaluate_archive.py llff FSGS \
  --gt-root /path/to/llff_gt --device cuda --out /tmp/llff_fsgs_recomputed.json

python code/evaluation/reevaluate_archive.py dtu CoR-GS \
  --gt-root /path/to/dtu_gt --mask-zip /path/to/corgs_25view_masks.zip \
  --device cuda --out /tmp/dtu_corgs_recomputed.json
```

DTU mask 包会校验 SHA-256；LLFF 的数值文件名渲染由每景 `run_manifest.json` 映射到真实测试图。可用 `--select fern` 或 `--select scan8` 先做单景检查。

## 准备输入

1. 获取 LLFF 八景 `images_4` 与 DTU 15 scan 处理包，按照 `data/llff/protocol/`、`data/dtu/protocol/splits/cor_gs_3view/` 固定三训练图和测试视角。下载来源与数据包哈希见 `data/dtu/protocol/data_download_manifest.json`。GT 和 DTU mask 需要从原数据包准备，不能从渲染图推回。
2. LLFF 为每景用三训练图生成 FSGS MVS 初始化点云；DTU 的原版 3DGS、CoR-GS、R4 同样使用三图 MVS。输入点云 SHA-256 分别记录在 `data/llff/protocol/input_manifest.json` 和 DTU 运行清单中。DTU DNGaussian / DropoutGS 作者路线改用约 1000 个随机点，其采样盒依赖处理包中的 49 相机 COLMAP 模型。
3. R4 需要 Marigold base 和 14k 步 RGB-D predictor UNet 权重；配置及权重 SHA-256 见[代码与数据索引](CODE_AND_DATA.md)。只提供代码和渲染结果不能重训 R4。

## 训练代码与配置

| 路线 | 入口与参数记录 |
|---|---|
| LLFF FSGS / CoR-GS / DNGaussian / DropoutGS | `code/compare/<方法>/llff_mvs10k/source/`；同分支 `config/experiment_config.json`；`code/compare/_llff_shared/runner_templates/` |
| LLFF Difix3D+ 60k | `code/compare/Difix3D_60k/llff_mvs_60k/runner/`；同分支 `config/run_audit.json` 和 `experiment_config.json`。30k 基础训练后续训 30k，使用最终 60k 原始 3D 渲染 |
| LLFF / DTU 原版 3DGS | `code/compare/vanilla_3DGS/runner/run_scene.py`；`experiment_config.json` 和每景 `run_manifest.json` |
| LLFF R4 | `code/ours/llff_mvs10k/run_pipeline.py`、`train_r4_memory.py`；每景 `output/llff/our_R4/<场景>/` 的参数与清单 |
| DTU CoR-GS / R4 / 诊断适配 | `code/compare/_dtu_shared/train_scene_dtu.py`；R4 入口在 `code/ours/dtu/`；`data/dtu/protocol/` 与每景运行清单 |
| DTU DNGaussian / DropoutGS 随机点 / 6k | `code/compare/_dtu_author_runner/run_one_official.sh`；各方法 `dtu_author/source/`、每景 `run_manifest.json` |

代码快照保留了实验机的 `/data/lilinwei/...`、`/tmp/lilinwei/...` 路径。移植时要把数据、模型、Python 环境和输出路径映射到目标机器；请以运行清单中的命令、实际读入点云哈希和 `cfg_args` 检查映射。原工作站记录使用带 CUDA 的 PyTorch 及各方法匹配的自定义光栅器；`d23e` 环境为 Python 3.8.20 / PyTorch 2.0.1+cu118，predictor 的 `mast3r` 环境为 Python 3.10.19 / PyTorch 2.10.0+cu128 / diffusers 0.35.1。这里没有在全新机器完成一键重训验证。

## 指标与论文表述

LLFF 的 10k 方法共享三图 MVS、seed 2024 与 43 张留出图评测；Difix3D+ 采用同类 MVS 输入和同一评测，但训练 **60k**。它不能作为等预算 10k 排名依据。R4 和 Difix3D+ 可见留出/测试相机位姿，但不读取测试 RGB。

DTU 五行共享 25-view mask 评测，但训练初始化和预算不同。DNGaussian / DropoutGS 为随机点 / 6k；其他主表方法为 MVS / 10k。R4 伪目标使用八个留出相机位姿，其中五个与评测相机重合，不读取评测 RGB。`output/dtu/_diagnostics/` 的 MVS / 10k 本地适配不代表作者 DTU 配置。两表均先平均景内测试视角，再对场景等权平均；DTU 的 `±` 为景间样本标准差。
