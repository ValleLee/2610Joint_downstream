# 代码与数据索引

我们方法的算法细节与实际损失见 [联合预测器指导 3DGS 的方法说明](JOINT_PREDICTOR_3DGS_METHOD.md)。

`code/compare/` 按方法保存实验时的代码快照；分支名表示本地实际测试路线，不表示所有方法共享训练超参数。原始仓库许可证保留在各源码目录。`code/ours/predictor/` 是冻结 RGB-D predictor 的推理代码；它的权重不随 Git 包发布。

| 实验 | 代码与运行入口 | 配置和证据 |
|---|---|---|
| LLFF FSGS、CoR-GS、DNGaussian、DropoutGS | `code/compare/<方法>/llff_mvs10k/source/` 和 `code/compare/_llff_shared/runner_templates/` | `code/compare/<方法>/llff_mvs10k/config/`、`output/llff/<方法>/` |
| LLFF Difix3D+ 60k | `code/compare/Difix3D_60k/llff_mvs_60k/`，含 30k + 30k 适配器、运行脚本和视角映射 | 该分支 `config/run_audit.json`、`output/llff/Difix3D_60k/` |
| 原版 Graphdeco 3DGS | `code/compare/vanilla_3DGS/source/`、`runner/` | `code/compare/vanilla_3DGS/experiment_config.json`、两数据集下的 `run_manifest.json` |
| DTU CoR-GS | `code/compare/CoR-GS/dtu_author/source/`、`code/compare/_dtu_shared/` | `data/dtu/protocol/`、`output/dtu/CoR-GS/` |
| DTU DNGaussian、DropoutGS | `code/compare/<方法>/dtu_author/source/` 和 `code/compare/_dtu_author_runner/` | `output/dtu/<方法>/` 中的随机点 / 6k 运行清单 |
| DTU 本地 MVS/10k 适配 | `code/compare/<方法>/dtu_diagnostic/` 与共享 DTU 入口 | `output/dtu/_diagnostics/`，不纳入作者配置主表 |
| 我们的方法 R4 | `code/ours/{llff_mvs10k,dtu,reconstruction,predictor}/` | `data/` 中的协议及 `output/*/our_R4/` |
| 统一评测 | `code/evaluation/{llff,dtu}/` | `output/*/source_metrics/` 中的原始汇总 |

## 外部数据与权重

- LLFF：八景 `images_4`、相机文件与固定 3-view split；每景实际 MVS PLY 的 SHA-256 和初始化审计见 `data/llff/protocol/input_manifest.json`、`actual_initialization_audit.json`。
- DTU：15 个 CoR-GS 处理后的 scan、三训练图 split、GT 与 25-view mask。数据来源和包哈希见 `data/dtu/protocol/data_download_manifest.json`，视角名单见 `data/dtu/protocol/splits/cor_gs_3view/`。DNGaussian/DropoutGS 随机点采样盒来自本地 49 相机 COLMAP 稀疏模型；这项额外几何范围信息须保留在论文设置描述中。
- R4 predictor：14k 步 RGB-D UNet 权重 `diffusion_pytorch_model.safetensors`，SHA-256 `5b24319bc96edef789687858a5132dbdece38c2fe64cce8424154d3aaddc290d`；还需要 Marigold base。配置在 `code/ours/predictor/config/`，大权重未纳入 Git。
- 最终 RGB 渲染 PNG 已归档；GT 和 mask 属于外部数据。取得相应数据后可用 `code/evaluation/` 的原始评测实现重新计算指标。

第三方代码快照、原始绝对路径和运行环境属于历史证据；移植到新机器时需修改路径并安装匹配的 CUDA 扩展。各方法真实运行参数以其 `experiment_config.json`、每景 `run_manifest.json` 和 `cfg_args` 为准。
