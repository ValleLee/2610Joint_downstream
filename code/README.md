# 代码目录

`compare/` 按方法和数据集路线分支，`ours/` 保存 R4 重建与冻结 RGB-D predictor，`evaluation/` 保存两套统一评测的原始实现。原始训练脚本中的绝对路径是实验记录，移植时按 `docs/REPRODUCE.md` 映射。

| 方法 | LLFF | DTU 主表 | DTU 诊断 |
|---|---|---|---|
| vanilla_3DGS | `compare/vanilla_3DGS/llff_mvs10k/` | `compare/vanilla_3DGS/dtu_author/` | — |
| FSGS | `compare/FSGS/llff_mvs10k/` | — | `compare/FSGS/dtu_diagnostic/` |
| CoR-GS | `compare/CoR-GS/llff_mvs10k/` | `compare/CoR-GS/dtu_author/` | — |
| DNGaussian | `compare/DNGaussian/llff_mvs10k/` | `compare/DNGaussian/dtu_author/` | `compare/DNGaussian/dtu_diagnostic/` |
| DropoutGS | `compare/DropoutGS/llff_mvs10k/` | `compare/DropoutGS/dtu_author/` | `compare/DropoutGS/dtu_diagnostic/` |
| Difix3D+ | `compare/Difix3D_60k/llff_mvs_60k/` | — | `compare/Difix3D_10k/dtu_diagnostic/` |
| our_R4 | `ours/llff_mvs10k/` | `ours/dtu/` | — |

原版 3DGS 的两个数据集分支共用 `compare/vanilla_3DGS/source/` 与 `runner/`；DTU 作者路线 DNGaussian/DropoutGS 的运行脚本在 `compare/_dtu_author_runner/`，其他 DTU 输入准备脚本在 `compare/_dtu_shared/`。每个方法的运行时配置还见 `output/<数据集>/<方法>/<场景>/` 与 `data/`。
