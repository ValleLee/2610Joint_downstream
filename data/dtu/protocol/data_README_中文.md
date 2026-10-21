# DTU 数据与 split 说明

本实验不下载 DTU 官方 123GB Rectified 全量数据。原图、相机数据、完整 mask 来源包和 COLMAP 工作区缓存在 `/tmp/lilinwei/1007fsgsmvs_dtu/`；本机只保存来源清单、结果复算所需的 375 张小 mask 归档与最终实验结果。

当前按需下载 CoR-GS 发布的 DTU 3/6/9-view 处理包（Google Drive ID：`1ibu2-Naueh-Vb0ILj0DCU-h4Pxi5EdYY`）。归档已下载到 `/tmp`、SHA-256 为 `edb5c8619ef3a1192d3f89d53a1fcb5cfb24c914ccbefe90b09c6ca1c1c29079`，且 ZIP 完整性检查通过。压缩包覆盖全部 15 个目标 scan，每个场景均有 49 张图、相机内外参和 poses_bounds。未经检查不得直接把包内点云作为初始化：每个场景仍须核对它是否确由训练索引 `[25,22,28]` 的三张图生成；本实验会从三张训练图构建唯一共享 MVS PLY。

场景：`scan8`、`scan21`、`scan30`、`scan31`、`scan34`、`scan38`、`scan40`、`scan41`、`scan45`、`scan55`、`scan63`、`scan82`、`scan103`、`scan110`、`scan114`。

每场景 49 个 COLMAP 图像名按字典序排序。训练索引 `[25,22,28]`；显式排除 `[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]`；本轮初始结果评测其余 31 个。每场景 49 个名字、相机记录、训练图 hash 和 3/15/31 名单已经冻结到 `splits/cor_gs_3view/` 与 `dataset_manifest.json`。CoR-GS 原始代码还将 6/9 视角协议中的六个候选训练位 `[0,8,13,40,44,48]` 从三视角测试集中排除，因此其原始代码评测 **25** 个视角。两种划分的指标应分别标注。

mask 来源：CoR-GS 处理包不含 mask，且其 README 所连的 Google Drive 文件目前返回 404。此前核查的 [Spann3R DTU 测试包](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md)只与目标集重合 3 个 scan，[IDR 公开数据入口](https://github.com/lioryariv/idr) 的归档只重合 5 个 scan，因此初始 31 视角结果采用全图 RGB。后续找到 [DNGaussian 官方 RegNeRF mask 备份](https://github.com/Fictionarry/DNGaussian) 的 `submission_data.zip`，约 38 MB，覆盖全部 15 个目标场景的 CoR-GS 原始 25 个测试视角。包暂存于 `/tmp/lilinwei/1007fsgsmvs_dtu/downloads/dngaussian_submission_data.zip`，SHA-256 为 `cd65d10cc0126e3f266a0f30e407c26fda3ecf61ce13318613261d9dfeef8b23`，ZIP CRC 检查通过。评测用 `rect_NNN_3_r5000.png` 对应 mask 编号 `NNN-1`，已目视核对三种文件组织形式与 GT 对齐。详细来源和校验结果见 `download_manifest.json`；新计算的 25 视角物体 mask 指标与既有 31 视角全图指标分别保存。
