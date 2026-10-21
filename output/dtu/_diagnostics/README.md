# DTU 本地 MVS / 10k 适配诊断

这里保存 FSGS、DNGaussian、DropoutGS、Difix3D+ 在三训练图 MVS / 10k 下的本地适配实验，每个方法 15 scan。数字与最终 RGB 渲染完整保留，汇总见 [summary.csv](summary.csv)。它们不进入上一层的“各方法自身配置”主表：FSGS 没有独立的作者 DTU 路线；DNGaussian、DropoutGS 的作者 DTU 路线是随机点 / 6k；Difix3D+ 这里是本地短程改编。

指标仍使用同一套 CoR-GS 25-view mask 评测。每个 scan 保存 31 张已渲染图，其中固定 25 张用于指标；逐视角评测名单在 `metrics/metrics_corgs25_mask.json` 中。
