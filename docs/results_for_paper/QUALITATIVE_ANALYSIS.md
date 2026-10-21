# 典型视角的 RGB 与深度比较

主图把同一测试相机的 RGB 和深度上下排列。红框与左下角放大图对所有方法使用同一原图坐标。RGB 是最终运行输出的逐字节副本；各方法的深度由归档 Gaussian PLY 在同一相机上重新渲染。图中 **Reference** 列上方是 RGB GT，下方是 **Depth Anything V2 从该 RGB GT 预测的相对深度代理**，不是测量的深度 GT。高斯深度在每个视角的所有方法间共用一个相机深度色标；DA-V2 数值是相对、近似逆深度，独立归一化。两者都用近暖、远冷的视觉方向，但颜色数值不能直接对照。色标与来源说明保留在本文和图注，不叠加在图片上。源文件、SHA-256 与显示范围见 [asset_manifest.json](asset_manifest.json)、[depth_manifest.json](depth_manifest.json)、[da_reference_manifest.json](da_reference_manifest.json) 和 [depth_display_ranges.json](depth_display_ranges.json)。

## LLFF：`trex` 与 `room`

![LLFF RGB 与深度主图](figures/llff_rgb_depth_main.png)

[扩展 RGB+depth 图](figures/llff_rgb_depth_expanded.png)另外展示 CoR-GS、Difix3D+。按当前选图要求，LLFF 图中不展示 DNGaussian，但 [LLFF 主表](LLFF_TABLE.md)仍保留其指标。Difix3D+ 为 30k+30k，其他图中方法为统一 MVS 初始化和 10k 优化。

- **`trex / DJI_20200223_163604_605`：** 普通 3DGS 在恐龙骨架和上方弧形建筑处出现大范围漂浮与颜色混叠，深度也缺少连续结构。其 RGB 中明显的青绿色来自原始归档结果：从最终 PLY 和同一相机重渲染 RGB，与归档图的 PSNR 为 **59.12 dB**，没有对该列做冷暖色调整。稀疏视角方法恢复了更完整的骨架与背景层次。Ours 的肋骨和头部轮廓较清楚，但 FSGS、DropoutGS 同样给出较完整的结果，不能把这张图描述成 Ours 对所有方法的明显胜出。
- **`room / DJI_20200226_143912_008`：** 观察墙面、屏幕轮廓、桌面及椅子。普通 3DGS 在左侧和桌面有严重拖影；FSGS 的结构已较稳定，DropoutGS 在屏幕和桌面仍有明显涂抹。Ours 的屏幕边界与周围墙面较干净。深度图可检查墙面、桌面、屏幕之间的层次；当前 Ours 也存在局部空洞和表面不平整，不能称为准确深度真值。

LLFF 八场景宏平均：Ours PSNR **19.7992 dB**，DropoutGS **19.8039 dB**，差 **0.0047 dB**；FSGS 的 LPIPS **0.2700** 优于 Ours 的 **0.2856**。两个场景的选择用于解释典型结构变化，完整结论应以主表为准。

**候选图注：** *Qualitative comparison on LLFF under three-view MVS initialization. Each pair of rows shows RGB and model-rendered camera depth at the same held-out view. Gaussian depths share a color range per view, determined by the 2nd–98th percentiles of valid depths across methods. The Reference depth is a Depth Anything V2 prediction from RGB ground truth, normalized independently using its 1st–99th percentiles; it is a relative structural reference, not measured ground truth. Warm colors indicate nearer regions. All shown methods use 10k steps; Difix3D+ (30k+30k) appears in the expanded panel.*

## DTU：`scan30` 与 `scan21`

![DTU RGB 与深度主图](figures/dtu_rgb_depth_main.png)

- **`scan30 / rect_010_3_r5000`：** Ours 对南瓜外轮廓、果梗和表面纹理恢复较完整。普通 3DGS 的 RGB 与深度在物体外有漂浮伪影；其他方法虽恢复主体，但背景和物体边缘仍有不同程度的残影。新图用同一放大 ROI 内跨方法有效深度的 2%–98% 分位数确定高斯深度色标（**6.73–7.82**），因此果梗和表面的相对变化比之前的全图色标更可见。该视角的前景 mask PSNR：Ours **24.584 dB**，DropoutGS **23.382 dB**。
- **`scan21 / rect_010_3_r5000`：** Ours 对砖房外墙、窗框的 RGB 结构较清楚；DNGaussian 明显模糊，CoR-GS 与 DropoutGS 在局部边缘有更强伪影。深度展示建筑与背景的分离，但由于没有测量深度，不能由此定量判定几何精度。该视角的前景 mask PSNR：Ours **18.229 dB**，CoR-GS **14.224 dB**，DropoutGS **13.075 dB**。

之前 DTU 深度“只剩大轮廓”有两处原因。首先，底层 rasterizer 返回的是透明度加权深度和累积 alpha，旧脚本漏掉了 **depth / alpha**，低覆盖像素被错误压向近处；现已重新导出全部方法的深度。其次，按整幅图着色时背景跨度淹没了物体表面起伏。现在固定各方法共用的 ROI 范围，`scan30` 为 **6.73–7.82**、`scan21` 为 **5.83–8.40** 相机单位，并保留原始 TIFF 数值。`scan30` 果梗和表面、`scan21` 窗框与墙面的深度变化更容易看见。深度只表示几何，RGB 上的颜色纹理不会全部出现在深度图；当前 DTU 渲染分辨率约为 396×297，细节也受此限制。

DTU 全集宏平均 Ours PSNR **20.325 dB**，DropoutGS **20.322 dB**，差 **0.003 dB**；CoR-GS 的 SSIM **0.856**、LPIPS **0.121** 仍优于 Ours 的 **0.815**、**0.162**。DNGaussian 和 DropoutGS 的 DTU 作者路线为 random-point/6k；3DGS、CoR-GS 和 Ours 为 MVS/10k。图文须明确训练路线差异。

**候选图注：** *RGB and depth on DTU scan30 and scan21. Gaussian camera depths share one range per view, determined by the 2nd–98th percentiles of valid depths across methods within the common crop and applied to the full image. The Reference depth is a Depth Anything V2 prediction from RGB ground truth, normalized independently using its 1st–99th percentiles; it is not measured or metric depth ground truth. Warm colors indicate nearer regions. DTU RGB metrics use the common 25-view object-mask evaluation; training routes differ as specified in the table caption.*

## 深度生成与可解释性

`render_depth.py` 从各方法最终 PLY 读取高斯的位置、尺度、旋转、透明度及可用的 SH 系数，通过同一个带 depth 输出的 Gaussian rasterizer 在归档测试相机上渲染。该 rasterizer 输出 `D=Σ(zᵢwᵢ)` 和 `alpha=Σwᵢ`；我们将 **`D/alpha`** 作为有效像素的期望相机深度。对于只归档几何字段的 DNGaussian 和 DropoutGS，颜色 SH 设为零；这不改变 depth 或 alpha。alpha < 0.5 的像素置为 NaN。LLFF 从全图、DTU 从共享红框 ROI 的跨方法有效深度求 2% 与 98% 分位数，再固定应用于该视角所有方法的完整深度图。原始 float32 深度保存在 `assets/*/*/*_depth.tiff`。

`make_da_reference.py` 使用 Depth Anything V2 ViT-L（输入尺寸 518）对本目录逐字节复制的四张 `GT.png` 推理，并保存 `GT_da_depth.tiff`。原先旧实验目录的 DA 参考图与当前视角不完全覆盖且分辨率有差异，所以本图全部重新对**当前 GT RGB**运行同一权重。DA-V2 输出独立使用 1%–99% 分位数着色。它仅是结构参考，不能用于直接比较高斯相机深度的绝对数值或计算深度误差。

抽查相机对齐：用同一 PLY 和相机重新渲染 RGB，与归档 RGB 比较，LLFF `trex` Ours 为 **55.20 dB**，DTU `scan30` Ours 为 **58.77 dB**。这验证了重渲染相机与归档视角一致；深度图本身仍是模型输出，不能充当真实几何精度评估。
