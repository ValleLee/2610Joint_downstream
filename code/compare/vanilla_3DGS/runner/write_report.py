#!/usr/bin/env python3
"""生成图表、代表性渲染拼图和中文实验总结。"""
import json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; MET=OUT/'metrics'
MET.mkdir(parents=True,exist_ok=True)
METRICS=['PSNR','SSIM','LPIPS']
DIRECTIONS={'PSNR':'↑','SSIM':'↑','LPIPS':'↓'}

def f(x): return f'{x:.4f}'
def macro(summary,method,metric): return summary['metrics'][method][metric]
def table_line(method,values): return '| '+method+' | '+' | '.join(f"{f(values[m]['mean'])} ± {f(values[m]['scene_sample_std'])}" for m in METRICS)+' |'

def plot_summary(dtu,llff):
    fig,axes=plt.subplots(2,3,figsize=(13,7.2))
    colors={'vanilla_3DGS':'#de8f05','our_R4':'#0173b2'}
    for row,(dataset,summary) in enumerate([('DTU',dtu),('LLFF',llff)]):
        for col,metric in enumerate(METRICS):
            ax=axes[row,col]
            for x,method in enumerate(['vanilla_3DGS','our_R4']):
                v=summary['metrics'][method][metric]
                ax.bar(x,v['mean'],yerr=v['scene_sample_std'],capsize=4,color=colors[method],width=.58)
                ax.text(x,v['mean'],f"{v['mean']:.3f}",ha='center',va='bottom',fontsize=8)
            ax.set_xticks([0,1],['Vanilla 3DGS','Our R4'],rotation=15,ha='right')
            ax.set_title(f'{dataset}  {metric} {DIRECTIONS[metric]}')
            ax.grid(axis='y',alpha=.25)
            if metric=='LPIPS': ax.set_ylim(bottom=0)
    fig.suptitle('Shared three-view MVS initialization, seed 2024, 10k iterations\nError bars show between-scene sample standard deviation',fontsize=12)
    fig.tight_layout()
    fig.savefig(MET/'macro_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)


def plot_scenes(dataset,summary):
    rows=[r for r in summary['per_scene'] if r['method'] in ('vanilla_3DGS','our_R4')]
    scenes=sorted({r['scene'] for r in rows})
    fig,axes=plt.subplots(1,3,figsize=(max(14,len(scenes)*.85),5))
    for ax,metric in zip(axes,METRICS):
        for i,method in enumerate(['vanilla_3DGS','our_R4']):
            values=[next(r[metric] for r in rows if r['scene']==scene and r['method']==method) for scene in scenes]
            ax.plot(range(len(scenes)),values,marker='o',label='Vanilla 3DGS' if method=='vanilla_3DGS' else 'Our R4')
        ax.set_title(f'{metric} {DIRECTIONS[metric]}');ax.set_xticks(range(len(scenes)),scenes,rotation=65,ha='right');ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{dataset}: scene-level metric means (each scene equally weighted)')
    fig.tight_layout();fig.savefig(MET/f'{dataset.lower()}_per_scene_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)

def contact_sheet(dataset,scene,name,gtpath,vanillapath,r4path):
    paths=[gtpath,vanillapath,r4path];labels=['Canonical GT','Vanilla 3DGS','Our R4']
    ims=[]
    for p in paths:
        with Image.open(p) as im: ims.append(im.convert('RGB').copy())
    w,h=ims[0].size
    if any(x.size!=(w,h) for x in ims): raise ValueError(f'{dataset}/{scene}/{name} image sizes differ')
    title_h=32; canvas=Image.new('RGB',(w*3,h+title_h),'white');draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(ims,labels)):
        draw.text((i*w+8,8),label,fill='black');canvas.paste(im,(i*w,title_h))
    path=MET/f'{dataset.lower()}_{scene}_render_comparison.png';canvas.save(path);return path

def make_tables(dtu,llff):
    dtus=[];llffs=[]
    for method,v in dtu['metrics'].items():
        if method=='delta_our_R4_minus_vanilla_3DGS':continue
        dtus.append((method,v))
    for method,v in llff['metrics'].items():
        if method=='delta_our_R4_minus_vanilla_3DGS':continue
        llffs.append((method,v))
    text=['# 三视角 MVS 初始化下的原始 3DGS 对照：结果总结与分析','',
          '## 对照目的','',
          '本次在 15 个 DTU scan 和 8 个 LLFF 场景上补跑 Graphdeco 原始 3DGS：每场景严格 3 张训练图、同一份 MVS 初始点云、seed 2024、10,000 次迭代。主对比对象是同项目、同初始化的 `our_method`（R4）。', '',
          '两行之间的训练视角、测试视角、输入像素、MVS 点云 hash、随机种子、迭代数和场景聚合方式均核对一致。方法本身不同：vanilla 使用原始 RGB L1 + DSSIM 和标准 densification；R4 还使用 pose-only 伪目标与深度/伪视角训练逻辑。因此差异描述为完整方法包的配对差异，不能仅凭这两行归因到一个特定损失，也不构成多 seed 显著性检验。', '',
          '## 汇总指标','',
          '数值为“先对每场景的测试视角取均值、再对场景等权平均”；± 后为 15 个 DTU scan 或 8 个 LLFF 场景之间的样本标准差，不是随机种子标准差。', '',
          '| 数据集 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','|---|---|---:|---:|---:|']
    for dataset,arr in [('DTU',dtus),('LLFF',llffs)]:
        for method,v in arr:text.append(f"| {dataset} | {method} | {f(v['PSNR']['mean'])} ± {f(v['PSNR']['scene_sample_std'])} | {f(v['SSIM']['mean'])} ± {f(v['SSIM']['scene_sample_std'])} | {f(v['LPIPS']['mean'])} ± {f(v['LPIPS']['scene_sample_std'])} |")
    old_dtu=json.loads((Path('/data/lilinwei/3D/1006difix/1007fsgsmvs/dtu/output/metrics/summary_corgs25_mask.json')).read_text())
    all_dtu=[('Vanilla 3DGS（本次新增）',dtu['metrics']['vanilla_3DGS'])]
    all_dtu.extend((name,old_dtu['summary'][name]) for name in ['FSGS','CoR-GS','DropoutGS','DNGaussian','Difix3D_10k'])
    all_dtu.append(('Our R4',dtu['metrics']['our_R4']))
    text += ['', '## DTU 同一 mask 口径下的完整方法背景','',
             '下表将此前六种方法（五个外部 baseline 与 R4）的修正后 CoR-GS 25-view mask 结果，与本次新增 vanilla 3DGS 并列。逐场景核对运行 manifest 后，七种方法均使用相同三视角训练划分、逐字节相同的 MVS 初始化和 seed 2024，并按同一逐物体 mask 公式计算；R4 额外使用 pose-only 相机位姿先验。', '',
             '| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','|---|---:|---:|---:|']
    for name,v in all_dtu:
        text.append(f"| {name} | {f(v['PSNR']['mean'])} ± {f(v['PSNR']['scene_sample_std'])} | {f(v['SSIM']['mean'])} ± {f(v['SSIM']['scene_sample_std'])} | {f(v['LPIPS']['mean'])} ± {f(v['LPIPS']['scene_sample_std'])} |")
    text += ['', '## 配对差值（R4 − Vanilla 3DGS）','',
             '| 数据集 | ΔPSNR | ΔSSIM | ΔLPIPS | R4 在更多场景占优的指标 |','|---|---:|---:|---:|---|']
    for dataset,s in [('DTU',dtu),('LLFF',llff)]:
        ds=s['metrics']['delta_our_R4_minus_vanilla_3DGS']
        wins={}
        for metric in METRICS:
            wins[metric]=sum(1 for r in s['per_scene'] if r['method']=='our_R4' and any(q['scene']==r['scene'] and q['method']=='vanilla_3DGS' and ((r[metric]>q[metric]) if metric!='LPIPS' else (r[metric]<q[metric])) for q in s['per_scene']))
        better=', '.join(f'{m} {wins[m]}/{len([r for r in s["per_scene"] if r["method"]=="our_R4"])}' for m in METRICS)
        text.append(f"| {dataset} | {ds['PSNR']:+.4f} | {ds['SSIM']:+.4f} | {ds['LPIPS']:+.4f} | {better} |")
    paper_dtu={'PSNR':17.65,'SSIM':0.816,'LPIPS':0.146}
    local_dtu={m:dtu['metrics']['vanilla_3DGS'][m]['mean'] for m in METRICS}
    text += ['', '![DTU 与 LLFF 汇总图](metrics/macro_comparison.png)','',
             '逐场景曲线：`metrics/dtu_per_scene_comparison.png`、`metrics/llff_per_scene_comparison.png`。代表性渲染：`metrics/dtu_scan8_render_comparison.png`、`metrics/llff_fern_render_comparison.png`。完整逐场景与逐视角 CSV/JSON 在 `metrics/`。','',
             '## 评测口径','',
             '- **DTU**：15 scans × 25 个 CoR-GS 官方测试视角；PSNR 使用严格前景 mask，SSIM/LPIPS 按 CoR-GS 对 mask 外置白的实现。不能与历史 31-view 未掩码结果或其他全图口径直接混合。','- **LLFF**：8 场景，共 43 个 LLFF holdout 测试视角；canonical GT 为 `images_4`，指标为 RGB PSNR、skimage SSIM 和 VGG LPIPS。','- 训练图像只来自三个指定训练视角。R4 使用每场景 8 个额外相机位姿作为 pose-only 伪目标，没有使用这些测试视角的 RGB；因此 R4 接收了 vanilla 没有的相机位姿先验。','- 初始化云均来自当前 `1007fsgsmvs` 实验的 MVS PLY。LLFF 这一批输入与另一个较早的 `FSGS_LLFF_data/*/3_views/dense/fused.ply` 不同（点数及 SHA 均不同）；因此旧 LLFF baseline 数字不并入本报告主表。','',
             f"CoR-GS 论文 Table 3 中 3DGS 的 DTU 数值为 `17.65 / 0.816 / 0.146`；本次 vanilla 为 `{local_dtu['PSNR']:.4f} / {local_dtu['SSIM']:.4f} / {local_dtu['LPIPS']:.4f}`，数值接近，可作为评测口径的合理性检查。该对照使用本项目指定的 10k 预算和共享初始化，不能称为对论文 vanilla 配置的逐项复现。", '',
             '## 消融结论边界','',
             '本轮只补跑 vanilla 3DGS，没有新增 R4 单因素消融。vanilla 是“完整方法包关闭”的控制线，不等同于任何一项独立消融。预先定义的 RGB 伪监督、伪深度、真实深度、伪标签历史和目标位姿数量消融逻辑见 [`docs/消融逻辑与对比边界.md`](../docs/消融逻辑与对比边界.md)；不得把计划中的变体描述为已有结果。', '',
             '## 可复现文件','',
             '- 训练代码副本及上游 commit、本地改动清单：`code/graphdeco-3dgs/`、`code/source_manifest.json`。','- 每场景初始/最终 Gaussian 点云、所有测试渲染、运行参数与输入 hash：`DTU/<scan>/`、`LLFF/<scene>/`。','- 数据划分和训练视角：`data/splits/`；初始 MVS PLY 副本：`data/initialization/`。','- 协议与 GPU、临时存储约定：[`docs/运行与指标协议.md`](../docs/运行与指标协议.md)。','']
    (OUT/'结果总结与分析.md').write_text('\n'.join(text),encoding='utf-8')

def main():
    dtu=json.loads((MET/'dtu_summary.json').read_text());llff=json.loads((MET/'llff_summary.json').read_text())
    plot_summary(dtu,llff);plot_scenes('DTU',dtu);plot_scenes('LLFF',llff)
    firstdt=json.loads((ROOT/'data/splits/DTU/scan8/scan8.json').read_text())['test_names'][0]
    firstll=json.loads((ROOT/'data/splits/LLFF/fern/fern.json').read_text())['test_names'][0]
    dm=json.loads((ROOT/'data/splits/DTU/scan8/scan8.json').read_text())
    lm=json.loads((ROOT/'data/splits/LLFF/fern/fern.json').read_text())
    contact_sheet('DTU','scan8',firstdt,Path(dm['canonical_gt_dir'])/firstdt,OUT/'DTU/scan8/renders'/firstdt,Path('/data/lilinwei/3D/1006difix/1007fsgsmvs/dtu/output/our_method/scan8/renders')/firstdt)
    stem=Path(firstll).stem
    contact_sheet('LLFF','fern',firstll,Path(lm['image_mapping'][firstll]),OUT/'LLFF/fern/renders'/f'{stem}.png',Path('/data/lilinwei/3D/1006difix/1007fsgsmvs/output/fern/test/ours_10000/renders')/f'{stem}.png')
    make_tables(dtu,llff)
    print('WROTE',OUT/'结果总结与分析.md')
if __name__=='__main__':main()
