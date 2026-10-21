#!/usr/bin/env python3
"""对齐 DTU CoR-GS mask 指标与 LLFF public-common 指标，比较 vanilla 3DGS 和 R4。"""
from __future__ import annotations
import csv, hashlib, json, math, sys
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZipFile
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/data/lilinwei/3D/1006difix/1007fsgsmvs')
DTU=PROJECT/'dtu'
MASK_ZIP=DTU/'data/eval_masks/corgs_25view_masks.zip'
MASK_SHA='988f1a1957083191b45f7d4ea55a0b74566fc4286f1c224be9d19aa2a143ffbd'
SCANS=[f'scan{x}' for x in [8,21,30,31,34,38,40,41,45,55,63,82,103,110,114]]
LLFF=['fern','flower','fortress','horns','leaves','orchids','room','trex']
GPU=4

def rgb(path:Path)->np.ndarray:
    with Image.open(path) as im: return np.asarray(im.convert('RGB'),dtype=np.uint8)

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dtu_names(scan):
    split=json.loads((DTU/'data/splits/cor_gs_3view'/f'{scan}.json').read_text())
    cfg=json.loads((DTU/'code/experiment_config.json').read_text())
    reserved={25,22,28,40,44,48,0,8,13}; excluded=set(cfg['excluded_indices'])
    names=split['camera_names']
    return [n for i,n in enumerate(names) if i not in reserved|excluded]

def dtu_mask(z:ZipFile,scan,name,size):
    frame=int(name.split('_')[1])-1
    prefix=f'submission_data/idrmasks/{scan}/'
    candidates=[prefix+f'{frame:03d}.png',prefix+'mask/'+f'{frame:03d}.png']
    member=[x for x in candidates if x in z.NameToInfo]
    if len(member)!=1: raise ValueError(f'{scan}/{name}: mask candidates {member}')
    with Image.open(BytesIO(z.read(member[0]))) as im:
        if im.size!=(1600,1200): raise ValueError(f'{member[0]} mask shape {im.size}')
        return np.asarray(im.resize(size),dtype=np.float32)[:,:,:3]/255.0,member[0]

def stats(rows,metric,method):
    vals=[r[metric] for r in rows if r['method']==method]
    return {'mean':float(mean(vals)),'scene_sample_std':float(stdev(vals)) if len(vals)>1 else 0.0}

def load_lpips():
    corgs=DTU/'code/methods/CoR-GS'
    sys.path.insert(0,str(corgs))
    from lpipsPyTorch.modules.lpips import LPIPS
    model=LPIPS('vgg','0.1').cuda().eval()
    return model

def compute_llff(lpips_model):
    scene_rows=[]; view_rows=[]
    for scene in LLFF:
        manifest=json.loads((ROOT/'output/LLFF'/scene/'run_manifest.json').read_text())
        test=manifest['test_names']; image_map=manifest['image_mapping']
        for method,render_root in [
            ('vanilla_3DGS',ROOT/'output/LLFF'/scene/'renders'),
            ('our_R4',PROJECT/'output'/scene/'test/ours_10000/renders')]:
            vals=[]
            for name in test:
                stem=Path(name).stem
                pred_path=render_root/f'{stem}.png'
                gt_path=Path(image_map[name])
                p0=rgb(pred_path);g0=rgb(gt_path)
                if p0.shape!=g0.shape: raise ValueError(f'LLFF/{scene}/{name}: render/GT mismatch {p0.shape} vs {g0.shape}')
                p=p0.astype(np.float32)/255.;g=g0.astype(np.float32)/255.
                mse=float(np.mean((p-g)**2));psnr=-10*math.log10(max(mse,1e-12))
                ssim=float(structural_similarity(p,g,channel_axis=2,data_range=1.0))
                a=torch.from_numpy(p.transpose(2,0,1).copy()).unsqueeze(0).cuda();b=torch.from_numpy(g.transpose(2,0,1).copy()).unsqueeze(0).cuda()
                with torch.no_grad(): lp=float(lpips_model(a,b).mean().item())
                row={'dataset':'LLFF','scene':scene,'method':method,'view':name,'PSNR':psnr,'SSIM':ssim,'LPIPS':lp}
                vals.append(row);view_rows.append(row)
            scene_rows.append({'dataset':'LLFF','scene':scene,'method':method,'view_count':len(vals),
                               **{m:mean(r[m] for r in vals) for m in ('PSNR','SSIM','LPIPS')}})
        print(f'LLFF {scene} complete',flush=True)
    save_table('LLFF',scene_rows,view_rows)

def compute_dtu(lpips_model):
    if sha(MASK_ZIP)!=MASK_SHA: raise ValueError('mask ZIP SHA256 mismatch')
    scene_rows=[];view_rows=[]
    with ZipFile(MASK_ZIP) as z:
        for scan in SCANS:
            names=dtu_names(scan)
            for name in names:
                gt=rgb(DTU/'output/ground_truth'/scan/name).astype(np.float32)/255.
                mask,member=dtu_mask(z,scan,name,(gt.shape[1],gt.shape[0]))
                binary=mask==1.0
                for method,pred_path in [
                    ('vanilla_3DGS',ROOT/'output/DTU'/scan/'renders'/name),
                    ('our_R4',DTU/'output/our_method'/scan/'renders'/name)]:
                    pred=rgb(pred_path).astype(np.float32)/255.
                    if pred.shape!=gt.shape: raise ValueError(f'DTU/{scan}/{name}: render/GT mismatch')
                    mse=float(np.mean((pred[binary]-gt[binary])**2));psnr=-10*math.log10(max(mse,1e-12))
                    mp=pred*mask+(1-mask);mg=gt*mask+(1-mask)
                    a=torch.from_numpy(mp.transpose(2,0,1).copy()).unsqueeze(0).cuda();b=torch.from_numpy(mg.transpose(2,0,1).copy()).unsqueeze(0).cuda()
                    with torch.no_grad():
                        # Use CoR-GS's own 3DGS-native SSIM implementation, as in its metrics_dtu.py.
                        if 'corgs_ssim' not in globals():
                            from utils.loss_utils import ssim as corgs_ssim
                            globals()['corgs_ssim']=corgs_ssim
                        ssim=float(corgs_ssim(a,b).item());lp=float(lpips_model(a,b).mean().item())
                    row={'dataset':'DTU','scene':scan,'method':method,'view':name,'mask_member':member,
                         'mask_fraction':float(binary.mean()),'PSNR':psnr,'SSIM':ssim,'LPIPS':lp}
                    view_rows.append(row)
            for method in ('vanilla_3DGS','our_R4'):
                subset=[r for r in view_rows if r['scene']==scan and r['method']==method]
                if len(subset)!=25: raise ValueError(f'{scan}/{method}: expected 25 evaluated views; got {len(subset)}')
                scene_rows.append({'dataset':'DTU','scene':scan,'method':method,'view_count':25,
                                   **{m:mean(r[m] for r in subset) for m in ('PSNR','SSIM','LPIPS')}})
            print(f'DTU {scan} complete',flush=True)
    save_table('DTU',scene_rows,view_rows)

def save_table(dataset,scene_rows,view_rows):
    out=ROOT/'output/metrics';out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('per_scene',scene_rows),('per_view',view_rows)]:
        with (out/f'{dataset.lower()}_{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()));writer.writeheader();writer.writerows(rows)
    summary={'dataset':dataset,'aggregation':'mean per scene, then equal-weight macro average over scenes',
             'metrics':{},'per_scene':scene_rows}
    for method in ('vanilla_3DGS','our_R4'):
        subset=[r for r in scene_rows if r['method']==method]
        summary['metrics'][method]={m:stats(subset,m,method) for m in ('PSNR','SSIM','LPIPS')}
    summary['metrics']['delta_our_R4_minus_vanilla_3DGS']={}
    for m in ('PSNR','SSIM','LPIPS'):
        summary['metrics']['delta_our_R4_minus_vanilla_3DGS'][m]=summary['metrics']['our_R4'][m]['mean']-summary['metrics']['vanilla_3DGS'][m]['mean']
    if dataset=='DTU':
        summary['evaluation_protocol']='CoR-GS 25 test views; foreground PSNR (strict mask==1 for RGB channels), CoR-GS 3DGS SSIM and VGG LPIPS after whitening outside the soft mask'
        summary['mask_sha256']=MASK_SHA
    else:
        summary['evaluation_protocol']='LLFF holdout canonical images_4; PSNR, skimage SSIM(channel_axis=2), VGG LPIPS'
    (out/f'{dataset.lower()}_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(dataset,json.dumps(summary['metrics'],ensure_ascii=False),flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['DTU','LLFF','all'],default='all');p.add_argument('--gpu',type=int,default=4);a=p.parse_args()
    if a.gpu==6: raise ValueError('GPU 6 is excluded')
    torch.cuda.set_device(0)
    lpips_model=load_lpips()
    if a.dataset in ('LLFF','all'): compute_llff(lpips_model)
    if a.dataset in ('DTU','all'):
        sys.path.insert(0,str(DTU/'code/methods/CoR-GS'))
        compute_dtu(lpips_model)
if __name__=='__main__': main()
