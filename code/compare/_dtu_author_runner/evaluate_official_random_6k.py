#!/usr/bin/env python3
"""按 CoR-GS 原始 DTU mask 指标评估官方随机初始化重跑结果。"""
import csv
import argparse
import json
import statistics
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

BASE = Path('/data/lilinwei/3D/1006difix/1007fsgsmvs/dtu')
OUT = BASE / 'output/official_random_6k'
COMMON = BASE / 'code'
sys.path.insert(0, str(COMMON))
import evaluate_corgs25_mask as common
sys.path.insert(0, str(COMMON / 'methods/CoR-GS'))
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS

ALL_SCANS = ('scan8','scan21','scan30','scan31','scan34','scan38','scan40','scan41',
             'scan45','scan55','scan63','scan82','scan103','scan110','scan114')
METHODS = ('DNGaussian','DropoutGS')
DEVICE = 'cuda'
lpips_model = LPIPS('vgg','0.1').to(DEVICE).eval()
parser=argparse.ArgumentParser()
parser.add_argument('--scans',nargs='+',choices=ALL_SCANS,default=ALL_SCANS)
SCANS=parser.parse_args().scans

def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0

rows=[]
with ZipFile(common.MASK_ZIP) as archive:
  for scan in SCANS:
    views=[]
    for name in common.official_names(scan):
      gt=rgb(BASE/'output/ground_truth'/scan/name)
      mask=common.load_mask(archive,scan,name,(gt.shape[1],gt.shape[0]))
      views.append((name,gt,mask))
    for method in METHODS:
      per_view=[]
      render_dir=OUT/method/scan/'renders'
      for name,gt,mask in views:
        pred=rgb(render_dir/name)
        if pred.shape != gt.shape: raise ValueError(f'{method}/{scan}/{name}: image shape mismatch')
        mask_bin=(mask==1.0)
        mse=float(np.mean((pred[mask_bin]-gt[mask_bin])**2))
        if mse<=0: raise ValueError(f'{method}/{scan}/{name}: invalid masked MSE')
        pred_white=pred*mask+(1.0-mask)
        gt_white=gt*mask+(1.0-mask)
        a=torch.from_numpy(pred_white.transpose(2,0,1).copy()).unsqueeze(0).to(DEVICE)
        b=torch.from_numpy(gt_white.transpose(2,0,1).copy()).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
          ss=float(ssim(a,b).item())
          lp=float(lpips_model(a,b).mean().item())
        per_view.append({'view':name,'mask_member':common.mask_path(archive,scan,name),
                         'PSNR':-10*np.log10(mse),'SSIM':ss,'LPIPS':lp})
      row={'method':method,'scan':scan,'view_count':25,
           **{m:statistics.mean(v[m] for v in per_view) for m in ('PSNR','SSIM','LPIPS')}}
      rows.append(row)
      (OUT/method/scan/'metrics_corgs25_mask.json').write_text(json.dumps({
        'protocol':'CoR-GS upstream DTU 25-view + object mask',
        'mask_archive_sha256':common.MASK_SHA256,'mean':{k:row[k] for k in ('PSNR','SSIM','LPIPS')},
        'per_view':per_view},ensure_ascii=False,indent=2)+'\n')
      print(method,scan,*(f'{k}={row[k]:.4f}' for k in ('PSNR','SSIM','LPIPS')),flush=True)

metric_dir=OUT/'metrics'; metric_dir.mkdir(parents=True,exist_ok=True)
with (metric_dir/'per_scene_corgs25_mask.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['method','scan','view_count','PSNR','SSIM','LPIPS'])
  writer.writeheader(); writer.writerows(rows)
summary={'protocol':'CoR-GS source-code DTU split: 25 test views per scan',
 'metric_code':'CoR-GS metrics_dtu.py: foreground PSNR; white-background 3DGS SSIM and VGG LPIPS',
 'mask_archive_sha256':common.MASK_SHA256,'scene_count':15,'views_per_scene':25,
 'summary':{method:{m:{'mean':statistics.mean(r[m] for r in rows if r['method']==method),
                                  'scene_sample_std':statistics.stdev(r[m] for r in rows if r['method']==method) if len(SCANS)>1 else None}
                     for m in ('PSNR','SSIM','LPIPS')} for method in METHODS}}
summary['scene_count']=len(SCANS)
(metric_dir/'summary_corgs25_mask.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('EVALUATION_COMPLETE',flush=True)
