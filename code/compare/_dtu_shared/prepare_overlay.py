#!/usr/bin/env python3
"""给所有方法构建同一数据视图与共享 MVS PLY 的临时 overlay。"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

PROJECT=Path(__file__).resolve().parent.parent
TEMP=Path('/tmp/lilinwei/1007fsgsmvs_dtu')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def prepare(scan,source_root,cloud):
 split_p=PROJECT/'data/splits/cor_gs_3view'/f'{scan}.json'
 cloud_m=PROJECT/'output/initialization'/scan/'manifest.json'
 if not split_p.is_file() or not cloud_m.is_file() or not cloud.is_file(): raise FileNotFoundError('缺少冻结 split 或归档 MVS PLY')
 split=json.loads(split_p.read_text()); init=json.loads(cloud_m.read_text())
 cloud_hash=sha(cloud)
 if init.get('status')!='complete' or init.get('sha256')!=cloud_hash: raise ValueError('MVS 点云与已归档清单 hash 不一致')
 if init.get('training_names')!=split['train_names']: raise ValueError('MVS 训练图名与 split 不一致')
 src=source_root/scan
 if len(list((src/'images').glob('*')))!=49: raise ValueError(f'{scan}: 原始图像目录不是 49 张')
 overlay=TEMP/'overlay'/'DTU'/scan
 manifest=overlay/'dtu_overlay_manifest.json'
 if overlay.exists():
  if manifest.is_file():
   old=json.loads(manifest.read_text())
   if old.get('shared_mvs_sha256')==cloud_hash and old.get('split_sha256')==sha(split_p): return overlay
  raise FileExistsError(f'已有 overlay 与当前点云不匹配，拒绝覆盖: {overlay}')
 overlay.parent.mkdir(parents=True,exist_ok=True); overlay.mkdir()
 for child in src.iterdir():
  if child.name in {'3_views','6_views','9_views'}: continue
  (overlay/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
 dense=overlay/'3_views/dense'; dense.mkdir(parents=True)
 (dense/'fused.ply').symlink_to(cloud.resolve())
 manifest.write_text(json.dumps({'scan':scan,'source':str(src),'split':str(split_p),'split_sha256':sha(split_p),
  'train_names':split['train_names'],'test_names':split['test_names'],'shared_mvs':str(cloud),
  'shared_mvs_sha256':cloud_hash,'blocked_alternate_clouds':['6_views','9_views','bundled_3_views']},indent=2,ensure_ascii=False)+'\n')
 return overlay

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--scan',required=True); p.add_argument('--source-root',type=Path,required=True); p.add_argument('--cloud',type=Path,required=True)
 a=p.parse_args(); print(prepare(a.scan,a.source_root,a.cloud))
