#!/usr/bin/env python3
"""读取 CoR-GS 处理包中的 COLMAP 相机并冻结 DTU 3/15/31 split。"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SCANS=['scan8','scan21','scan30','scan31','scan34','scan38','scan40','scan41','scan45','scan55','scan63','scan82','scan103','scan110','scan114']
TRAIN=[25,22,28]
EXCLUDED=[3,4,5,6,7,16,17,18,19,20,21,36,37,38,39]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def read_cameras(path):
 out={}
 for line in path.read_text().splitlines():
  s=line.strip()
  if not s or s.startswith('#'): continue
  x=s.split(); out[int(x[0])]={'model':x[1],'width':int(x[2]),'height':int(x[3]),'params':[float(v) for v in x[4:]]}
 return out

def read_image_records(path):
 out=[]
 for line in path.read_text().splitlines():
  s=line.strip()
  if not s or s.startswith('#'): continue
  x=s.split()
  if len(x)<10 or Path(x[-1]).suffix.lower() not in {'.png','.jpg','.jpeg'}: continue
  try:
   image_id=int(x[0]); q=[float(v) for v in x[1:5]]; t=[float(v) for v in x[5:8]]; cid=int(x[8])
  except ValueError: continue
  out.append({'image_id':image_id,'qvec_wxyz':q,'tvec':t,'camera_id':cid,'name':x[9]})
 return sorted(out,key=lambda v:v['name'])

def make(root:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True)
 all_scenes=[]
 for scan in SCANS:
  scene=root/scan; images_dir=scene/'images'; sparse=scene/'sparse/0'
  cams=read_cameras(sparse/'cameras.txt'); records=read_image_records(sparse/'images.txt')
  disk=sorted(p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'})
  names=[r['name'] for r in records]
  if len(names)!=49 or len(set(names))!=49 or names!=disk: raise ValueError(f'{scan}: 相机模型/图像文件不一致 ({len(names)} vs {len(disk)})')
  if any(r['camera_id'] not in cams for r in records): raise ValueError(f'{scan}: camera_id 不存在')
  train_names=[names[i] for i in range(49) if i in TRAIN]
  excluded_names=[names[i] for i in range(49) if i in EXCLUDED]
  test_names=[names[i] for i in range(49) if i not in TRAIN+EXCLUDED]
  expected_train=sorted([names[i] for i in TRAIN])
  if train_names!=expected_train or len(excluded_names)!=15 or len(test_names)!=31: raise ValueError(f'{scan}: 协议分割错误')
  data={'dataset':'DTU CoR-GS processed package','scan':scan,'camera_order':'lexicographic image filename, as in CoR-GS readDTUSceneInfo',
    'camera_count':49,'camera_names':names,'cameras':cams,'image_records':records,
    'train_indices':TRAIN,'train_names':train_names,'train_image_sha256':{n:sha(images_dir/n) for n in train_names},
    'excluded_indices':EXCLUDED,'excluded_names':excluded_names,'test_indices':[i for i in range(49) if i not in TRAIN+EXCLUDED],
    'test_names':test_names,'test_count':31,'mask_status':'pending alternate mask package inspection',
    'source_scene':str(scene),'source_sparse_sha256':{n:sha(sparse/n) for n in ('cameras.txt','images.txt','cameras.bin','images.bin') if (sparse/n).is_file()},
    'corgs_bundle_3view_cloud':str(scene/'3_views/dense/fused.ply'),'corgs_bundle_3view_cloud_sha256':sha(scene/'3_views/dense/fused.ply') if (scene/'3_views/dense/fused.ply').is_file() else None,
    'note':'本项目仍需重新生成本场景唯一共享 MVS PLY；bundle 点云仅作为来源记录，未经 hash/视角核验不作为训练输入。'}
  (out/f'{scan}.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
  all_scenes.append({'scan':scan,'camera_count':len(names),'train_count':len(train_names),'excluded_count':len(excluded_names),'test_count':len(test_names),'split':str(out/f'{scan}.json')})
 (out.parent/'dataset_manifest.json').write_text(json.dumps({'dataset':'DTU','source_package':'CoR-GS DTU 3/6/9-view bundle','source_root':str(root),'scans':all_scenes,'split_protocol':'3 training indices [25,22,28], 15 excluded, remaining 31 test'},indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'data/splits/cor_gs_3view'); a=p.parse_args(); make(a.dataset_root,a.output)
