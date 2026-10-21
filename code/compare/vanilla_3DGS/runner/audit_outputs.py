#!/usr/bin/env python3
"""核查最终场景、训练视角、MVS 初始化 hash、点云与渲染完整性。"""
import csv,hashlib,json
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/data/lilinwei/3D/1006difix/1007fsgsmvs')
DTU_SCENES=[f'scan{x}' for x in [8,21,30,31,34,38,40,41,45,55,63,82,103,110,114]]
LLFF_SCENES=['fern','flower','fortress','horns','leaves','orchids','room','trex']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def ply_vertices(path):
 with path.open('rb') as f:
  for line in f:
   if line.startswith(b'element vertex '): return int(line.split()[-1])
   if line.strip()==b'end_header': break
 raise ValueError(f'No PLY vertex count in {path}')

rows=[]
for dataset,scenes in [('DTU',DTU_SCENES),('LLFF',LLFF_SCENES)]:
 for scene in scenes:
  out=ROOT/'output'/dataset/scene
  split=json.loads((ROOT/'data/splits'/dataset/scene/f'{scene}.json').read_text())
  run=json.loads((out/'run_manifest.json').read_text())
  assert run['gpu'] in {0,1,2,3,4,5,7},(dataset,scene,run['gpu'])
  assert run['seed']==2024 and run['iterations']==10000,(dataset,scene)
  assert run.get('training_arguments_file')=='cfg_args' and (out/'cfg_args').is_file(),(dataset,scene,'missing archived training arguments')
  assert run['train_names']==split['train_names'] and run['test_names']==split['test_names'],(dataset,scene,'split mismatch')
  init=out/'initial_point_cloud.ply'; init_copy=ROOT/'data/initialization'/dataset/scene/'initial_point_cloud.ply'
  assert init.is_file() and init_copy.is_file()
  assert sha(init)==sha(init_copy)==run['initial_cloud_sha256'],(dataset,scene,'MVS hash mismatch')
  if dataset=='DTU':
   expected=PROJECT/'dtu/output/initialization'/scene/'initial_point_cloud.ply'
  else:
   expected=PROJECT/'output/initialization/fsgs_official_mvs'/scene/'initial_point_cloud.ply'
  assert sha(expected)==run['initial_cloud_sha256'],(dataset,scene,'does not match current R4 input')
  final=out/'final_point_cloud.ply'; assert final.is_file() and sha(final)==run['final_point_cloud_sha256']
  initial_points=ply_vertices(init)
  assert initial_points==run['initial_point_count'],(dataset,scene,'initial point count mismatch')
  renders=sorted((out/'renders').glob('*.png'))
  assert len(renders)==len(split['test_names'])==run['rendered_views'],(dataset,scene,len(renders),len(split['test_names']))
  for name in split['test_names']:
   p=out/'renders'/f'{Path(name).stem}.png';assert p.is_file(),(dataset,scene,name)
   if dataset=='DTU': gt=PROJECT/'dtu/output/ground_truth'/scene/name
   else: gt=Path(split['image_mapping'][name])
   with Image.open(p) as a,Image.open(gt) as b: assert a.size==b.size,(dataset,scene,name,a.size,b.size)
  log=(out/'train.log').read_text(errors='replace')
  assert '10000/10000' in log and '100%|' in log,(dataset,scene,'training did not reach the 10k step')
  rows.append({'dataset':dataset,'scene':scene,'train_views':len(split['train_names']),'test_views':len(split['test_names']),
               'gpu':run['gpu'],'initial_points':initial_points,
               'initial_sha256':run['initial_cloud_sha256'],'final_points_sha256':run['final_point_cloud_sha256'],
               'render_count':len(renders),'elapsed_seconds':run['elapsed_seconds']})
print('AUDIT_PASS scenes=',len(rows),'DTU=',sum(r['dataset']=='DTU' for r in rows),'LLFF=',sum(r['dataset']=='LLFF' for r in rows))
print('test renders=',sum(r['render_count'] for r in rows),'all initial MVS hashes match current R4 inputs')
metric_checks={}
for dataset,scene_count,view_count in [('DTU',15,25),('LLFF',8,43)]:
 met=ROOT/'output/metrics'
 with (met/f'{dataset.lower()}_per_scene.csv').open(newline='') as f: scene_rows=list(csv.DictReader(f))
 with (met/f'{dataset.lower()}_per_view.csv').open(newline='') as f: view_rows=list(csv.DictReader(f))
 summary=json.loads((met/f'{dataset.lower()}_summary.json').read_text())
 assert len(scene_rows)==scene_count*2,(dataset,'unexpected per-scene metric row count')
 expected_views=(scene_count*view_count if dataset=='DTU' else view_count)*2
 assert len(view_rows)==expected_views,(dataset,'unexpected per-view metric row count')
 assert {r['method'] for r in scene_rows}=={'vanilla_3DGS','our_R4'}
 assert set(summary['metrics']) >= {'vanilla_3DGS','our_R4','delta_our_R4_minus_vanilla_3DGS'}
 metric_checks[dataset]={'scene_metric_rows':len(scene_rows),'view_metric_rows':len(view_rows)}
for name in ['macro_comparison.png','dtu_per_scene_comparison.png','llff_per_scene_comparison.png',
             'dtu_scan8_render_comparison.png','llff_fern_render_comparison.png']:
 p=ROOT/'output/metrics'/name
 assert p.is_file() and p.stat().st_size>0,('missing metric visualization',name)
prior_method_checks={'methods':{},'scans_checked':15}
prior_root=PROJECT/'dtu/output'
for method in ['FSGS','CoR-GS','DropoutGS','DNGaussian','Difix3D_10k','our_method']:
 checked=0
 for scan in DTU_SCENES:
  current=json.loads((ROOT/'output/DTU'/scan/'run_manifest.json').read_text())
  prior=json.loads((prior_root/method/scan/'run_manifest.json').read_text())
  assert prior['seed']==current['seed']==2024,(method,scan,'seed mismatch')
  assert prior['iterations']==current['iterations']==10000,(method,scan,'iteration mismatch')
  assert prior['train_names']==current['train_names'],(method,scan,'training split mismatch')
  assert prior['initial_cloud_sha256']==current['initial_cloud_sha256'],(method,scan,'MVS initialization mismatch')
  checked+=1
 prior_method_checks['methods'][method]={'scans_verified':checked,'seed':2024,'iterations':10000,
      'same_three_view_training_split':True,'same_mvs_initialization_sha256':True}
(ROOT/'output/audit_manifest.json').write_text(json.dumps({'status':'pass','scene_count':len(rows),'test_render_count':sum(r['render_count'] for r in rows),'metric_checks':metric_checks,'prior_dtu_method_parity':prior_method_checks,'scenes':rows},indent=2,ensure_ascii=False)+'\n')
