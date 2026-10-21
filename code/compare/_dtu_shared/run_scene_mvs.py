#!/usr/bin/env python3
"""从 CoR-GS 固定三视角协议生成并归档单场景唯一 MVS 初始化。"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
TEMP = Path('/tmp/lilinwei/1007fsgsmvs_dtu')
OUTPUT = PROJECT / 'output' / 'initialization'
COLMAP = Path('/data/lilinwei/.conda/envs/colmap/bin/colmap')
PYTHON = Path('/data/lilinwei/.conda/envs/d23e/bin/python')
PIPELINE = CODE / 'mvs_colmap_dtu.py'
EXPECTED_TRAIN = [25, 22, 28]
EXCLUDED = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def point_count(path: Path) -> int:
    with path.open('rb') as f:
        for line in f:
            s=line.decode('ascii','replace').strip()
            if s.startswith('element vertex '): return int(s.split()[-1])
            if s=='end_header': break
    raise ValueError(f'PLY 缺少 vertex 数: {path}')


def run(scan: str, gpu: int, dataset_root: Path, allow_gpu: bool) -> Path:
    if not allow_gpu:
        raise RuntimeError('GPU 任务被闸门阻止；确认另一组实验结束后，显式传 --allow-gpu')
    if gpu == 6 or gpu not in range(8): raise ValueError('GPU 6 禁用，只接受 0-5 或 7')
    split_path=PROJECT/'data'/'splits'/'cor_gs_3view'/f'{scan}.json'
    split=json.loads(split_path.read_text())
    if split['train_indices'] != EXPECTED_TRAIN or split['excluded_indices'] != EXCLUDED or len(split['test_names']) != 31:
        raise ValueError(f'{scan} split 不符合已冻结的 CoR-GS 协议')
    source=dataset_root/scan
    if not (source/'sparse/0').is_dir() or not (source/'images').is_dir():
        raise FileNotFoundError(f'数据缺少 sparse/0 或 images: {source}')
    source_names=sorted(p.name for p in (source/'images').iterdir() if p.is_file())
    if source_names != split['camera_names']:
        raise ValueError(f'{scan} 图像名与冻结 49 视角列表不符')
    selected=[name for i,name in enumerate(source_names) if i in EXPECTED_TRAIN]
    if set(selected) != set(split['train_names']):
        raise ValueError(f'{scan} MVS 实际选图与 manifest 不同')

    dest=OUTPUT/scan; dest.mkdir(parents=True,exist_ok=True)
    cloud=dest/'initial_point_cloud.ply'; manifest_path=dest/'manifest.json'
    if cloud.exists() or manifest_path.exists():
        if cloud.is_file() and manifest_path.is_file():
            old=json.loads(manifest_path.read_text())
            if old.get('status')=='complete' and old.get('sha256')==sha256(cloud) and old.get('training_names')==selected:
                return cloud
        raise FileExistsError(f'存在未核验归档，拒绝覆盖: {dest}')

    stamp=time.strftime('%Y%m%d_%H%M%S')
    attempt=TEMP/'mvs_attempts'/scan/f'attempt_{stamp}'
    staged=attempt/'dataset'/scan
    (staged/'sparse/0').mkdir(parents=True)
    (staged/'images').mkdir()
    # 仅拷贝三张选中 RGB；稀疏模型中的其余相机位姿只用于登记编号，
    # actual MVS pipeline receives exactly the three names in created/images.txt.
    for f in (source/'sparse/0').iterdir():
        if f.is_file(): shutil.copy2(f,staged/'sparse/0'/f.name)
    image_hashes={}
    for name in selected:
        src=source/'images'/name
        shutil.copy2(src,staged/'images'/name)
        image_hashes[name]=sha256(src)

    env=os.environ.copy(); env['PATH']=str(COLMAP.parent)+os.pathsep+env.get('PATH','')
    env['CUDA_VISIBLE_DEVICES']=str(gpu); env['OMP_NUM_THREADS']='4'; env['OPENBLAS_NUM_THREADS']='1'
    command=[str(PYTHON),'-u',str(PIPELINE),'--scene',scan,'--base_path',str(attempt/'dataset')+'/', '--n_views','3']
    log=attempt/'mvs.log'; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        f.write(f'gpu={gpu}\nsource={source}\ntrain_indices={EXPECTED_TRAIN}\ntrain_names={json.dumps(selected)}\n')
        f.write('command='+json.dumps(command,ensure_ascii=False)+'\n'); f.flush()
        result=subprocess.run(command,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
    mvs=staged/'3_views'; fused=mvs/'dense/fused.ply'
    generated=[line.split()[9] for line in (mvs/'created/images.txt').read_text().splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
    if result.returncode or set(generated)!=set(selected) or len(generated)!=3:
        raise RuntimeError(f'{scan} MVS 未使用且仅使用三张训练图；详见 {log}')
    if not fused.is_file() or fused.stat().st_size==0:
        raise RuntimeError(f'{scan} MVS 未产生有效 fused.ply；详见 {log}')
    n=point_count(fused)
    if n<=0: raise RuntimeError(f'{scan} MVS 点云为空')
    tmp=cloud.with_suffix('.ply.tmp'); shutil.copy2(fused,tmp); tmp.replace(cloud)
    archived_log=dest/'mvs.log'; shutil.copy2(log,archived_log)
    payload={'scan':scan,'status':'complete','method':'CoR-GS DTU COLMAP PatchMatch pipeline; no random fallback',
      'train_indices':EXPECTED_TRAIN,'training_names':selected,'training_image_sha256':image_hashes,
      'excluded_indices':EXCLUDED,'test_count':31,'generated_mvs_image_names':generated,
      'triangulated_model_pose_count':len(generated),'point_count':n,'point_cloud':str(cloud),
      'sha256':sha256(cloud),'gpu':gpu,'log':str(archived_log),'temporary_log':str(log),
      'split_manifest':str(split_path),'split_manifest_sha256':sha256(split_path),
      'source_dataset_root':str(source),'finished_at':time.strftime('%Y-%m-%d %H:%M:%S %z')}
    manifest_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    return cloud

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scan',required=True)
    p.add_argument('--gpu',required=True,type=int)
    p.add_argument('--dataset-root',required=True,type=Path)
    p.add_argument('--allow-gpu',action='store_true',help='另一组 GPU 实验结束并确认后才可传入')
    a=p.parse_args(); print(run(a.scan,a.gpu,a.dataset_root,a.allow_gpu))
