#!/usr/bin/env python3
"""运行 Graphdeco 原始 3DGS 的一个冻结三视角 / MVS 场景。"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/data/lilinwei/3D/1006difix/1007fsgsmvs')
DTU = PROJECT / 'dtu'
TEMP = Path('/tmp/lilinwei/1007fsgsmvs_3dgs')
STAGE = TEMP / 'data'
RUNS = TEMP / 'runs'
LLFF_ROOT = Path('/data/lilinwei/3D/D23E/Aref/FSGS_LLFF_data')
LLFF_INIT = PROJECT / 'output/initialization/fsgs_official_mvs'
DTU_INIT = DTU / 'output/initialization'
SCENES = {
    'DTU': [f'scan{x}' for x in [8,21,30,31,34,38,40,41,45,55,63,82,103,110,114]],
    'LLFF': ['fern','flower','fortress','horns','leaves','orchids','room','trex'],
}
ALLOWED_GPUS = {0,1,2,3,4,5,7}
PYTHON = Path('/data/lilinwei/.conda/envs/d23e/bin/python')
REPO = ROOT / 'code/graphdeco-3dgs'
EXT_ROOT = Path('/data/lilinwei/3D/D23E/tools/gaussian-splatting/submodules')


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def ply_vertices(path: Path) -> int:
    with path.open('rb') as stream:
        for raw in stream:
            line=raw.decode('ascii','ignore').strip()
            if line.startswith('element vertex '): return int(line.split()[-1])
            if line=='end_header': break
    raise ValueError(f'PLY header has no vertex count: {path}')


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    dst.symlink_to(src.resolve())


def stage_scene(dataset: str, scene: str) -> tuple[Path,dict]:
    root = STAGE / dataset / scene
    if root.exists(): shutil.rmtree(root)
    sparse = root / 'sparse/0'
    images = root / 'images'
    sparse.mkdir(parents=True); images.mkdir(parents=True)
    if dataset == 'DTU':
        overlay = Path('/tmp/lilinwei/1007fsgsmvs_dtu/overlay/DTU') / scene
        image_dir = overlay / 'images_4'
        split_doc = json.loads((DTU / f'data/splits/cor_gs_3view/{scene}.json').read_text())
        camera_names = split_doc['camera_names']
        train_names = list(split_doc['train_names'])
        reserved = {25,22,28,40,44,48,0,8,13}
        excluded = set(json.loads((DTU/'code/experiment_config.json').read_text())['excluded_indices'])
        test_names = [n for i,n in enumerate(camera_names) if i not in reserved | excluded]
        initial = DTU_INIT / scene / 'initial_point_cloud.ply'
        source_sparse = overlay / 'sparse/0'
        input_resolution = 1
        gt_base = DTU / 'output/ground_truth' / scene
        if len(test_names) != 25: raise ValueError(f'{scene}: CoR-GS protocol expects 25 views')
        source_map = {name: image_dir/name for name in sorted(p.name for p in image_dir.glob('*') if p.is_file())}
    elif dataset == 'LLFF':
        source = LLFF_ROOT / scene
        report = json.loads((source/'fsgs_do3_prepare_report.json').read_text())
        image_dir = source/'images_4'
        train_names = report['fsgs_train_names']
        test_names = report['fsgs_test_names']
        source_sparse = source/'sparse/0'
        initial = LLFF_INIT / scene / 'initial_point_cloud.ply'
        input_resolution = 1
        gt_base = PROJECT / 'output' / scene / 'test/ours_10000/gt'
        sys.path.insert(0, str(REPO))
        from scene.colmap_loader import read_extrinsics_binary
        extr = read_extrinsics_binary(str(source_sparse/'images.bin'))
        cam_order = [extr[k].name for k in sorted(extr)]
        rgb_order = sorted(p.name for p in image_dir.glob('*.png'))
        if len(cam_order) != len(rgb_order): raise ValueError(f'{scene}: camera / images_4 count mismatch')
        source_map = dict(zip(cam_order, [image_dir/n for n in rgb_order]))
    else: raise ValueError(dataset)

    if not initial.is_file(): raise FileNotFoundError(initial)
    if len(train_names) != 3 or set(train_names) & set(test_names):
        raise ValueError(f'{dataset}/{scene}: invalid frozen split')
    for name, src in source_map.items():
        if src.is_file(): link(src, images/name)
    for key in ('cameras.bin','images.bin','cameras.txt','images.txt'):
        src=source_sparse/key
        if src.is_file(): link(src, sparse/key)
    # Keep the original Graphdeco scene loader and GaussianModel; its initial PLY is the shared three-view MVS cloud.
    link(initial, sparse/'points3D.ply')
    split = {'dataset':dataset,'scene':scene,'train_names':train_names,'test_names':test_names,
             'all_camera_names':sorted(source_map),'image_mapping':{k:str(v) for k,v in source_map.items()},
             'initial_cloud':str(initial),'initial_cloud_sha256':sha(initial),'initial_point_count':ply_vertices(initial),'canonical_gt_dir':str(gt_base),
             'resolution':input_resolution,'seed':2024,'iterations':10000}
    (sparse/'frozen_split.json').write_text(json.dumps(split,indent=2,ensure_ascii=False)+'\n')
    copied = ROOT/'data/initialization'/dataset/scene/'initial_point_cloud.ply'
    copied.parent.mkdir(parents=True,exist_ok=True)
    if not copied.exists(): shutil.copy2(initial,copied)
    if sha(copied)!=split['initial_cloud_sha256']: raise ValueError(f'{scene}: archived MVS hash mismatch')
    if len([n for n in train_names if (images/n).is_file()]) != 3: raise ValueError(f'{scene}: missing train RGB')
    if len([n for n in test_names if (images/n).is_file()]) != len(test_names): raise ValueError(f'{scene}: missing eval RGB')
    return root,split


def run(dataset: str, scene: str, gpu: int) -> None:
    if gpu not in ALLOWED_GPUS: raise ValueError('Only GPUs 0-5 and 7 are allowed; GPU 6 is unavailable')
    source, mapping=stage_scene(dataset,scene)
    model=RUNS/dataset/scene
    logdir=ROOT/'output'/dataset/scene/'logs'
    logdir.mkdir(parents=True,exist_ok=True); model.parent.mkdir(parents=True,exist_ok=True)
    log_path=logdir/'train.log'
    env=os.environ.copy()
    ext_paths=[EXT_ROOT/'diff-gaussian-rasterization',EXT_ROOT/'simple-knn',REPO]
    env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONHASHSEED':'2024','BASELINE_SEED':'2024',
                'OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','PYTHONNOUSERSITE':'1',
                'PYTHONPATH':os.pathsep.join(map(str,ext_paths))})
    base=[str(PYTHON),str(REPO/'train.py'),'--source_path',str(source),'--model_path',str(model),
          '--images','images','--resolution','1','--eval','--iterations','10000',
          '--position_lr_max_steps','10000','--densify_until_iter','10000',
          '--test_iterations','10000','--save_iterations','10000','--checkpoint_iterations','10000',
          '--disable_viewer','--quiet']
    render=[str(PYTHON),str(REPO/'render.py'),'--source_path',str(source),'--model_path',str(model),
            '--images','images','--resolution','1','--eval','--iteration','10000','--skip_train','--quiet']
    t0=time.time()
    with log_path.open('w',encoding='utf-8') as log:
        log.write(f'DATASET={dataset} SCENE={scene} GPU={gpu} SEED=2024\n')
        log.write(f'INITIAL_PLY={mapping["initial_cloud"]}\nINITIAL_SHA256={mapping["initial_cloud_sha256"]}\n')
        log.write('TRAIN_COMMAND='+' '.join(base)+'\n'); log.flush()
        subprocess.run(base,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        log.write('\nRENDER_COMMAND='+' '.join(render)+'\n'); log.flush()
        subprocess.run(render,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        elapsed=time.time()-t0
        final=model/'point_cloud/iteration_10000/point_cloud.ply'
        render_dir=model/'test/ours_10000/renders'; gt_dir=model/'test/ours_10000/gt'
        if not final.is_file(): raise FileNotFoundError(final)
        preds=sorted(render_dir.glob('*.png')); gts=sorted(gt_dir.glob('*.png'))
        if len(preds)!=len(mapping['test_names']) or len(gts)!=len(preds):
            raise ValueError(f'{scene}: rendered {len(preds)} test views, expected {len(mapping["test_names"])}')
        # Persist the final model and renders only; keep all bulky intermediate training artifacts in /tmp.
        out=ROOT/'output'/dataset/scene
        out.mkdir(parents=True,exist_ok=True)
        out_renders=out/'renders'; out_renders.mkdir(exist_ok=True)
        from PIL import Image
        import numpy as np
        for i,name in enumerate(mapping['test_names']):
            pred=render_dir/f'{i:05d}.png'; gt=gt_dir/f'{i:05d}.png'
            if not pred.is_file() or not gt.is_file(): raise FileNotFoundError(f'{scene}: missing indexed render {i}')
            canonical=Path(mapping['image_mapping'][name]) if dataset=='LLFF' else Path(mapping['canonical_gt_dir'])/name
            with Image.open(gt) as a, Image.open(canonical) as b:
                if a.size != b.size or not np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB'))):
                    raise ValueError(f'{scene}/{name}: renderer GT differs from canonical image pixels')
            shutil.copy2(pred,out_renders/(Path(name).stem+'.png'))
        shutil.copy2(final,out/'final_point_cloud.ply')
        shutil.copy2(source/'sparse/0/points3D.ply',out/'initial_point_cloud.ply')
        shutil.copy2(model/'cfg_args',out/'cfg_args')
        (out/'run_manifest.json').write_text(json.dumps({**mapping,'gpu':gpu,'physical_gpu':gpu,
            'elapsed_seconds':elapsed,'final_point_cloud_sha256':sha(out/'final_point_cloud.ply'),
            'rendered_views':len(preds),'source_repo':'graphdeco-inria/gaussian-splatting',
            'initial_point_count':mapping['initial_point_count'],
            'source_commit':'54c035f','source_tree':'copied locally; local adapter only freezes train/test names and points images to images_4'},
            indent=2,ensure_ascii=False)+'\n')
        shutil.copy2(log_path,out/'train.log')
        # Training model/checkpoints remain temporary until metrics are written; cleanup is performed by finalize.py.
        print(f'COMPLETE {dataset}/{scene} gpu={gpu} seconds={elapsed:.1f}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=SCENES,required=True);p.add_argument('--scene',required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args()
    if a.scene not in SCENES[a.dataset]: raise ValueError(f'{a.scene} is not a configured {a.dataset} scene')
    run(a.dataset,a.scene,a.gpu)
if __name__=='__main__': main()
