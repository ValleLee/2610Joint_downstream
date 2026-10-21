#!/usr/bin/env python3
"""Assemble the paper package from the selected local runs.

Run on the original workstation. Large artifacts are hard linked because /data
does not support reflinks and is nearly full. They remain ordinary Git files.
Do not modify the source artifacts in place after assembling this package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = Path('/data/lilinwei/3D')
LLFF_BATCH = SRC / 'baselines/output/1007_fsgs_mvs_baselines'
OURS = SRC / '1006difix/1007fsgsmvs'
DTU = OURS / 'dtu'
GRAPH = OURS / '3dGS'
SCENES = ('fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex')
SCANS = (8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114)
LLFF_METHODS = ('FSGS', 'CoR-GS', 'DNGaussian', 'DropoutGS', 'Difix3D_60k')
MAIN_DTU = ('vanilla_3DGS', 'CoR-GS', 'DNGaussian', 'DropoutGS', 'our_R4')
DIAGNOSTIC_DTU = ('FSGS', 'DNGaussian', 'DropoutGS', 'Difix3D_10k')
PROVENANCE = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def move_old(source: str, target: str) -> None:
    src, dst = ROOT / source, ROOT / target
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def small_copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def linked(source: Path, target: Path, publish: bool) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
    if source.stat().st_ino != target.stat().st_ino or source.stat().st_dev != target.stat().st_dev:
        raise RuntimeError(f'Expected identical hard link: {target}')
    record = {
        'path': str(target.relative_to(ROOT)), 'source': str(source),
        'bytes': source.stat().st_size, 'sha256': sha256(source),
        'publish_on_git': publish,
    }
    PROVENANCE.append(record)
    return record


def relocate_code() -> None:
    for method in LLFF_METHODS:
        move_old(f'code/llff/methods/{method}', f'code/compare/{method}/llff_mvs10k/source')
        move_old(f'configs/llff/methods/{method}', f'code/compare/{method}/llff_mvs10k/config')
        move_old(f'results/llff/methods/{method}', f'output/llff/{method}/metrics')
    difix_src = LLFF_BATCH / 'Difix3D_60k'
    difix_dst = ROOT / 'code/compare/Difix3D_60k/llff_mvs_60k'
    if not (difix_dst / 'source').exists():
        shutil.copytree(difix_src / 'code', difix_dst / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if not (difix_dst / 'config').exists():
        shutil.copytree(difix_src / 'config', difix_dst / 'config')
    for name in ('run_scene.sh', 'worker.sh', 'compute_metrics.py'):
        small_copy(difix_src / 'configs' / name, difix_dst / 'runner' / name)
    for path in (difix_src / 'data_mapping').glob('*.json'):
        small_copy(path, difix_dst / 'data_mapping' / path.name)
    for path in (difix_src / 'metrics').glob('*.json'):
        small_copy(path, ROOT / 'output/llff/Difix3D_60k/metrics' / path.name)
    for name in ('DIFIX3D_60K_METRICS.csv', 'DIFIX3D_60K_METRICS.json'):
        small_copy(LLFF_BATCH / name, ROOT / 'output/llff/source_metrics' / name)
    move_old('code/compare/Difix3D_10k/llff_mvs10k/source/upstream', 'code/compare/Difix3D_60k/upstream')
    stale_code = ROOT / 'code/compare/Difix3D_10k'
    if stale_code.exists():
        shutil.rmtree(stale_code)
    stale_output = ROOT / 'output/llff/Difix3D_10k'
    if stale_output.exists():
        shutil.rmtree(stale_output)
    move_old('code/graphdeco_3dgs', 'code/compare/vanilla_3DGS/source')
    move_old('code/graphdeco_scripts', 'code/compare/vanilla_3DGS/runner')
    move_old('code/dtu/methods/CoR-GS', 'code/compare/CoR-GS/dtu_author/source')
    move_old('code/dtu/official_random_6k', 'code/compare/_dtu_author_runner')
    move_old('code/r4/fsgs_base', 'code/ours/reconstruction/fsgs_base')
    move_old('code/r4/llff', 'code/ours/llff_mvs10k')
    move_old('code/r4/teacher_inference', 'code/ours/predictor')
    move_old('code/dtu/r4_support', 'code/ours/reconstruction/dtu_support')
    move_old('code/llff/runner_templates', 'code/compare/_llff_shared/runner_templates')
    move_old('code/llff/compute_unified_metrics_original.py', 'code/evaluation/llff/compute_unified_metrics_original.py')
    for name in ('evaluate_corgs25_mask.py', 'derive_corgs_25view_metrics.py'):
        move_old(f'code/dtu/{name}', f'code/evaluation/dtu/{name}')
    for name in ('train_r4_memory_dtu.py', 'fsgs_r4_memory_teacher.py', 'fsgs_r4_mipnerf_depth.py', 'prepare_r4_pose_targets.py'):
        move_old(f'code/dtu/{name}', f'code/ours/dtu/{name}')
    for name in ('train_scene_dtu.py', 'run_scene_mvs.py', 'run_paper_native_profiles.py', 'prepare_method_inputs.py', 'prepare_overlay.py', 'mvs_colmap_dtu.py', 'prepare_splits.py'):
        move_old(f'code/dtu/{name}', f'code/compare/_dtu_shared/{name}')
    move_old('configs/llff', 'data/llff/protocol')
    move_old('configs/dtu', 'data/dtu/protocol')
    move_old('configs/r4', 'code/ours/predictor/config')
    move_old('code/ours/predictor/config/teacher_run_config.yaml', 'code/ours/predictor/config/predictor_run_config.yaml')
    move_old('code/ours/predictor/config/teacher_checkpoint', 'code/ours/predictor/config/predictor_checkpoint')
    move_old('configs/graphdeco_source_manifest.json', 'code/compare/vanilla_3DGS/source_manifest.json')
    move_old('configs/graphdeco_experiment_config.json', 'code/compare/vanilla_3DGS/experiment_config.json')
    move_old('configs/main_table_protocols.json', 'data/main_table_protocols.json')
    move_old('results/llff/COMPARISON_WITH_1007FSGSMVS.csv', 'output/llff/source_metrics/COMPARISON_WITH_1007FSGSMVS.csv')
    move_old('results/llff/COMPARISON_WITH_1007FSGSMVS.json', 'output/llff/source_metrics/COMPARISON_WITH_1007FSGSMVS.json')
    move_old('results/llff/UNIFIED_METRICS.csv', 'output/llff/source_metrics/UNIFIED_METRICS.csv')
    move_old('results/llff/UNIFIED_METRICS.json', 'output/llff/source_metrics/UNIFIED_METRICS.json')
    move_old('results/llff/crosscheck', 'output/llff/source_metrics/crosscheck')
    move_old('results/dtu/graphdeco', 'output/dtu/source_metrics/graphdeco')
    move_old('results/dtu/mvs10k', 'output/dtu/source_metrics/mvs10k')
    move_old('results/dtu/official_random_6k', 'output/dtu/source_metrics/official_random_6k')
    move_old('results/dtu/audits', 'output/dtu/source_metrics/audits')
    # Original tarballs are local archival inputs. Extract the executable source
    # into the public method branches, omitting datasets, assets, binaries, and Git.
    for method, archive_name in (('DNGaussian', 'DNGaussian_22a588c.tar.gz'), ('DropoutGS', 'DropoutGS_5ff8d1a.tar.gz')):
        archive = ROOT / 'code/compare/_dtu_author_runner/repositories' / archive_name
        out = ROOT / 'code/compare' / method / 'dtu_author/source'
        if not archive.is_file():
            raise FileNotFoundError(archive)
        if not out.exists():
            out.mkdir(parents=True)
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    path = Path(member.name)
                    if not member.isfile() or path.is_absolute() or '..' in path.parts:
                        continue
                    if any(p in {'.git', 'assets', 'data', 'output', 'outputs', 'checkpoints', 'weights', '__pycache__'} for p in path.parts):
                        continue
                    if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.pdf', '.ply', '.pth', '.pt', '.so', '.npy', '.npz'}:
                        continue
                    if member.size > 2_000_000:
                        continue
                    target = out / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as stream, target.open('wb') as dst:
                        shutil.copyfileobj(stream, dst)
    for method in ('FSGS', 'DNGaussian', 'DropoutGS', 'Difix3D_10k'):
        src = DTU / 'code/methods' / ('Difix3D' if method == 'Difix3D_10k' else method)
        dst = ROOT / 'code/compare' / method / 'dtu_diagnostic/source'
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.so', '*.pth', '*.pt', '*.ply', '*.png', 'assets', 'data', 'output'))


def scene_files(dataset: str, method: str, scene: str, source: Path, main: bool = True) -> None:
    base = ROOT / 'output' / dataset / (method if main else f'_diagnostics/{method}') / scene
    base.mkdir(parents=True, exist_ok=True)
    files = []
    render_src = source / ('test/ours_10000/renders' if dataset == 'llff' and method == 'our_R4' else 'eval/ours_60000/renders' if dataset == 'llff' and method == 'Difix3D_60k' else 'eval/ours_10000/renders' if dataset == 'llff' and method not in ('vanilla_3DGS',) else 'renders')
    if dataset == 'dtu':
        render_src = source / 'renders'
    if not render_src.is_dir():
        raise FileNotFoundError(render_src)
    renders = sorted(p for p in render_src.glob('*.png') if not p.name.endswith('_depth.png'))
    if not renders:
        raise RuntimeError(f'No RGB renders: {render_src}')
    for path in renders:
        files.append(linked(path, base / 'renders' / path.name, True))
    small_names = ('run_manifest.json', 'metrics_corgs25_mask.json', 'metrics.json', 'data_mapping.json', 'cfg_args', 'cfg.yml', 'cameras.json', 'data_split.json', 'joint_summary.json', 'per_view.json', 'results.json', 'manifest.json', 'opt_args')
    for name in small_names:
        if (source / name).is_file():
            target = base / ('metrics' if 'metrics' in name or name in ('joint_summary.json', 'per_view.json', 'results.json') else 'config' if name in ('cfg_args', 'cfg.yml', 'opt_args') else '') / name
            small_copy(source / name, target)
            files.append({'path': str(target.relative_to(ROOT)), 'source': str(source / name), 'bytes': target.stat().st_size, 'sha256': sha256(target), 'publish_on_git': True})
    if dataset == 'llff' and not (base / 'run_manifest.json').exists():
        inputs = json.loads((ROOT / 'data/llff/protocol/input_manifest.json').read_text())['scenes'][scene]
        config = ('code/compare/Difix3D_60k/llff_mvs_60k/config/experiment_config.json' if method == 'Difix3D_60k'
                  else 'code/ours/llff_mvs10k/config.json' if method == 'our_R4'
                  else f'code/compare/{method}/llff_mvs10k/config/experiment_config.json')
        normalized = {
            'assembled_manifest': True, 'dataset': 'LLFF', 'method': method, 'scene': scene,
            'seed': 2024, 'iterations': 60000 if method == 'Difix3D_60k' else 10000,
            'training_rgb_count': 3, 'evaluation_rgb_count': len(inputs['test_names']),
            'train_names': inputs['train_names'], 'test_names': inputs['test_names'],
            'initial_cloud_sha256': inputs['mvs_sha256'], 'source_config': config,
            'source_output': str(source),
        }
        target = base / 'run_manifest.json'
        target.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + '\n')
    if dataset == 'llff':
        target = base / 'run_manifest.json'
        if not any(item['path'] == str(target.relative_to(ROOT)) for item in files):
            files.append({'path': str(target.relative_to(ROOT)), 'source': 'derived from data/llff/protocol/input_manifest.json and method config', 'bytes': target.stat().st_size, 'sha256': sha256(target), 'publish_on_git': True})
    if dataset == 'llff':
        if method == 'vanilla_3DGS':
            local = [('final_point_cloud.ply', 'point_cloud/final_point_cloud.ply')]
        else:
            point_iter = 60000 if method == 'Difix3D_60k' else 10000
            local = [(f'point_cloud/iteration_{point_iter}/point_cloud.ply', 'point_cloud/point_cloud.ply')]
            if method == 'CoR-GS':
                local.append(('point_cloud_gs2/iteration_10000/point_cloud.ply', 'point_cloud/point_cloud_gs2.ply'))
            for name in ('chkpnt10000.pth', 'ckpt_10000_rank0.pt', 'ckpt_60000_rank0.pt'):
                if (source / name).is_file():
                    local.append((name, f'checkpoints/{name}'))
    elif method in ('DNGaussian', 'DropoutGS') and main:
        local = [('point_cloud/point_cloud.ply', 'point_cloud/point_cloud.ply'), ('checkpoint_6000.pth.gz', 'checkpoints/checkpoint_6000.pth.gz')]
    else:
        local = [('final_point_cloud.ply', 'point_cloud/final_point_cloud.ply'), ('final_point_cloud.ply.gz', 'point_cloud/final_point_cloud.ply.gz')]
    for src_rel, dst_rel in local:
        path = source / src_rel
        if path.is_file():
            files.append(linked(path, base / dst_rel, False))
    (base / 'artifact_manifest.json').write_text(json.dumps({'dataset': dataset, 'method': method, 'scene': scene, 'main_table': main, 'rgb_render_count': len(renders), 'files': files}, ensure_ascii=False, indent=2) + '\n')


def assemble_outputs() -> None:
    for scene in SCENES:
        scene_files('llff', 'vanilla_3DGS', scene, GRAPH / 'output/LLFF' / scene)
        for method in LLFF_METHODS:
            scene_files('llff', method, scene, LLFF_BATCH / ('Difix3D_60k' if method == 'Difix3D_60k' else method) / 'outputs' / scene)
        scene_files('llff', 'our_R4', scene, OURS / 'output' / scene)
    for scan in SCANS:
        name = f'scan{scan}'
        scene_files('dtu', 'vanilla_3DGS', name, GRAPH / 'output/DTU' / name)
        for method in ('CoR-GS', 'our_R4'):
            original = 'our_method' if method == 'our_R4' else method
            scene_files('dtu', method, name, DTU / 'output' / original / name)
        for method in ('DNGaussian', 'DropoutGS'):
            scene_files('dtu', method, name, DTU / 'output/official_random_6k' / method / name)
        for method in DIAGNOSTIC_DTU:
            scene_files('dtu', f'{method}_mvs10k', name, DTU / 'output' / method / name, False)


def summaries() -> None:
    llff_csv = ROOT / 'output/llff/source_metrics/COMPARISON_WITH_1007FSGSMVS.csv'
    with llff_csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cross = json.loads((ROOT / 'output/llff/source_metrics/crosscheck/vanilla_ours_common_4.json').read_text())['results']['vanilla_3DGS']
    macro = cross.get('macro_average', cross.get('macro'))
    if macro is None:
        raise RuntimeError(f'No Graphdeco cross-check macro: {cross.keys()}')
    vanilla = {'method': 'vanilla_3DGS', 'n_scenes': 8, 'n_views': 43, **{k: macro[k] for k in ('PSNR', 'SSIM', 'LPIPS')}}
    ordered = [vanilla] + [{k: ('our_R4' if k == 'method' and row[k] == '1007fsgsmvs (ours)' else row[k]) for k in row} for row in rows if row['method'] != 'Difix3D_10k']
    difix = next(row for row in csv.DictReader((ROOT / 'output/llff/source_metrics/DIFIX3D_60K_METRICS.csv').open(newline='')) if row['step'] == '60000')
    ordered.insert(-1, {'method': 'Difix3D_60k', 'n_scenes': difix['n_scenes'], 'n_views': difix['n_views'], **{k: difix[k] for k in ('PSNR', 'SSIM', 'LPIPS')}})
    with (ROOT / 'output/llff/summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('method', 'n_scenes', 'n_views', 'PSNR', 'SSIM', 'LPIPS'))
        writer.writeheader(); writer.writerows(ordered)
    dtu_graph = json.loads((ROOT / 'output/dtu/source_metrics/graphdeco/dtu_summary.json').read_text())['metrics']
    dtu_mvs = json.loads((ROOT / 'output/dtu/source_metrics/mvs10k/summary_corgs25_mask.json').read_text())['summary']
    dtu_author = json.loads((ROOT / 'output/dtu/source_metrics/official_random_6k/summary_corgs25_mask.json').read_text())['summary']
    with (ROOT / 'output/dtu/summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('method', 'protocol', 'n_scans', 'views_per_scan', 'PSNR_mean', 'PSNR_scene_std', 'SSIM_mean', 'SSIM_scene_std', 'LPIPS_mean', 'LPIPS_scene_std'))
        for method, protocol, record in (
            ('vanilla_3DGS', 'mvs10k', dtu_graph['vanilla_3DGS']),
            ('CoR-GS', 'mvs10k_author', dtu_mvs['CoR-GS']),
            ('DNGaussian', 'rand6k_author', dtu_author['DNGaussian']),
            ('DropoutGS', 'rand6k_author', dtu_author['DropoutGS']),
            ('our_R4', 'mvs10k', dtu_graph['our_R4']),
        ):
            writer.writerow((method, protocol, 15, 25, *[record[k][field] for k in ('PSNR', 'SSIM', 'LPIPS') for field in ('mean', 'scene_sample_std')]))
    with (ROOT / 'output/dtu/_diagnostics/summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(('method', 'protocol', 'PSNR', 'SSIM', 'LPIPS'))
        for method in DIAGNOSTIC_DTU:
            record = dtu_mvs[method]
            writer.writerow((f'{method}_mvs10k', 'mvs10k_local_adaptation', *[record[k]['mean'] for k in ('PSNR', 'SSIM', 'LPIPS')]))


def main() -> None:
    relocate_code()
    assemble_outputs()
    summaries()
    from derive_scene_summaries import main as derive_scene_summaries
    derive_scene_summaries()
    (ROOT / 'SOURCE_MANIFEST.json').write_text(json.dumps({'description': 'Original workstation artifacts; large artifacts are local hard links. publish_on_git marks intended release files.', 'files': PROVENANCE}, ensure_ascii=False, indent=2) + '\n')
    print(f'Assembled {len(PROVENANCE)} source files, {sum(r["bytes"] for r in PROVENANCE) / 1e9:.2f} GB logical size')


if __name__ == '__main__':
    main()
