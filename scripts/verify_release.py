#!/usr/bin/env python3
"""Verify the published evidence and independently recompute both main tables."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('PSNR', 'SSIM', 'LPIPS')
LLFF = ('fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex')
DTU = tuple(f'scan{x}' for x in (8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114))
LLFF_METHODS = ('vanilla_3DGS', 'FSGS', 'CoR-GS', 'DNGaussian', 'DropoutGS', 'Difix3D_60k', 'our_R4')
DTU_METHODS = ('vanilla_3DGS', 'CoR-GS', 'DNGaussian', 'DropoutGS', 'our_R4')
DTU_DIAGNOSTICS = ('FSGS_mvs10k', 'DNGaussian_mvs10k', 'DropoutGS_mvs10k', 'Difix3D_10k_mvs10k')


def read_json(path: str | Path):
    return json.loads((ROOT / path).read_text())


def read_csv(path: str | Path) -> list[dict]:
    with (ROOT / path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(a: float, b: float, label: str, tol: float = 1e-8) -> None:
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol):
        raise AssertionError(f'{label}: {a} != {b}')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts() -> tuple[int, int]:
    manifests = sorted((ROOT / 'output').rglob('artifact_manifest.json'))
    assert len(manifests) == 56 + 135, f'expected 56 LLFF and 135 DTU runs, found {len(manifests)}'
    published = 0
    optional = 0
    all_paths = set()
    for path in manifests:
        record = json.loads(path.read_text())
        base = path.parent
        renders = sorted((base / 'renders').glob('*.png'))
        assert len(renders) == record['rgb_render_count'], path
        assert all(p.is_file() and p.stat().st_size > 0 for p in renders), path
        listed_renders = {ROOT / item['path'] for item in record['files'] if item['path'].endswith('.png')}
        assert set(renders) == listed_renders, path
        for item in record['files']:
            rel = item['path']
            assert rel not in all_paths, f'duplicate artifact path: {rel}'
            all_paths.add(rel)
            target = ROOT / rel
            if not item['publish_on_git'] and not target.exists():
                optional += 1
                continue
            assert target.is_file() and target.stat().st_size == item['bytes'], rel
            assert sha256(target) == item['sha256'], rel
            if item['publish_on_git']:
                published += 1
            else:
                optional += 1
    source = read_json('SOURCE_MANIFEST.json')['files']
    assert len({item['path'] for item in source}) == len(source)
    assert {item['path'] for item in source if item['publish_on_git']} <= all_paths
    for item in source:
        target = ROOT / item['path']
        if target.exists():
            assert target.stat().st_size == item['bytes'], item['path']
        else:
            assert not item['publish_on_git'], item['path']
    return published, optional


def verify_llff() -> None:
    rows = {row['method']: row for row in read_csv('output/llff/summary.csv')}
    assert tuple(rows) == LLFF_METHODS
    inputs = read_json('data/llff/protocol/input_manifest.json')['scenes']
    assert set(inputs) == set(LLFF)
    assert sum(len(inputs[s]['test_names']) for s in LLFF) == 43
    for method in LLFF_METHODS:
        assert int(rows[method]['n_scenes']) == 8 and int(rows[method]['n_views']) == 43
        scene_records = []
        for scene in LLFF:
            base = ROOT / 'output/llff' / method / scene
            assert len(list((base / 'renders').glob('*.png'))) == len(inputs[scene]['test_names']), f'{method}/{scene}'
            normalized = read_json(base / 'metrics/summary.json')
            assert normalized['method'] == method and normalized['scene'] == scene
            assert normalized['evaluated_views'] == len(inputs[scene]['test_names'])
            scene_records.append(normalized)
        for metric in METRICS:
            close(statistics.mean(x[metric] for x in scene_records), rows[method][metric], f'LLFF normalized {method} {metric}', 1e-5)
    common = {row['method']: row for row in read_csv('output/llff/source_metrics/COMPARISON_WITH_1007FSGSMVS.csv')}
    for method in ('FSGS', 'CoR-GS', 'DNGaussian', 'DropoutGS'):
        record = read_json(f'output/llff/{method}/metrics/public_common_eval.json')['results']
        assert set(record) == set(LLFF)
        for metric in METRICS:
            value = statistics.mean(record[s][metric] for s in LLFF)
            close(value, rows[method][metric], f'LLFF {method} {metric}')
            close(value, common[method][metric], f'LLFF original {method} {metric}')
        config = read_json(f'code/compare/{method}/llff_mvs10k/config/experiment_config.json')
        assert config['seed'] == 2024 and config['total_training_steps'] == 10000
        for scene in LLFF:
            assert config['input_manifests'][scene]['sha256'] == inputs[scene]['mvs_sha256']
    difix = read_json('output/llff/Difix3D_60k/metrics/public_common_eval_60000.json')['results']
    assert set(difix) == set(LLFF)
    difix_config = read_json('code/compare/Difix3D_60k/llff_mvs_60k/config/experiment_config.json')
    difix_audit = read_json('code/compare/Difix3D_60k/llff_mvs_60k/config/run_audit.json')
    assert difix_config['seed'] == 2024 and difix_config['base_steps'] == 30000 and difix_config['total_steps'] == 60000
    assert difix_audit['audit']['all_mvs_hashes_match'] and difix_audit['audit']['all_render_gt_counts_match']
    for scene in LLFF:
        assert len(difix[scene]['view_mapping']) == len(inputs[scene]['test_names'])
        assert {x['canonical_test_image'] for x in difix[scene]['view_mapping']} == set(inputs[scene]['test_names'])
    for metric in METRICS:
        close(statistics.mean(difix[s][metric] for s in LLFF), rows['Difix3D_60k'][metric], f'LLFF Difix 60k {metric}')
    vanilla = read_json('output/llff/source_metrics/crosscheck/vanilla_ours_common_4.json')['results']['vanilla_3DGS']['scenes']
    assert set(vanilla) == set(LLFF)
    for metric in METRICS:
        close(statistics.mean(vanilla[s][metric] for s in LLFF), rows['vanilla_3DGS'][metric], f'LLFF 3DGS {metric}')
    for metric in METRICS:
        values = []
        for scene in LLFF:
            per_view = read_json(f'output/llff/our_R4/{scene}/metrics/per_view.json')['ours_10000'][metric]
            assert {Path(x).stem for x in per_view} == {Path(x).stem for x in inputs[scene]['test_names']}
            values.append(statistics.mean(per_view.values()))
        close(statistics.mean(values), rows['our_R4'][metric], f'LLFF R4 {metric}', 1e-5)
    for scene in LLFF:
        gs = read_json(f'output/llff/vanilla_3DGS/{scene}/run_manifest.json')
        assert gs['initial_cloud_sha256'] == inputs[scene]['mvs_sha256']
        assert gs['seed'] == 2024 and gs['iterations'] == 10000
    assert '60k' in (ROOT / 'output/llff/README.md').read_text()


def verify_dtu() -> None:
    rows = {row['method']: row for row in read_csv('output/dtu/summary.csv')}
    assert tuple(rows) == DTU_METHODS
    source_specs = (
        ('output/dtu/source_metrics/graphdeco/dtu_per_scene.csv', ('vanilla_3DGS', 'our_R4')),
        ('output/dtu/source_metrics/mvs10k/per_scene_corgs25_mask.csv', ('CoR-GS',)),
        ('output/dtu/source_metrics/official_random_6k/per_scene_corgs25_mask.csv', ('DNGaussian', 'DropoutGS')),
    )
    for rel, methods in source_specs:
        source = read_csv(rel)
        for method in methods:
            scenes = [row for row in source if row['method'] == method]
            assert {row.get('scan', row.get('scene')) for row in scenes} == set(DTU)
            assert all(int(row['view_count']) == 25 for row in scenes)
            assert int(rows[method]['n_scans']) == 15 and int(rows[method]['views_per_scan']) == 25
            for metric in METRICS:
                scores = [float(row[metric]) for row in scenes]
                close(statistics.mean(scores), rows[method][f'{metric}_mean'], f'DTU {method} {metric}')
                close(statistics.stdev(scores), rows[method][f'{metric}_scene_std'], f'DTU {method} {metric} std')
    for method in DTU_METHODS:
        scene_records = []
        for scan in DTU:
            base = ROOT / 'output/dtu' / method / scan
            expected = 31 if method in ('CoR-GS', 'our_R4') else 25
            assert len(list((base / 'renders').glob('*.png'))) == expected, f'{method}/{scan}'
            normalized = read_json(base / 'metrics/summary.json')
            assert normalized['method'] == method and normalized['scene'] == scan and normalized['evaluated_views'] == 25
            scene_records.append(normalized)
            if method in ('CoR-GS', 'DNGaussian', 'DropoutGS', 'our_R4'):
                metrics = read_json(base / 'metrics/metrics_corgs25_mask.json')
                assert len(metrics['per_view']) == 25
        for metric in METRICS:
            close(statistics.mean(x[metric] for x in scene_records), rows[method][f'{metric}_mean'], f'DTU normalized {method} {metric}')
            close(statistics.stdev(x[metric] for x in scene_records), rows[method][f'{metric}_scene_std'], f'DTU normalized {method} {metric} std')
    for method in DTU_DIAGNOSTICS:
        for scan in DTU:
            base = ROOT / 'output/dtu/_diagnostics' / method / scan
            assert len(list((base / 'renders').glob('*.png'))) == 31
            assert len(read_json(base / 'metrics/metrics_corgs25_mask.json')['per_view']) == 25
    for scan in DTU:
        gs = read_json(f'output/dtu/vanilla_3DGS/{scan}/run_manifest.json')
        cor = read_json(f'output/dtu/CoR-GS/{scan}/run_manifest.json')
        ours = read_json(f'output/dtu/our_R4/{scan}/run_manifest.json')
        assert len({x['initial_cloud_sha256'] for x in (gs, cor, ours)}) == 1
        assert len({tuple(x['train_names']) for x in (gs, cor, ours)}) == 1
        assert all(x['seed'] == 2024 and x['iterations'] == 10000 for x in (gs, cor, ours))
        for method in ('DNGaussian', 'DropoutGS'):
            run = read_json(f'output/dtu/{method}/{scan}/run_manifest.json')
            assert run['official_seed'] == 0 and run['iterations'] == 6000
            assert run['test_view_count'] == run['render_count'] == 25
    assert set(x['method'] for x in read_csv('output/dtu/_diagnostics/summary.csv')) == set(DTU_DIAGNOSTICS)


if __name__ == '__main__':
    published, optional = verify_artifacts()
    verify_llff()
    verify_dtu()
    print(f'OK: {published} published artifact hashes; {optional} optional local files; LLFF 7 methods / 8 scenes / 43 views; DTU 5 main methods + 4 diagnostics / 15 scans / 25 evaluated views')
