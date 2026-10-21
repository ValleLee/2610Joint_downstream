#!/usr/bin/env python3
"""Make a small, uniform per-scene metrics index from archived source results."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('PSNR', 'SSIM', 'LPIPS')
LLFF = ('fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex')
DTU = tuple(f'scan{x}' for x in (8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114))


def load(rel):
    return json.loads((ROOT / rel).read_text())


def rows(rel):
    with (ROOT / rel).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_scene(base: Path, value: dict) -> None:
    path = base / 'metrics/summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    manifest = base / 'artifact_manifest.json'
    record = json.loads(manifest.read_text())
    rel = str(path.relative_to(ROOT))
    record['files'] = [item for item in record['files'] if item['path'] != rel]
    record['files'].append({'path': rel, 'source': value['derived_from'], 'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'publish_on_git': True})
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    cross = load('output/llff/source_metrics/crosscheck/vanilla_ours_common_4.json')['results']['vanilla_3DGS']['scenes']
    for method in ('vanilla_3DGS', 'FSGS', 'CoR-GS', 'DNGaussian', 'DropoutGS', 'Difix3D_60k', 'our_R4'):
        if method == 'vanilla_3DGS':
            source = 'output/llff/source_metrics/crosscheck/vanilla_ours_common_4.json'
            per_scene = cross
        elif method == 'Difix3D_60k':
            source = 'output/llff/Difix3D_60k/metrics/public_common_eval_60000.json'
            per_scene = load(source)['results']
        elif method == 'our_R4':
            source = 'output/llff/our_R4/<scene>/metrics/per_view.json'
            per_scene = {}
            for scene in LLFF:
                data = load(source.replace('<scene>', scene))['ours_10000']
                per_scene[scene] = {'n_views': len(data['PSNR']), **{m: statistics.mean(data[m].values()) for m in METRICS}}
        else:
            source = f'output/llff/{method}/metrics/public_common_eval.json'
            per_scene = load(source)['results']
        for scene in LLFF:
            record = per_scene[scene]
            write_scene(ROOT / 'output/llff' / method / scene,
                        {'dataset': 'llff', 'method': method, 'scene': scene, 'evaluated_views': int(record['n_views']),
                         'PSNR': record['PSNR'], 'SSIM': record['SSIM'], 'LPIPS': record['LPIPS'],
                         'derived_from': source.replace('<scene>', scene)})
        (ROOT / 'output/llff' / method / 'summary.json').write_text(json.dumps(
            {'dataset': 'llff', 'method': method, 'scenes': list(LLFF),
             'macro': {m: statistics.mean(per_scene[s][m] for s in LLFF) for m in METRICS},
             'derived_from': source}, ensure_ascii=False, indent=2) + '\n')
    dtu_sources = (
        ('output/dtu/source_metrics/graphdeco/dtu_per_scene.csv', ('vanilla_3DGS', 'our_R4'), False),
        ('output/dtu/source_metrics/mvs10k/per_scene_corgs25_mask.csv', ('CoR-GS',), False),
        ('output/dtu/source_metrics/official_random_6k/per_scene_corgs25_mask.csv', ('DNGaussian', 'DropoutGS'), False),
        ('output/dtu/source_metrics/mvs10k/per_scene_corgs25_mask.csv', ('FSGS', 'DNGaussian', 'DropoutGS', 'Difix3D_10k'), True),
    )
    for source, methods, diagnostic in dtu_sources:
        all_rows = rows(source)
        for method in methods:
            output_method = f'{method}_mvs10k' if diagnostic else method
            selected = {row.get('scan', row.get('scene')): row for row in all_rows if row['method'] == method}
            if set(selected) != set(DTU):
                raise ValueError(f'incomplete DTU coverage: {output_method}')
            parent = ROOT / 'output/dtu' / ('_diagnostics' if diagnostic else '') / output_method
            for scene in DTU:
                record = selected[scene]
                write_scene(parent / scene,
                            {'dataset': 'dtu', 'method': output_method, 'scene': scene, 'evaluated_views': int(record['view_count']),
                             'PSNR': float(record['PSNR']), 'SSIM': float(record['SSIM']), 'LPIPS': float(record['LPIPS']),
                             'derived_from': source})
            (parent / 'summary.json').write_text(json.dumps(
                {'dataset': 'dtu', 'method': output_method, 'scenes': list(DTU),
                 'macro': {m: statistics.mean(float(selected[s][m]) for s in DTU) for m in METRICS},
                 'scene_sample_std': {m: statistics.stdev(float(selected[s][m]) for s in DTU) for m in METRICS},
                 'derived_from': source}, ensure_ascii=False, indent=2) + '\n')
    print('Derived uniform per-scene metrics for 56 LLFF and 135 DTU runs')


if __name__ == '__main__':
    main()
