#!/usr/bin/env python3
"""Recompute archived RGB metrics from final renders and separately obtained GT.

Requires Pillow, NumPy, PyTorch and the original metric dependencies. It writes
only to --out, leaving the archived measurements unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
MASK_SHA256 = '988f1a1957083191b45f7d4ea55a0b74566fc4286f1c224be9d19aa2a143ffbd'
METRICS = ('PSNR', 'SSIM', 'LPIPS')


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rgb(path: Path, np, Image):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0


def llff_pairs(method: str, scene: str):
    base = ROOT / 'output/llff' / method / scene
    run = load(base / 'run_manifest.json')
    renders = {p.name.lower(): p for p in (base / 'renders').glob('*.png')}
    pairs = []
    for index, test_name in enumerate(run['test_names']):
        stem = Path(test_name).stem.lower()
        choices = (f'{stem}.png', f'{index:05d}.png', f'{index:04d}.png', f'{index}.png')
        matches = [renders[name] for name in choices if name in renders]
        matches = list({p.name: p for p in matches}.values())
        if len(matches) != 1:
            raise ValueError(f'{method}/{scene}: cannot map {test_name} to exactly one render')
        pairs.append((test_name, matches[0]))
    if {path.name for _, path in pairs} != {path.name for path in renders.values()}:
        raise ValueError(f'{method}/{scene}: extra render files')
    return pairs


def evaluate_llff(args, np, Image, torch):
    source = ROOT / 'code/ours/reconstruction/fsgs_base'
    sys.path.insert(0, str(source))
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips
    out = {}
    for scene in args.select or ('fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'):
        views = []
        for test_name, render in llff_pairs(args.method, scene):
            gt = args.gt_root / scene / (Path(test_name).stem + '.png')
            pred_arr, gt_arr = rgb(render, np, Image), rgb(gt, np, Image)
            if pred_arr.shape != gt_arr.shape:
                raise ValueError(f'{scene}/{test_name}: image shape mismatch')
            a = torch.from_numpy(pred_arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
            b = torch.from_numpy(gt_arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
            with torch.no_grad():
                views.append({'view': test_name, 'render': render.name,
                              'PSNR': float(psnr(a, b).mean()), 'SSIM': float(ssim(a, b)),
                              'LPIPS': float(lpips(a, b, net_type='vgg'))})
        out[scene] = {'n_views': len(views), 'mean': {m: statistics.mean(v[m] for v in views) for m in METRICS}, 'per_view': views}
    return out


def evaluate_dtu(args, np, Image, torch):
    if args.mask_zip is None or sha256(args.mask_zip) != MASK_SHA256:
        raise ValueError('DTU requires the CoR-GS 25-view mask ZIP with the recorded SHA-256')
    source = ROOT / 'code/compare/CoR-GS/dtu_author/source'
    sys.path.insert(0, str(source))
    from utils.loss_utils import ssim
    from lpipsPyTorch.modules.lpips import LPIPS
    model = LPIPS('vgg', '0.1').to(args.device).eval()
    out = {}
    vanilla_views = None
    if args.method == 'vanilla_3DGS':
        with (ROOT / 'output/dtu/source_metrics/graphdeco/dtu_per_view.csv').open(newline='') as stream:
            vanilla_views = [row for row in csv.DictReader(stream) if row['method'] == 'vanilla_3DGS']
    with ZipFile(args.mask_zip) as archive:
        for scan in args.select or tuple(f'scan{n}' for n in (8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114)):
            base = ROOT / 'output/dtu' / args.method / scan
            if vanilla_views is not None:
                eval_views = [row for row in vanilla_views if row['scene'] == scan]
            else:
                stored = load(base / 'metrics/metrics_corgs25_mask.json')
                if stored['mask_archive_sha256'] != MASK_SHA256:
                    raise ValueError(f'{scan}: mask mismatch')
                eval_views = stored['per_view']
            if len(eval_views) != 25:
                raise ValueError(f'{scan}: expected 25 evaluation views')
            views = []
            for item in eval_views:
                name = item['view']
                pred = rgb(base / 'renders' / name, np, Image)
                gt = rgb(args.gt_root / scan / name, np, Image)
                if pred.shape != gt.shape:
                    raise ValueError(f'{scan}/{name}: image shape mismatch')
                with Image.open(BytesIO(archive.read(item['mask_member']))) as image:
                    mask = np.asarray(image.resize((gt.shape[1], gt.shape[0])), dtype=np.float32)[:, :, :3] / 255.0
                foreground = mask == 1.0
                mse = float(np.mean((pred[foreground] - gt[foreground]) ** 2))
                if mse <= 0:
                    raise ValueError(f'{scan}/{name}: invalid foreground MSE')
                a = torch.from_numpy((pred * mask + (1.0 - mask)).transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
                b = torch.from_numpy((gt * mask + (1.0 - mask)).transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    views.append({'view': name, 'PSNR': -10.0 * math.log10(mse),
                                  'SSIM': float(ssim(a, b).item()), 'LPIPS': float(model(a, b).mean().item())})
            out[scan] = {'n_views': len(views), 'mean': {m: statistics.mean(v[m] for v in views) for m in METRICS}, 'per_view': views}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('llff', 'dtu'))
    parser.add_argument('method', help='directory name under output/<dataset>/')
    parser.add_argument('--gt-root', type=Path, required=True, help='LLFF: <scene>/<test>.png; DTU: <scan>/<view>.png')
    parser.add_argument('--mask-zip', type=Path, help='required for DTU; CoR-GS 25-view masks')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--select', nargs='+', help='optional scenes/scans for a targeted check')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    if args.dataset == 'dtu' and args.method not in ('vanilla_3DGS', 'CoR-GS', 'DNGaussian', 'DropoutGS', 'our_R4'):
        raise ValueError('choose a DTU main-table method directory')
    if args.dataset == 'llff' and args.method not in ('vanilla_3DGS', 'FSGS', 'CoR-GS', 'DNGaussian', 'DropoutGS', 'Difix3D_60k', 'our_R4'):
        raise ValueError('choose an LLFF main-table method directory')
    scenes = evaluate_llff(args, np, Image, torch) if args.dataset == 'llff' else evaluate_dtu(args, np, Image, torch)
    result = {'dataset': args.dataset, 'method': args.method, 'scenes': scenes,
              'macro': {m: statistics.mean(v['mean'][m] for v in scenes.values()) for m in METRICS}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['macro'], indent=2))


if __name__ == '__main__':
    main()
