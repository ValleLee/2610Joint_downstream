#!/usr/bin/env python3
"""将官方 DNGaussian DPT 的归一化 16-bit 深度 PNG 转成其 loader 可读的 8-bit PNG。"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

src, dst = map(Path, sys.argv[1:3])
files = sorted(src.glob("depth_*.png"))
if len(files) != 49:
    raise SystemExit(f"期望每个 DTU scan 有 49 张 DPT 深度图，实际为 {len(files)}: {src}")
dst.mkdir(parents=True, exist_ok=True)
rows = []
for path in files:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.dtype != np.uint16 or array.ndim != 2:
        raise SystemExit(f"深度图格式异常: {path}: {array.dtype}, {array.shape}")
    # DNGaussian 的官方 DPT 生成脚本先逐图 min-max 归一化到 uint16；
    # 当前官方 PILtoTorch 只接受 uint8，因此保留相同归一化范围并降到 8-bit。
    depth8 = np.rint(array.astype(np.float32) * (255.0 / 65535.0)).clip(0, 255).astype(np.uint8)
    out = dst / path.name
    Image.fromarray(depth8, mode="L").save(out, compress_level=3)
    rows.append({"name": path.name, "source_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                 "output_sha256": __import__("hashlib").sha256(out.read_bytes()).hexdigest(),
                 "shape": list(depth8.shape), "source_dtype": "uint16", "output_dtype": "uint8"})
if len(list(dst.glob("depth_*.png"))) != 49:
    raise SystemExit("8-bit 深度图数量校验失败")
(dst / "conversion_manifest.json").write_text(json.dumps({
    "conversion": "per-image normalized uint16 DPT PNG to uint8 by round(value*255/65535)",
    "source": str(src), "files": rows
}, ensure_ascii=False, indent=2) + "\n")
print(f"DEPTH8_READY {dst} count={len(rows)}")
