#!/usr/bin/env python3
"""从已有 31 视角指标导出 CoR-GS 原始代码的 25 测试视角指标。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
CONFIG = json.loads((CODE / "experiment_config.json").read_text())
PAPER_RESERVED = {25, 22, 28, 40, 44, 48, 0, 8, 13}
EXCLUDED = set(CONFIG["excluded_indices"])
METRICS = ("PSNR", "SSIM", "LPIPS")


def main() -> None:
    rows = []
    extra_names_by_scan = {}
    for scan in CONFIG["scans"]:
        split_path = PROJECT / "data/splits/cor_gs_3view" / f"{scan}.json"
        split = json.loads(split_path.read_text())
        names = sorted(record["name"] for record in split["image_records"])
        assert len(names) == 49 and len(set(names)) == 49
        official_test = {name for i, name in enumerate(names)
                         if i not in PAPER_RESERVED | EXCLUDED}
        current_test = set(split["test_names"])
        extra = current_test - official_test
        assert len(official_test) == 25 and len(current_test) == 31
        assert len(extra) == 6 and official_test < current_test
        assert set(split["train_names"]) == {names[i] for i in (25, 22, 28)}
        extra_names_by_scan[scan] = sorted(extra)

        for method in CONFIG["methods"]:
            path = PROJECT / "output" / method / scan / "metrics.json"
            recorded = json.loads(path.read_text())
            by_name = {row["view"]: row for row in recorded["per_view"]}
            assert len(by_name) == 31 and set(by_name) == current_test
            selected = [by_name[name] for name in sorted(official_test)]
            rows.append({"method": method, "scan": scan,
                         "view_count": len(selected),
                         **{metric: mean(row[metric] for row in selected)
                            for metric in METRICS}})

    output = PROJECT / "output/metrics"
    output.mkdir(exist_ok=True)
    with (output / "per_scene_corgs_25view_unmasked.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "scan", "view_count", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in CONFIG["methods"]:
        subset = [row for row in rows if row["method"] == method]
        assert len(subset) == len(CONFIG["scans"])
        summary[method] = {
            metric: {"mean": mean(row[metric] for row in subset),
                     "scene_sample_std": stdev(row[metric] for row in subset)}
            for metric in METRICS
        }
    payload = {
        "protocol": "CoR-GS upstream DTU reader 3-view split: 25 test views",
        "masking": "unmasked RGB; still not directly comparable to paper Table 3",
        "source_metrics": "existing output/<method>/<scan>/metrics.json from 31-view evaluation",
        "training_indices": [25, 22, 28],
        "reserved_nine_indices": sorted(PAPER_RESERVED),
        "excluded_fifteen_indices": sorted(EXCLUDED),
        "extra_six_in_previous_31_view_protocol": sorted(PAPER_RESERVED - {25, 22, 28}),
        "extra_six_names_by_scan": extra_names_by_scan,
        "scene_count": len(CONFIG["scans"]),
        "test_views_per_scene": 25,
        "summary": summary,
    }
    (output / "summary_corgs_25view_unmasked.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for method, values in summary.items():
        print(method, " ".join(f"{key}={values[key]['mean']:.4f}" for key in METRICS))


if __name__ == "__main__":
    main()
