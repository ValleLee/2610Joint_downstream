#!/usr/bin/env python3
"""等待其他训练结束，独立复核两方法的 15 场景论文式随机初始化配置。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import threading
import time

from run_corrected_baselines import GPUS, gpu_idle
from train_scene_dtu import PROJECT, TEMP, PY_GS

CODE = PROJECT / "code"
SCANS = json.loads((CODE / "experiment_config.json").read_text())["scans"]
METHODS = ("DNGaussian", "DropoutGS")
OUT = PROJECT / "output/diagnostics"
LOGS = TEMP / "logs/paper_native_6k"
STATUS = TEMP / "status/paper_native_6k.json"
LOCK = threading.Lock()
STATE = {"state": "waiting_for_other_gpu_jobs", "planned": len(SCANS) * len(METHODS),
         "complete": [], "failed": []}


def save_status() -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(STATE, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATUS)


def run_scene(scan: str, available: Queue) -> None:
    gpu = available.get()
    try:
        for method in METHODS:
            dest = OUT / "paper_native_6k" / method / f"{scan}.json"
            if dest.is_file():
                with LOCK:
                    STATE["complete"].append(f"{method}/{scan}")
                    save_status()
                continue
            LOGS.mkdir(parents=True, exist_ok=True)
            cmd = [str(PY_GS), str(CODE / "recheck_native_dtu_hparams.py"),
                   "--method", method, "--scan", scan, "--gpu", str(gpu), "--paper-profile"]
            try:
                with (LOGS / f"{method}_{scan}.out").open("w") as stream:
                    subprocess.run(cmd, cwd=PROJECT.parent, stdout=stream, stderr=subprocess.STDOUT, check=True)
                report_path = TEMP / "paper_dtu_profile" / method / scan / "recheck_result.json"
                report = json.loads(report_path.read_text())
                if (report["initialization"], report["iterations"], report["native_gt_exact"]) != ("random", 6000, 31):
                    raise ValueError(f"{method}/{scan}: 不是经原生 GT 核对的随机初始化 6k 结果")
                print("DIAGNOSTIC", method, scan, f"PSNR={report['masked_psnr_25view']:.4f}",
                      "gpu", gpu, flush=True)
                with LOCK:
                    STATE["complete"].append(f"{method}/{scan}")
                    save_status()
            except Exception as error:
                print("FAILED", method, scan, repr(error), flush=True)
                with LOCK:
                    STATE["failed"].append({"method": method, "scan": scan, "error": str(error)})
                    save_status()
    finally:
        available.put(gpu)


def run_mvs_6k_pilot() -> None:
    """scan8 固定 DTU 超参数和 6k，仅更换随机/MVS 初始化。"""
    STATE["state"] = "controlled_mvs_6k_scan8"
    save_status()
    LOGS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for method, gpu in zip(METHODS, (0, 1)):
            cmd = [str(PY_GS), str(CODE / "recheck_native_dtu_hparams.py"),
                   "--method", method, "--scan", "scan8", "--gpu", str(gpu), "--mvs-6k"]
            log = (LOGS / f"mvs_6k_{method}_scan8.out").open("w")
            future = executor.submit(subprocess.run, cmd, cwd=PROJECT.parent, stdout=log,
                                     stderr=subprocess.STDOUT, check=True)
            futures[future] = (method, log)
        for future in as_completed(futures):
            method, log = futures[future]
            try:
                future.result()
            finally:
                log.close()
            print("CONTROLLED_MVS_6K", method, "scan8", flush=True)
    rows = []
    for method in METHODS:
        mvs = json.loads((TEMP / "mvs_6k" / method / "scan8/recheck_result.json").read_text())
        rnd = json.loads((TEMP / "paper_dtu_profile" / method / "scan8/recheck_result.json").read_text())
        main = json.loads((PROJECT / "output" / method / "scan8/recheck_result.json").read_text())
        if mvs["iterations"] != rnd["iterations"] or mvs["dtu_hparams"][mvs["dtu_hparams"].index("--iterations") + 1] != "6000":
            raise ValueError(f"{method}: 6k 受控对照参数错误")
        rows.append({"method": method, "scan": "scan8", "seed": 2024,
                     "same_test_protocol": "CoR-GS 25-view object-mask foreground PSNR",
                     "random_1000_points_6k": rnd["masked_psnr_25view"],
                     "three_view_mvs_6k": mvs["masked_psnr_25view"],
                     "three_view_mvs_10k": main["masked_psnr_25view"],
                     "random_minus_mvs_6k": rnd["masked_psnr_25view"] - mvs["masked_psnr_25view"],
                     "mvs_10k_minus_6k": main["masked_psnr_25view"] - mvs["masked_psnr_25view"],
                     "mvs_initial_cloud_sha256": mvs["initial_cloud_sha256"],
                     "native_gt_exact": mvs["native_gt_exact"]})
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"scope": "scan8 controlled diagnostic; same native DTU hyperparameters and 6k for random versus shared 3-view MVS",
               "random_box_caveat": "random profile derives its sampling box from provided sparse/0/points3D.bin, which has tracks from 49 cameras in the local scan8 source",
               "rows": rows}
    (OUT / "scan8_初始化与步数受控对照.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# scan8 初始化与训练步数受控对照", "",
             "所有分数均用相同 25 测试视角、canonical GT、mask 和 CoR-GS 前景 PSNR；seed 2024。", "",
             "| 方法 | 随机 1,000 点 / 6k | 三视角 MVS / 6k | 三视角 MVS / 10k |",
             "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['random_1000_points_6k']:.4f} | "
                     f"{row['three_view_mvs_6k']:.4f} | {row['three_view_mvs_10k']:.4f} |")
    lines += ["", "随机 6k 与 MVS 6k 只改初始化来源；两种 MVS 设置只改训练时长及迭代相关的学习率/稠密化终点。",
              "随机初始化的采样盒来自所给 49 相机稀疏模型，故它仍不满足严格三张图几何先验限制。此对照仅覆盖 scan8，不能直接推广为 15 场景均值。", ""]
    (OUT / "scan8_初始化与步数受控对照_中文.md").write_text("\n".join(lines))


def main() -> None:
    save_status()
    while not gpu_idle():
        time.sleep(60)
    run_mvs_6k_pilot()
    STATE["state"] = "running"
    save_status()
    available: Queue = Queue()
    for gpu in GPUS:
        available.put(gpu)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = [executor.submit(run_scene, scan, available) for scan in SCANS]
        for future in as_completed(futures):
            future.result()
    if STATE["failed"]:
        STATE["state"] = "failed"
        save_status()
        raise RuntimeError(f"{len(STATE['failed'])} 个论文式诊断失败")
    STATE["state"] = "evaluating"
    save_status()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "5"
    env["PYTHONUNBUFFERED"] = "1"
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "evaluate.out").open("w") as stream:
        subprocess.run([str(PY_GS), str(CODE / "evaluate_paper_native_diagnostics.py")],
                       cwd=PROJECT.parent, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    STATE["state"] = "complete"
    save_status()
    print("PAPER_NATIVE_6K_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
