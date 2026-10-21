#!/usr/bin/env python3
"""按队列串行训练一个 GPU worker；多个 worker 由 GNU screen 并行运行。"""
import json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
QUEUE_DIR=Path('/tmp/lilinwei/1007fsgsmvs_3dgs/queues')
LOG_DIR=Path('/tmp/lilinwei/1007fsgsmvs_3dgs/logs')

def main():
    gpu=int(sys.argv[1])
    if gpu not in {0,1,2,3,4,5,7}: raise SystemExit('GPU 6 disabled')
    queue=json.loads((QUEUE_DIR/f'gpu{gpu}.json').read_text())
    status=[]
    with (LOG_DIR/f'worker_gpu{gpu}.log').open('a',buffering=1) as log:
        for dataset,scene in queue:
            output=ROOT/'output'/dataset/scene
            manifest=output/'run_manifest.json'
            if manifest.is_file():
                prior=json.loads(manifest.read_text())
                if (output/'final_point_cloud.ply').is_file() and len(list((output/'renders').glob('*.png'))) == prior['rendered_views']:
                    row={'dataset':dataset,'scene':scene,'gpu':gpu,'returncode':0,'skipped':'already complete'}
                    status.append(row)
                    (QUEUE_DIR/f'gpu{gpu}_status.json').write_text(json.dumps(status,indent=2)+'\n')
                    log.write(f'SKIP {dataset}/{scene}: output already complete\n')
                    continue
            start=time.time();log.write(f'START {dataset}/{scene} gpu={gpu}\n')
            cmd=['/data/lilinwei/.conda/envs/d23e/bin/python',str(ROOT/'scripts/run_scene.py'),
                 '--dataset',dataset,'--scene',scene,'--gpu',str(gpu)]
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            row={'dataset':dataset,'scene':scene,'gpu':gpu,'returncode':result.returncode,
                 'elapsed_seconds':time.time()-start}
            status.append(row)
            (QUEUE_DIR/f'gpu{gpu}_status.json').write_text(json.dumps(status,indent=2)+'\n')
            log.write(f"{'DONE' if result.returncode==0 else 'FAILED'} {dataset}/{scene} rc={result.returncode}\n")
    return 0 if all(x['returncode']==0 for x in status) else 1
if __name__=='__main__': raise SystemExit(main())
