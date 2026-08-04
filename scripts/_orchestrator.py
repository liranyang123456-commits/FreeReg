# -*- coding: utf-8 -*-
"""
FreeRegNet 训练完成自动执行器
================================
后台常驻: 监控多物体训练日志, 训练完成后自动:
  ① 用全量权重重跑 NN+ICP 级联评测 → 最终精度/速度对比表
  ② 启动 Qwen-VL 主干训练 (验证 VLM 增益)
  ③ 写最终汇总 outputs/freereg_net/FINAL_RESULTS.md
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'e:\Free_coordinate_MR_Registration')
LOG = ROOT / 'outputs/freereg_net/train_full_log.txt'
FINAL = ROOT / 'outputs/freereg_net/FINAL_RESULTS.md'
DONE_MARK = '最佳 ADD-S@10%d = '


def run(cmd, logfile):
    print(f'[orchestrator] RUN: {cmd}', flush=True)
    with open(logfile, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, shell=True, cwd=str(ROOT),
                           stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def wait_training():
    print('[orchestrator] 等待训练完成...', flush=True)
    while True:
        if LOG.exists():
            txt = LOG.read_text(encoding='utf-8', errors='ignore')
            if DONE_MARK in txt:
                # 训练脚本结尾打印 "最佳 ADD-S@10%d = X"
                for line in txt.splitlines():
                    if DONE_MARK in line:
                        print('[orchestrator] 训练完成:', line.strip(), flush=True)
                return
        time.sleep(60)


def main():
    wait_training()
    # ① 全量权重级联评测
    rc = run('conda run --no-capture-output -n py3d python scripts/'
             'run_freereg_cascade.py --ckpt outputs/freereg_net/'
             'freereg_net_multi.pth --scenes 1 2 4 5 6 --max 60 '
             '--with_engine',
             ROOT / 'outputs/freereg_net/cascade_full_log.txt')
    # ② Qwen-VL 主干训练 (短, 验证 VLM 增益)
    rc2 = run('conda run --no-capture-output -n py3d python scripts/'
              'run_freereg_net_train.py --backbone qwen_vl --scenes 1 2 '
              '--epochs 5 --batch 8 --out freereg_net_qwen.pth',
              ROOT / 'outputs/freereg_net/qwen_train_log.txt')
    # ③ 汇总
    lines = ['# FreeRegNet 最终训练与级联汇总',
             f'生成时间: {time.strftime("%Y-%m-%d %H:%M")}', '']
    for name, lf in [('训练日志', LOG),
                     ('级联评测', ROOT / 'outputs/freereg_net/cascade_full_log.txt'),
                     ('Qwen训练', ROOT / 'outputs/freereg_net/qwen_train_log.txt')]:
        lines.append(f'\n## {name}')
        if lf.exists():
            tail = lf.read_text(encoding='utf-8', errors='ignore').splitlines()
            keep = [l for l in tail if any(
                k in l for k in ('ep', 'ADD', '级联', '纯', '最佳', 'best',
                                 '模型', '参数', '样本'))]
            lines.extend(keep[-25:] if keep else tail[-15:])
        else:
            lines.append('(未生成)')
    FINAL.write_text('\n'.join(lines), encoding='utf-8')
    print('[orchestrator] 完成, 汇总:', FINAL, flush=True)
    print('[orchestrator] ORCHESTRATOR_DONE', flush=True)


if __name__ == '__main__':
    main()
