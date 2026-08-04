# -*- coding: utf-8 -*-
"""
真实视频人工 GT —— ② 双人交互标注工具 (大屏+放大镜版)
==========================================================

cv2 交互窗口 (自动最大化): 显示提案叠加图, 点击记录 2D 解剖标志点+标签。
放大镜: 鼠标移动即跟随显示光标邻域 4x 放大视图 (可调/可关)。

键位:
  左键      标注当前点 (记入, 自动保存)
  空格 / n  下一张          b      上一张
  s         跳过本帧        u      撤销上一条
  1-5       标签 (1隆突 2右主 3左主 4右叶 5左叶)
  z         放大镜开关      +/-    放大倍数 2x-8x
  f         全屏/窗口切换   q/ESC  保存退出

输出: outputs/gt_pkg/annotations_<reader>.csv (每条即时落盘, 防丢失)

用法:
    python scripts/run_gt_annotate.py --reader R1
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

OUT = Path('outputs/gt_pkg')
LABELS = {ord('1'): 'carina', ord('2'): 'right_main', ord('3'): 'left_main',
          ord('4'): 'right_lobar', ord('5'): 'left_lobar'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reader', type=str, required=True)
    args = ap.parse_args()
    frames = sorted((OUT / 'overlays').glob('f*.png'))
    if not frames:
        print('提案包不存在, 先运行 run_gt_gtprep.py')
        return

    out_csv = OUT / f'annotations_{args.reader}.csv'
    records = []
    if out_csv.exists():  # 断点续标
        with open(out_csv, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                records.append(dict(frame=int(r['frame']), u=int(r['u']),
                                    v=int(r['v']), label=r['label']))
        print(f'载入已有标注 {len(records)} 条 (断点续标)')

    def autosave():
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['frame', 'u', 'v', 'label'])
            w.writeheader()
            w.writerows(records)

    state = {'click': None, 'mouse': (0, 0), 'mag': True, 'zoom': 4,
             'fullscreen': True}
    i = 0
    cur_label = 'carina'
    win = f'GT annotate [{args.reader}]'

    def on_mouse(event, x, y, flags, param):
        state['mouse'] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state['click'] = (x, y)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win, on_mouse)
    print(f'标注开始 [{args.reader}]: {len(frames)}帧 (z=放大镜开关, '
          f'+/-倍数, f=全屏)')

    def draw_magnifier(disp):
        if not state['mag']:
            return disp
        mx, my = state['mouse']
        H, W = disp.shape[:2]
        r = 40
        x0, y0 = max(0, mx - r), max(0, my - r)
        x1, y1 = min(W, mx + r), min(H, my + r)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return disp
        crop = disp[y0:y1, x0:x1]
        z = state['zoom']
        big = cv2.resize(crop, ((x1 - x0) * z, (y1 - y0) * z),
                         interpolation=cv2.INTER_NEAREST)
        bh, bw = big.shape[:2]
        # 放到右上角, 避免遮挡中心
        px, py = W - bw - 10, 10
        if py + bh > H - 10:
            py = H - bh - 10
        cv2.rectangle(disp, (px - 2, py - 2), (px + bw + 2, py + bh + 2),
                      (255, 255, 255), 2)
        disp[py:py + bh, px:px + bw] = big
        # 十字线
        cxm, cym = px + (mx - x0) * z, py + (my - y0) * z
        cv2.drawMarker(disp, (int(cxm), int(cym)), (0, 255, 255),
                       cv2.MARKER_CROSS, 14, 1)
        return disp

    while 0 <= i < len(frames):
        img = cv2.imread(str(frames[i]))
        fi = int(frames[i].stem[1:])
        disp = img.copy()
        for r in records:
            if r['frame'] == fi:
                cv2.circle(disp, (r['u'], r['v']), 8, (255, 0, 255), 2)
                cv2.putText(disp, r['label'], (r['u'] + 10, r['v'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if state['click'] is not None:
            cv2.circle(disp, state['click'], 8, (255, 0, 255), 1)
        cv2.putText(disp,
                    f'[{i + 1}/{len(frames)}] label={cur_label} '
                    f'(1-5换标签 左键标点 空格/n下张 b上张 s跳过 u撤销 '
                    f'z放大镜{state["zoom"]}x f全屏 q保存)',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 1)
        disp = draw_magnifier(disp)
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k in LABELS:
            cur_label = LABELS[k]
        elif k == ord('z'):
            state['mag'] = not state['mag']
        elif k in (ord('+'), ord('=')):
            state['zoom'] = min(8, state['zoom'] + 1)
        elif k in (ord('-'), ord('_')):
            state['zoom'] = max(2, state['zoom'] - 1)
        elif k == ord('f'):
            state['fullscreen'] = not state['fullscreen']
            cv2.setWindowProperty(
                win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if state['fullscreen']
                else cv2.WINDOW_NORMAL)
        elif k in (ord('n'), 32):
            if state['click'] is not None:
                records.append(dict(frame=fi, u=state['click'][0],
                                    v=state['click'][1], label=cur_label))
                state['click'] = None
                autosave()
            i += 1
        elif k == ord('b'):
            i = max(0, i - 1)
        elif k == ord('s'):
            i += 1
        elif k == ord('u'):
            if records:
                records.pop()
                autosave()
        elif k == ord('q') or k == 27:
            break
    autosave()
    cv2.destroyAllWindows()
    print(f'保存 {len(records)} 条标注 → {out_csv}')


if __name__ == '__main__':
    main()
