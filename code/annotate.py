"""Ground truth for tracking clips: one 'x,y,w,h' line per frame, 'nan,nan,nan,nan' = target not visible.

python annotate.py --source clips/c1.mp4 --out clips/c1_gt.txt --every 10
    draws a box every 10th frame (ENTER/SPACE = accept, c = target not visible);
    frames in between are linearly interpolated.
python annotate.py --source clips/c1.mp4 --out clips/c1_gt.txt --every 10 --redo
    reopens an existing file and shows only the keyframes marked 'not visible' (same --every as before).
python annotate.py --source clips/c1.mp4 --out clips/c1_gt.txt --every 10 --frames 30,40
    reopens an existing file and relabels just these keyframes.
python annotate.py --mot ../MOT15/train/TUD-Campus/gt/gt.txt --id 1 --out clips/tud1_gt.txt
    converts one pedestrian of a MOTChallenge sequence (pick an id visible in frame 1)."""
import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--source', help='video to annotate')
ap.add_argument('--every', type=int, default=10)
ap.add_argument('--mot', help='MOTChallenge gt.txt to convert instead')
ap.add_argument('--id', type=int, help='object id in the MOT file')
ap.add_argument('--out', required=True)
ap.add_argument('--redo', action='store_true', help='revisit only the keyframes of --out marked not visible')
ap.add_argument('--frames', help='comma-separated keyframes of --out to relabel, e.g. 30,32')
args = ap.parse_args()

if args.mot:
    d = np.loadtxt(args.mot, delimiter=',')
    gt = np.full((int(d[:, 0].max()), 4), np.nan)
    for row in d[d[:, 1] == args.id]:
        gt[int(row[0]) - 1] = row[2:6]
else:
    cap = cv2.VideoCapture(args.source)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt, keys = np.full((n, 4), np.nan), sorted(set(range(0, n, args.every)) | {n - 1})
    if args.redo or args.frames:
        gt = np.loadtxt(args.out, delimiter=',', ndmin=2)
        todo = [int(k) for k in args.frames.split(',')] if args.frames else [k for k in keys if not np.isfinite(gt[k]).all()]
        if set(todo) - set(keys):
            raise SystemExit(f'not keyframes for --every {args.every}: {sorted(set(todo) - set(keys))}')
        gt[[i for i in range(n) if i not in keys]] = np.nan  # interpolated frames are recomputed below
    else:
        todo = keys
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i not in todo:
            continue
        prev = [k for k in keys if k < i and np.isfinite(gt[k]).all()]
        if prev:  # previous box as a hint only: an empty selection still means 'not visible'
            x, y, w, h = gt[prev[-1]].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 160, 160), 1)
        cv2.putText(frame, f'frame {i + 1}/{n}: drag a NEW box + ENTER (grey = previous, not saved); c = not visible',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        while True:  # an empty selection is saved as 'not visible' only after confirming with c
            box = cv2.selectROI('annotate', frame, showCrosshair=False)
            if box[2] > 0:
                gt[i] = box
                break
            ask = frame.copy()
            cv2.putText(ask, 'no box: c = target NOT visible, any other key = draw again', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('annotate', ask)
            if cv2.waitKey(0) & 0xFF == ord('c'):
                break
    cv2.destroyAllWindows()
    hidden = [k for k in keys if not np.isfinite(gt[k]).all()]
    print(f'{len(hidden)}/{len(keys)} keyframes marked not visible: {hidden}')
    for a, b in zip(keys, keys[1:]):  # interpolate only between two visible keyframes
        if np.isfinite(gt[a]).all() and np.isfinite(gt[b]).all():
            for i in range(a + 1, b):
                w = (i - a) / (b - a)
                gt[i] = (1 - w) * gt[a] + w * gt[b]

np.savetxt(args.out, gt, fmt='%.1f', delimiter=',')
print(f'wrote {args.out}: {np.isfinite(gt).all(axis=1).sum()}/{len(gt)} frames with a box')
