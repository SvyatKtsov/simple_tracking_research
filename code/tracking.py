"""Single-object tracking on CPU: correlation-filter, Siamese-CNN and transformer trackers from OpenCV,
with optional ORB re-detection after tracking failure.

python tracking.py --tracker csrt                         # camera: drag a box, ENTER; r = reselect, q = quit
python tracking.py --record clips/c1.mp4 --seconds 20     # record a test clip (any OS, no ffmpeg needed)
python tracking.py --compare --source clips/c1.mp4 --gt clips/c1_gt.txt --tag C1   # -> results/trackers.csv"""
import argparse
import csv
import hashlib
import re
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE / 'models'
ZOO = 'https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/'
NANO = 'https://github.com/HonglinChu/SiamTrackers/raw/248663fde6bf7c40190cf10ee396d5662919ecd3/NanoTrack/models/nanotrackv2/'
MODEL_FILES = {  # name: (URL pinned to a commit, sha256)
    'nano_backbone.onnx': (NANO + 'nanotrack_backbone_sim.onnx', '530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4'),
    'nano_head.onnx': (NANO + 'nanotrack_head_sim.onnx', '0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9'),
    'vittrack.onnx': (ZOO + 'object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx', '2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d'),
}
FIELDS = ['tracker', 'tag', 'source', 'recover', 'skip', 'gray', 'frames', 'mean_ms', 'p95_ms', 'fps',
          'tracked_pct', 'lost_events', 'recoveries', 'auc', 'prec20', 'miou']


def model(name):
    """Downloads a model into ./models on first use and verifies its checksum."""
    url, sha = MODEL_FILES[name]
    path = MODELS / name
    if not path.exists():
        MODELS.mkdir(exist_ok=True)
        print('downloading', url)
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(url, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha:
            tmp.unlink()
            raise SystemExit(f'checksum mismatch for {name}')
        tmp.rename(path)
    return str(path)


def nano():
    p = cv2.TrackerNano_Params()
    p.backbone, p.neckhead = model('nano_backbone.onnx'), model('nano_head.onnx')
    return cv2.TrackerNano_create(p)


def vit():
    p = cv2.TrackerVit_Params()
    p.net = model('vittrack.onnx')
    return cv2.TrackerVit_create(p)


TRACKERS = {  # online-learned classical trackers -> offline-trained deep trackers
    'mil': cv2.TrackerMIL_create,    # Multiple Instance Learning: boosted classifier updated online
    'kcf': cv2.TrackerKCF_create,    # Kernelized Correlation Filter: ridge regression solved in the Fourier domain
    'csrt': cv2.TrackerCSRT_create,  # CSR-DCF: correlation filter + spatial/channel reliability maps
    'nano': nano,                    # NanoTrack: lightweight Siamese CNN (template vs search region)
    'vit': vit,                      # VitTrack: lightweight one-stream vision transformer
}


def open_source(src):
    if src.isdigit():  # DirectShow opens webcams much faster than the default backend on Windows
        return cv2.VideoCapture(int(src), cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    return cv2.VideoCapture(src)  # video file or image sequence, e.g. img/%08d.jpg


def frames(cap, skip=1, gray=False):
    """Yields (index, frame). skip=N keeps every Nth frame (= faster motion); gray drops colour."""
    i = -1
    while True:
        ok, f = cap.read()
        if not ok:
            return
        i += 1
        if i % skip == 0:
            yield i, cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) if gray else f


def clamp(box, frame):
    H, W = frame.shape[:2]
    x, y, w, h = map(int, box)
    x, y = min(max(x, 0), W - 2), min(max(y, 0), H - 2)
    return x, y, max(min(w, W - x), 2), max(min(h, H - y), 2)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def load_gt(path):
    """One 'x,y,w,h' line per frame (OTB/LaSOT format); 'nan' or w<=0 = target not visible."""
    return np.array([re.split(r'[,\s]+', line.strip())[:4] for line in open(path) if line.strip()], dtype=float)


def visible(g):
    return np.isfinite(g).all() and g[2] > 0 and g[3] > 0


class OrbRecovery:
    """Remembers ORB descriptors of the target every `every` frames; when the tracker fails it searches the
    full frame (ratio-test matching + RANSAC homography) and re-initialises the tracker at the match."""
    def __init__(self, every=30, keep=10):
        self.orb, self.bf = cv2.ORB_create(2000), cv2.BFMatcher(cv2.NORM_HAMMING)
        self.history, self.every, self.count = deque(maxlen=keep), every, 0

    def remember(self, frame, box):
        self.count += 1
        if self.count % self.every != 1:
            return
        x, y, w, h = box
        kps, des = self.orb.detectAndCompute(frame[y:y + h, x:x + w], None)
        if des is not None and len(kps) >= 10:  # keypoints back to frame coordinates
            self.history.append((np.float32([(k.pt[0] + x, k.pt[1] + y) for k in kps]), des, (w, h)))

    def recover(self, frame):
        kps, des = self.orb.detectAndCompute(frame, None)
        if des is None or len(des) < 10:
            return None
        for pts, des1, (w, h) in reversed(self.history):
            good = [m[0] for m in self.bf.knnMatch(des1, des, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) < 10:
                continue
            src = pts[[m.queryIdx for m in good]]
            dst = np.float32([kps[m.trainIdx].pt for m in good])
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None and mask.sum() >= 8:
                cx, cy = dst[mask.ravel() == 1].mean(axis=0)
                return int(cx - w / 2), int(cy - h / 2), w, h
        return None


def track(name, it, frame, box, recover=False, show=True, gt=None, writer=None):
    """Runs one tracker over the frame iterator. Per-frame time includes recovery attempts.
    Returns per-frame records and why it stopped ('end', 'q' or 'r')."""
    tracker = TRACKERS[name]()
    tracker.init(frame, box)
    orb = OrbRecovery() if recover else None
    recs, path = [], deque(maxlen=50)
    for idx, frame in it:
        t0 = time.perf_counter()
        found, out = tracker.update(frame)
        recovered = False
        if found:
            out = clamp(out, frame)
            if orb:
                orb.remember(frame, out)
        elif orb and (new := orb.recover(frame)):
            out = clamp(new, frame)
            tracker = TRACKERS[name]()
            tracker.init(frame, out)
            recovered = True
        else:
            out = None
        recs.append({'idx': idx, 'ms': (time.perf_counter() - t0) * 1e3, 'found': found, 'recovered': recovered, 'box': out})
        if not (show or writer):
            continue
        if gt is not None and idx < len(gt) and visible(gt[idx]):
            x, y, w, h = gt[idx].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 1)  # ground truth: blue
        if out:
            x, y, w, h = out
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            path.append((x + w // 2, y + h // 2))
        else:
            cv2.putText(frame, 'LOST', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for a, b in zip(path, list(path)[1:]):
            cv2.line(frame, a, b, (0, 0, 255), 2)
        cv2.putText(frame, f'{name}  {recs[-1]["ms"]:.1f} ms', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer:
            writer.write(frame)
        if show:
            cv2.imshow('Tracking', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('r')):
                return recs, chr(key)
    return recs, 'end'


def summary(name, recs, gt=None):
    t = np.array([r['ms'] for r in recs])
    oks = [r['found'] for r in recs]
    row = {'tracker': name, 'frames': len(t), 'mean_ms': round(float(t.mean()), 2),
           'p95_ms': round(float(np.percentile(t, 95)), 2), 'fps': round(1000 / float(t.mean()), 1),
           'tracked_pct': round(100 * float(np.mean(oks)), 1),
           'lost_events': sum(a and not b for a, b in zip([True] + oks, oks)),
           'recoveries': sum(r['recovered'] for r in recs)}
    if gt is not None:  # OTB protocol: success AUC over IoU thresholds, precision at 20 px centre error
        ious, errs = [], []
        for r in recs:
            if r['idx'] >= len(gt) or not visible(gt[r['idx']]):
                continue
            g, b = gt[r['idx']], r['box']
            ious.append(iou(b, g) if b else 0.0)
            errs.append(np.hypot(b[0] + b[2] / 2 - g[0] - g[2] / 2, b[1] + b[3] / 2 - g[1] - g[3] / 2) if b else np.inf)
        ious, errs = np.array(ious), np.array(errs)
        if len(ious):
            row.update(auc=round(float(np.mean([np.mean(ious > th) for th in np.linspace(0, 1, 21)])), 3),
                       prec20=round(float(np.mean(errs <= 20)), 3), miou=round(float(ious.mean()), 3))
    return row


def select(frame, init, g):
    if init:
        return tuple(map(int, init.split(',')))
    if g is not None:
        return tuple(g.astype(int))
    box = cv2.selectROI('Tracking', frame, showCrosshair=False)
    if box[2] == 0:
        raise SystemExit
    return box


def make_writer(path, frame, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame.shape[1::-1])


def record(path, seconds, src):
    cap = open_source(src)
    if src.isdigit():  # many USB webcams reach 30 FPS at 640x480 only with MJPG
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f'cannot read from {src}')
    t0, n = time.time(), 0  # measure the real rate: CAP_PROP_FPS reports what the driver claims, not what it delivers
    while ok and time.time() - t0 < 2:
        ok, frame = cap.read()
        n += 1
    fps = n / (time.time() - t0)
    print(f'camera delivers {fps:.1f} FPS')
    if fps < 20:
        print('warning: low FPS, usually auto-exposure in dim light; add light for a real-time clip')
    writer, end = make_writer(path, frame, round(fps)), time.time() + seconds
    while ok and time.time() < end:
        writer.write(frame)
        cv2.putText(frame, f'REC {end - time.time():.0f}s  q = stop', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Record', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ok, frame = cap.read()
    writer.release()
    print('wrote', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', default='csrt', choices=TRACKERS)
    ap.add_argument('--compare', nargs='*', choices=TRACKERS, help='run these trackers (default: all) on the same clip')
    ap.add_argument('--source', default='0', help='camera index, video file or image sequence (img/%%08d.jpg)')
    ap.add_argument('--gt', help='ground-truth file, one x,y,w,h line per frame (also gives the initial box)')
    ap.add_argument('--init', help='initial box x,y,w,h (otherwise ground truth or drag one)')
    ap.add_argument('--recover', default='none', choices=['none', 'orb'])
    ap.add_argument('--skip', type=int, default=1, help='use every Nth frame (simulates faster motion / lower FPS)')
    ap.add_argument('--gray', action='store_true', help='drop colour (ablation)')
    ap.add_argument('--tag', default='', help='condition label stored in the CSV, e.g. C3-illumination')
    ap.add_argument('--save', help='write the annotated video here (per tracker in compare mode)')
    ap.add_argument('--record', help='record the camera to this file and exit')
    ap.add_argument('--seconds', type=float, default=20)
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--out', default=HERE / 'results' / 'trackers.csv')
    ap.add_argument('--threads', type=int, help='limit OpenCV worker threads (default: all cores), e.g. 1 = single-core budget')
    args = ap.parse_args()
    if args.threads:
        cv2.setNumThreads(args.threads)
    if args.record:
        record(args.record, args.seconds, args.source)
        sys.exit()

    gt = load_gt(args.gt) if args.gt else None
    names = TRACKERS if args.compare == [] else args.compare or [args.tracker]
    show, recover = not args.no_show, args.recover == 'orb'
    if args.compare is not None and args.source.isdigit():
        raise SystemExit('--compare needs a video file so every tracker sees the same frames')

    src = Path(args.source)
    src_name = src.parent.parent.name if '%' in src.name else src.name  # seq/img/%08d.jpg -> seq
    rows, box = [], None
    for name in names:
        cap = open_source(args.source)
        it = frames(cap, args.skip, args.gray)
        first = next(it, None)
        while first and gt is not None and not args.init and not (first[0] < len(gt) and visible(gt[first[0]])):
            first = next(it, None)  # start where the target first appears (OTB: sequences start with the target)
        if first is None:
            raise SystemExit(f'cannot read from {args.source}' + (' or no visible ground-truth box' if gt is not None else ''))
        frame = first[1]
        box = box or select(frame, args.init, gt[first[0]] if gt is not None else None)
        save = args.save and (Path(args.save).with_stem(f'{Path(args.save).stem}_{name}') if len(names) > 1 else args.save)
        writer = save and make_writer(save, frame, (cap.get(cv2.CAP_PROP_FPS) or 30) / args.skip)
        while True:
            recs, why = track(name, it, frame, box, recover, show, gt, writer)
            if recs:
                rows.append({**summary(name, recs, gt), 'tag': args.tag, 'source': src_name,
                             'recover': args.recover, 'skip': args.skip, 'gray': args.gray})
                print(rows[-1])
            if why != 'r':
                break
            frame = next(it)[1]
            box = select(frame, None, None)
        if writer:
            writer.release()
        if why == 'q':
            break

    if args.compare is not None and rows:
        out = Path(args.out)
        out.parent.mkdir(exist_ok=True)
        new = not out.exists()
        with open(out, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print('appended to', out)
    cv2.destroyAllWindows()
