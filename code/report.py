"""Turns results/trackers.csv into the README tables (results/tables.md) and Figure 1 (docs/media/speed_accuracy.png).

python report.py"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORDER = ['mil', 'kcf', 'csrt', 'nano', 'vit']
NAMES = {'mil': 'MIL', 'kcf': 'KCF', 'csrt': 'CSRT', 'nano': 'NanoTrack', 'vit': 'VitTrack'}
COLOR = {'classifier': 'tab:blue', 'corr. filter': 'tab:green', 'Siamese CNN': 'tab:red', 'transformer': 'tab:purple'}
FAMILY = {'mil': 'classifier', 'kcf': 'corr. filter', 'csrt': 'corr. filter', 'nano': 'Siamese CNN', 'vit': 'transformer'}


def variant(r):
    if r['recover'] == 'orb':
        return 'orb'
    if r['gray'] == 'True':
        return 'gray'
    return 'default' if r['skip'] == '1' else f'skip{r["skip"]}'


rows = list(csv.DictReader(open(HERE / 'results' / 'trackers.csv')))
runs = defaultdict(dict)  # (tracker, variant) -> tag -> row (last run wins)
for r in rows:
    runs[r['tracker'], variant(r)][r['tag']] = r
trackers = [t for t in ORDER if (t, 'default') in runs]
tags = sorted({tag for t in trackers for tag in runs[t, 'default']})


def mean(t, v, key, only=None):
    vals = [float(r[key]) for tag, r in runs.get((t, v), {}).items() if r[key] != '' and (only is None or tag in only)]
    return np.mean(vals) if vals else np.nan


def f(x, d=3):
    return '–' if np.isnan(x) else f'{x:.{d}f}'


out = ['**Table 1. Overall** (mean over clips, default setting).', '',
       '| Tracker | Family | mean ms | p95 ms | FPS | Success AUC | Prec@20 | Lost events |', '|---|---|---|---|---|---|---|---|']
for t in trackers:
    ms = mean(t, 'default', 'mean_ms')
    lost = sum(int(r['lost_events']) for r in runs[t, 'default'].values())
    out.append(f'| {NAMES[t]} | {FAMILY[t]} | {f(ms, 1)} | {f(mean(t, "default", "p95_ms"), 1)} | {f(1000 / ms, 0)} | '
               f'{f(mean(t, "default", "auc"))} | {f(mean(t, "default", "prec20"))} | {lost} |')

out += ['', '**Table 2. Success AUC per condition.**', '', '| Tracker | ' + ' | '.join(tags) + ' |', '|---|' + '---|' * len(tags)]
for t in trackers:
    out.append(f'| {NAMES[t]} | ' + ' | '.join(f(mean(t, 'default', 'auc', {tag})) for tag in tags) + ' |')

variants = [v for v in ['gray', 'skip2', 'skip4', 'orb'] if any((t, v) in runs for t in trackers)]
out += ['', '**Table 3. Ablations** (Success AUC, Δ vs. default; every column uses the same clips).', '',
        '| Tracker | default | ' + ' | '.join(variants) + (' | ORB recoveries | ORB cost (ms)' if 'orb' in variants else '') + ' |',
        '|---|' + '---|' * (len(variants) + 1 + 2 * ('orb' in variants))]
for t in trackers:
    common = set(runs[t, 'default']).intersection(*(runs.get((t, v), {}) for v in variants))
    b = mean(t, 'default', 'auc', common)
    cells = [f(b)]
    for v in variants:
        a = mean(t, v, 'auc', common)
        cells.append('–' if np.isnan(a) else f'{a:.3f} ({a - b:+.3f})')
    if 'orb' in variants:
        cells.append(str(sum(int(r['recoveries']) for tag, r in runs[t, 'orb'].items() if tag in common)))
        cells.append(f(mean(t, 'orb', 'mean_ms', common) - mean(t, 'default', 'mean_ms', common), 1))
    out.append(f'| {NAMES[t]} | ' + ' | '.join(cells) + ' |')

(HERE / 'results' / 'tables.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out), '\n\nwrote results/tables.md')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for t in trackers:
        x, y = 1000 / mean(t, 'default', 'mean_ms'), mean(t, 'default', 'auc')
        ax.scatter(x, y, s=50, color=COLOR[FAMILY[t]], label=FAMILY[t], zorder=3)
        ax.annotate(NAMES[t], (x, y), textcoords='offset points', xytext=(6, 6))
    by_family = dict(zip(*ax.get_legend_handles_labels()[::-1]))  # one legend entry per family
    ax.legend(by_family.values(), by_family.keys(), loc='lower right', fontsize=8)
    ax.set(xscale='log', xlabel='FPS on CPU (log scale)', ylabel='Success AUC', title='Speed vs. accuracy')
    ax.set_xticks([20, 50, 100, 200])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.25, y=0.15)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    (HERE / 'docs' / 'media').mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / 'docs' / 'media' / 'speed_accuracy.png', dpi=150)
    print('wrote docs/media/speed_accuracy.png')
except ImportError:
    print('pip install matplotlib for Figure 1')
