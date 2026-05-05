"""
make_splits.py
==============
Step 2 of the pipeline.
Reads clips.json and produces train/val/test split files.
Stratified by fine-grained label to ensure class balance.

Usage:
    python src/make_splits.py --config config.yaml
"""

import json
import argparse
import yaml
import random
from pathlib import Path
from collections import defaultdict, Counter


def stratified_split(clips, train_ratio, val_ratio, seed):
    random.seed(seed)
    by_label = defaultdict(list)
    for clip in clips:
        by_label[clip['label_fine']].append(clip)

    train, val, test = [], [], []

    for label, label_clips in by_label.items():
        random.shuffle(label_clips)
        n = len(label_clips)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        n_test = n - n_train - n_val

        if n_test < 1:
            if n >= 3:
                n_train, n_val, n_test = n - 2, 1, 1
            elif n == 2:
                n_train, n_val, n_test = 1, 1, 0
            else:
                n_train, n_val, n_test = 1, 0, 0

        train.extend(label_clips[:n_train])
        val.extend(label_clips[n_train:n_train + n_val])
        test.extend(label_clips[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def print_split_stats(train, val, test, fine_to_idx):
    all_labels = sorted(fine_to_idx.keys())
    train_counts = Counter(c['label_fine'] for c in train)
    val_counts = Counter(c['label_fine'] for c in val)
    test_counts = Counter(c['label_fine'] for c in test)

    header = f"{'Class':<35} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}"
    lines = [header, '-' * len(header)]

    for label in all_labels:
        tr = train_counts.get(label, 0)
        v = val_counts.get(label, 0)
        te = test_counts.get(label, 0)
        lines.append(f"{label:<35} {tr:>6} {v:>6} {te:>6} {tr+v+te:>6}")

    lines.append('-' * len(header))
    lines.append(f"{'TOTAL':<35} {len(train):>6} {len(val):>6} {len(test):>6} {len(train)+len(val)+len(test):>6}")
    return '\n'.join(lines)


def main(args):
    config = yaml.safe_load(open(args.config))
    clips_path = Path(config['paths']['clips_index'])
    splits_dir = Path(config['paths']['splits_dir'])
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not clips_path.exists():
        print(f"[ERROR] clips.json not found at {clips_path}")
        print("Run parse_annotations.py first.")
        return

    with open(clips_path) as f:
        data = json.load(f)

    clips = data['clips']
    fine_to_idx = data['fine_to_idx']
    print(f"Loaded {len(clips)} clips with {data['n_fine_classes']} fine-grained classes")

    train, val, test = stratified_split(
        clips,
        config['split']['train'],
        config['split']['val'],
        config['split']['seed']
    )

    print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    stats = print_split_stats(train, val, test, fine_to_idx)
    print(f"\n{stats}")

    for name, split_clips in [('train', train), ('val', val), ('test', test)]:
        out_path = splits_dir / f"{name}.json"
        with open(out_path, 'w') as f:
            json.dump(split_clips, f, indent=2)
        print(f"Saved {out_path}")

    with open(splits_dir / 'split_stats.txt', 'w') as f:
        f.write(stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args)
