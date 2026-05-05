"""
parse_annotations.py
====================
Reads all CSV annotation files and produces a single clips.json index.

Label mapping sourced directly from AnnChorTemplate.json VIA annotator file,
attribute "BalletSteps" options. 

Usage:
    python src/parse_annotations.py --config config.yaml
"""

import csv
import json
import re
import argparse
import yaml
from pathlib import Path
from collections import Counter


# Authoritative mapping from AnnChorTemplate.json "BalletSteps" options.
# Integer -> fine-grained name and general group.
# 19 (Backwards) is a parallel tag, not a movement class — excluded.
LABEL_MAP = {
    0:  {"fine": "ExtDerriereOnRight",    "general": "ExtDerriere"},
    1:  {"fine": "ExtDerriereOnLeft",     "general": "ExtDerriere"},
    2:  {"fine": "Courus",                "general": "Courus"},
    3:  {"fine": "EchappeSecond",         "general": "Echappe"},
    4:  {"fine": "CabrioleDevantRight",   "general": "CabrioleDevant"},
    5:  {"fine": "CabrioleDevantLeft",    "general": "CabrioleDevant"},
    6:  {"fine": "GrandJeteRight",        "general": "GrandJete"},
    7:  {"fine": "GrandJeteLeft",         "general": "GrandJete"},
    8:  {"fine": "PirouetteRight",        "general": "Pirouette"},
    9:  {"fine": "PirouetteLeft",         "general": "Pirouette"},
    10: {"fine": "SissonneFRight",        "general": "Sissonne"},
    11: {"fine": "SissonneFLeft",         "general": "Sissonne"},
    12: {"fine": "ExtSecondRight",        "general": "ExtensionSecond"},
    13: {"fine": "ExtSecondLeft",         "general": "ExtensionSecond"},
    14: {"fine": "TourEnLair",            "general": "TourEnLair"},
    15: {"fine": "CabrioleDerriereRight", "general": "CabrioleDerriere"},
    16: {"fine": "CabrioleDerriereLeft",  "general": "CabrioleDerriere"},
    17: {"fine": "BalanceDCRight",        "general": "Balance"},
    18: {"fine": "BalanceDCLeft",         "general": "Balance"},
    # 19 = Backwards — parallel annotation tag, excluded
}


def parse_metadata(metadata_str):
    try:
        return json.loads(metadata_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_file_list(file_list_str):
    try:
        paths = json.loads(file_list_str)
        if paths:
            raw = paths[0]
            match = re.search(r'\\([A-Z0-9]+)\\([A-Z0-9]+\.mp4)', raw)
            if match:
                return match.group(1), match.group(2)
    except (json.JSONDecodeError, TypeError, IndexError):
        pass
    return None, None


def parse_csv(csv_path, performance_id):
    clips = []
    video_meta = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        lines = f.readlines()

    data_lines = [l for l in lines if not l.startswith('#')]
    fieldnames = ['metadata_id', 'file_list', 'temporal_segment_start',
                  'temporal_segment_end', 'metadata']

    for row in csv.DictReader(data_lines, fieldnames=fieldnames):
        metadata = parse_metadata(row.get('metadata', '{}'))
        group, filename = parse_file_list(row.get('file_list', '[]'))

        if group is None:
            continue

        video_rel_path = f"{group}/{filename}"

        if 'OnlyOneDancer?' in metadata:
            video_meta[video_rel_path] = {'only_one_dancer': metadata['OnlyOneDancer?'] == '1'}
            continue

        if 'BalletSteps' not in metadata:
            continue

        start_str = row.get('temporal_segment_start', '').strip()
        end_str = row.get('temporal_segment_end', '').strip()
        if not start_str or not end_str:
            continue

        try:
            start, end = float(start_str), float(end_str)
        except ValueError:
            continue

        label_int = int(metadata['BalletSteps'])

        # Skip Backwards (19) — not a movement class
        if label_int == 19:
            continue

        label_info = LABEL_MAP.get(label_int, {
            'fine': f"Unknown_{label_int}",
            'general': f"Unknown_{label_int}"
        })

        clips.append({
            'clip_id': f"{performance_id}_{row['metadata_id']}",
            'performance_id': performance_id,
            'video_rel_path': video_rel_path,
            'start': start,
            'end': end,
            'duration': round(end - start, 4),
            'label_int': label_int,
            'label_fine': label_info['fine'],
            'label_general': label_info['general'],
            'metadata_id': row['metadata_id'],
        })

    for clip in clips:
        meta = video_meta.get(clip['video_rel_path'], {})
        clip['only_one_dancer'] = meta.get('only_one_dancer', None)

    return clips


def main(args):
    config = yaml.safe_load(open(args.config))
    ann_dir = Path(config['paths']['raw_annotations'])
    out_path = Path(config['paths']['clips_index'])
    min_dur = config['dataset']['min_clip_duration']

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ann_dir.exists():
        print(f"[ERROR] Annotation directory not found: {ann_dir}")
        return

    csv_files = sorted(ann_dir.glob('*.csv'))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {ann_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    all_clips = []
    unknown_labels = set()

    for csv_path in csv_files:
        performance_id = csv_path.stem
        clips = parse_csv(csv_path, performance_id)
        before = len(clips)
        clips = [c for c in clips if c['duration'] >= min_dur]
        dropped = before - len(clips)
        print(f"  {performance_id}: {len(clips)} clips" + (f" ({dropped} dropped)" if dropped else ""))
        for c in clips:
            if c['label_fine'].startswith('Unknown_'):
                unknown_labels.add(c['label_int'])
        all_clips.extend(clips)

    fine_labels = sorted(set(c['label_fine'] for c in all_clips))
    general_labels = sorted(set(c['label_general'] for c in all_clips))
    fine_to_idx = {l: i for i, l in enumerate(fine_labels)}
    general_to_idx = {l: i for i, l in enumerate(general_labels)}

    for clip in all_clips:
        clip['label_fine_idx'] = fine_to_idx[clip['label_fine']]
        clip['label_general_idx'] = general_to_idx[clip['label_general']]

    print(f"\n{'='*50}")
    print(f"Total clips:          {len(all_clips)}")
    print(f"Fine-grained classes: {len(fine_to_idx)}")
    print(f"General classes:      {len(general_to_idx)}")

    if unknown_labels:
        print(f"\n[WARNING] Unknown label integers: {sorted(unknown_labels)}")
    else:
        print("\nAll labels mapped successfully.")

    print("\nFine-grained class distribution:")
    for label, count in sorted(Counter(c['label_fine'] for c in all_clips).items(), key=lambda x: -x[1]):
        print(f"  [{fine_to_idx[label]:2d}] {label:<30} {count:5d} clips")

    print("\nGeneral class distribution:")
    for label, count in sorted(Counter(c['label_general'] for c in all_clips).items(), key=lambda x: -x[1]):
        print(f"  [{general_to_idx[label]:2d}] {label:<30} {count:5d} clips")

    output = {
        'clips': all_clips,
        'fine_to_idx': fine_to_idx,
        'general_to_idx': general_to_idx,
        'idx_to_fine': {v: k for k, v in fine_to_idx.items()},
        'idx_to_general': {v: k for k, v in general_to_idx.items()},
        'n_fine_classes': len(fine_to_idx),
        'n_general_classes': len(general_to_idx),
        'total_clips': len(all_clips),
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args)
