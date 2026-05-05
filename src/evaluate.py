"""
evaluate.py
===========
Summarize and compare results across all experiments.

Usage:
    python src/evaluate.py --config config.yaml
    python src/evaluate.py --config config.yaml --exp exp1 exp3
"""

import json
import argparse
import yaml
from pathlib import Path
from collections import Counter


def load_results(logs_dir, exp_names):
    results = {}
    for exp_name in exp_names:
        path = logs_dir / f"{exp_name}_results.json"
        if path.exists():
            with open(path) as f:
                results[exp_name] = json.load(f)
        else:
            print(f"[WARN] No results found for {exp_name}")
    return results


def print_summary_table(all_results):
    if not all_results:
        print("No results to display.")
        return
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*70}")
    print(f"{'Experiment':<35} {'Acc':>7} {'MacroF1':>9} {'Params':>10}")
    print('-' * 70)
    for exp_name, res in sorted(all_results.items()):
        print(f"{res['experiment']:<35} "
              f"{res['test_accuracy']:>7.4f} "
              f"{res['test_macro_f1']:>9.4f} "
              f"{res['n_params']:>10,}")
    print(f"{'='*70}")


def print_per_class_table(results):
    print(f"\nPer-class F1 — {results['experiment']}")
    print(f"{'Class':<35} {'F1':>6}")
    print('-' * 45)
    for cls, f1 in sorted(results['per_class_f1'].items(), key=lambda x: -x[1]):
        print(f"{cls:<35} {f1:>6.4f}")


def print_confusion_matrix(results):
    cm = results['confusion_matrix']
    names = [n[:12] for n in results['class_names']]
    print(f"\nConfusion Matrix — {results['experiment']}")
    header = ' ' * 14 + '  '.join(f"{n:>12}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{names[i]:>12}  " + '  '.join(f"{v:>12}" for v in row))


def main(args):
    config = yaml.safe_load(open(args.config))
    logs_dir = Path(config['paths']['results_dir']) / 'logs'

    if args.exp:
        exp_names = []
        for key in args.exp:
            exp_cfg = config['experiments'].get(key)
            if exp_cfg:
                exp_names.append(exp_cfg['name'])
            else:
                print(f"[WARN] Unknown experiment key: {key}")
    else:
        exp_names = [v['name'] for v in config['experiments'].values()]

    all_results = load_results(logs_dir, exp_names)
    print_summary_table(all_results)

    for exp_name, res in sorted(all_results.items()):
        print_per_class_table(res)

    if all_results:
        best = max(all_results.values(), key=lambda r: r['test_macro_f1'])
        print_confusion_matrix(best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--exp', nargs='+', default=None)
    args = parser.parse_args()
    main(args)
