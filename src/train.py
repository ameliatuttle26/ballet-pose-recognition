"""
train.py
========
Training and evaluation script.

Usage:
    python src/train.py --config config.yaml --exp exp1
"""

import json
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset import BalletPoseDataset
from models import build_model, input_dim_for


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, macro_f1, all_preds, all_labels


def main(args):
    config = yaml.safe_load(open(args.config))
    exp_config = config['experiments'][args.exp]
    train_config = config['train']

    exp_name = exp_config['name']
    set_seed(train_config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")

    splits_dir = Path(config['paths']['splits_dir'])
    keypoints_dir = Path(config['paths']['keypoints_dir'])
    results_dir = Path(config['paths']['results_dir'])
    logs_dir = results_dir / 'logs'
    models_dir = results_dir / 'models'
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = exp_config['model']
    input_type = exp_config['input']
    n_frames = exp_config.get('n_frames', config['dataset']['n_frames'])
    label_type = exp_config['label_type']

    def make_dataset(split):
        return BalletPoseDataset(
            split_json=splits_dir / f"{split}.json",
            keypoints_dir=keypoints_dir,
            label_type=label_type,
            input_type=input_type,
            n_frames=n_frames,
        )

    train_ds = make_dataset('train')
    val_ds = make_dataset('val')
    test_ds = make_dataset('test')

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Classes: {train_ds.n_classes}")

    train_loader = DataLoader(train_ds, batch_size=train_config['batch_size'],
                              shuffle=True, num_workers=train_config['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'],
                            shuffle=False, num_workers=train_config['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=train_config['batch_size'],
                             shuffle=False, num_workers=train_config['num_workers'])

    in_dim = input_dim_for(input_type, n_joints=config['dataset']['n_joints'])
    model = build_model(model_name, n_frames, in_dim, train_ds.n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} | Parameters: {n_params:,}")

    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'],
                                 weight_decay=train_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    best_val_f1 = 0
    patience_counter = 0
    history = []
    best_ckpt = models_dir / f"{exp_name}_best.pt"

    print(f"\n{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7} {'VaF1':>7}")
    print('-' * 50)

    for epoch in range(1, train_config['epochs'] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(va_f1)

        history.append({
            'epoch': epoch,
            'train_loss': round(tr_loss, 4),
            'train_acc': round(tr_acc, 4),
            'val_loss': round(va_loss, 4),
            'val_acc': round(va_acc, 4),
            'val_f1': round(va_f1, 4),
        })

        print(f"{epoch:>6} {tr_loss:>8.4f} {tr_acc:>7.3f} {va_loss:>8.4f} {va_acc:>7.3f} {va_f1:>7.3f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= train_config['patience']:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nLoading best checkpoint from {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_acc, test_f1, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device)

    with open(config['paths']['clips_index']) as f:
        clips_data = json.load(f)
    idx_to_label = clips_data[f'idx_to_{label_type}']
    class_names = [idx_to_label[str(i)] for i in range(train_ds.n_classes)]

    print(f"\n{'='*50}")
    print(f"TEST RESULTS — {exp_name}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Macro F1:  {test_f1:.4f}")
    print(f"\nPer-class report:")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(test_labels, test_preds)
    per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
    per_class_results = {class_names[i]: round(float(per_class_f1[i]), 4)
                         for i in range(len(class_names))}

    results = {
        'experiment': exp_name,
        'timestamp': datetime.now().isoformat(),
        'config': exp_config,
        'n_params': n_params,
        'test_accuracy': round(test_acc, 4),
        'test_macro_f1': round(test_f1, 4),
        'per_class_f1': per_class_results,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'history': history,
    }

    results_path = logs_dir / f"{exp_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--exp', required=True)
    args = parser.parse_args()
    main(args)
