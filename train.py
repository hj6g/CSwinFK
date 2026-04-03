import os
import csv
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, SEED,
    LAST_K, OUTPUT_DIR, CKPT_DIR, NUM_CLASSES, EMBED_DIM
)
from utils import set_seed
from dataset import build_dataloaders
from models.cswinfk_model import CNN_SwinTiny_CAtest
from engine import eval_loss_acc
from metrics import evaluate


def main(seed=SEED, data_dir=''):
    if not data_dir:
        raise ValueError("Please provide dataset path using --data_dir")

    set_seed(seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    _, _, train_ld, val_ld = build_dataloaders(
        data_dir=data_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN_SwinTiny_CA(
        n_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0

        for x, y, _ in train_ld:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            run_loss += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        sch.step()

        train_loss = run_loss / max(1, len(train_ld))
        train_acc = correct / max(1, total)
        val_loss, val_acc = eval_loss_acc(model, val_ld, device, loss_fn)

        print(
            f'Epoch {ep:3d}/{EPOCHS}: '
            f'loss {train_loss:.4f}  '
            f'val_loss {val_loss:.4f}  '
            f'train_acc {train_acc:.4f}  '
            f'val_acc {val_acc:.4f}'
        )

        if ep > EPOCHS - LAST_K and val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    best_path = os.path.join(CKPT_DIR, f'best_seed{seed}.pth')
    torch.save(best_state, best_path)
    model.load_state_dict(best_state)

    acc, cm, prec, sen, f1, spe = evaluate(model, val_ld, device)

    print('\n=========== FINAL RESULTS ===========')
    print(f'Accuracy     : {acc:.4f}')
    print(f'Sensitivity  : {sen:.4f}')
    print(f'Specificity  : {spe:.4f}')
    print(f'Precision    : {prec:.4f}')
    print(f'F1-score     : {f1:.4f}')
    print('Confusion Matrix:\n', cm)

    result = {
        "seed": seed,
        "acc": float(acc),
        "sens": float(sen),
        "spec": float(spe),
        "prec": float(prec),
        "f1": float(f1),
    }

    with open(os.path.join(OUTPUT_DIR, 'metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['seed', 'acc', 'sens', 'spec', 'prec', 'f1']
        )
        writer.writeheader()
        writer.writerow(result)

    print('Saved metrics to outputs/.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    args = parser.parse_args()

    main(seed=args.seed, data_dir=args.data_dir)
