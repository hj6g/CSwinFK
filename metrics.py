import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def get_preds_and_labels(model, loader, device):
    model.eval()
    y_true, y_score = [], []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x = batch['image']
                y = batch['label']
            else:
                x, y = batch[:2]

            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)[:, 1]

            y_score.extend(probs.cpu().numpy())
            y_true.extend(y.numpy())

    return np.array(y_true), np.array(y_score)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x = batch['image']
                y = batch['label']
            else:
                x, y = batch[:2]

            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)

            y_pred.extend(preds.cpu().tolist())
            y_true.extend(y.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    prec = precision_score(y_true, y_pred, zero_division=0)
    sen = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spe = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else float('nan')

    return acc, cm, prec, sen, f1, spe
