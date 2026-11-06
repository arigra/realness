"""The crop classifier of the classification study (docs/notes/2026-11-05-classification.md):
a small CNN on standardised target crops (src/crops.py), trained with class-balanced batches,
early-stopped on balanced accuracy over real val crops, scored on real test crops.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LR, BATCH, EVAL_EVERY, PATIENCE, MAX_STEPS = 1e-3, 64, 200, 5, 4000


class CropNet(nn.Module):
    def __init__(self, n_classes=3, w=16):
        super().__init__()
        block = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(1, b), nn.SiLU())
        self.body = nn.Sequential(block(1, w), block(w, w), nn.MaxPool2d(2), block(w, 2 * w), nn.MaxPool2d(2),
                                  block(2 * w, 4 * w), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        return self.head(self.body(x))


def scores(pred, y, n_classes=3):
    """Balanced accuracy (mean per-class recall over present classes), macro F1, confusion (true x pred)."""
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y.tolist(), pred.tolist()):
        conf[t, p] += 1
    present = conf.sum(1) > 0
    recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
    prec = conf.diagonal() / np.maximum(conf.sum(0), 1)
    f1 = 2 * prec * recall / np.maximum(prec + recall, 1e-9)
    return {"bal_acc": float(recall[present].mean()), "macro_f1": float(f1[present].mean()),
            "confusion": conf.tolist()}


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    out = torch.cat([model(x[i:i + 512].to(device)).argmax(1).cpu() for i in range(0, len(x), 512)])
    model.train()
    return out


def train(x, y, val, device, seed):
    """x (N, 1, h, w), y (N,) training crops; val = (x, y) real val crops. Returns the model at
    its best val balanced accuracy, and how training stopped."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = CropNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    counts = torch.bincount(y, minlength=3).float()
    weight = (1.0 / counts.clamp(min=1))[y]                  # class-balanced sampling
    best, best_state, best_step, bad = -1.0, None, 0, 0
    for step in range(1, MAX_STEPS + 1):
        idx = torch.multinomial(weight, BATCH, replacement=True, generator=g)
        xb = x[idx]
        if torch.rand((), generator=g) < 0.5:                 # range is not mirror-symmetric; Doppler sign is
            xb = xb.flip(-1)
        loss = F.cross_entropy(model(xb.to(device)), y[idx].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            v = scores(predict(model, val[0], device), val[1])["bal_acc"]
            if v > best:
                best, best_step, bad = v, step, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"val_bal_acc": best, "best_step": best_step, "stopped_at": step}


def shortcut_baseline(f_train, y_train, f_test, y_test):
    """What speed and range alone give (the leak the crops close): a class-balanced logistic
    regression on (|speed|, range)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(f_train.numpy())
    clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(f_train.numpy()), y_train.numpy())
    return scores(torch.tensor(clf.predict(sc.transform(f_test.numpy()))), y_test)
