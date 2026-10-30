"""One consistent multiclass metric definition and paired stage comparisons."""

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)


def score(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    if p.ndim != 2 or len(p) != len(y) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid evaluation probabilities")
    if not np.allclose(p.sum(1), 1, rtol=0, atol=2e-6):
        raise ValueError("Probabilities do not sum to one")
    nclass = p.shape[1]
    if not np.array_equal(np.unique(y), np.arange(nclass)):
        raise ValueError("All classes must be present for macro OvR AUROC evaluation")
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(nclass))
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=np.arange(nclass), zero_division=0)
    auc = [float(roc_auc_score(y == c, p[:, c])) for c in range(nclass)]
    ap = [float(average_precision_score(y == c, p[:, c])) for c in range(nclass)]
    result = dict(accuracy=float((pred == y).mean()), correct=int((pred == y).sum()),
                  macro_f1=float(f1.mean()), macro_auroc=float(np.mean(auc)), macro_ap=float(np.mean(ap)),
                  mcc=float(matthews_corrcoef(y, pred)), macro_precision=float(pr.mean()),
                  macro_recall=float(rc.mean()), balanced_accuracy=float(rc.mean()),
                  nll=float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
                  support=sup.tolist(), confusion_matrix=cm.tolist(), per_class_correct=cm.diagonal().tolist(),
                  per_class_precision=pr.tolist(), per_class_recall=rc.tolist(), per_class_f1=f1.tolist(),
                  per_class_auroc=auc, per_class_ap=ap)
    result["composite"] = float(np.mean([result[k] for k in ("accuracy", "macro_f1", "macro_auroc")]))
    return result


def paired(y, before, after):
    a = before.argmax(1); b = after.argmax(1)
    fixed = (a != y) & (b == y); damaged = (a == y) & (b != y)
    k = before.shape[1]
    return dict(corrected=int(fixed.sum()), damaged=int(damaged.sum()),
                net_correct=int(fixed.sum() - damaged.sum()), changed=int((a != b).sum()),
                per_class_corrected=[int((fixed & (y == c)).sum()) for c in range(k)],
                per_class_damaged=[int((damaged & (y == c)).sum()) for c in range(k)])
