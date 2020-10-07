# Use scikit-learn to calculate roc-auc and f1 score

from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

def get_auroc(labels, logits):
    return roc_auc_score(labels, logits)

def get_f1(labels, logits):
    return f1_score(labels, np.around(logits))