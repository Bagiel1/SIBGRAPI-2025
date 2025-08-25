import numpy as np
from sklearn.model_selection import StratifiedKFold

def compute_confidence_interval(values):
    mean= np.mean(values)
    std= np.std(values, ddof=1)
    return mean, std

def create_folds(features, labels, n_folds=10):
    kf= StratifiedKFold(n_splits=n_folds, shuffle=False)
    return list(kf.split(features, labels))