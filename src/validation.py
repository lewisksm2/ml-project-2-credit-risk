import numpy as np

from model import (
    evaluate_model,
    train_model
    )

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

#---------------------------------------------------------------------------------
# Cross Validation
#---------------------------------------------------------------------------------

def run_cv(model, X, y, k: int = 5, random_state: int = 42):
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    accuracy_scores = []
    roc_auc_scores = []

    for (train_index, test_index) in skf.split(X,y):
        model_copy = clone(model)
        trained_model = train_model(model_copy, X.iloc[train_index], y.iloc[train_index])
        
        results = evaluate_model(trained_model, X.iloc[test_index], y.iloc[test_index])
        
        accuracy_scores.append(results["accuracy"])
        if results["roc_auc"] is not None:
            roc_auc_scores.append(results["roc_auc"])
   
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores, ddof=1)
    
    mean_roc_auc = np.mean(roc_auc_scores)
    std_roc_auc = np.std(roc_auc_scores, ddof=1)
    
    
    return {
        "accuracy": (mean_accuracy, std_accuracy),
        "roc_auc": (mean_roc_auc, std_roc_auc)        
        }