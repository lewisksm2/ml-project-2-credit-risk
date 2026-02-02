from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
#---------------------------------------------------------------------------------
# Model Constructors
#---------------------------------------------------------------------------------


def get_logistic_model(C: float, random_state: int = 42) -> LogisticRegression:
    
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
        )
    
    return model

def get_random_forest_model(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    random_state: int = 42
    ) -> RandomForestClassifier:
    
    model = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=1
        )
    
    return model

#---------------------------------------------------------------------------------
# Training
#---------------------------------------------------------------------------------

def train_model(model, X, y):
    
    model.fit(X,y)
    
    return model


def evaluate_model(model, X, y):
    
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:,1]
    else:
        y_prob = None
        
    results = {}
    
    results["accuracy"] = accuracy_score(y,y_pred)
    
    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(y,y_prob)
    else:
        results["roc_auc"] = None
        
    results["confusion_matrix"] = confusion_matrix(y, y_pred)
    
    return results
    