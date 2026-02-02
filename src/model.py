from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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