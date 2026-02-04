import numpy as np
import pandas as pd

from validation import (
    run_cv
    )

from model import (
    get_logistic_model,
    get_random_forest_model
    )


def logistic_tuning(X,y,k: int = 5, grid_size: int = 7, alpha: float = 0.5) -> float:
    
    C_values = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    rows = []
    
    for _ in range(3):
        print(f"Testing C values between {C_values[0]} and {C_values[-1]}.")
        results = []
        for C in C_values:  
            
            model = get_logistic_model(C)
            result = run_cv(model,X,y,k, random_state=42+_)
            mean, std = result["roc_auc"]
            score = mean - alpha * std # This score will pick out models with high roc auc accuracy mean, and low standard deviation
            results.append(score)
            rows.append({"iter" : _,
                         "C" : C,
                         "mean_auc" : mean,
                         "mean_std" : std,
                         "score" : score                                    
                })
        best_C_value = C_values[np.argmax(results)]

        new_logC_values = np.linspace(np.log(best_C_value/10), np.log(best_C_value*10), grid_size)
        
        C_values = np.exp(new_logC_values) # We zoom in at the best_C_value
            
    history_df = pd.DataFrame(rows)
    
    history_df.to_csv("results/logistic_tuning.csv", index=False)

    return best_C_value, history_df
            

