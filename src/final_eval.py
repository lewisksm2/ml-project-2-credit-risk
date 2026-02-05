import pandas as pd

from tuning import (
    logistic_tuning,
    random_forest_tuning
    )

from model import (
    get_logistic_model,
    get_random_forest_model,
    evaluate_model,
    train_model
    )

from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/cleaned_data.csv"
TARGET_COL = "credit_risk"

def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    return df

def split_features_target(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def make_train_test_split(X, y, test_size: float = 0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y       
        )

def print_results(name, results):
    print(f"\n=== {name} Results ===")
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    if results["roc_auc"] is not None:
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
    
    print("Confusion Matrix:")
    print(results["confusion_matrix"])


def main():
    print("Loading data...")
    
    df = load_data()
    
    X, y = split_features_target(df)
    
    X_train, X_test, y_train, y_test = make_train_test_split(X,y)
    
    print("Tuning logistic regression model...")
    
    best_logistic_row = logistic_tuning(X_train,y_train)
    
    print("Tuning random forest model...")
    
    best_rf_row = random_forest_tuning(X_train, y_train)
    
    print("Training models...")
    
    best_logistic_model = get_logistic_model(best_logistic_row["C"])

    best_max_depth = None
    if isinstance(best_rf_row["max_depth"],int):
        best_max_depth = best_rf_row["max_depth"]

    best_rf_model = get_random_forest_model(n_estimators=int(best_rf_row["n_estimators"]), max_depth=best_max_depth, min_samples_leaf=int(best_rf_row["min_samples_leaf"]))
    
    train_model(best_logistic_model, X_train, y_train)
    train_model(best_rf_model, X_train, y_train)

    print("Evaluating models...")

    logistic_results = evaluate_model(best_logistic_model, X_test, y_test)
    rf_results = evaluate_model(best_rf_model, X_test, y_test)
    
    print_results("Logistic Regression", logistic_results)
    print_results("Random Forest", rf_results)
 
    
if __name__ == "__main__":
    main()