import pandas as pd
import numpy as np
import pathlib

categorical_cols=[
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
    ]

numeric_cols = [
    "number_credits",
    "people_liable",
    "duration",
    "amount",
    "installment_rate",
    "age",
    "credit_risk",
    "present_residence"    
    ]

log_cols = [
    "duration",
    "amount",
    "age"
    ]

filename = 'german_credit.csv'
output = 'cleaned_data.csv'
    
project_root = pathlib.Path(__file__).resolve().parents[1]

raw_data_dir = project_root / 'data' / 'raw'
cleaned_data_dir = project_root / 'data' / 'processed'
    
raw_file_path = raw_data_dir / filename
cleaned_file_path = cleaned_data_dir / output




def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame, output_path = None) -> pd.DataFrame:
    
    df = df.copy()
    
    df = pd.get_dummies(df, categorical_cols, drop_first = True)
    
    df[log_cols] = np.log1p(df[log_cols])
    
    if output_path:
        df.to_csv(output_path, index = False)
   
    return df


if __name__ == '__main__':
    df = load_data(raw_file_path)
    clean_data(df, cleaned_file_path)