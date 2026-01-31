import requests
import os
import pathlib

url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"

def download_data():
    
    filename = 'german_credit.csv'
    
    project_root = pathlib.Path(__file__).resolve().parents[1]

    raw_data_dir = project_root / 'data' / 'raw'
    
    file_path = raw_data_dir / filename

    if file_path.exists():
        print(f"File already exists at {file_path}")
        return

    print("Downloading data...")
    
    response = requests.get(url)
    
    if response.status_code == 200:
        
        with open(file_path,'wb') as f:
            f.write(response.content)
        print(f"File '{filename}' successfully downloaded and saved to: {file_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    



if __name__ == '__main__':
    download_data()