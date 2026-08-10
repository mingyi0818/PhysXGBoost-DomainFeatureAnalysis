from huggingface_hub import hf_hub_download
import os
import sys

local_dir = "D:/datasets/WeedSense"
os.makedirs(local_dir, exist_ok=True)

print("Downloading train_data.csv...")
try:
    path = hf_hub_download(
        repo_id='baselab/weedsense',
        repo_type='dataset',
        filename='train/train_data.csv',
        local_dir=local_dir,
    )
    print(f'Downloaded: {path}')
    import pandas as pd
    df = pd.read_csv(path)
    print(f'Total rows: {len(df)}')
    print(f'Species: {df["species"].nunique()}')
    print(df['species'].value_counts().to_string())
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
