"""
Investigate DeepWeeds dataset structure
"""
from datasets import load_dataset

print("Loading deepweeds dataset...")
ds = load_dataset("deepweeds")

print(f"\nKeys: {list(ds.keys())}")
print(f"\nFeatures: {ds['train'].features}")
print(f"\nFirst item keys: {list(ds['train'][0].keys())}")
print(f"\nFirst item: {ds['train'][0]}")
print(f"\nDataset description:")
print(ds['train'].description)
print(f"\nDataset info:")
print(ds['train'].info)