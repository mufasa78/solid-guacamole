import os
import pandas as pd
from dataset_utils import load_dataset_with_required_columns, get_available_datasets

def test_load_dataset():
    """
    Test the load_dataset_with_required_columns function
    """
    print("Testing load_dataset_with_required_columns function...")
    
    # Test with a dataset that has the required columns
    df, message = load_dataset_with_required_columns('datasets/books_fixed.csv')
    print(f"Test 1: {message}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print()
    
    # Test with a dataset that doesn't have the required columns
    df, message = load_dataset_with_required_columns('datasets/books.csv')
    print(f"Test 2: {message}")
    if df is not None:
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    print()

def test_get_available_datasets():
    """
    Test the get_available_datasets function
    """
    print("Testing get_available_datasets function...")
    
    datasets = get_available_datasets('datasets')
    print(f"Found {len(datasets)} datasets")
    
    for i, dataset in enumerate(datasets):
        print(f"Dataset {i+1}: {dataset['name']}")
        print(f"  Path: {dataset['path']}")
        print(f"  Has required columns: {dataset['has_required_columns']}")
        print(f"  Has fixed version: {dataset['has_fixed_version']}")
        if dataset['has_fixed_version']:
            print(f"  Fixed path: {dataset['fixed_path']}")
        print(f"  Records: {dataset['records']}")
        print()

if __name__ == "__main__":
    test_load_dataset()
    print("-" * 50)
    test_get_available_datasets()
