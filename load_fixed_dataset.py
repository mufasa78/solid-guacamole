import pandas as pd
import os
import sys
import streamlit as st

def load_dataset(dataset_path):
    """
    Load a dataset and check if it has the required 'text' and 'label' columns
    
    Parameters:
    dataset_path: Path to the dataset
    
    Returns:
    df: DataFrame containing the dataset
    """
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Check if the dataset has the required columns
        if 'text' in df.columns and 'label' in df.columns:
            print(f"Dataset loaded successfully: {dataset_path}")
            print(f"Total samples: {len(df)}")
            print(f"Label distribution:\n{df['label'].value_counts()}")
            return df
        else:
            # Try to find a fixed version of the dataset
            fixed_path = dataset_path.replace('.csv', '_fixed.csv')
            if os.path.exists(fixed_path):
                print(f"Dataset does not have required columns. Loading fixed version: {fixed_path}")
                df = pd.read_csv(fixed_path)
                print(f"Fixed dataset loaded successfully: {fixed_path}")
                print(f"Total samples: {len(df)}")
                print(f"Label distribution:\n{df['label'].value_counts()}")
                return df
            else:
                print(f"Error: Dataset does not have required 'text' and 'label' columns: {dataset_path}")
                print("Please run fix_all_datasets.py to fix all datasets")
                return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def main():
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset '{dataset_path}' not found!")
            return
        
        # Load the dataset
        df = load_dataset(dataset_path)
        
        if df is not None:
            # Display the first few rows
            print("\nFirst few rows:")
            print(df.head())
    else:
        print("Usage: python load_fixed_dataset.py dataset_path")

if __name__ == "__main__":
    main()
