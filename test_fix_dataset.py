import pandas as pd
from fix_dataset_format import fix_dataset_format

# Test the fix_dataset_format function
print("Testing fix_dataset_format function...")

# Test with a dataset that doesn't have the required columns
try:
    fixed_df = fix_dataset_format('datasets/books.csv')
    if fixed_df is not None:
        print(f"Successfully fixed dataset. Shape: {fixed_df.shape}")
        print(f"Columns: {fixed_df.columns.tolist()}")
    else:
        print("Failed to fix dataset.")
except Exception as e:
    print(f"Error: {str(e)}")
