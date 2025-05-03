import pandas as pd

# Try to read the crosslingual_literary.csv file
try:
    df = pd.read_csv('datasets/crosslingual_literary.csv')
    print("Successfully read crosslingual_literary.csv")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head(2))
except Exception as e:
    print(f"Error reading crosslingual_literary.csv: {e}")

# Try to read the novels_0.1.4.csv file
try:
    df = pd.read_csv('datasets/novels_0.1.4.csv')
    print("\nSuccessfully read novels_0.1.4.csv")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head(2))
except Exception as e:
    print(f"Error reading novels_0.1.4.csv: {e}")
