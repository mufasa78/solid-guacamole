import pandas as pd
import os
import sys

def check_csv_file(file_path):
    """
    Check if a CSV file can be properly read and processed
    
    Parameters:
    file_path: Path to the CSV file
    
    Returns:
    success: Boolean indicating if the file can be read
    df: DataFrame containing the data if successful, None otherwise
    error: Error message if unsuccessful, None otherwise
    """
    try:
        # Try to read the file with different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully read {file_path} with encoding {encoding}")
                return True, df, None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return False, None, str(e)
        
        if df is None:
            return False, None, "Failed to read file with any encoding"
        
    except Exception as e:
        return False, None, str(e)

def main():
    # Get the datasets directory
    datasets_dir = 'datasets'
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files in {datasets_dir}")
    
    # Check each file
    for file_name in csv_files:
        file_path = os.path.join(datasets_dir, file_name)
        print(f"\nChecking {file_path}...")
        
        success, df, error = check_csv_file(file_path)
        
        if success:
            # Print basic information about the DataFrame
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First few rows:")
            print(df.head(2))
        else:
            print(f"Error reading {file_path}: {error}")

if __name__ == "__main__":
    main()
