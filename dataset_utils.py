import pandas as pd
import os

def load_dataset_with_required_columns(dataset_path):
    """
    Load a dataset and ensure it has the required 'text' and 'label' columns
    
    Parameters:
    dataset_path: Path to the dataset
    
    Returns:
    df: DataFrame containing the dataset with 'text' and 'label' columns
    message: Message about the loading process
    """
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Check if the dataset has the required columns
        if 'text' in df.columns and 'label' in df.columns:
            return df, f"Dataset loaded successfully: {dataset_path}"
        else:
            # Try to find a fixed version of the dataset
            fixed_path = dataset_path.replace('.csv', '_fixed.csv')
            if os.path.exists(fixed_path):
                df = pd.read_csv(fixed_path)
                return df, f"Using fixed version of the dataset: {fixed_path}"
            else:
                return None, f"Error: Dataset does not have required 'text' and 'label' columns: {dataset_path}"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def get_available_datasets(datasets_dir='datasets'):
    """
    Get a list of available datasets with information about whether they have the required columns
    
    Parameters:
    datasets_dir: Directory containing the datasets
    
    Returns:
    datasets: List of dictionaries with information about each dataset
    """
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    
    # Check each dataset
    datasets = []
    for file_name in csv_files:
        dataset_path = os.path.join(datasets_dir, file_name)
        try:
            df = pd.read_csv(dataset_path)
            has_required_columns = 'text' in df.columns and 'label' in df.columns
            
            # Check if a fixed version exists
            fixed_path = dataset_path.replace('.csv', '_fixed.csv')
            has_fixed_version = os.path.exists(fixed_path)
            
            datasets.append({
                'name': file_name,
                'path': dataset_path,
                'has_required_columns': has_required_columns,
                'has_fixed_version': has_fixed_version,
                'fixed_path': fixed_path if has_fixed_version else None,
                'records': len(df)
            })
        except:
            # Skip datasets that can't be loaded
            pass
    
    return datasets
