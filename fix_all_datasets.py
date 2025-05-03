import pandas as pd
import os
import re
import sys
from datetime import datetime

def clean_text(text):
    """
    Clean and preprocess text
    
    Parameters:
    text: Text to clean
    
    Returns:
    cleaned_text: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', str(text))
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def fix_dataset_format(input_file, output_dir=None, overwrite=False):
    """
    Fix a dataset to ensure it has 'text' and 'label' columns
    
    Parameters:
    input_file: Path to the input CSV file
    output_dir: Directory to save the fixed dataset (default: same as input)
    overwrite: Whether to overwrite the original file (default: False)
    
    Returns:
    result: Dictionary with information about the fix operation
    """
    result = {
        'file': input_file,
        'status': 'unknown',
        'message': '',
        'output_file': '',
        'records': 0,
        'fixed': False
    }
    
    try:
        print(f"Checking dataset: {input_file}")
        
        # Load the dataset
        df = pd.read_csv(input_file)
        result['records'] = len(df)
        
        # Check if the dataset already has the required columns
        if 'text' in df.columns and 'label' in df.columns:
            result['status'] = 'ok'
            result['message'] = "Dataset already has 'text' and 'label' columns"
            print(result['message'])
            return result
        
        # Determine the output file path
        file_name = os.path.basename(input_file)
        if overwrite:
            output_file = input_file
        else:
            if output_dir:
                output_file = os.path.join(output_dir, file_name.replace('.csv', '_fixed.csv'))
            else:
                output_file = input_file.replace('.csv', '_fixed.csv')
        
        result['output_file'] = output_file
        
        # Create a new DataFrame for the fixed data
        fixed_df = pd.DataFrame()
        
        # Fix the 'text' column
        if 'text' not in df.columns:
            # Try to find columns that might contain text content
            text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                            ['title', 'name', 'description', 'content', 'text', 'summary', 'author'])]
            
            if text_columns:
                print(f"Creating 'text' column from: {', '.join(text_columns)}")
                # Combine the text columns
                fixed_df['text'] = df[text_columns].apply(
                    lambda row: ' '.join([str(val) for val in row if pd.notna(val)]), 
                    axis=1
                )
            else:
                # If no suitable text columns found, use the first string column
                string_columns = [col for col in df.columns if df[col].dtype == 'object']
                if string_columns:
                    print(f"No obvious text columns found. Using '{string_columns[0]}' as 'text' column")
                    fixed_df['text'] = df[string_columns[0]]
                else:
                    print("No suitable columns found for 'text'. Using row indices as placeholder text")
                    fixed_df['text'] = [f"Sample {i}" for i in range(len(df))]
        else:
            fixed_df['text'] = df['text']
        
        # Clean the text
        fixed_df['text'] = fixed_df['text'].apply(clean_text)
        
        # Fix the 'label' column
        if 'label' not in df.columns:
            # Try to find columns that might contain label information
            label_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                             ['label', 'category', 'class', 'genre', 'tag', 'type', 'emotion'])]
            
            if label_columns:
                print(f"Creating 'label' column from: {label_columns[0]}")
                # Use the first label column
                label_col = label_columns[0]
                
                # Check if the label column contains numeric values
                if pd.api.types.is_numeric_dtype(df[label_col]):
                    fixed_df['label'] = df[label_col]
                else:
                    # Map categorical labels to integers
                    label_mapping = {label: i for i, label in enumerate(sorted(df[label_col].unique()))}
                    fixed_df['label'] = df[label_col].map(label_mapping)
                    
                    # Save the label mapping for reference
                    label_mapping_df = pd.DataFrame({
                        'original_label': list(label_mapping.keys()),
                        'numeric_label': list(label_mapping.values())
                    })
                    mapping_file = output_file.replace('.csv', '_label_mapping.csv')
                    label_mapping_df.to_csv(mapping_file, index=False)
                    print(f"Saved label mapping to {mapping_file}")
            else:
                # If no suitable label columns found, use a simple numeric label
                print("No suitable columns found for 'label'. Using column index modulo 8 as label")
                fixed_df['label'] = df.index % 8  # Using 8 categories as in the genre_mapping
        else:
            fixed_df['label'] = df['label']
        
        # Ensure label is an integer
        fixed_df['label'] = fixed_df['label'].astype(int)
        
        # Save the fixed dataset
        fixed_df.to_csv(output_file, index=False)
        
        result['status'] = 'fixed'
        result['message'] = f"Fixed dataset saved to {output_file}"
        result['fixed'] = True
        print(result['message'])
        print(f"Total samples: {len(fixed_df)}")
        print(f"Label distribution:\n{fixed_df['label'].value_counts()}")
        
        return result
    
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Error fixing dataset: {str(e)}"
        print(result['message'])
        return result

def fix_all_datasets(datasets_dir='datasets', output_dir=None, overwrite=False):
    """
    Fix all datasets in a directory
    
    Parameters:
    datasets_dir: Directory containing the datasets
    output_dir: Directory to save the fixed datasets (default: same as input)
    overwrite: Whether to overwrite the original files (default: False)
    
    Returns:
    results: List of dictionaries with information about each fix operation
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(datasets_dir) if f.endswith('.csv') and not f.endswith('_fixed.csv')]
    
    print(f"Found {len(csv_files)} CSV files in {datasets_dir}")
    
    # Fix each dataset
    results = []
    for file_name in csv_files:
        input_file = os.path.join(datasets_dir, file_name)
        result = fix_dataset_format(input_file, output_dir, overwrite)
        results.append(result)
    
    return results

def generate_report(results, report_file='dataset_fix_report.html'):
    """
    Generate an HTML report of the fix operations
    
    Parameters:
    results: List of dictionaries with information about each fix operation
    report_file: Path to the report file
    """
    # Create the HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Fix Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .ok {{ color: green; }}
            .fixed {{ color: blue; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Dataset Fix Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <table>
            <tr>
                <th>File</th>
                <th>Status</th>
                <th>Records</th>
                <th>Output File</th>
                <th>Message</th>
            </tr>
    """
    
    for result in results:
        status_class = result['status']
        html += f"""
            <tr>
                <td>{result['file']}</td>
                <td class="{status_class}">{result['status'].upper()}</td>
                <td>{result['records']}</td>
                <td>{result['output_file'] if result['fixed'] else '-'}</td>
                <td>{result['message']}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    # Save the report
    with open(report_file, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {report_file}")

def main():
    # Parse command-line arguments
    datasets_dir = 'datasets'
    output_dir = None
    overwrite = False
    
    if len(sys.argv) > 1:
        datasets_dir = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'overwrite':
        overwrite = True
    
    # Fix all datasets
    results = fix_all_datasets(datasets_dir, output_dir, overwrite)
    
    # Generate a report
    generate_report(results)
    
    # Print summary
    fixed_count = sum(1 for result in results if result['fixed'])
    ok_count = sum(1 for result in results if result['status'] == 'ok')
    error_count = sum(1 for result in results if result['status'] == 'error')
    
    print("\nSummary:")
    print(f"Total datasets: {len(results)}")
    print(f"Already OK: {ok_count}")
    print(f"Fixed: {fixed_count}")
    print(f"Errors: {error_count}")
    print("\nSee dataset_fix_report.html for details")

if __name__ == "__main__":
    main()
