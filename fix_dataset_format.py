import pandas as pd
import os
import sys
import re

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

def fix_dataset_format(input_file, output_file=None):
    """
    Fix a dataset to ensure it has 'text' and 'label' columns
    
    Parameters:
    input_file: Path to the input CSV file
    output_file: Path to the output CSV file (default: input_file with '_fixed' suffix)
    
    Returns:
    df: DataFrame containing the fixed data
    """
    print(f"Fixing dataset format: {input_file}")
    
    # Set default output file if not provided
    if output_file is None:
        output_file = input_file.replace('.csv', '_fixed.csv')
    
    # Load the dataset
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if the dataset already has the required columns
    if 'text' in df.columns and 'label' in df.columns:
        print("Dataset already has 'text' and 'label' columns")
        return df
    
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
                         ['label', 'category', 'class', 'genre', 'tag', 'type'])]
        
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
    
    print(f"Fixed dataset saved to {output_file}")
    print(f"Total samples: {len(fixed_df)}")
    print(f"Label distribution:\n{fixed_df['label'].value_counts()}")
    
    return fixed_df

def main():
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found!")
            return
        
        # Fix the dataset format
        fix_dataset_format(input_file, output_file)
    else:
        print("Usage: python fix_dataset_format.py input_file.csv [output_file.csv]")
        print("If output_file is not provided, input_file_fixed.csv will be used")

if __name__ == "__main__":
    main()
