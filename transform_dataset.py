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

def transform_dataset(input_file, output_file=None):
    """
    Transform a dataset to include 'text' and 'label' columns
    
    Parameters:
    input_file: Path to the input CSV file
    output_file: Path to the output CSV file (default: input_file with '_transformed' suffix)
    
    Returns:
    df: DataFrame containing the transformed data
    """
    print(f"Transforming dataset: {input_file}")
    
    # Set default output file if not provided
    if output_file is None:
        output_file = input_file.replace('.csv', '_transformed.csv')
    
    # Load the dataset
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if the dataset already has the required columns
    if 'text' in df.columns and 'label' in df.columns:
        print("Dataset already has 'text' and 'label' columns")
        return df
    
    # Create a new DataFrame for the transformed data
    transformed_df = pd.DataFrame()
    
    # Create the 'text' column based on available columns
    if 'text' not in df.columns:
        # Try to find columns that might contain text content
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                        ['title', 'name', 'description', 'content', 'text', 'summary', 'author'])]
        
        if text_columns:
            print(f"Creating 'text' column from: {', '.join(text_columns)}")
            # Combine the text columns
            transformed_df['text'] = df[text_columns].apply(
                lambda row: ' '.join([str(val) for val in row if pd.notna(val)]), 
                axis=1
            )
        else:
            # If no suitable text columns found, use the first string column
            string_columns = [col for col in df.columns if df[col].dtype == 'object']
            if string_columns:
                print(f"No obvious text columns found. Using '{string_columns[0]}' as 'text' column")
                transformed_df['text'] = df[string_columns[0]]
            else:
                print("No suitable columns found for 'text'. Using row indices as placeholder text")
                transformed_df['text'] = [f"Sample {i}" for i in range(len(df))]
    else:
        transformed_df['text'] = df['text']
    
    # Clean the text
    transformed_df['text'] = transformed_df['text'].apply(clean_text)
    
    # Create the 'label' column based on available columns
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
                transformed_df['label'] = df[label_col]
            else:
                # Map categorical labels to integers
                label_mapping = {label: i for i, label in enumerate(sorted(df[label_col].unique()))}
                transformed_df['label'] = df[label_col].map(label_mapping)
                
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
            print("No suitable columns found for 'label'. Using a default label of 0")
            transformed_df['label'] = 0
    else:
        transformed_df['label'] = df['label']
    
    # Save the transformed dataset
    transformed_df.to_csv(output_file, index=False)
    
    print(f"Transformed dataset saved to {output_file}")
    print(f"Total samples: {len(transformed_df)}")
    print(f"Label distribution:\n{transformed_df['label'].value_counts()}")
    
    return transformed_df

def main():
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found!")
            return
        
        # Transform the dataset
        transform_dataset(input_file, output_file)
    else:
        print("Usage: python transform_dataset.py input_file.csv [output_file.csv]")
        print("If output_file is not provided, input_file_transformed.csv will be used")

if __name__ == "__main__":
    main()
