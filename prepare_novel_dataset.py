import pandas as pd
import numpy as np
import re
import os
import jieba
from sklearn.model_selection import train_test_split

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
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def prepare_novel_dataset(input_file, output_file, text_column='text', label_column='label', min_length=10, max_samples=None):
    """
    Prepare a novel dataset for transformer models
    
    Parameters:
    input_file: Path to the input CSV file
    output_file: Path to the output CSV file
    text_column: Name of the column containing the text
    label_column: Name of the column containing the label
    min_length: Minimum length of text to include
    max_samples: Maximum number of samples to include per class
    
    Returns:
    df: DataFrame containing the prepared data
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    
    # Check if the required columns exist
    if text_column not in df.columns:
        # If using novels_0.1.4.csv, we'll create a text column from the description
        if 'name' in df.columns and 'tags' in df.columns:
            print(f"Creating text column from name and tags...")
            df[text_column] = df['name'] + ". " + df['tags'].apply(lambda x: x.replace("'", "").replace("[", "").replace("]", "").replace(",", " "))
        else:
            raise ValueError(f"Column {text_column} not found in the dataset")
    
    if label_column not in df.columns:
        # If using novels_0.1.4.csv, we'll create a label column from the genres
        if 'genres' in df.columns:
            print(f"Creating label column from genres...")
            
            # Extract the first genre for each novel
            def extract_first_genre(genres_str):
                try:
                    genres = eval(genres_str)
                    if genres and len(genres) > 0:
                        return genres[0]
                    return "unknown"
                except:
                    return "unknown"
            
            df[label_column] = df['genres'].apply(extract_first_genre)
        else:
            raise ValueError(f"Column {label_column} not found in the dataset")
    
    # Clean the text
    print("Cleaning text...")
    df[text_column] = df[text_column].apply(clean_text)
    
    # Remove rows with empty text or text shorter than min_length
    print(f"Removing rows with text shorter than {min_length} characters...")
    df = df[df[text_column].apply(lambda x: len(str(x)) >= min_length)]
    
    # Balance the dataset if max_samples is provided
    if max_samples is not None:
        print(f"Balancing dataset with max {max_samples} samples per class...")
        balanced_df = pd.DataFrame()
        for label in df[label_column].unique():
            class_df = df[df[label_column] == label]
            if len(class_df) > max_samples:
                class_df = class_df.sample(max_samples, random_state=42)
            balanced_df = pd.concat([balanced_df, class_df])
        df = balanced_df
    
    # Map labels to integers
    print("Mapping labels to integers...")
    label_mapping = {label: i for i, label in enumerate(sorted(df[label_column].unique()))}
    df['label_id'] = df[label_column].map(label_mapping)
    
    # Save the label mapping
    label_mapping_df = pd.DataFrame({
        'label': list(label_mapping.keys()),
        'label_id': list(label_mapping.values())
    })
    label_mapping_df.to_csv(output_file.replace('.csv', '_label_mapping.csv'), index=False)
    
    # Select only the necessary columns
    df = df[[text_column, 'label_id']]
    df.columns = ['text', 'label']
    
    # Save the prepared dataset
    print(f"Saving prepared dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"Final shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df

def main():
    # Create output directory if it doesn't exist
    os.makedirs('prepared_datasets', exist_ok=True)
    
    # Prepare the crosslingual_literary dataset
    try:
        prepare_novel_dataset(
            input_file='datasets/crosslingual_literary.csv',
            output_file='prepared_datasets/prepared_literary.csv',
            text_column='text',
            label_column='emotion',
            min_length=5
        )
    except Exception as e:
        print(f"Error preparing crosslingual_literary.csv: {e}")
    
    # Prepare the novels dataset
    try:
        prepare_novel_dataset(
            input_file='datasets/novels_0.1.4.csv',
            output_file='prepared_datasets/prepared_novels.csv',
            text_column='text',
            label_column='label',
            min_length=10,
            max_samples=1000
        )
    except Exception as e:
        print(f"Error preparing novels_0.1.4.csv: {e}")

if __name__ == "__main__":
    main()
