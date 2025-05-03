import pandas as pd
import os
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

def main():
    print("Converting books dataset to the required format...")
    
    # Check if the books.csv file exists
    if not os.path.exists('datasets/books.csv'):
        print("Error: datasets/books.csv file not found!")
        return
    
    # Load the books.csv file
    print("Loading books.csv...")
    books_df = pd.read_csv('datasets/books.csv')
    print(f"Loaded {len(books_df)} books")
    
    # Display the columns
    print(f"Columns in books.csv: {books_df.columns.tolist()}")
    
    # Check if book_tags.csv exists for genre information
    has_tags = os.path.exists('datasets/book_tags.csv') and os.path.exists('datasets/tags.csv')
    
    if has_tags:
        print("Loading book_tags.csv and tags.csv for genre information...")
        book_tags_df = pd.read_csv('datasets/book_tags.csv')
        tags_df = pd.read_csv('datasets/tags.csv')
        
        # Merge tags with book_tags to get tag names
        book_tags_with_names = pd.merge(
            book_tags_df, 
            tags_df, 
            left_on='tag_id', 
            right_on='tag_id'
        )
        
        # Get the most common tag for each book
        top_tags = book_tags_with_names.sort_values('count', ascending=False).drop_duplicates('goodreads_book_id')
        
        # Merge with books to add tag information
        books_with_tags = pd.merge(
            books_df,
            top_tags[['goodreads_book_id', 'tag_name']],
            left_on='book_id',
            right_on='goodreads_book_id',
            how='left'
        )
        
        # Create a mapping of tag names to numeric labels
        unique_tags = books_with_tags['tag_name'].dropna().unique()
        tag_mapping = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}
        
        # Save the tag mapping for reference
        tag_mapping_df = pd.DataFrame({
            'tag_name': list(tag_mapping.keys()),
            'label': list(tag_mapping.values())
        })
        tag_mapping_df.to_csv('datasets/tag_mapping.csv', index=False)
        print(f"Saved tag mapping with {len(tag_mapping)} categories")
        
        # Create the text field by combining title and authors
        books_with_tags['text'] = books_with_tags.apply(
            lambda row: f"{row['title']} by {row['authors']}", 
            axis=1
        )
        
        # Clean the text
        books_with_tags['text'] = books_with_tags['text'].apply(clean_text)
        
        # Map tag names to numeric labels
        books_with_tags['label'] = books_with_tags['tag_name'].map(tag_mapping)
        
        # Drop rows with missing labels
        books_with_tags = books_with_tags.dropna(subset=['label'])
        
        # Select only the necessary columns
        transformed_df = books_with_tags[['text', 'label']]
        
    else:
        print("No tag information found. Creating basic dataset...")
        
        # Create the text field by combining title and authors
        books_df['text'] = books_df.apply(
            lambda row: f"{row['title']} by {row['authors']}", 
            axis=1
        )
        
        # Clean the text
        books_df['text'] = books_df['text'].apply(clean_text)
        
        # Create a simple label based on the book_id (just for demonstration)
        # In a real scenario, you would want to use meaningful categories
        books_df['label'] = books_df['book_id'] % 8  # Using 8 categories as in the genre_mapping
        
        # Select only the necessary columns
        transformed_df = books_df[['text', 'label']]
    
    # Save the transformed dataset
    output_file = 'datasets/books_transformed.csv'
    transformed_df.to_csv(output_file, index=False)
    
    print(f"Transformed dataset saved to {output_file}")
    print(f"Total samples: {len(transformed_df)}")
    print(f"Label distribution:\n{transformed_df['label'].value_counts()}")

if __name__ == "__main__":
    main()
