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
    print("Creating books dataset with text and label columns...")
    
    # Check if the required files exist
    if not os.path.exists('datasets/books.csv'):
        print("Error: datasets/books.csv file not found!")
        return
    
    if not os.path.exists('datasets/book_tags.csv'):
        print("Error: datasets/book_tags.csv file not found!")
        return
    
    if not os.path.exists('datasets/tags.csv'):
        print("Error: datasets/tags.csv file not found!")
        return
    
    # Load the datasets
    print("Loading datasets...")
    books_df = pd.read_csv('datasets/books.csv')
    book_tags_df = pd.read_csv('datasets/book_tags.csv')
    tags_df = pd.read_csv('datasets/tags.csv')
    
    print(f"Loaded {len(books_df)} books")
    print(f"Loaded {len(book_tags_df)} book-tag associations")
    print(f"Loaded {len(tags_df)} tags")
    
    # Get the most common tags
    print("Finding the most common tags...")
    tag_counts = book_tags_df['tag_id'].value_counts()
    common_tags = tag_counts.head(8).index.tolist()
    
    # Get the tag names for the common tags
    common_tag_names = tags_df[tags_df['tag_id'].isin(common_tags)]
    print("Most common tags:")
    for _, row in common_tag_names.iterrows():
        print(f"  {row['tag_id']}: {row['tag_name']}")
    
    # Create a mapping of tag_id to label (0-7)
    tag_mapping = {tag_id: i for i, tag_id in enumerate(common_tags)}
    
    # Save the tag mapping for reference
    tag_mapping_df = pd.DataFrame({
        'tag_id': list(tag_mapping.keys()),
        'label': list(tag_mapping.values())
    })
    tag_mapping_df = pd.merge(tag_mapping_df, tags_df[['tag_id', 'tag_name']], on='tag_id')
    tag_mapping_df.to_csv('datasets/tag_label_mapping.csv', index=False)
    print(f"Saved tag mapping with {len(tag_mapping)} categories")
    
    # Filter book_tags to only include the common tags
    filtered_book_tags = book_tags_df[book_tags_df['tag_id'].isin(common_tags)]
    
    # Get the most relevant tag for each book (highest count)
    book_primary_tags = filtered_book_tags.sort_values('count', ascending=False).drop_duplicates('goodreads_book_id')
    
    # Map tag_id to label
    book_primary_tags['label'] = book_primary_tags['tag_id'].map(tag_mapping)
    
    # Merge with books to get the book details
    books_with_labels = pd.merge(
        books_df,
        book_primary_tags[['goodreads_book_id', 'label']],
        left_on='book_id',
        right_on='goodreads_book_id',
        how='inner'
    )
    
    # Create the text field by combining title and authors
    books_with_labels['text'] = books_with_labels.apply(
        lambda row: f"{row['title']} by {row['authors']}", 
        axis=1
    )
    
    # Clean the text
    books_with_labels['text'] = books_with_labels['text'].apply(clean_text)
    
    # Select only the necessary columns
    final_df = books_with_labels[['text', 'label']]
    
    # Save the dataset
    output_file = 'datasets/books_with_labels.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"Created dataset saved to {output_file}")
    print(f"Total samples: {len(final_df)}")
    print(f"Label distribution:\n{final_df['label'].value_counts()}")

if __name__ == "__main__":
    main()
