import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import os
import time
from transformers import get_linear_schedule_with_warmup
from transformer_models import BERTClassifier, batch_encode_for_transformer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('datasets/expanded_novels.csv')
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution: \n{df['label'].value_counts()}")
    
    # Use all data for training, validation, and testing
    train_df = df.copy()
    val_df = df.copy()
    test_df = df.copy()
    print("Using the same data for train, val, and test.")
    
    # Create BERT model
    model = BERTClassifier(
        pretrained_model_name='bert-base-chinese',
        num_classes=len(df['label'].unique()),
        dropout=0.1
    ).to(device)
    print("Model created: BERT")
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Preprocess data
    print("Preprocessing data...")
    max_length = 128  # Maximum sequence length
    
    # Process train data
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].values
    
    # Tokenize all texts
    print("Tokenizing texts...")
    encodings = tokenizer(
        train_texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    labels = torch.tensor(train_labels).to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # Train model
    print("Starting training...")
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        train_preds = preds.cpu().numpy()
        train_trues = labels.cpu().numpy()
        
        # Calculate metrics
        train_acc = accuracy_score(train_trues, train_preds)
        train_f1 = f1_score(train_trues, train_preds, average='weighted')
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, 1)
        test_preds = preds.cpu().numpy()
        test_trues = labels.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(test_trues, test_preds)
    f1 = f1_score(test_trues, test_preds, average='weighted')
    report = classification_report(test_trues, test_preds)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Classification Report:\n{report}")
    
    # Save model
    model_filename = "Deep_BERT_Transformer"
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = f"{model_filename}.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer info
    tokenizer_info = {
        'model_name': model.model_name,
        'tokenizer_class': tokenizer.__class__.__name__
    }
    
    with open(f"{model_filename}_tokenizer_info.pkl", 'wb') as f:
        pickle.dump(tokenizer_info, f)
    
    print(f"Model saved as {model_path}")
    print(f"Tokenizer info saved as {model_filename}_tokenizer_info.pkl")
    
    # Save processed data for app use
    processed_data = {
        'model_name': model.model_name,
        'tokenizer_class': tokenizer.__class__.__name__,
        'max_length': max_length,
        'class_distribution': df['label'].value_counts().to_dict(),
        'num_classes': len(df['label'].unique()),
        'class_names': {
            0: "玄幻",
            1: "武侠",
            2: "都市",
            3: "言情",
            4: "科幻",
            5: "历史",
            6: "游戏",
            7: "悬疑"
        }
    }
    
    with open(f'{model_filename}_processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed data saved for app use as {model_filename}_processed_data.pkl")

if __name__ == "__main__":
    main()
