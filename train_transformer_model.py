import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import os
import time
from transformers import get_linear_schedule_with_warmup
from transformer_models import BERTClassifier, RoBERTaClassifier, AutoTransformerClassifier, preprocess_for_transformer, batch_encode_for_transformer
from utils import save_model

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_transformer_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device, patience=5):
    """
    Train a transformer-based model

    Parameters:
    model: The transformer model
    train_dataloader: DataLoader for training data
    val_dataloader: DataLoader for validation data
    optimizer: Optimizer
    scheduler: Learning rate scheduler
    num_epochs: Number of training epochs
    device: Device to train on (cuda/cpu)
    patience: Early stopping patience

    Returns:
    Training history
    """
    criterion = nn.CrossEntropyLoss()

    # Initialize variables for tracking training progress
    history = {
        'loss': [], 'acc': [], 'f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_trues = []

        for batch in train_dataloader:
            # Get batch data
            if len(batch) == 3 and 'token_type_ids' in batch[0]:
                batch_dict, labels = batch[0], batch[1]
                input_ids = batch_dict['input_ids'].to(device)
                attention_mask = batch_dict['attention_mask'].to(device)
                token_type_ids = batch_dict['token_type_ids'].to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                batch_dict, labels = batch[0], batch[1]
                input_ids = batch_dict['input_ids'].to(device)
                attention_mask = batch_dict['attention_mask'].to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Record loss and predictions
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_trues.extend(labels.cpu().numpy())

        # Calculate training metrics
        train_loss = train_loss / len(train_dataloader)
        train_acc = accuracy_score(train_trues, train_preds)
        train_f1 = f1_score(train_trues, train_preds, average='weighted')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Get batch data
                if len(batch) == 3 and 'token_type_ids' in batch[0]:
                    batch_dict, labels = batch[0], batch[1]
                    input_ids = batch_dict['input_ids'].to(device)
                    attention_mask = batch_dict['attention_mask'].to(device)
                    token_type_ids = batch_dict['token_type_ids'].to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    batch_dict, labels = batch[0], batch[1]
                    input_ids = batch_dict['input_ids'].to(device)
                    attention_mask = batch_dict['attention_mask'].to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Record loss and predictions
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_loss = val_loss / len(val_dataloader)
        val_acc = accuracy_score(val_trues, val_preds)
        val_f1 = f1_score(val_trues, val_preds, average='weighted')

        # Update history
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history

def evaluate_transformer_model(model, test_dataloader, device):
    """
    Evaluate a transformer-based model

    Parameters:
    model: The transformer model
    test_dataloader: DataLoader for test data
    device: Device to evaluate on (cuda/cpu)

    Returns:
    Predictions and evaluation metrics
    """
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in test_dataloader:
            # Get batch data
            if len(batch) == 3 and 'token_type_ids' in batch[0]:
                batch_dict, labels = batch[0], batch[1]
                input_ids = batch_dict['input_ids'].to(device)
                attention_mask = batch_dict['attention_mask'].to(device)
                token_type_ids = batch_dict['token_type_ids'].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                batch_dict, labels = batch[0], batch[1]
                input_ids = batch_dict['input_ids'].to(device)
                attention_mask = batch_dict['attention_mask'].to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, average='weighted')
    report = classification_report(all_trues, all_preds)

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'classification_report': report
    }

    return all_preds, metrics

def save_transformer_model(model, tokenizer, model_name):
    """
    Save a transformer model and its tokenizer

    Parameters:
    model: The transformer model
    tokenizer: The tokenizer used with the model
    model_name: Name to save the model under
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save model
    model_path = f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)

    # Save tokenizer info
    tokenizer_info = {
        'model_name': model.model_name,
        'tokenizer_class': tokenizer.__class__.__name__
    }

    with open(f"{model_name}_tokenizer_info.pkl", 'wb') as f:
        pickle.dump(tokenizer_info, f)

    print(f"Model saved as {model_path}")
    print(f"Tokenizer info saved as {model_name}_tokenizer_info.pkl")

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('datasets/expanded_novels.csv')
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution: \n{df['label'].value_counts()}")

    # Split data - handle small datasets
    if len(df) < 30:  # For very small datasets
        # Just use the same data for train, val, and test
        train_df = df.copy()
        val_df = df.copy()
        test_df = df.copy()
        print("Dataset is too small for splitting. Using the same data for train, val, and test.")
    else:
        # For larger datasets, use stratified split
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Define model configurations
    models_config = {
        'BERT': {
            'model_class': BERTClassifier,
            'params': {
                'pretrained_model_name': 'bert-base-chinese',
                'num_classes': len(df['label'].unique()),
                'dropout': 0.1
            },
            'training_params': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'epochs': 10,
                'patience': 3,
                'warmup_steps': 0
            }
        },
        'RoBERTa': {
            'model_class': RoBERTaClassifier,
            'params': {
                'pretrained_model_name': 'hfl/chinese-roberta-wwm-ext',
                'num_classes': len(df['label'].unique()),
                'dropout': 0.1
            },
            'training_params': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'epochs': 10,
                'patience': 3,
                'warmup_steps': 0
            }
        }
    }

    # Train models
    for model_name, config in models_config.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} model...")
        print(f"{'='*50}")

        # Create model
        model = config['model_class'](**config['params']).to(device)
        print(f"Model created: {model_name}")

        # Get tokenizer
        tokenizer = model.get_tokenizer()
        print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

        # Preprocess data
        print("Preprocessing data...")
        max_length = 128  # Maximum sequence length

        # Process train data
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].values
        train_encodings = batch_encode_for_transformer(train_texts, tokenizer, max_length=max_length)

        # Process validation data
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].values
        val_encodings = batch_encode_for_transformer(val_texts, tokenizer, max_length=max_length)

        # Process test data
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].values
        test_encodings = batch_encode_for_transformer(test_texts, tokenizer, max_length=max_length)

        # Create data loaders
        train_dataset = TensorDataset(train_encodings, torch.tensor(train_labels))
        val_dataset = TensorDataset(val_encodings, torch.tensor(val_labels))
        test_dataset = TensorDataset(test_encodings, torch.tensor(test_labels))

        batch_size = config['training_params']['batch_size']
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        # Set up optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=config['training_params']['learning_rate'])

        # Create scheduler with warmup
        total_steps = len(train_dataloader) * config['training_params']['epochs']
        warmup_steps = config['training_params']['warmup_steps']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Train model
        print("Starting training...")
        history = train_transformer_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['training_params']['epochs'],
            device=device,
            patience=config['training_params']['patience']
        )

        # Evaluate model
        print(f"\nEvaluating {model_name} model on test set...")
        preds, metrics = evaluate_transformer_model(model, test_dataloader, device)

        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['f1']:.4f}")
        print(f"Classification Report:\n{metrics['classification_report']}")

        # Save model
        model_filename = f"Deep_{model_name}_Transformer"
        save_transformer_model(model, tokenizer, model_filename)

        # Save training history
        with open(f"{model_filename}_history.pkl", 'wb') as f:
            pickle.dump(history, f)

        # Save evaluation results
        with open(f"{model_filename}_evaluation.pkl", 'wb') as f:
            pickle.dump(metrics, f)

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
