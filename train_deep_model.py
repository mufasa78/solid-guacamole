import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os
from models import CNNModel, RNNModel, TransformerModel
from preprocess import preprocess_text, build_vocab, tokenize_and_pad
from utils import save_model
from train import train_model, evaluate_model

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载扩展数据集
df = pd.read_csv('datasets/expanded_novels.csv')
print(f"Loaded {len(df)} samples")
print(f"Class distribution: \n{df['label'].value_counts()}")

# 预处理文本
print("Preprocessing text...")
df['processed_text'] = df['text'].apply(preprocess_text)

# 构建词汇表
print("Building vocabulary...")
vocab = build_vocab(df['processed_text'], min_freq=1)
print(f"Vocabulary size: {len(vocab)}")

# 将文本转换为序列并填充
max_len = 200  # 设置最大长度
print(f"Tokenizing and padding sequences to max length {max_len}...")
X = df['processed_text'].apply(lambda x: tokenize_and_pad(x, vocab, max_len)).tolist()
y = df['label'].values

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# 定义模型参数
models_config = {
    'CNN': {
        'model_class': CNNModel,
        'params': {
            'vocab_size': len(vocab),
            'embedding_dim': 200,
            'num_filters': 256,
            'filter_sizes': [2, 3, 4, 5],
            'num_classes': len(df['label'].unique()),
            'dropout': 0.5
        },
        'training_params': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'patience': 10,
            'optimizer_type': 'adam'
        }
    },
    'RNN': {
        'model_class': RNNModel,
        'params': {
            'vocab_size': len(vocab),
            'embedding_dim': 200,
            'hidden_dim': 256,
            'num_layers': 2,
            'num_classes': len(df['label'].unique()),
            'dropout': 0.5,
            'bidirectional': True
        },
        'training_params': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'patience': 10,
            'optimizer_type': 'adam'
        }
    },
    'Transformer': {
        'model_class': TransformerModel,
        'params': {
            'vocab_size': len(vocab),
            'embedding_dim': 200,
            'nhead': 8,
            'dim_feedforward': 512,
            'num_layers': 4,
            'num_classes': len(df['label'].unique()),
            'dropout': 0.3,
            'max_len': max_len
        },
        'training_params': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'patience': 15,
            'optimizer_type': 'adam'
        }
    }
}

# 训练所有模型
for model_name, config in models_config.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name} model...")
    print(f"{'='*50}")
    
    # 创建模型
    model = config['model_class'](**config['params']).to(device)
    print(f"Model architecture:\n{model}")
    
    # 训练模型
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        **config['training_params']
    )
    
    # 评估模型
    print(f"\nEvaluating {model_name} model on test set...")
    preds, metrics = evaluate_model(model, X_test, y_test, device)
    
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")
    print(f"Classification Report:\n{metrics['classification_report']}")
    
    # 保存模型
    model_filename = f"Deep_{model_name}_Model"
    save_model(model, vocab, model_filename)
    print(f"Model saved as {model_filename}.pt")
    
    # 保存训练历史
    with open(f"{model_filename}_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    
    # 保存评估结果
    with open(f"{model_filename}_evaluation.pkl", 'wb') as f:
        pickle.dump(metrics, f)

print("\nAll models trained and evaluated successfully!")

# 保存处理后的数据以便在应用中使用
processed_data = {
    'vocab': vocab,
    'max_len': max_len,
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

with open('deep_processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)
print("Processed data saved for app use")
