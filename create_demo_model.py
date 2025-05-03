import pandas as pd
import torch
import os
import pickle
from models import CNNModel
from preprocess import preprocess_text, build_vocab, tokenize_and_pad
from utils import save_model

# 加载示例数据
df = pd.read_csv('datasets/sample_novels.csv')
print(f"Loaded {len(df)} samples")

# 预处理文本
df['processed_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing completed")

# 构建词汇表
vocab = build_vocab(df['processed_text'])
print(f"Vocabulary built with {len(vocab)} tokens")

# 创建一个简单的CNN模型用于演示
model = CNNModel(
    vocab_size=len(vocab),
    embedding_dim=100,
    num_filters=128,
    filter_sizes=[3, 4, 5],
    num_classes=8,
    dropout=0.3
)

# 保存模型和词汇表
model_name = "Demo_CNN_Model"
save_model(model, vocab, model_name)
print(f"Demo model saved as {model_name}.pt")

# 保存处理后的数据以便在应用中使用
with open('processed_sample_data.pkl', 'wb') as f:
    pickle.dump({'df': df, 'vocab': vocab}, f)
print("Processed data saved for app use")
