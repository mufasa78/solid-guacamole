import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    基于CNN的文本分类模型
    """
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 多个卷积层，每个卷积核大小不同，捕捉不同长度的特征
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        # 应用多个卷积层
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, num_filters, seq_len-k+1)]
        
        # 最大池化
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [(batch_size, num_filters)]
        
        # 拼接所有卷积结果
        cat = torch.cat(pooled, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        
        # Dropout和全连接层
        cat = self.dropout(cat)
        out = self.fc(cat)  # [batch_size, num_classes]
        
        return out

class RNNModel(nn.Module):
    """
    基于RNN (LSTM/GRU) 的文本分类模型
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, bidirectional=True, dropout=0.5):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # 如果是双向LSTM，输出维度需要乘2
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # 通过LSTM层
        output, (hidden, cell) = self.lstm(x)  # output: [batch_size, seq_len, hidden_dim*2]
        
        # 使用最后一个时间步的输出
        if self.lstm.bidirectional:
            # 如果是双向LSTM，拼接最后一个时间步的两个方向的隐藏状态
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # [batch_size, hidden_dim*2]
        else:
            hidden = hidden[-1,:,:]  # [batch_size, hidden_dim]
        
        hidden = self.dropout(hidden)
        out = self.fc(hidden)  # [batch_size, num_classes]
        
        return out

class TransformerEncoderLayer(nn.Module):
    """
    自定义的Transformer编码器层
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
    def forward(self, src, src_mask=None, return_attention=False):
        # 多头自注意力
        src2 = self.norm1(src)
        src2, attn_weights = self.self_attn(src2, src2, src2, attn_mask=src_mask, need_weights=return_attention, average_attn_weights=not return_attention)
        src = src + self.dropout(src2)
        
        # 前馈网络
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout(src2)
        
        if return_attention:
            return src, attn_weights
        return src

class TransformerModel(nn.Module):
    """
    基于Transformer的文本分类模型
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # 保存最后一层的注意力权重，用于可视化
        self.last_attention_weights = None
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.embedding_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # 应用Transformer层
        for i, layer in enumerate(self.transformer_layers):
            x, attention_weights = layer(x, return_attention=True)
            if i == len(self.transformer_layers) - 1:
                self.last_attention_weights = attention_weights
        
        x = self.norm(x)
        
        # 使用[CLS]位置的输出进行分类，这里简化为使用序列的平均值
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        
        return x
    
    def get_attention_weights(self, x):
        """
        获取最后一层注意力权重，用于可视化
        """
        # 运行前向传播以获取注意力权重
        _ = self.forward(x)
        return self.last_attention_weights

class PositionalEncoding(nn.Module):
    """
    Transformer位置编码
    """
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

# 导入math模块（位置编码需要）
import math
