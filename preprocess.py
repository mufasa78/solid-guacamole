import jieba
import re
import os
import numpy as np
from collections import Counter

def preprocess_text(text):
    """
    对中文文本进行预处理

    参数:
    text: 输入的原始文本

    返回:
    处理后的分词列表
    """
    if not isinstance(text, str):
        return []

    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)

    # 去除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 保留数字和英文，只替换为特殊标记，而不是完全删除
    text = re.sub(r'\d+', ' <NUM> ', text)
    text = re.sub(r'[a-zA-Z]+', ' <ENG> ', text)

    # 替换标点符号为空格，而不是完全删除
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9<>]', ' ', text)

    # 使用jieba进行分词，添加自定义词典以提高分词准确性
    jieba.load_userdict('custom_dict.txt') if os.path.exists('custom_dict.txt') else None
    words = jieba.lcut(text)

    # 去除停用词（保留单字符词语，因为在中文中单字也可能有重要含义）
    # 加载停用词表（如果存在）
    stop_words = set()
    if os.path.exists('stopwords.txt'):
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            stop_words = set([line.strip() for line in f])

    # 过滤停用词，但保留单字符的重要词语
    words = [word for word in words if word.strip() and (word not in stop_words or (len(word) == 1 and word.isalpha()))]

    return words

def build_vocab(texts, max_vocab_size=50000, min_freq=1):
    """
    构建词汇表

    参数:
    texts: 预处理后的文本列表
    max_vocab_size: 最大词汇表大小
    min_freq: 最小词频，只有出现次数大于等于min_freq的词才会被加入词汇表

    返回:
    词汇表（词到索引的映射）
    """
    # 统计所有词语频率
    all_words = []
    for text in texts:
        all_words.extend(text)

    # 计数并过滤低频词
    counter = Counter(all_words)
    filtered_words = [(word, count) for word, count in counter.items() if count >= min_freq]

    # 选择最常见的词语
    sorted_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)
    if len(sorted_words) > max_vocab_size - 2:  # 留出<PAD>和<UNK>的位置
        sorted_words = sorted_words[:max_vocab_size - 2]

    # 构建词汇表
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words:
        vocab[word] = len(vocab)

    return vocab

def tokenize_and_pad(texts, vocab, max_length):
    """
    将文本转换为token id并进行填充

    参数:
    texts: 预处理后的文本列表或单个文本
    vocab: 词汇表
    max_length: 序列最大长度

    返回:
    填充后的token id数组
    """
    # 处理单个文本的情况
    if isinstance(texts, list) and not isinstance(texts[0], list):
        # 单个文本
        tokens = [vocab.get(word, vocab['<UNK>']) for word in texts[:max_length]]

        # 填充
        if len(tokens) < max_length:
            tokens = tokens + [vocab['<PAD>']] * (max_length - len(tokens))

        return np.array(tokens, dtype=np.int32)

    # 处理文本列表的情况
    result = np.zeros((len(texts), max_length), dtype=np.int32)

    for i, text in enumerate(texts):
        # 将词语转换为索引
        tokens = [vocab.get(word, vocab['<UNK>']) for word in text[:max_length]]

        # 填充
        if len(tokens) < max_length:
            tokens = tokens + [vocab['<PAD>']] * (max_length - len(tokens))

        result[i] = tokens

    return result
