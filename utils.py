import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import time
from io import BytesIO
import base64
from matplotlib.colors import LinearSegmentedColormap

def save_model(model, vocab, model_name):
    """
    保存模型和词汇表

    参数:
    model: 训练好的模型
    vocab: 词汇表
    model_name: 模型名称
    """
    # 创建保存目录（如果不存在）
    os.makedirs('models', exist_ok=True)

    # 保存模型
    torch.save(model.state_dict(), f'{model_name}.pt')

    # 保存词汇表
    with open(f'{model_name}_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

def load_model(model_path, model_type='CNN'):
    """
    加载模型和词汇表

    参数:
    model_path: 模型文件路径
    model_type: 模型类型（'CNN', 'RNN', 'Transformer'）

    返回:
    加载的模型和词汇表
    """
    # 构建词汇表路径
    vocab_path = model_path.replace('.pt', '_vocab.pkl')

    try:
        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # 根据模型类型创建模型实例
        from models import CNNModel, RNNModel, TransformerModel

        if 'CNN' in model_path:
            model = CNNModel(
                vocab_size=len(vocab),
                embedding_dim=100,
                num_filters=128,
                filter_sizes=[3, 4, 5],
                num_classes=8
            )
        elif 'RNN' in model_path:
            model = RNNModel(
                vocab_size=len(vocab),
                embedding_dim=100,
                hidden_dim=128,
                num_layers=2,
                num_classes=8,
                bidirectional=True
            )
        elif 'Transformer' in model_path:
            model = TransformerModel(
                vocab_size=len(vocab),
                embedding_dim=100,
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                num_classes=8
            )
        else:
            # 默认使用CNN模型
            model = CNNModel(
                vocab_size=len(vocab),
                embedding_dim=100,
                num_filters=128,
                filter_sizes=[3, 4, 5],
                num_classes=8
            )

        # 加载模型参数
        model.load_state_dict(torch.load(model_path))
        model.eval()

        return model, vocab

    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None, None

def plot_training_curves(history):
    """
    绘制训练曲线

    参数:
    history: 训练历史记录

    返回:
    matplotlib图形
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    axes[0].plot(history['loss'], label='训练集')
    axes[0].plot(history['val_loss'], label='验证集')
    axes[0].set_title('损失函数')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # 准确率曲线
    axes[1].plot(history['acc'], label='训练集')
    axes[1].plot(history['val_acc'], label='验证集')
    axes[1].set_title('准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    """
    绘制混淆矩阵

    参数:
    cm: 混淆矩阵
    class_names: 类别名称列表

    返回:
    matplotlib图形
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 数据归一化以提高可视化效果
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理可能的NaN值

    # 创建热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax)

    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('混淆矩阵 (绝对数量)')

    plt.tight_layout()
    return fig

def plot_normalized_confusion_matrix(cm, class_names):
    """
    绘制归一化的混淆矩阵

    参数:
    cm: 混淆矩阵
    class_names: 类别名称列表

    返回:
    matplotlib图形
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 数据归一化以提高可视化效果
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理可能的NaN值

    # 创建热力图
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax, vmin=0, vmax=1)

    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('混淆矩阵 (百分比)')

    plt.tight_layout()
    return fig

def plot_class_distribution(label_counts, class_names=None, title='类别分布', figsize=(12, 10)):
    """
    绘制类别分布的可视化图表

    参数:
    label_counts: 标签计数Series或字典
    class_names: 类别名称列表或字典 (可选)
    title: 图表标题
    figsize: 图形大小

    返回:
    matplotlib图形
    """
    # 创建一个包含两个子图的图形：条形图和饼图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 确保label_counts是Series类型
    import pandas as pd
    if not isinstance(label_counts, pd.Series):
        label_counts = pd.Series(label_counts)

    # 排序并获取标签和计数
    sorted_counts = label_counts.sort_index()
    labels = sorted_counts.index.tolist()
    counts = sorted_counts.values

    # 如果提供了类别名称，则使用它们
    if class_names is not None:
        if isinstance(class_names, dict):
            display_labels = [class_names.get(label, str(label)) for label in labels]
        else:
            display_labels = [class_names[label] if label < len(class_names) else str(label) for label in labels]
    else:
        display_labels = [str(label) for label in labels]

    # 创建一个漂亮的颜色映射
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))

    # 绘制条形图
    bars = ax1.bar(range(len(labels)), counts, color=colors)

    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # 设置条形图属性
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(display_labels, rotation=45, ha='right')
    ax1.set_xlabel('类别')
    ax1.set_ylabel('样本数量')
    ax1.set_title(f'{title} (条形图)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 为条形图添加百分比标签
    total = sum(counts)
    for i, bar in enumerate(bars):
        percentage = counts[i] / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                f'{percentage:.1f}%', ha='center', va='center',
                color='white' if percentage > 10 else 'black', fontweight='bold')

    # 绘制饼图
    wedges, texts, autotexts = ax2.pie(
        counts,
        labels=display_labels,
        autopct='%1.1f%%',
        textprops={'fontsize': 9},
        colors=colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        shadow=True,
        startangle=90
    )

    # 设置饼图属性
    ax2.set_title(f'{title} (饼图)')
    ax2.axis('equal')  # 确保饼图是圆形的

    # 为饼图添加图例
    ax2.legend(wedges, [f'{label} ({count})' for label, count in zip(display_labels, counts)],
              title="类别 (样本数)", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 调整布局
    plt.tight_layout()

    return fig

def plot_attention_weights(tokens, attention_weights, figsize=(10, 8)):
    """
    可视化注意力权重

    参数:
    tokens: 输入文本分词列表
    attention_weights: 注意力权重矩阵
    figsize: 图形大小

    返回:
    matplotlib图形
    """
    # 转换为numpy数组，如果是PyTorch张量
    if hasattr(attention_weights, 'detach'):
        attention_weights = attention_weights.detach().cpu().numpy()

    # 如果是一维数组，转换为方阵
    if len(attention_weights.shape) == 1:
        n = int(np.sqrt(len(attention_weights)))
        attention_weights = attention_weights.reshape(n, n)

    # 如果注意力权重是高维张量，取其平均值
    if len(attention_weights.shape) > 2:
        # 如果是多头注意力，取所有头的平均
        attention_weights = attention_weights.mean(axis=0)

    # 裁剥权重矩阵以匹配令牌长度
    if attention_weights.shape[0] > len(tokens):
        attention_weights = attention_weights[:len(tokens), :len(tokens)]
    elif attention_weights.shape[0] < len(tokens):
        tokens = tokens[:attention_weights.shape[0]]

    fig, ax = plt.subplots(figsize=figsize)

    # 创建热力图
    im = ax.imshow(attention_weights, cmap='YlOrRd')

    # 添加列标签（tokens）
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)

    # 连接格点线
    ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    # 添加数值标注
    threshold = attention_weights.max() / 2.
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                   ha="center", va="center",
                   color="white" if attention_weights[i, j] > threshold else "black")

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('注意力权重', rotation=-90, va='bottom')

    ax.set_title('注意力可视化')
    plt.tight_layout()

    return fig

def visualize_model_architecture(model_type):
    """
    生成模型架构示意图

    参数:
    model_type: 模型类型 ('CNN', 'RNN', 'Transformer')

    返回:
    base64编码的图像字符串
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    if model_type == 'CNN':
        # CNN模型可视化
        components = [
            {'name': '输入层', 'color': '#FFD580'},
            {'name': '词嵌入层', 'color': '#C3B1E1'},
            {'name': '卷积层 (3x3)', 'color': '#87CEEB'},
            {'name': '卷积层 (4x4)', 'color': '#98FB98'},
            {'name': '卷积层 (5x5)', 'color': '#FFA07A'},
            {'name': '池化层', 'color': '#D3D3D3'},
            {'name': '拼接层', 'color': '#ADD8E6'},
            {'name': 'Dropout层', 'color': '#F08080'},
            {'name': '全连接层', 'color': '#90EE90'},
            {'name': '输出层', 'color': '#FFB6C1'}
        ]

        title = 'CNN模型架构'

    elif model_type == 'RNN':
        # RNN模型可视化
        components = [
            {'name': '输入层', 'color': '#FFD580'},
            {'name': '词嵌入层', 'color': '#C3B1E1'},
            {'name': 'LSTM层 (正向)', 'color': '#87CEEB'},
            {'name': 'LSTM层 (反向)', 'color': '#98FB98'},
            {'name': '输出连接', 'color': '#ADD8E6'},
            {'name': 'Dropout层', 'color': '#F08080'},
            {'name': '全连接层', 'color': '#90EE90'},
            {'name': '输出层', 'color': '#FFB6C1'}
        ]

        title = 'RNN(LSTM)模型架构'

    elif model_type == 'Transformer':
        # Transformer模型可视化
        components = [
            {'name': '输入层', 'color': '#FFD580'},
            {'name': '词嵌入层', 'color': '#C3B1E1'},
            {'name': '位置编码', 'color': '#E6E6FA'},
            {'name': '自注意力机制', 'color': '#87CEEB'},
            {'name': '前馈神经网络', 'color': '#98FB98'},
            {'name': '通道加法 & 层正则化', 'color': '#D3D3D3'},
            {'name': '平均池化', 'color': '#ADD8E6'},
            {'name': 'Dropout层', 'color': '#F08080'},
            {'name': '全连接层', 'color': '#90EE90'},
            {'name': '输出层', 'color': '#FFB6C1'}
        ]

        title = 'Transformer模型架构'

    elif model_type == 'BERT':
        # BERT模型可视化
        components = [
            {'name': '输入层', 'color': '#FFD580'},
            {'name': '词嵌入层', 'color': '#C3B1E1'},
            {'name': '位置嵌入层', 'color': '#E6E6FA'},
            {'name': '分段嵌入层', 'color': '#FFDAB9'},
            {'name': 'BERT编码器层 (x12)', 'color': '#87CEEB'},
            {'name': '多头自注意力', 'color': '#98FB98'},
            {'name': '前馈神经网络', 'color': '#D3D3D3'},
            {'name': '层正则化', 'color': '#ADD8E6'},
            {'name': '[CLS]标记表示', 'color': '#F08080'},
            {'name': 'Dropout层', 'color': '#FFA07A'},
            {'name': '全连接层', 'color': '#90EE90'},
            {'name': '输出层', 'color': '#FFB6C1'}
        ]

        title = 'BERT模型架构'

    elif model_type == 'RoBERTa':
        # RoBERTa模型可视化
        components = [
            {'name': '输入层', 'color': '#FFD580'},
            {'name': '词嵌入层', 'color': '#C3B1E1'},
            {'name': '位置嵌入层', 'color': '#E6E6FA'},
            {'name': 'RoBERTa编码器层 (x12)', 'color': '#87CEEB'},
            {'name': '多头自注意力', 'color': '#98FB98'},
            {'name': '前馈神经网络', 'color': '#D3D3D3'},
            {'name': '层正则化', 'color': '#ADD8E6'},
            {'name': '[CLS]标记表示', 'color': '#F08080'},
            {'name': 'Dropout层', 'color': '#FFA07A'},
            {'name': '全连接层', 'color': '#90EE90'},
            {'name': '输出层', 'color': '#FFB6C1'}
        ]

        title = 'RoBERTa模型架构'

    else:
        # 默认通用模型可视化
        components = [
            {'name': '输入层', 'color': '#FFD580'},
            {'name': '特征提取层', 'color': '#C3B1E1'},
            {'name': '全连接层', 'color': '#90EE90'},
            {'name': '输出层', 'color': '#FFB6C1'}
        ]

        title = '模型架构'

    # 绘制模型组件
    num_components = len(components)
    y_positions = np.linspace(0.1, 0.9, num_components)
    box_height = 0.7 / num_components

    for i, (pos, comp) in enumerate(zip(y_positions, components)):
        rect = plt.Rectangle((0.1, pos - box_height / 2), 0.8, box_height,
                             facecolor=comp['color'], alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.5, pos, comp['name'], ha='center', va='center', fontsize=12)

    # 添加箭头
    arrow_positions = y_positions[:-1] + np.diff(y_positions) / 2
    for pos in arrow_positions:
        ax.annotate('', xy=(0.5, pos - box_height / 2), xytext=(0.5, pos + box_height / 2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16)
    ax.axis('off')

    # 转换为base64字符串
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return img_str

def create_demo_model(model_type, vocab_size=5000, embedding_dim=100, num_classes=8):
    """
    创建演示模型用于测试

    参数:
    model_type: 模型类型 ('CNN', 'RNN', 'Transformer', 'BERT', 'RoBERTa')
    vocab_size: 词汇表大小
    embedding_dim: 词嵌入维度
    num_classes: 类别数量

    返回:
    创建的模型和词汇表
    """
    from models import CNNModel, RNNModel, TransformerModel

    # 创建简单的演示词汇表
    vocab = {str(i): i for i in range(1, vocab_size + 1)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = vocab_size + 1

    # 创建模型
    if model_type == 'CNN':
        model = CNNModel(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            num_filters=128,
            filter_sizes=[3, 4, 5],
            num_classes=num_classes,
            dropout=0.3
        )
    elif model_type == 'RNN':
        model = RNNModel(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            bidirectional=True,
            dropout=0.3
        )
    elif model_type == 'Transformer':
        model = TransformerModel(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_classes=num_classes,
            dropout=0.1
        )
    elif model_type == 'BERT' or model_type == 'RoBERTa':
        try:
            # 尝试导入transformer模型
            if model_type == 'BERT':
                from transformer_models import BERTClassifier
                model = BERTClassifier(
                    pretrained_model_name="bert-base-chinese",
                    num_classes=num_classes,
                    dropout=0.1
                )
            else:  # RoBERTa
                from transformer_models import RoBERTaClassifier
                model = RoBERTaClassifier(
                    pretrained_model_name="hfl/chinese-roberta-wwm-ext",
                    num_classes=num_classes,
                    dropout=0.1
                )
            # 对于transformer模型，我们不使用自定义词汇表，而是使用预训练模型的tokenizer
            tokenizer = model.get_tokenizer()
            return model, tokenizer
        except ImportError:
            print(f"无法导入transformer模型，请确保已安装transformers库")
            # 回退到CNN模型
            model = CNNModel(
                vocab_size=len(vocab),
                embedding_dim=embedding_dim,
                num_filters=128,
                filter_sizes=[3, 4, 5],
                num_classes=num_classes,
                dropout=0.3
            )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model, vocab

def generate_sample_data(num_samples=100, vocab_size=1000, max_length=100, num_classes=8):
    """
    生成示例数据集用于测试

    参数:
    num_samples: 生成的样本数量
    vocab_size: 词汇表大小
    max_length: 序列最大长度
    num_classes: 类别数量

    返回:
    X: 样本特征数组
    y: 样本标签数组
    """
    # 生成随机序列数据作为样本特征
    lengths = np.random.randint(10, max_length, num_samples)
    X = []

    for length in lengths:
        # 生成随机整数序列作为样本
        sequence = np.random.randint(1, vocab_size, length)
        # 填充到最大长度
        padded_sequence = np.pad(sequence, (0, max_length - length), 'constant')
        X.append(padded_sequence)

    X = np.array(X)

    # 生成随机类别标签
    y = np.random.randint(0, num_classes, num_samples)

    return X, y
