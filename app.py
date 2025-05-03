import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import jieba
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import io
import base64
from models import CNNModel, RNNModel, TransformerModel
from preprocess import preprocess_text, build_vocab, tokenize_and_pad
from train import train_model, evaluate_model
from utils import load_model, save_model, plot_training_curves, plot_confusion_matrix, plot_normalized_confusion_matrix, \
    visualize_model_architecture, plot_attention_weights, generate_sample_data, create_demo_model, plot_class_distribution
from dataset_utils import load_dataset_with_required_columns, get_available_datasets

# Import transformer models
try:
    from transformer_models import BERTClassifier, RoBERTaClassifier, AutoTransformerClassifier, preprocess_for_transformer
    from transformer_utils import load_transformer_model, classify_text_with_transformer, classify_and_explain
    TRANSFORMER_MODELS_AVAILABLE = True
except ImportError:
    TRANSFORMER_MODELS_AVAILABLE = False

# 设置页面配置
st.set_page_config(
    page_title="网络小说题材分类系统",
    page_icon="📚",
    layout="wide"
)

# 初始化session state
if 'vocab' not in st.session_state:
    st.session_state.vocab = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# 页面标题
st.title("基于深度学习的网络小说题材分类系统")

# 文本分类函数
def classify_text(input_text, model, vocab, is_transformer_model=False, tokenizer=None):
    """
    使用模型对输入文本进行分类

    参数:
    input_text: 输入文本
    model: 训练好的模型
    vocab: 词汇表
    is_transformer_model: 是否是transformer模型（BERT/RoBERTa）
    tokenizer: transformer模型的tokenizer
    """
    with st.spinner("正在分类..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        if is_transformer_model and tokenizer is not None:
            # 使用transformer模型的分类方法
            result = classify_and_explain(input_text, model, tokenizer)
            predicted_class = result['predicted_class']
            probabilities = np.zeros(len(genre_mapping))
            for i, (class_name, prob) in enumerate(result['top_classes']):
                class_idx = [k for k, v in genre_mapping.items() if v == class_name][0]
                probabilities[class_idx] = prob / 100.0

            # 显示分词结果
            with st.expander("查看分词结果"):
                tokens = tokenizer.tokenize(input_text)[:50]
                st.write("\n".join([f"{i+1}. {token}" for i, token in enumerate(tokens)]))
                if len(tokens) > 50:
                    st.write(f"... 共 {len(tokens)} 个词")

            # 注意力权重暂不支持
            attention_weights = None
        else:
            # 传统模型的分类方法
            # 预处理文本
            processed_text = preprocess_text(input_text)

            # 显示分词结果
            with st.expander("查看分词结果"):
                st.write("\n".join([f"{i+1}. {word}" for i, word in enumerate(processed_text[:50])]))
                if len(processed_text) > 50:
                    st.write(f"... 共 {len(processed_text)} 个词")

            # tokenize
            tokenized_text = tokenize_and_pad([processed_text], vocab, 200)

            # 预测
            with torch.no_grad():
                inputs = torch.tensor(tokenized_text).to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # 获取预测的类别和概率
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
                predicted_class = predicted.item()

                # 如果是Transformer模型，尝试获取注意力权重
                attention_weights = None
                if hasattr(model, 'get_attention_weights'):
                    attention_weights = model.get_attention_weights(inputs)

        # 显示预测结果
        st.subheader("分类结果")
        st.success(f"预测题材: **{genre_mapping.get(predicted_class, '未知')}**")

        # 显示各类别概率
        st.subheader("各题材概率")
        # 检查probabilities是否是PyTorch张量，如果是则转换为numpy数组
        if hasattr(probabilities, 'cpu'):
            probabilities_np = probabilities.cpu().numpy()
        else:
            probabilities_np = probabilities  # 已经是numpy数组

        probs_df = pd.DataFrame({
            '题材': [genre_mapping.get(i, f'类别{i}') for i in range(len(probabilities))],
            '概率': probabilities_np
        })
        probs_df = probs_df.sort_values('概率', ascending=False)

        # 创建一个更美观的概率分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 使用渐变色调
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(probs_df)))

        # 水平条形图 - 更容易阅读长标签
        bars = ax1.barh(probs_df['题材'], probs_df['概率'], color=colors)
        ax1.set_xlabel('概率')
        ax1.set_ylabel('题材类别')
        ax1.set_title('各题材类别概率分布 (条形图)')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        ax1.set_xlim(0, 1)

        # 为条形添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width + 0.01 if width < 0.3 else width - 0.05
            label_color = 'black' if width < 0.3 else 'white'
            ax1.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', va='center', ha='left' if width < 0.3 else 'right',
                    color=label_color, fontweight='bold')

        # 饼图 - 显示比例关系
        wedges, texts, autotexts = ax2.pie(
            probs_df['概率'],
            labels=probs_df['题材'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1},
            textprops={'fontsize': 9}
        )

        # 突出显示最高概率的扇区
        wedges[0].set_edgecolor('black')
        wedges[0].set_linewidth(2)

        ax2.set_title('各题材类别概率分布 (饼图)')
        ax2.axis('equal')  # 确保饼图是圆形的

        plt.tight_layout()
        st.pyplot(fig)

        # 显示概率表格
        st.write("概率详情:")
        formatted_probs = probs_df.copy()
        formatted_probs['概率'] = formatted_probs['概率'].apply(lambda x: f"{x:.2%}")
        st.dataframe(formatted_probs)

        # 如果是Transformer模型且有注意力权重，显示注意力可视化
        if attention_weights is not None:
            st.subheader("注意力可视化")
            st.write("下图显示了模型在处理文本时关注的重点。越亮的区域表示模型赋予该部分更高的关注度。")

            # 取前20个词进行可视化，避免图像过大
            tokens_to_show = processed_text[:20]
            if attention_weights.shape[1] > 20:
                attention_weights_to_show = attention_weights[0, :20, :20]
            else:
                attention_weights_to_show = attention_weights[0, :attention_weights.shape[1], :attention_weights.shape[1]]
                tokens_to_show = processed_text[:attention_weights.shape[1]]

            fig = plot_attention_weights(tokens_to_show, attention_weights_to_show)
            st.pyplot(fig)

# 侧边栏
st.sidebar.header("系统功能")
page = st.sidebar.selectbox(
    "选择功能模块",
    ["数据处理", "模型训练", "模型评估", "文本分类", "模型比较", "帮助和可视化"]
)

# 定义类别映射
genre_mapping = {
    0: "玄幻",
    1: "武侠",
    2: "都市",
    3: "言情",
    4: "科幻",
    5: "历史",
    6: "游戏",
    7: "悬疑"
}

# 数据处理页面
if page == "数据处理":
    st.header("数据处理")

    # 数据加载方式选择
    data_source = st.radio(
        "选择数据来源",
        ["上传数据文件", "使用预设数据集", "生成示例数据"]
    )

    if data_source == "上传数据文件":
        # 上传数据文件
        uploaded_file = st.file_uploader("上传数据集 (CSV格式: 文本内容和标签)", type=["csv"])

        if uploaded_file is not None:
            # 加载数据
            try:
                # 保存上传的文件到临时位置
                temp_file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)

                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 检查并修复数据集格式
                try:
                    df, message = load_dataset_with_required_columns(temp_file_path)
                except Exception as e:
                    st.error(f"数据集加载错误: {str(e)}")
                    st.info("如果出现 'name 'pd' is not defined' 错误，请确保已正确导入pandas库。")
                    df, message = None, f"加载失败: {str(e)}"

                if df is not None:
                    st.success(f"{message}，共 {len(df)} 条记录")

                    # 显示数据集基本信息
                    st.subheader("数据集预览")
                    st.dataframe(df.head())

                    # 数据集统计信息
                    st.subheader("数据集统计")
                    label_counts = df['label'].value_counts()
                    fig = plot_class_distribution(
                        label_counts,
                        class_names=genre_mapping,
                        title='各题材类别数量统计',
                        figsize=(12, 6)
                    )
                    st.pyplot(fig)

                    # 数据预处理
                    if st.button("开始数据预处理"):
                        with st.spinner("正在进行数据预处理..."):
                            # 文本预处理
                            df['processed_text'] = df['text'].apply(preprocess_text)

                            # 构建词汇表
                            vocab = build_vocab(df['processed_text'])
                            st.session_state.vocab = vocab

                            # 保存预处理后的数据
                            st.session_state.df = df

                            st.success(f"预处理完成！词汇表大小: {len(vocab)}")

                            # 可下载预处理后的数据
                            csv = df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">下载预处理后的数据</a>'
                            st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error(message)
                    st.info("上传的数据集需要包含 'text' 和 'label' 列。您可以使用 fix_dataset_format.py 脚本修复数据集格式。")

                    # 提供修复选项
                    if st.button("自动修复数据集格式"):
                        with st.spinner("正在修复数据集格式..."):
                            # 导入修复函数和必要的库
                            import pandas as pd
                            from fix_dataset_format import fix_dataset_format

                            # 修复数据集
                            fixed_df = fix_dataset_format(temp_file_path)

                            if fixed_df is not None:
                                df = fixed_df
                                st.success(f"数据集格式已修复，共 {len(df)} 条记录")

                                # 显示数据集基本信息
                                st.subheader("数据集预览")
                                st.dataframe(df.head())

                                # 数据集统计信息
                                st.subheader("数据集统计")
                                label_counts = df['label'].value_counts()
                                fig = plot_class_distribution(
                                    label_counts,
                                    class_names=genre_mapping,
                                    title='各题材类别数量统计',
                                    figsize=(12, 6)
                                )
                                st.pyplot(fig)

                                # 数据预处理
                                if st.button("开始数据预处理", key="preprocess_fixed"):
                                    with st.spinner("正在进行数据预处理..."):
                                        # 文本预处理
                                        df['processed_text'] = df['text'].apply(preprocess_text)

                                        # 构建词汇表
                                        vocab = build_vocab(df['processed_text'])
                                        st.session_state.vocab = vocab

                                        # 保存预处理后的数据
                                        st.session_state.df = df

                                        st.success(f"预处理完成！词汇表大小: {len(vocab)}")
                            else:
                                st.error("无法修复数据集格式，请确保数据集包含有效的文本列")
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")

    elif data_source == "使用预设数据集":
        # 获取datasets目录下的所有CSV文件及其信息
        available_datasets = get_available_datasets('datasets')

        if not available_datasets:
            st.warning("没有找到预设数据集，请先上传数据文件或生成示例数据")
        else:
            # 创建一个更有信息的选择列表
            dataset_options = []
            for dataset in available_datasets:
                status = ""
                if dataset['has_required_columns']:
                    status = "✓ 格式正确"
                elif dataset['has_fixed_version']:
                    status = "⚠️ 已修复"
                else:
                    status = "❌ 格式错误"

                option = f"{dataset['name']} ({status}, {dataset['records']}条记录)"
                dataset_options.append(option)

            # 选择数据集
            selected_option = st.selectbox("选择预设数据集", dataset_options)

            # 从选项中提取数据集名称
            selected_dataset_name = selected_option.split(' (')[0]
            selected_dataset = next((d for d in available_datasets if d['name'] == selected_dataset_name), None)

            if selected_dataset:
                # 加载选择的数据集
                try:
                    # 使用我们的工具函数加载数据集
                    dataset_path = selected_dataset['path']
                    try:
                        df, message = load_dataset_with_required_columns(dataset_path)
                    except Exception as e:
                        st.error(f"数据集加载错误: {str(e)}")
                        st.info("如果出现 'name 'pd' is not defined' 错误，请确保已正确导入pandas库。")
                        df, message = None, f"加载失败: {str(e)}"

                    if df is not None:
                        st.success(f"{message}，共 {len(df)} 条记录")

                        # 显示数据集基本信息
                        st.subheader("数据集预览")
                        st.dataframe(df.head())

                        # 数据集统计信息
                        st.subheader("数据集统计")
                        label_counts = df['label'].value_counts()
                        fig = plot_class_distribution(
                            label_counts,
                            class_names=genre_mapping,
                            title='各题材类别数量统计',
                            figsize=(12, 6)
                        )
                        st.pyplot(fig)

                        # 数据预处理
                        if st.button("开始数据预处理"):
                            with st.spinner("正在进行数据预处理..."):
                                # 文本预处理
                                df['processed_text'] = df['text'].apply(preprocess_text)

                                # 构建词汇表
                                vocab = build_vocab(df['processed_text'])
                                st.session_state.vocab = vocab

                                # 保存预处理后的数据
                                st.session_state.df = df

                                st.success(f"预处理完成！词汇表大小: {len(vocab)}")

                                # 可下载预处理后的数据
                                csv = df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">下载预处理后的数据</a>'
                                st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.error(message)
                        st.info("请运行 fix_all_datasets.py 脚本修复所有数据集")
                except Exception as e:
                    st.error(f"数据集加载失败: {str(e)}")

    elif data_source == "生成示例数据":
        st.subheader("生成示例数据")

        # 设置生成参数
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.slider("生成样本数量", 100, 2000, 1000)
            vocab_size = st.slider("词汇表大小", 1000, 10000, 5000)

        with col2:
            max_length = st.slider("最大序列长度", 50, 500, 200)
            num_classes = st.slider("类别数量", 2, 8, 8)

        # 生成示例数据
        if st.button("生成示例数据"):
            with st.spinner("正在生成示例数据..."):
                # 生成示例数据
                X, y = generate_sample_data(num_samples=num_samples, vocab_size=vocab_size, max_length=max_length, num_classes=num_classes)

                # 创建演示词汇表
                vocab = {str(i): i for i in range(1, vocab_size + 1)}
                vocab['<PAD>'] = 0
                vocab['<UNK>'] = vocab_size + 1

                # 将数据保存到会话状态
                st.session_state.X_sample = X
                st.session_state.y_sample = y
                st.session_state.vocab = vocab

                # 创建简单的DataFrame
                text_samples = [' '.join([f'word{token}' if token > 0 else '<PAD>' for token in sample[:20]]) + '...' for sample in X]
                df = pd.DataFrame({
                    'text': text_samples,
                    'processed_text': [['word'+str(token) if token > 0 else '<PAD>' for token in sample[:20]] for sample in X],
                    'label': y
                })

                # 保存到会话状态
                st.session_state.df = df

                st.success(f"成功生成示例数据！共 {len(X)} 条记录，词汇表大小: {len(vocab)}")

                # 显示数据集统计信息
                st.subheader("数据集统计")
                label_counts = pd.Series(y).value_counts().sort_index()
                fig = plot_class_distribution(
                    label_counts,
                    class_names=genre_mapping,
                    title='生成的样本各类别数量分布',
                    figsize=(12, 6)
                )
                st.pyplot(fig)

# 模型训练页面
elif page == "模型训练":
    st.header("模型训练")

    # 检查是否已有预处理数据
    if 'df' not in st.session_state or st.session_state.vocab is None:
        st.info("请先在'数据处理'页面加载或生成数据并进行预处理")
        st.warning("您需要先处理数据才能训练模型。请前往'数据处理'页面选择数据集或生成示例数据。")
    else:
        # 模型选择
        model_options = ["CNN", "RNN (LSTM)", "Transformer"]

        # Add transformer models if available
        if TRANSFORMER_MODELS_AVAILABLE:
            model_options.extend(["BERT", "RoBERTa"])

        model_type = st.selectbox(
            "选择模型类型",
            model_options
        )

        # 训练参数设置
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("训练轮数", 1, 100, 20)
            batch_size = st.slider("批次大小", 8, 128, 16)
            test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
            patience = st.slider("早停耐心值", 1, 20, 5, help="如果验证集性能在这么多轮内没有提升，则停止训练")

        with col2:
            learning_rate = st.number_input("学习率", 0.0001, 0.1, 0.001, format="%.4f")
            max_length = st.slider("最大序列长度", 50, 500, 200)
            num_classes = len(st.session_state.df['label'].unique())
            optimizer_type = st.selectbox("优化器类型", ["adam", "sgd", "rmsprop"], index=0)

        # 模型特定参数
        if model_type == "CNN":
            st.subheader("CNN 模型参数")
            embedding_dim = st.slider("词嵌入维度", 50, 300, 200)
            num_filters = st.slider("卷积核数量", 32, 512, 256)
            filter_sizes = st.multiselect("卷积核大小", [2, 3, 4, 5, 6], default=[2, 3, 4, 5])
            dropout = st.slider("Dropout比例", 0.0, 0.7, 0.5)

        elif model_type == "RNN (LSTM)":
            st.subheader("RNN 模型参数")
            embedding_dim = st.slider("词嵌入维度", 50, 300, 200)
            hidden_dim = st.slider("隐藏层维度", 64, 512, 256)
            num_layers = st.slider("LSTM层数", 1, 5, 2)
            bidirectional = st.checkbox("双向LSTM", value=True)
            dropout = st.slider("Dropout比例", 0.0, 0.7, 0.5)

        elif model_type == "Transformer":
            st.subheader("Transformer 模型参数")
            embedding_dim = st.slider("词嵌入维度", 50, 300, 200)
            hidden_dim = st.slider("隐藏层维度", 64, 512, 256)
            num_layers = st.slider("Transformer层数", 1, 6, 4)
            num_heads = st.slider("注意力头数", 1, 8, 8)
            dropout = st.slider("Dropout比例", 0.0, 0.5, 0.3)

        elif model_type == "BERT" and TRANSFORMER_MODELS_AVAILABLE:
            st.subheader("BERT 模型参数")
            pretrained_model_name = st.selectbox(
                "预训练模型",
                ["bert-base-chinese", "hfl/chinese-bert-wwm-ext", "hfl/chinese-macbert-base"],
                index=0
            )
            max_length = st.slider("最大序列长度", 32, 512, 128)
            dropout = st.slider("Dropout比例", 0.0, 0.5, 0.1)
            learning_rate = st.number_input("学习率", 1e-6, 1e-3, 2e-5, format="%.6f")

        elif model_type == "RoBERTa" and TRANSFORMER_MODELS_AVAILABLE:
            st.subheader("RoBERTa 模型参数")
            pretrained_model_name = st.selectbox(
                "预训练模型",
                ["hfl/chinese-roberta-wwm-ext", "hfl/chinese-roberta-wwm-ext-large"],
                index=0
            )
            max_length = st.slider("最大序列长度", 32, 512, 128)
            dropout = st.slider("Dropout比例", 0.0, 0.5, 0.1)
            learning_rate = st.number_input("学习率", 1e-6, 1e-3, 2e-5, format="%.6f")

        if st.button("开始训练"):
            with st.spinner(f"正在训练{model_type}模型..."):
                df = st.session_state.df
                vocab = st.session_state.vocab

                # 准备训练集和测试集
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)

                X_train = tokenize_and_pad(train_df['processed_text'], vocab, max_length)
                y_train = train_df['label'].values

                X_test = tokenize_and_pad(test_df['processed_text'], vocab, max_length)
                y_test = test_df['label'].values

                # 根据模型类型创建模型
                vocab_size = len(vocab)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                if model_type == "CNN":
                    model = CNNModel(
                        vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        num_filters=num_filters,
                        filter_sizes=filter_sizes,
                        num_classes=num_classes,
                        dropout=dropout
                    )
                elif model_type == "RNN (LSTM)":
                    model = RNNModel(
                        vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_classes=num_classes,
                        bidirectional=bidirectional,
                        dropout=dropout
                    )
                elif model_type == "Transformer":
                    model = TransformerModel(
                        vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        num_classes=num_classes,
                        dropout=dropout
                    )
                elif model_type == "BERT" and TRANSFORMER_MODELS_AVAILABLE:
                    st.info("使用BERT模型进行训练。这将使用预训练的BERT模型，而不是从头训练。")
                    model = BERTClassifier(
                        pretrained_model_name=pretrained_model_name,
                        num_classes=num_classes,
                        dropout=dropout
                    )
                    # 使用transformer_train.py中的训练函数
                    st.warning("BERT模型训练需要使用特殊的训练流程，请使用train_transformer_model.py脚本进行训练。")
                    st.stop()
                elif model_type == "RoBERTa" and TRANSFORMER_MODELS_AVAILABLE:
                    st.info("使用RoBERTa模型进行训练。这将使用预训练的RoBERTa模型，而不是从头训练。")
                    model = RoBERTaClassifier(
                        pretrained_model_name=pretrained_model_name,
                        num_classes=num_classes,
                        dropout=dropout
                    )
                    # 使用transformer_train.py中的训练函数
                    st.warning("RoBERTa模型训练需要使用特殊的训练流程，请使用train_transformer_model.py脚本进行训练。")
                    st.stop()

                # 训练模型
                model.to(device)
                history = train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_test,
                    y_val=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=device,
                    patience=patience,
                    optimizer_type=optimizer_type
                )

                # 保存模型和训练历史
                model_name = f"{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.models[model_name] = model
                st.session_state.training_history[model_name] = history
                st.session_state.current_model = model_name

                # 保存模型到文件
                save_model(model, vocab, model_name)

                st.success(f"模型训练完成！最终测试集准确率: {history['val_acc'][-1]:.4f}")

                # 显示训练过程曲线
                st.subheader("训练过程")
                fig = plot_training_curves(history)
                st.pyplot(fig)

# 模型评估页面
elif page == "模型评估":
    st.header("模型评估")

    # 检查是否有训练好的模型
    if not st.session_state.models:
        st.warning("请先在'模型训练'页面训练模型")
    else:
        # 选择模型
        if st.session_state.current_model:
            default_model = st.session_state.current_model
        else:
            default_model = list(st.session_state.models.keys())[0]

        model_name = st.selectbox(
            "选择要评估的模型",
            list(st.session_state.models.keys()),
            index=list(st.session_state.models.keys()).index(default_model)
        )

        model = st.session_state.models[model_name]

        # 上传评估数据或使用原始测试集
        use_original_test = st.checkbox("使用原始测试集", value=True)

        if use_original_test:
            if 'df' not in st.session_state:
                st.warning("请先在'数据处理'页面上传并预处理数据")
            else:
                # 使用原始数据的测试集
                from sklearn.model_selection import train_test_split
                df = st.session_state.df
                train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

                if st.button("开始评估"):
                    with st.spinner("正在评估模型..."):
                        # 准备评估数据
                        X_test = tokenize_and_pad(test_df['processed_text'], st.session_state.vocab, 200)
                        y_test = test_df['label'].values

                        # 评估模型
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        y_pred, metrics = evaluate_model(model, X_test, y_test, device)

                        # 显示评估结果
                        st.subheader("评估结果")
                        st.write(f"测试集准确率: {metrics['accuracy']:.4f}")
                        st.write(f"F1分数: {metrics['f1']:.4f}")

                        # 显示混淆矩阵
                        st.subheader("混淆矩阵")
                        tab1, tab2 = st.tabs(["绝对数量", "百分比"])
                        with tab1:
                            fig = plot_confusion_matrix(metrics['confusion_matrix'], list(genre_mapping.values()))
                            st.pyplot(fig)
                        with tab2:
                            fig = plot_normalized_confusion_matrix(metrics['confusion_matrix'], list(genre_mapping.values()))
                            st.pyplot(fig)

                        # 显示分类报告
                        st.subheader("分类报告")
                        st.text(metrics['classification_report'])
        else:
            # 上传评估数据
            uploaded_file = st.file_uploader("上传评估数据集 (CSV格式: 文本内容和标签)", type=["csv"])

            if uploaded_file is not None:
                # 加载数据
                try:
                    # 保存上传的文件到临时位置
                    temp_file_path = os.path.join("temp", uploaded_file.name)
                    os.makedirs("temp", exist_ok=True)

                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 检查并修复数据集格式
                    try:
                        eval_df, message = load_dataset_with_required_columns(temp_file_path)
                    except Exception as e:
                        st.error(f"数据集加载错误: {str(e)}")
                        st.info("如果出现 'name 'pd' is not defined' 错误，请确保已正确导入pandas库。")
                        eval_df, message = None, f"加载失败: {str(e)}"

                    if eval_df is not None:
                        st.success(f"{message}，共 {len(eval_df)} 条记录")

                        if st.button("开始评估"):
                            with st.spinner("正在评估模型..."):
                                # 文本预处理
                                eval_df['processed_text'] = eval_df['text'].apply(preprocess_text)

                                # 准备评估数据
                                X_test = tokenize_and_pad(eval_df['processed_text'], st.session_state.vocab, 200)
                                y_test = eval_df['label'].values

                                # 评估模型
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                y_pred, metrics = evaluate_model(model, X_test, y_test, device)

                                # 显示评估结果
                                st.subheader("评估结果")
                                st.write(f"测试集准确率: {metrics['accuracy']:.4f}")
                                st.write(f"F1分数: {metrics['f1']:.4f}")

                                # 显示混淆矩阵
                                st.subheader("混淆矩阵")
                                tab1, tab2 = st.tabs(["绝对数量", "百分比"])
                                with tab1:
                                    fig = plot_confusion_matrix(metrics['confusion_matrix'], list(genre_mapping.values()))
                                    st.pyplot(fig)
                                with tab2:
                                    fig = plot_normalized_confusion_matrix(metrics['confusion_matrix'], list(genre_mapping.values()))
                                    st.pyplot(fig)

                                # 显示分类报告
                                st.subheader("分类报告")
                                st.text(metrics['classification_report'])
                    else:
                        st.error(message)
                        st.info("上传的数据集需要包含 'text' 和 'label' 列。您可以使用 fix_dataset_format.py 脚本修复数据集格式。")

                        # 提供修复选项
                        if st.button("自动修复数据集格式"):
                            with st.spinner("正在修复数据集格式..."):
                                # 导入修复函数和必要的库
                                import pandas as pd
                                from fix_dataset_format import fix_dataset_format

                                # 修复数据集
                                fixed_df = fix_dataset_format(temp_file_path)

                                if fixed_df is not None:
                                    eval_df = fixed_df
                                    st.success(f"数据集格式已修复，共 {len(eval_df)} 条记录")

                                    if st.button("开始评估", key="eval_fixed"):
                                        with st.spinner("正在评估模型..."):
                                            # 文本预处理
                                            eval_df['processed_text'] = eval_df['text'].apply(preprocess_text)

                                            # 准备评估数据
                                            X_test = tokenize_and_pad(eval_df['processed_text'], st.session_state.vocab, 200)
                                            y_test = eval_df['label'].values

                                            # 评估模型
                                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                            y_pred, metrics = evaluate_model(model, X_test, y_test, device)

                                            # 显示评估结果
                                            st.subheader("评估结果")
                                            st.write(f"测试集准确率: {metrics['accuracy']:.4f}")
                                            st.write(f"F1分数: {metrics['f1']:.4f}")

                                            # 显示混淆矩阵
                                            st.subheader("混淆矩阵")
                                            tab1, tab2 = st.tabs(["绝对数量", "百分比"])
                                            with tab1:
                                                fig = plot_confusion_matrix(metrics['confusion_matrix'], list(genre_mapping.values()))
                                                st.pyplot(fig)
                                            with tab2:
                                                fig = plot_normalized_confusion_matrix(metrics['confusion_matrix'], list(genre_mapping.values()))
                                                st.pyplot(fig)

                                            # 显示分类报告
                                            st.subheader("分类报告")
                                            st.text(metrics['classification_report'])
                                else:
                                    st.error("无法修复数据集格式，请确保数据集包含有效的文本列")
                except Exception as e:
                    st.error(f"数据加载失败: {str(e)}")

# 文本分类页面
elif page == "文本分类":
    st.header("文本分类")

    # 增加多种输入方式
    input_method = st.radio(
        "选择输入方式",
        ["直接输入文本", "上传文本文件 (.txt)", "上传CSV文件"]
    )

    # 准备输入文本
    input_text = ""

    if input_method == "直接输入文本":
        input_text = st.text_area("输入小说片段进行分类", height=200)
    elif input_method == "上传文本文件 (.txt)":
        uploaded_file = st.file_uploader("上传文本文件", type=["txt"])
        if uploaded_file is not None:
            # 读取文本文件内容
            try:
                # 尝试不同编码
                try:
                    # 首先尝试UTF-8编码
                    input_text = uploaded_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    # 如果UTF-8解码失败，尝试GBK编码
                    uploaded_file.seek(0)  # 重置文件指针
                    input_text = uploaded_file.read().decode('gbk', errors='ignore')

                st.success(f'文件"{uploaded_file.name}"加载成功！')

                # 显示文件内容预览
                st.subheader("文件内容预览")
                st.text_area("文件内容", input_text, height=200, disabled=True)
            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")
    elif input_method == "上传CSV文件":
        uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])
        if uploaded_file is not None:
            try:
                # 保存上传的文件到临时位置
                temp_file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)

                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 检查并修复数据集格式
                try:
                    df, message = load_dataset_with_required_columns(temp_file_path)
                except Exception as e:
                    st.error(f"数据集加载错误: {str(e)}")
                    st.info("如果出现 'name 'pd' is not defined' 错误，请确保已正确导入pandas库。")
                    df, message = None, f"加载失败: {str(e)}"

                if df is not None:
                    st.success(f"{message}")

                    # 显示数据预览
                    st.subheader("数据预览")
                    st.dataframe(df.head())

                    # 选择要分类的文本
                    selected_index = st.selectbox("选择要分类的文本", range(len(df)))
                    input_text = df.iloc[selected_index]['text']

                    # 显示选择的文本
                    st.text_area("选择的文本", input_text, height=200, disabled=True)
                else:
                    st.error(message)
                    st.info("上传的数据集需要包含 'text' 列。您可以使用 fix_dataset_format.py 脚本修复数据集格式。")

                    # 提供修复选项
                    if st.button("自动修复数据集格式"):
                        with st.spinner("正在修复数据集格式..."):
                            # 导入修复函数和必要的库
                            import pandas as pd
                            from fix_dataset_format import fix_dataset_format

                            # 修复数据集
                            fixed_df = fix_dataset_format(temp_file_path)

                            if fixed_df is not None:
                                df = fixed_df
                                st.success(f"数据集格式已修复，共 {len(df)} 条记录")

                                # 显示数据预览
                                st.subheader("数据预览")
                                st.dataframe(df.head())

                                # 选择要分类的文本
                                selected_index = st.selectbox("选择要分类的文本", range(len(df)), key="fixed_text_select")
                                input_text = df.iloc[selected_index]['text']

                                # 显示选择的文本
                                st.text_area("选择的文本", input_text, height=200, disabled=True, key="fixed_text_area")
                            else:
                                st.error("无法修复数据集格式，请确保数据集包含有效的文本列")
            except Exception as e:
                st.error(f"数据加载失败: {str(e)}")

    # 选择模型类型
    if TRANSFORMER_MODELS_AVAILABLE:
        model_type = st.radio(
            "选择模型类型",
            ["传统模型", "Transformer模型"],
            index=0
        )
    else:
        model_type = "传统模型"

    if model_type == "传统模型":
        # 检查是否有训练好的模型
        if not st.session_state.models:
            # 尝试加载保存的模型
            model_files = [f for f in os.listdir('.') if f.endswith('.pt') and not f.startswith('Deep_BERT') and not f.startswith('Deep_RoBERTa')]

            if not model_files:
                st.warning("没有可用的模型，请先在'模型训练'页面训练模型或在'帮助和可视化'页面创建演示模型")
            else:
                model_name = st.selectbox("选择保存的模型", model_files)
                model, vocab = load_model(model_name)

                if model and vocab:
                    st.success(f"已加载模型: {model_name}")
                    st.session_state.models[model_name] = model
                    st.session_state.vocab = vocab
                    st.session_state.current_model = model_name

                    # 进行分类
                    if st.button("开始分类") and input_text:
                        classify_text(input_text, model, vocab)
        else:
            # 选择模型
            if st.session_state.current_model:
                default_model = st.session_state.current_model
            else:
                default_model = list(st.session_state.models.keys())[0]

            model_name = st.selectbox(
                "选择要使用的模型",
                list(st.session_state.models.keys()),
                index=list(st.session_state.models.keys()).index(default_model)
            )

            model = st.session_state.models[model_name]

            # 进行分类
            if st.button("开始分类") and input_text:
                classify_text(input_text, model, st.session_state.vocab)

    elif model_type == "Transformer模型" and TRANSFORMER_MODELS_AVAILABLE:
        # 检查磁盘上的transformer模型文件
        transformer_models = [f.replace(".pt", "") for f in os.listdir() if f.endswith('.pt') and (f.startswith('Deep_BERT') or f.startswith('Deep_RoBERTa'))]

        if not transformer_models:
            st.warning("没有找到可用的Transformer模型，请先使用train_transformer_model.py脚本训练模型")

            # 提供示例模型选项
            if st.button("使用预训练BERT模型进行演示"):
                with st.spinner("正在加载预训练BERT模型..."):
                    try:
                        model = BERTClassifier(num_classes=8)
                        tokenizer = model.get_tokenizer()
                        st.success("预训练BERT模型加载成功！现在可以进行文本分类了。")

                        # 进行分类
                        if input_text:
                            classify_text(input_text, model, None, is_transformer_model=True, tokenizer=tokenizer)
                    except Exception as e:
                        st.error(f"加载预训练模型失败: {str(e)}")
        else:
            selected_model = st.selectbox("选择Transformer模型", transformer_models)

            # 加载transformer模型
            try:
                model, tokenizer = load_transformer_model(selected_model + ".pt")

                if model is not None and tokenizer is not None:
                    st.success(f"已加载模型: {selected_model}")

                    # 进行分类
                    if st.button("开始分类") and input_text:
                        classify_text(input_text, model, None, is_transformer_model=True, tokenizer=tokenizer)
                else:
                    st.error("模型加载失败，请检查模型文件是否完整")
            except Exception as e:
                st.error(f"加载模型失败: {str(e)}")

# 模型比较页面
elif page == "模型比较":
    st.header("模型比较")

    # 检查是否有多个训练好的模型
    if len(st.session_state.models) < 2:
        st.warning("需要至少两个模型才能进行比较，请先在'模型训练'页面训练模型")
    else:
        st.write("本页面允许您直观地比较不同模型的性能，帮助您选择最适合您数据的模型。")

        # 选择比较方式
        compare_method = st.radio(
            "选择比较方式",
            ["性能指标比较", "训练过程比较", "混淆矩阵比较"],
            horizontal=True
        )

        # 选择要比较的模型
        models_to_compare = st.multiselect(
            "选择要比较的模型",
            list(st.session_state.models.keys()),
            default=list(st.session_state.models.keys())[:min(3, len(st.session_state.models))]
        )

        if len(models_to_compare) >= 2 and st.button("比较模型"):
            with st.spinner("正在比较模型..."):
                # 使用原始数据的测试集
                from sklearn.model_selection import train_test_split
                df = st.session_state.df
                train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

                # 准备评估数据
                X_test = tokenize_and_pad(test_df['processed_text'], st.session_state.vocab, 200)
                y_test = test_df['label'].values

                # 评估每个模型
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                results = {}

                for model_name in models_to_compare:
                    model = st.session_state.models[model_name]
                    y_pred, metrics = evaluate_model(model, X_test, y_test, device, return_prob=True)
                    results[model_name] = metrics

                # 根据选择的比较方式显示结果
                if compare_method == "性能指标比较":
                    st.subheader("模型性能指标比较")

                    # 准备合并数据
                    metrics_names = ["准确率", "F1分数", "精确率", "召回率"]
                    metrics_values = {
                        model_name: [results[model_name]['accuracy'],
                                   results[model_name]['f1'],
                                   results[model_name]['precision'],
                                   results[model_name]['recall']]
                        for model_name in results.keys()
                    }

                    # 制作显示多指标比较的雷达图
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, polar=True)
                    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
                    angles += angles[:1]  # 闭合图形

                    # 添加每个模型的雷达线
                    for i, model_name in enumerate(metrics_values.keys()):
                        values = metrics_values[model_name]
                        values += values[:1]  # 闭合图形
                        ax.plot(angles, values, linewidth=2, label=model_name)
                        ax.fill(angles, values, alpha=0.1)

                    # 设置图形属性
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(metrics_names)
                    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                    ax.yaxis.grid(True)
                    ax.set_ylim(0, 1)
                    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    plt.title("模型性能雷达图比较", va='bottom')
                    st.pyplot(fig)

                    # 详细指标表格
                    st.subheader("模型详细指标")
                    comparison_data = []
                    for model_name in results.keys():
                        comparison_data.append({
                            "模型": model_name,
                            "准确率": f"{results[model_name]['accuracy']:.4f}",
                            "F1分数": f"{results[model_name]['f1']:.4f}",
                            "精确率": f"{results[model_name]['precision']:.4f}",
                            "召回率": f"{results[model_name]['recall']:.4f}"
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)

                    # 各类别指标条形图
                    st.subheader("各类别指标条形图")
                    for model_name in models_to_compare:
                        st.write(f"#### {model_name} 模型各类别 F1 分数")
                        f1_per_class = results[model_name]['f1_per_class']
                        fig, ax = plt.subplots(figsize=(10, 6))
                        class_names = [genre_mapping.get(i, f'类别{i}') for i in range(len(f1_per_class))]
                        ax.barh(class_names, f1_per_class)
                        ax.set_xlabel('F1 分数')
                        ax.set_xlim(0, 1)
                        ax.grid(True, axis='x')
                        for i, v in enumerate(f1_per_class):
                            ax.text(v + 0.01, i, f"{v:.2f}", va='center')
                        plt.tight_layout()
                        st.pyplot(fig)

                elif compare_method == "训练过程比较":
                    st.subheader("训练过程比较")
                    if all(model_name in st.session_state.training_history for model_name in models_to_compare):

                        # 创建标签页
                        loss_tab, acc_tab, lr_tab = st.tabs(["损失函数", "准确率曲线", "学习率变化"])

                        with loss_tab:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for model_name in models_to_compare:
                                history = st.session_state.training_history[model_name]
                                ax.plot(history['loss'], label=f"{model_name} (训练)")
                                ax.plot(history['val_loss'], '--', label=f"{model_name} (验证)")

                            ax.set_xlabel('Epoch')
                            ax.set_ylabel('Loss')
                            ax.set_title('损失函数比较')
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)

                        with acc_tab:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for model_name in models_to_compare:
                                history = st.session_state.training_history[model_name]
                                ax.plot(history['acc'], label=f"{model_name} (训练)")
                                ax.plot(history['val_acc'], '--', label=f"{model_name} (验证)")

                            ax.set_xlabel('Epoch')
                            ax.set_ylabel('Accuracy')
                            ax.set_title('准确率比较')
                            ax.legend()
                            ax.grid(True)
                            ax.set_ylim(0, 1)
                            st.pyplot(fig)

                        with lr_tab:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for model_name in models_to_compare:
                                history = st.session_state.training_history[model_name]
                                if 'learning_rates' in history:
                                    ax.plot(history['learning_rates'], label=f"{model_name}")

                            ax.set_xlabel('Epoch')
                            ax.set_ylabel('Learning Rate')
                            ax.set_title('学习率变化曲线')
                            ax.legend()
                            ax.grid(True)
                            plt.yscale('log')
                            st.pyplot(fig)

                        # 显示收敛速度比较
                        st.subheader("训练收敛速度比较")
                        convergence_data = []

                        for model_name in models_to_compare:
                            history = st.session_state.training_history[model_name]
                            val_acc = history['val_acc']

                            # 找到验证准确率首次超过90%阈值的轮数
                            threshold_90 = next((i+1 for i, acc in enumerate(val_acc) if acc >= 0.9), "-")
                            threshold_95 = next((i+1 for i, acc in enumerate(val_acc) if acc >= 0.95), "-")

                            convergence_data.append({
                                "模型": model_name,
                                "最终准确率": f"{val_acc[-1]:.4f}",
                                "达到90%的轮数": threshold_90,
                                "达到95%的轮数": threshold_95,
                                "总训练轮数": len(val_acc)
                            })

                        st.table(pd.DataFrame(convergence_data))
                    else:
                        st.info("部分模型没有训练历史记录，无法比较训练过程")

                elif compare_method == "混淆矩阵比较":
                    st.subheader("混淆矩阵比较")

                    # 为每个模型显示混淆矩阵
                    for model_name in models_to_compare:
                        st.write(f"#### {model_name} 混淆矩阵")

                        tab1, tab2 = st.tabs(["绝对数量", "百分比"])
                        with tab1:
                            fig = plot_confusion_matrix(results[model_name]['confusion_matrix'], list(genre_mapping.values()))
                            st.pyplot(fig)
                        with tab2:
                            fig = plot_normalized_confusion_matrix(results[model_name]['confusion_matrix'], list(genre_mapping.values()))
                            st.pyplot(fig)

                # 错误分析
                st.subheader("常见错误模式分析")
                for model_name in models_to_compare:
                    errors = results[model_name]['error_analysis']
                    if len(errors) > 0:
                        most_common_errors = {}
                        for error in errors:
                            true_label = error['true_label']
                            pred_label = error['predicted_label']
                            error_pair = (true_label, pred_label)
                            most_common_errors[error_pair] = most_common_errors.get(error_pair, 0) + 1

                        # 显示前5个最常见的错误模式
                        top_errors = sorted(most_common_errors.items(), key=lambda x: x[1], reverse=True)[:5]

                        error_data = [{
                            "真实类别": genre_mapping.get(true_label, f"类别{true_label}"),
                            "预测类别": genre_mapping.get(pred_label, f"类别{pred_label}"),
                            "错误次数": count
                        } for (true_label, pred_label), count in top_errors]

                        st.write(f"**{model_name} 模型的常见错误模式**")
                        st.table(pd.DataFrame(error_data))
                    else:
                        st.write(f"**{model_name} 模型在测试集上没有错误！**")

                # 模型建议
                st.subheader("模型选择建议")
                best_model = max(results.items(), key=lambda x: x[1]['f1'])[0]
                st.success(f"基于全面的性能评估，我们建议使用 **{best_model}** 模型进行分类。该模型在F1分数等关键指标上表现最优。")
                st.info("注意：最终选择应考虑您的实际需求，比如模型大小、推理速度、特定类别的性能等因素。")

# 帮助和可视化页面
elif page == "帮助和可视化":
    st.header("模型架构可视化与示例生成")

    # 添加演示模型快速创建功能
    st.subheader("快速创建演示模型")
    st.write("下面可以快速创建演示模型用于测试，无需输入真实数据或训练即可直接使用。")

    col1, col2 = st.columns(2)
    with col1:
        model_options = ["CNN", "RNN", "Transformer"]
        if TRANSFORMER_MODELS_AVAILABLE:
            model_options.extend(["BERT", "RoBERTa"])

        demo_model_type = st.selectbox(
            "选择要创建的演示模型类型",
            model_options
        )
        demo_num_classes = st.slider("类别数量", 2, 10, 8, key="demo_num_classes")

    with col2:
        if demo_model_type in ["BERT", "RoBERTa"] and TRANSFORMER_MODELS_AVAILABLE:
            st.info("BERT和RoBERTa模型使用预训练的词嵌入，无需设置词汇表大小和嵌入维度")
            demo_vocab_size = 0  # 不使用
            demo_embedding_dim = 0  # 不使用
        else:
            demo_vocab_size = st.slider("词汇表大小", 1000, 10000, 5000, key="demo_vocab_size")
            demo_embedding_dim = st.slider("词嵌入维度", 50, 300, 100, key="demo_embedding_dim")

    if st.button("创建演示模型"):
        with st.spinner(f"正在创建 {demo_model_type} 演示模型..."):
            # 创建演示模型和词汇表
            model, vocab = create_demo_model(
                model_type=demo_model_type,
                vocab_size=demo_vocab_size,
                embedding_dim=demo_embedding_dim,
                num_classes=demo_num_classes
            )

            # 生成示例数据
            X, y = generate_sample_data(
                num_samples=1000,
                vocab_size=demo_vocab_size,
                max_length=200,
                num_classes=demo_num_classes
            )

            # 保存到会话状态
            model_name = f"Demo_{demo_model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.models[model_name] = model
            st.session_state.vocab = vocab
            st.session_state.current_model = model_name

            # 模拟训练历史数据
            fake_history = {
                'loss': [0.8, 0.6, 0.4, 0.3, 0.25],
                'val_loss': [0.85, 0.65, 0.45, 0.35, 0.3],
                'acc': [0.6, 0.7, 0.8, 0.85, 0.9],
                'val_acc': [0.55, 0.65, 0.75, 0.8, 0.85],
                'learning_rates': [0.001, 0.0008, 0.0006, 0.0004, 0.0002]
            }
            st.session_state.training_history[model_name] = fake_history

            # 创建简单的DataFrame
            text_samples = [' '.join([f'word{token}' if token > 0 else '<PAD>' for token in sample[:20]]) + '...' for sample in X]
            st.session_state.df = pd.DataFrame({
                'text': text_samples,
                'processed_text': [['word'+str(token) if token > 0 else '<PAD>' for token in sample[:20]] for sample in X],
                'label': y
            })

            st.success(f"成功创建 {demo_model_type} 演示模型 '{model_name}'")
            st.info("现在您可以在 '模型评估', '文本分类' 或 '模型比较' 页面使用演示模型")

    st.markdown("---")

    # 模型架构可视化
    st.subheader("模型架构可视化")
    model_options = ["CNN", "RNN", "Transformer"]
    if TRANSFORMER_MODELS_AVAILABLE:
        model_options.extend(["BERT", "RoBERTa"])

    model_type_viz = st.selectbox(
        "选择要可视化的模型类型",
        model_options
    )

    img_str = visualize_model_architecture(model_type_viz)
    st.image(f"data:image/png;base64,{img_str}", caption=f"{model_type_viz}模型架构示意图")

    # 生成示例数据
    st.subheader("示例数据生成")
    st.write("该功能允许您生成示例数据进行测试，非常适合在无真实数据情况下快速试验模型。")

    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.slider("生成样本数量", 10, 1000, 100, key="gen_num_samples")
        max_length = st.slider("序列最大长度", 10, 500, 100, key="gen_max_length")

    with col2:
        vocab_size = st.slider("词汇表大小", 100, 5000, 1000, key="gen_vocab_size")
        num_classes = st.slider("类别数量", 2, 10, 8, key="gen_num_classes")

    if st.button("生成示例数据"):
        with st.spinner("正在生成示例数据..."):
            X, y = generate_sample_data(num_samples, vocab_size, max_length, num_classes)

            # 保存数据到会话状态
            st.session_state.X_sample = X
            st.session_state.y_sample = y

            # 创建一个简单的词汇表
            sample_vocab = {str(i): i for i in range(1, vocab_size + 1)}
            sample_vocab['<PAD>'] = 0
            sample_vocab['<UNK>'] = vocab_size + 1
            st.session_state.vocab = sample_vocab

            # 深度学习无需真实词汇，只需要映射关系
            st.session_state.df = pd.DataFrame({
                'text': [''.join([f'word{token}' if token > 0 else '' for token in sample]) for sample in X],
                'label': y
            })

            # 显示数据统计信息
            st.success(f"示例数据生成成功! X形状: {X.shape}, y形状: {y.shape}")

            # 类别分布
            label_counts = pd.Series(y).value_counts().sort_index()
            fig = plot_class_distribution(
                label_counts,
                class_names=genre_mapping,
                title='生成的样本各类别数量分布',
                figsize=(12, 6)
            )
            st.pyplot(fig)

            # 序列长度分布
            seq_lengths = np.sum(X > 0, axis=1)
            fig, ax = plt.subplots(figsize=(10, 6))

            # 使用更美观的颜色和样式
            n, bins, patches = ax.hist(seq_lengths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

            # 添加核密度估计曲线
            from scipy import stats
            kde_x = np.linspace(seq_lengths.min(), seq_lengths.max(), 1000)
            kde = stats.gaussian_kde(seq_lengths)
            ax.plot(kde_x, kde(kde_x) * len(seq_lengths) * (bins[1] - bins[0]),
                   'r-', linewidth=2, label='密度估计')

            # 添加统计信息
            ax.axvline(seq_lengths.mean(), color='green', linestyle='dashed', linewidth=2, label=f'平均值: {seq_lengths.mean():.1f}')
            ax.axvline(np.median(seq_lengths), color='orange', linestyle='dashed', linewidth=2, label=f'中位数: {np.median(seq_lengths):.1f}')

            # 美化图表
            ax.set_xlabel('序列长度', fontsize=12)
            ax.set_ylabel('频数', fontsize=12)
            ax.set_title('生成的样本序列长度分布', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            # 添加统计数据文本框
            stats_text = (f"统计信息:\n"
                         f"样本数: {len(seq_lengths)}\n"
                         f"最小值: {seq_lengths.min()}\n"
                         f"最大值: {seq_lengths.max()}\n"
                         f"平均值: {seq_lengths.mean():.2f}\n"
                         f"中位数: {np.median(seq_lengths):.2f}\n"
                         f"标准差: {seq_lengths.std():.2f}")

            # 在图表右上角添加文本框
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            st.pyplot(fig)

    # 帮助和说明
    st.subheader("关于本系统")

    # 加载详细使用说明文档
    with open("templates/instructions.md", "r", encoding="utf-8") as f:
        instructions = f.read()

    if st.checkbox("显示详细使用说明", value=False):
        st.markdown(instructions)
    else:
        st.markdown("""
        ### 系统功能概述

        本系统为基于深度学习的网络小说题材分类系统，提供了以下功能：

        1. **数据处理**：上传和预处理网络小说文本数据
        2. **模型训练**：使用不同的深度学习模型进行训练
           - CNN: 卷积神经网络，操作更高效，适合捕获局部文本特征
           - RNN (LSTM): 递归神经网络，操作更适合捕获文本的序列信息
           - Transformer: 基于自注意力机制，可以更好地捕获过长距离的依赖关系
        3. **模型评估**：对训练好的模型进行性能评估
        4. **文本分类**：使用训练好的模型对新的文本进行分类
        5. **模型比较**：对多个模型的性能进行比较
        6. **帮助和可视化**：提供模型架构可视化和示例数据生成
        """)
        st.info("点击上方的复选框显示完整的使用说明文档")

    st.subheader("注意力可视化演示")
    st.markdown("""
    Transformer模型的一个关键实用功能是自注意力机制，它可以展示模型在处理文本时的关注点。
    下面提供了一个简单的示意图，展示了注意力权重的多头计算机制。
    """)

    # 创建一个示意的注意力热力图
    tokens = ["他", "拿", "起", "剑", "开始", "修炼", "术", "。"]
    attention_weights = np.random.rand(len(tokens), len(tokens))
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)  # 归一化

    fig = plot_attention_weights(tokens, attention_weights)
    st.pyplot(fig)

    st.info("注意: 上图为示意图，展示了注意力机制如何将不同单词进行加权关联。当使用真实模型分类时，注意力图会提供实际的单词关联强度。")

# 显示页脚
st.markdown("---")
st.markdown("基于深度学习的网络小说题材分类算法研究 © 2024")
