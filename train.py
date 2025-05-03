import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.001, device='cpu', patience=5, optimizer_type='adam'):
    """
    训练深度学习模型

    参数:
    model: 模型
    X_train: 训练数据
    y_train: 训练标签
    X_val: 验证数据
    y_val: 验证标签
    epochs: 训练轮数
    batch_size: 批次大小
    learning_rate: 学习率
    device: 设备 (cpu/cuda)
    patience: 早停的耐心值，如果验证集性能在这么多轮内没有提升，则停止训练
    optimizer_type: 优化器类型

    返回:
    训练历史记录
    """
    # 准备数据加载器，带有平行处理能力
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    # 定义优化器和损失函数
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = nn.CrossEntropyLoss()

    # 训练历史记录
    history = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': [],
        'f1': [],
        'val_f1': [],
        'learning_rates': []
    }

    # 早停变量
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_trues = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 记录损失和预测结果
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_trues.extend(labels.cpu().numpy())

        # 计算训练集指标
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_trues, train_preds)
        train_f1 = f1_score(train_trues, train_preds, average='weighted')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 记录损失和预测结果
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        # 计算验证集指标
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_trues, val_preds)
        val_f1 = f1_score(val_trues, val_preds, average='weighted')

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        old_lr = current_lr
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # 如果学习率发生变化，打印信息
        if old_lr != new_lr:
            print(f'Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}')

        # 记录历史
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)

        # 输出训练进度
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - F1: {train_f1:.4f} - '
              f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f} - '
              f'LR: {current_lr:.6f}')

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                # 恢复最佳模型
                model.load_state_dict(best_model_state)
                break

    # 如果使用了早停并保存了最佳模型，则恢复到最佳状态
    if best_model_state is not None and no_improve_epochs >= patience:
        model.load_state_dict(best_model_state)

    return history

def evaluate_model(model, X_test, y_test, device='cpu', return_prob=False, return_attention=False):
    """
    评估模型性能

    参数:
    model: 模型
    X_test: 测试数据
    y_test: 测试标签
    device: 设备 (cpu/cuda)
    return_prob: 是否返回概率分布
    return_attention: 是否返回注意力权重（如果模型支持）

    返回:
    预测结果和评估指标
    """
    # 准备数据加载器
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)

    # 评估模型
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []
    attention_weights = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 获取概率
            if return_prob:
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

            # 获取预测类别
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())

            # 如果模型支持注意力机制且需要获取注意力权重
            if return_attention and hasattr(model, 'get_attention_weights'):
                attn = model.get_attention_weights(inputs)
                if attn is not None:
                    attention_weights.extend(attn.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, average='weighted')
    f1_per_class = f1_score(all_trues, all_preds, average=None)
    cm = confusion_matrix(all_trues, all_preds)
    report = classification_report(all_trues, all_preds)

    # 精确率和召回率
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    precision = precision_score(all_trues, all_preds, average='weighted')
    recall = recall_score(all_trues, all_preds, average='weighted')

    # 错误分析
    errors = []
    for i, (true, pred) in enumerate(zip(all_trues, all_preds)):
        if true != pred:
            error_info = {
                'index': i,
                'true_label': true,
                'predicted_label': pred
            }
            if return_prob:
                error_info['probabilities'] = all_probs[i]
            errors.append(error_info)

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'error_analysis': errors
    }

    # 返回概率分布（如果需要）
    if return_prob:
        metrics['probabilities'] = all_probs

    # 返回注意力权重（如果需要且支持）
    if return_attention and attention_weights:
        metrics['attention_weights'] = attention_weights

    # 计算ROC-AUC（如果返回了概率）
    if return_prob and len(np.unique(all_trues)) > 1:
        try:
            from sklearn.preprocessing import label_binarize
            classes = np.unique(all_trues)
            if len(classes) == 2:
                metrics['roc_auc'] = roc_auc_score(all_trues, np.array(all_probs)[:, 1])
            else:
                # 多分类情况下的ROC-AUC
                y_test_bin = label_binarize(all_trues, classes=classes)
                metrics['roc_auc'] = roc_auc_score(y_test_bin, np.array(all_probs), multi_class='ovr', average='macro')
        except Exception as e:
            print(f"ROC-AUC计算错误: {str(e)}")
            metrics['roc_auc'] = None

    return all_preds, metrics
