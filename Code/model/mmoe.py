import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


# ==================== 数据预处理函数 ====================
def is_valid_date(date_text):
    """判断字符串是否为有效日期格式"""
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def cal_date_diff(date_str):
    """计算日期差异"""
    today = datetime.strptime('2014-6-11', '%Y-%m-%d')
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return abs(today - date).days


# ==================== MMoE 网络结构 ====================
class Expert(nn.Module):
    """专家网络"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(Expert, self).__init__()
        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Tower(nn.Module):
    """任务塔网络"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        super(Tower, self).__init__()
        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MMoE(nn.Module):
    """MMoE多任务学习网络"""

    def __init__(
            self,
            input_dim: int,
            num_experts: int,
            experts_output_dim: int,
            num_tasks: int,
            experts_hidden_dims=None,
            towers_hidden_dims=None,
            dropout_rate: float = 0.3
    ):
        super(MMoE, self).__init__()

        if towers_hidden_dims is None:
            towers_hidden_dims = [64, 32]
        if experts_hidden_dims is None:
            experts_hidden_dims = [256, 128]
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, experts_hidden_dims, experts_output_dim, dropout_rate)
            for _ in range(num_experts)
        ])

        # 门控网络（每个任务一个门控）
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_tasks)
        ])

        # 任务塔（每个任务一个塔）
        self.towers = nn.ModuleList([
            Tower(experts_output_dim, towers_hidden_dims, 1, dropout_rate)  # 每个塔输出1维，通过sigmoid转为概率
            for _ in range(num_tasks)
        ])

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, input_dim]
        Returns:
            每个任务的预测概率列表
        """
        batch_size = x.shape[0]

        # 计算各专家的输出
        experts_output = [expert(x) for expert in self.experts]  # 每个: [batch_size, experts_output_dim]
        experts_output = torch.stack(experts_output, dim=1)  # [batch_size, num_experts, experts_output_dim]

        # 对每个任务计算门控权重并加权专家输出
        task_outputs = []
        for i in range(self.num_tasks):
            # 计算门控权重
            gate_weights = self.gates[i](x)  # [batch_size, num_experts]
            gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]

            # 加权专家输出
            weighted_expert_output = (experts_output * gate_weights).sum(dim=1)  # [batch_size, experts_output_dim]

            # 通过任务塔
            task_output = self.towers[i](weighted_expert_output)  # [batch_size, 1]
            task_output = torch.sigmoid(task_output)  # 转换为概率
            task_outputs.append(task_output)

        return task_outputs


# ==================== 自定义数据集 ====================
class GISDataset(Dataset):
    """GIS局部放电数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, type_ids: np.ndarray):
        """
        Args:
            features: 特征矩阵 [n_samples, n_features]
            labels: 主标签（用于多分类任务）
            type_ids: 所有type_id的one-hot编码 [n_samples, n_type_ids]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.type_ids = torch.FloatTensor(type_ids)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'type_ids': self.type_ids[idx]
        }


# ==================== 数据处理类 ====================
class DataProcessor:
    """数据处理管道"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.type_id_encoders = {}  # 存储每个type_id的LabelEncoder
        self.scaler = StandardScaler()
        self.feature_names = None
        self.num_type_ids = 0

    def prepare_type_id_labels(self, data: pd.DataFrame, type_id_col: str = 'type_id') -> np.ndarray:
        """为每个type_id创建二元标签（存在/不存在）"""
        # 获取所有唯一的type_id
        unique_type_ids = sorted(data[type_id_col].unique())
        self.num_type_ids = len(unique_type_ids)

        # 为每个样本创建one-hot编码
        type_id_labels = np.zeros((len(data), self.num_type_ids))

        for i, type_id in enumerate(unique_type_ids):
            mask = data[type_id_col] == type_id
            type_id_labels[mask.values, i] = 1

        return type_id_labels

    def process_features(self, data: pd.DataFrame, categorical_cols: List[str],
                         embedding_dim: int = 8) -> Tuple[pd.DataFrame, Dict]:
        """处理特征，包括类别特征嵌入"""
        processed_data = data.copy()
        embeddings_info = {}

        # 处理类别特征
        for col in categorical_cols:
            if col in processed_data.columns:
                # 使用LabelEncoder编码
                le = LabelEncoder()
                encoded = le.fit_transform(processed_data[col].fillna('unknown'))
                processed_data[col] = encoded

                # 创建嵌入层
                num_categories = len(le.classes_)
                embedding_layer = nn.Embedding(num_categories, embedding_dim)

                # 获取嵌入向量
                with torch.no_grad():
                    embeddings = embedding_layer(torch.tensor(encoded, dtype=torch.long))
                    embeddings_np = embeddings.numpy()

                # 添加嵌入特征
                for i in range(embedding_dim):
                    processed_data[f'{col}_embed_{i}'] = embeddings_np[:, i]

                # 存储嵌入信息
                embeddings_info[col] = {
                    'encoder': le,
                    'embedding_layer': embedding_layer,
                    'num_categories': num_categories
                }

        # 移除原始类别列
        processed_data = processed_data.drop(columns=categorical_cols)

        return processed_data, embeddings_info

    def split_data(self, data: pd.DataFrame, datetime_col: str = 'datetime') -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间划分数据"""
        # 处理日期列
        if datetime_col in data.columns:
            data[datetime_col] = data[datetime_col].apply(
                lambda x: cal_date_diff(str(x)) if is_valid_date(str(x)) else -1
            )

        # 按日期划分
        train_data = data[data[datetime_col] < 575]
        valid_data = data[(575 <= data[datetime_col]) & (data[datetime_col] < 600)]
        test_data = data[data[datetime_col] >= 600]

        return train_data, valid_data, test_data


# ==================== 训练和评估函数 ====================
def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device, num_tasks: int, weights) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in dataloader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        type_ids = batch['type_ids'].to(device)

        # 前向传播
        task_outputs = model(features)

        # 计算损失
        loss = 0
        for i in range(num_tasks):
            # 每个任务的损失（二元交叉熵）
            task_loss = criterion(task_outputs[i].squeeze(), type_ids[:, i]) * weights[i]
            loss += task_loss

        # 计算主任务损失（多分类交叉熵）
        # 这里我们使用所有任务输出的加权和作为主任务的输入
        combined_output = torch.cat([output for output in task_outputs], dim=1)  # [batch_size, num_tasks]
        main_loss = nn.CrossEntropyLoss()(combined_output, labels)

        total_loss += (loss + main_loss).item()

        # 反向传播
        optimizer.zero_grad()
        (loss + main_loss).backward()
        optimizer.step()

        # 计算准确率（主任务）
        _, predicted = torch.max(combined_output, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device, num_tasks: int, weights) -> Tuple[float, float, Dict]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    task_accuracies = {f'task_{i}': 0.0 for i in range(num_tasks)}
    task_samples = {f'task_{i}': 0 for i in range(num_tasks)}

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            type_ids = batch['type_ids'].to(device)

            # 前向传播
            task_outputs = model(features)

            # 计算损失
            loss = 0
            for i in range(num_tasks):
                task_loss = criterion(task_outputs[i].squeeze(), type_ids[:, i]) * weights[i]
                loss += task_loss

            # 主任务损失
            combined_output = torch.cat([output for output in task_outputs], dim=1)
            main_loss = nn.CrossEntropyLoss()(combined_output, labels)

            total_loss += (loss + main_loss).item()

            # 计算主任务准确率
            _, predicted = torch.max(combined_output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 计算各任务准确率
            for i in range(num_tasks):
                task_pred = (task_outputs[i].squeeze() > 0.5).float()
                task_accuracies[f'task_{i}'] += (task_pred == type_ids[:, i]).sum().item()
                task_samples[f'task_{i}'] += type_ids[:, i].size(0)

    # 计算平均指标
    avg_loss = total_loss / len(dataloader)
    main_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # 计算各任务准确率
    for i in range(num_tasks):
        if task_samples[f'task_{i}'] > 0:
            task_accuracies[f'task_{i}'] /= task_samples[f'task_{i}']

    return avg_loss, main_accuracy, task_accuracies


def early_stopping_criterion(acc_history, patience, best_val_accuracy):
    acc_nums = len(acc_history)
    if acc_nums >= patience:
        for val in acc_history[acc_nums - 1:-patience:-1]:
            print("acc_history: ", str(acc_history))
            print("val: ", str(val))
            if val > best_val_accuracy:
                return False
        return True
    return False


# ==================== 主程序 ====================
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("加载数据...")
    data_path = 'G:\\PycharmProjects\\prps-prpd-ML-master\\Code\\data\\prps_fea_data.csv'
    data = pd.read_csv(data_path)

    # 2. 数据处理
    print("处理数据...")
    processor = DataProcessor()

    # 定义类别特征
    categorical_cols = ['devnum_id', 'channelnum_id']
    # categorical_cols = ['devnum_id', 'channelnum_id', 'alarm_id']

    # 移除原始类别列
    data = data.drop(columns='datastatus_id')

    # 处理特征
    processed_data, embeddings_info = processor.process_features(data, categorical_cols)

    # 准备标签
    # 主标签（用于多分类）
    label_encoder = LabelEncoder()
    processed_data['type_label'] = label_encoder.fit_transform(processed_data['type_id'])
    num_classes = len(label_encoder.classes_)
    num_samples = len(processed_data)

    # 各type_id的二元标签
    type_id_labels = processor.prepare_type_id_labels(processed_data)
    num_tasks = type_id_labels.shape[1]  # type_id的数量

    print(f"样本总数: {num_samples}")
    print(f"类别数量: {num_classes}")
    print(f"任务数量 (type_id): {num_tasks}")

    # 按时间划分数据
    train_data, valid_data, test_data = processor.split_data(processed_data)

    # 按样本数量计算权重
    weights = processed_data['type_label'].value_counts().apply(lambda x: x / num_samples).sort_index().tolist()

    # 3. 准备特征和标签
    # 移除不需要的列
    exclude_cols = ['type_id', 'type_label', 'datetime']
    feature_cols = [col for col in processed_data.columns if col not in exclude_cols]

    X_train = train_data[feature_cols].values
    X_valid = valid_data[feature_cols].values
    X_test = test_data[feature_cols].values

    y_train_main = train_data['type_label'].values
    y_valid_main = valid_data['type_label'].values
    y_test_main = test_data['type_label'].values

    # 获取type_id标签
    train_indices = train_data.index
    valid_indices = valid_data.index
    test_indices = test_data.index

    y_train_tasks = type_id_labels[train_indices]
    y_valid_tasks = type_id_labels[valid_indices]
    y_test_tasks = type_id_labels[test_indices]

    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # 4. 创建数据集和数据加载器
    print("创建数据加载器...")
    train_dataset = GISDataset(X_train, y_train_main, y_train_tasks)
    valid_dataset = GISDataset(X_valid, y_valid_main, y_valid_tasks)
    test_dataset = GISDataset(X_test, y_test_main, y_test_tasks)

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. 创建MMoE模型
    print("初始化MMoE模型...")
    input_dim = X_train.shape[1]
    num_experts = 5  # 专家数量
    experts_output_dim = 32  # 专家输出维度

    model = MMoE(
        input_dim=input_dim,
        num_experts=num_experts,
        experts_output_dim=experts_output_dim,
        num_tasks=num_tasks,
        experts_hidden_dims=[128, 64],
        towers_hidden_dims=[experts_output_dim, 16],
        dropout_rate=0.2
    ).to(device)

    # 6. 训练配置
    criterion = nn.BCELoss()  # 用于各任务的二元分类
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # 7. 训练循环
    print("开始训练...")
    num_epochs = 100
    best_val_accuracy = 0
    patience = 20
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    acc_history = []

    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, num_tasks, weights)

        # 验证
        val_loss, val_acc, task_accs = evaluate(model, valid_loader, criterion, device, num_tasks, weights)


        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # 学习率调整
        scheduler.step(val_loss)

        # 打印进度
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 显示前几个任务的准确率
        print("  各任务准确率:")
        for i in range(num_tasks):
            print(f"    Task {i}: {task_accs[f'task_{i}']:.4f}")

        # 早停检查
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
            }, 'best_mmoe_model.pth')
            print(f"  保存最佳模型，验证准确率: {val_acc:.4f}")
        else:
            # if early_stopping_criterion(acc_history, patience, best_val_accuracy):
            #     print(f"早停触发，在epoch {epoch + 1}")
            #     break
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，在epoch {epoch + 1}")
                break

        print("-" * 50)
        acc_history.append(val_acc)

    # 8. 测试最佳模型
    print("测试最佳模型...")
    # 加载最佳模型
    checkpoint = torch.load('best_mmoe_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_task_accs = evaluate(model, test_loader, criterion, device, num_tasks, weights)

    print(f"测试结果:")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  主任务准确率: {test_acc:.4f}")
    print(f"  各任务准确率:")
    for i in range(num_tasks):
        print(f"    任务 {i}: {test_task_accs[f'task_{i}']:.4f}")

    # 9. 可视化结果
    print("生成可视化图表...")
    # ==================== 中文显示配置 ====================
    # 方案1: 设置中文字体（推荐）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 损失曲线
    axes[0, 0].plot(train_losses, label='训练损失')
    axes[0, 0].plot(val_losses, label='验证损失')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 准确率曲线
    axes[0, 1].plot(train_accuracies, label='训练准确率')
    axes[0, 1].plot(val_accuracies, label='验证准确率')
    axes[0, 1].set_title('训练和验证准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 各任务准确率柱状图
    task_indices = list(range(num_tasks))
    task_acc_values = [test_task_accs[f'task_{i}'] for i in range(num_tasks)]
    axes[1, 0].bar(task_indices, task_acc_values)
    axes[1, 0].set_title('各type_id分类任务准确率')
    axes[1, 0].set_xlabel('任务索引')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].grid(True, axis='y')

    # 混淆矩阵（主任务）
    # 预测测试集
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            task_outputs = model(features)
            combined_output = torch.cat([output for output in task_outputs], dim=1)
            _, predicted = torch.max(combined_output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_labels, all_predictions)

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=label_encoder.classes_)
    disp.plot(ax=axes[1, 1], cmap='Blues', values_format='d')
    axes[1, 1].set_title('主任务混淆矩阵')

    plt.tight_layout()
    plt.savefig('mmoe_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("训练完成！结果已保存到 mmoe_results.png")


# ==================== 模型解释工具 ====================
class MMoEInterpreter:
    """MMoE模型解释工具"""

    def __init__(self, model: MMoE, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def analyze_gate_weights(self, X: np.ndarray, task_idx: int) -> pd.DataFrame:
        """分析门控网络权重"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            gate_weights = self.model.gates[task_idx][0](X_tensor)  # 获取线性层输出
            gate_weights = gate_weights.mean(dim=0)  # 平均权重

        # 创建权重DataFrame
        weights_df = pd.DataFrame({
            'expert': range(len(gate_weights)),
            'weight': gate_weights.numpy()
        }).sort_values('weight', ascending=False)

        return weights_df

    def get_feature_importance(self, X: np.ndarray, task_idx: int) -> pd.DataFrame:
        """分析特征重要性"""
        # 这里可以使用SHAP或简单的梯度分析
        # 简化的实现：基于门控网络的权重和专家网络的权重

        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        X_tensor.requires_grad = True

        # 获取任务输出
        task_outputs = self.model(X_tensor)
        output = task_outputs[task_idx].mean()

        # 计算梯度
        output.backward()
        gradients = X_tensor.grad.mean(dim=0).numpy()

        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(gradients)
        }).sort_values('importance', ascending=False)

        return importance_df


if __name__ == "__main__":
    main()



# 早停触发，在epoch 37
# 测试最佳模型...
# 测试结果:
#   测试损失: 1.8533
#   主任务准确率: 0.8475
#   各任务准确率:
#     任务 0: 0.9880
#     任务 1: 0.8983
#     任务 2: 0.9514
#     任务 3: 0.9972
#     任务 4: 0.8616
# 生成可视化图表...