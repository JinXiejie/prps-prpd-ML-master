import torch
import torch.nn as nn
import numpy as np



def quick_lstm_implementation():
    """快速实现的简化版本"""


    class SimplePhaseLSTM(nn.Module):
        """简化版LSTM：cyc_num × phase -> 1 × phase"""

        def __init__(self, phase_dim=360, hidden_dim=128):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=phase_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )

            # 输出层
            self.output = nn.Linear(hidden_dim * 2, phase_dim)

        def forward(self, x):
            """
            Args:
                x: [batch, cyc_num, phase_dim]

            Returns:
                output: [batch, 1, phase_dim]
            """
            # LSTM处理
            lstm_out, (hidden, cell) = self.lstm(x)  # [batch, cyc_num, hidden*2]

            # 取最后一个时间步
            last_output = lstm_out[:, -1:, :]  # [batch, 1, hidden*2]

            # 映射回phase维度
            output = self.output(last_output)  # [batch, 1, phase_dim]

            return output

    # 使用示例
    batch_size = 4
    cyc_num = 50  # 不定长，每个样本不同
    phase_dim = 360

    # 创建模拟数据
    data = torch.randn(batch_size, cyc_num, phase_dim)

    # 创建模型
    model = SimplePhaseLSTM(phase_dim=phase_dim)

    # 前向传播
    output = model(data)

    print(f"输入形状: {data.shape}")  # [4, 50, 360]
    print(f"输出形状: {output.shape}")  # [4, 1, 360]

    return model


# 运行简化版
model = quick_lstm_implementation()