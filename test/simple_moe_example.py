import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义单个专家网络，结构为一个两层的前馈神经网络
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 第一层
            nn.ReLU(),                         # 激活函数
            nn.Linear(hidden_dim, input_dim)   # 输出层
        )

    def forward(self, x):
        return self.net(x)  # 前向传播

# 定义门控网络，根据输入决定每个样本应该分配给哪些专家
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_experts)  # 线性层输出每个专家的得分

    def forward(self, x):
        # 使用 softmax 得到每个专家的权重（概率分布）
        return F.softmax(self.linear(x), dim=-1)

# 主体 MoE 模型
class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts  # 专家数量
        self.k = k  # 每个样本最多选择的专家数量（top-k）
        
        # 创建多个专家
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        
        # 初始化门控网络
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gate_scores = self.gate(x)  # [batch_size, num_experts]，每个样本的专家得分
        topk_vals, topk_idx = torch.topk(gate_scores, self.k, dim=1)  # 选择每个样本得分最高的 k 个专家

        output = torch.zeros_like(x)  # 初始化输出张量
        load = torch.zeros(self.num_experts, device=x.device)  # 记录每个专家被选中的次数，用于负载均衡损失

        # 遍历 top-k 专家
        for i in range(self.k):
            idx = topk_idx[:, i]  # 第 i 个被选中的专家编号 [batch_size]
            weight = topk_vals[:, i].unsqueeze(1)  # 对应专家的权重（注意需要 reshape）

            # 遍历所有专家
            for expert_id in range(self.num_experts):
                mask = (idx == expert_id)  # 找出哪些样本被分配给当前专家
                if mask.sum() == 0:
                    continue  # 如果没有样本分配给该专家，跳过

                selected_input = x[mask]  # 选中被该专家处理的样本
                expert_output = self.experts[expert_id](selected_input)  # 前向传播
                output[mask] += expert_output * weight[mask]  # 加权累加到输出中
                load[expert_id] += mask.sum()  # 统计当前专家处理了多少样本

        # 负载均衡损失：鼓励每个专家使用数量接近平均，避免所有流量都集中在少数专家
        load_balancing_loss = (load / load.sum()).pow(2).sum()

        return output, load_balancing_loss  # 返回最终输出和负载损失

# ========== 示例使用 ==========

batch_size = 8
input_dim = 16
hidden_dim = 32
num_experts = 4
k = 2

# 实例化模型
model = MoE(input_dim, hidden_dim, num_experts, k)
inputs = torch.randn(batch_size, input_dim)  # 构造一个随机输入
outputs, load_loss = model(inputs)  # 前向传播，得到模型输出和负载损失

# 构造一个假的回归任务损失（MSE）
task_loss = F.mse_loss(outputs, torch.randn_like(outputs))

# 总损失 = 任务损失 + 负载均衡正则项（加入一个系数，比如 0.01）
loss = task_loss + 0.01 * load_loss

# 反向传播
loss.backward()
