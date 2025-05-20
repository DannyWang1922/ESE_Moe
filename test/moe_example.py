import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, dim_model, hidden_dim):
        super(Expert, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_model)
        )
    
    def forward(self, x):
        return self.ffn(x)

class GatingNetwork(nn.Module):
    def __init__(self, dim_model, num_experts, top_k=1):
        super(GatingNetwork, self).__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim_model, num_experts)

    def forward(self, x):
        # x: [batch, seq_len, dim_model]
        logits = self.gate(x)  # [batch, seq_len, num_experts]
        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_scores = F.softmax(top_k_values, dim=-1)  # softmax over top-k
        
        return top_k_indices, top_k_scores

class MoEFeedForward(nn.Module):
    def __init__(self, dim_model, hidden_dim, num_experts, top_k=1):
        super(MoEFeedForward, self).__init__()
        self.experts = nn.ModuleList([Expert(dim_model, hidden_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(dim_model, num_experts, top_k)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        # x: [batch, seq_len, dim_model]
        batch_size, seq_len, dim_model = x.shape
        top_k_indices, top_k_scores = self.gating(x)  # [batch, seq_len, top_k]

        # Gather expert outputs
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # [batch, seq_len]
            score = top_k_scores[..., i].unsqueeze(-1)  # [batch, seq_len, 1]
            
            # Process each expert in batch (vectorized gather)
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)  # [batch, seq_len]
                if mask.sum() == 0:
                    continue
                selected_input = x[mask]  # [N_selected, dim_model]
                expert_out = self.experts[expert_id](selected_input)  # [N_selected, dim_model]
                output[mask] += expert_out * score[mask]
        
        return output

class TransformerBlockWithMoE(nn.Module):
    def __init__(self, dim_model, num_heads, hidden_dim, num_experts, dropout=0.1):
        super(TransformerBlockWithMoE, self).__init__()
        self.attn = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

        self.moe_ffn = MoEFeedForward(dim_model, hidden_dim, num_experts)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_output)

        moe_output = self.moe_ffn(x)
        x = self.norm2(x + moe_output)

        return x

# 模型参数
dim_model = 64
hidden_dim = 128
num_heads = 4
num_experts = 4
seq_len = 10
batch_size = 2

# 输入示例
x = torch.randn(batch_size, seq_len, dim_model)

# 构建一个带 MoE 的 Transformer Block
block = TransformerBlockWithMoE(dim_model, num_heads, hidden_dim, num_experts)

# 前向传播
out = block(x)

print(out.shape)  # [batch_size, seq_len, dim_model]
