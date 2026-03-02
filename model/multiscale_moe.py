import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleMoE(nn.Module):
    """多尺度层次化专家系统
    
    采用双层专家结构，主专家负责区分长短程力的大尺度特征，
    次级专家负责识别每种力类型内部的细分模式，如共价键、
    氢键、静电力、色散力等，从而形成完整的多尺度物理知识表示。
    """
    def __init__(self, hidden_channels, primary_experts=2, secondary_experts=3, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.primary_experts = primary_experts    # 主专家数量 (长/短程力)
        self.secondary_experts = secondary_experts  # 次级专家数量 (每种力类型的细分)
        
        # 主专家门控网络 - 区分长程/短程
        self.primary_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, primary_experts),
            nn.Softmax(dim=-1)
        )
        
        # 次级专家门控网络 - 每个主专家领域下的细分
        self.secondary_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, 96),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(96, secondary_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(primary_experts)
        ])
        
        # 次级专家网络 - 标量 (二维结构：[primary][secondary])
        self.secondary_experts_scalar = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels, hidden_channels)
                ) for _ in range(secondary_experts)
            ]) for _ in range(primary_experts)
        ])
        
        # 次级专家网络 - 向量 (二维结构：[primary][secondary])
        self.secondary_experts_vector = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels, hidden_channels, bias=False)
                ) for _ in range(secondary_experts)
            ]) for _ in range(primary_experts)
        ])
        
        # 物理解释器 - 为不同专家组合提供物理解释的标签
        # 例如: 0,0 = "共价键", 0,1 = "氢键", 1,0 = "静电力" 等
        self.physics_interpreter = nn.Sequential(
            nn.Linear(2, 16),
            nn.SiLU(),
            nn.Linear(16, primary_experts * secondary_experts)
        )
        
        # 多尺度整合层 - 标量
        self.integration_scalar = nn.Linear(hidden_channels, hidden_channels)
        
        # 多尺度整合层 - 向量
        self.integration_vector = nn.Linear(hidden_channels, hidden_channels, bias=False)
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, z=None):
        batch_size = scalar_short.size(0)
        
        # 组合短程和长程特征
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)
        vector_combined = torch.cat([vector_short, vector_long], dim=-1)
        
        # 1. 主专家选择 (长程/短程)
        primary_gates = self.primary_gate(scalar_combined)  # [batch_size, primary_experts]
        
        # 存储用于分析的专家权重
        expert_weights = torch.zeros(
            batch_size, self.primary_experts, self.secondary_experts, 
            device=scalar_combined.device
        )
        
        # 2. 标量特征的多尺度处理
        scalar_outputs = []
        
        for p in range(self.primary_experts):
            # 获取该主专家的次级专家权重
            secondary_gates = self.secondary_gates[p](scalar_combined)  # [batch_size, secondary_experts]
            
            # 存储专家权重用于分析
            expert_weights[:, p, :] = secondary_gates
            
            # 每个主专家下的次级专家输出
            p_outputs = []
            for s in range(self.secondary_experts):
                # 应用次级专家网络
                expert_out = self.secondary_experts_scalar[p][s](scalar_combined)
                
                # 计算主次级组合权重
                combined_weight = primary_gates[:, p:p+1] * secondary_gates[:, s:s+1]
                
                # 加权输出
                weighted_out = expert_out * combined_weight
                p_outputs.append(weighted_out)
            
            # 合并该主专家下的所有次级专家输出
            scalar_outputs.extend(p_outputs)
        
        # 3. 向量特征的多尺度处理
        vector_outputs = []
        
        for p in range(self.primary_experts):
            # 使用与标量相同的次级专家权重
            secondary_gates = self.secondary_gates[p](scalar_combined)
            
            # 每个主专家下的次级专家输出
            p_outputs = []
            for s in range(self.secondary_experts):
                # 应用次级专家网络
                expert_out = self.secondary_experts_vector[p][s](vector_combined)
                
                # 计算组合权重并扩展维度以匹配向量
                combined_weight = primary_gates[:, p:p+1] * secondary_gates[:, s:s+1]
                expanded_weight = combined_weight.unsqueeze(1)
                
                # 加权输出
                weighted_out = expert_out * expanded_weight
                p_outputs.append(weighted_out)
            
            # 合并该主专家下的所有次级专家输出
            vector_outputs.extend(p_outputs)
        
        # 4. 合并所有专家输出
        scalar_result = sum(scalar_outputs)
        vector_result = sum(vector_outputs)
        
        # 5. 通过多尺度整合层
        scalar_result = self.integration_scalar(scalar_result)
        vector_result = self.integration_vector(vector_result)
        
        # 6. 计算多尺度MOE损失
        
        # 专家使用频率 (扁平化查看主次级专家组合的频率)
        flat_weights = torch.zeros(
            batch_size, self.primary_experts * self.secondary_experts,
            device=scalar_combined.device
        )
        
        for p in range(self.primary_experts):
            for s in range(self.secondary_experts):
                idx = p * self.secondary_experts + s
                flat_weights[:, idx] = primary_gates[:, p] * expert_weights[:, p, s]
        
        # 计算专家使用分布
        expert_usage = flat_weights.mean(0)
        
        # 多尺度均衡损失 - 轻微促进专家均衡使用
        balance_factor = 0.05
        ideal_usage = torch.ones_like(expert_usage) / (self.primary_experts * self.secondary_experts)
        moe_loss = F.kl_div(
            (expert_usage + 1e-10).log(), ideal_usage, reduction='batchmean'
        ) * balance_factor
        
        # 7. 返回结果和多尺度专家权重（用于分析）
        return scalar_result, vector_result, moe_loss 