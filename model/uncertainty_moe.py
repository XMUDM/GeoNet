import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyAwareMoE(nn.Module):
    """不确定性感知自校准MOE模型
    
    每个专家同时输出预测和不确定性估计，系统根据不确定性
    动态调整专家权重，降低高不确定性专家的影响，实现自校准。
    适用于处理复杂或罕见的原子环境，如相变区、界面、缺陷等。
    """
    def __init__(self, hidden_channels, num_experts=2, dropout=0.1, uncertainty_weight=1.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_experts = num_experts
        self.uncertainty_weight = uncertainty_weight
        
        # 基础门控网络 - 生成初始专家权重
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_channels * 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 专家网络 - 标量预测与不确定性估计
        self.scalar_experts = nn.ModuleList([
            nn.ModuleDict({
                'prediction': nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels, hidden_channels)
                ),
                'uncertainty': nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels // 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels // 2, hidden_channels),
                    nn.Softplus()  # 确保不确定性为正值
                )
            }) for _ in range(num_experts)
        ])
        
        # 专家网络 - 向量预测与不确定性估计
        self.vector_experts = nn.ModuleList([
            nn.ModuleDict({
                'prediction': nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels, hidden_channels, bias=False)
                ),
                'uncertainty': nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels // 2, bias=False),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels // 2, hidden_channels, bias=False),
                    nn.Softplus()  # 确保不确定性为正值
                )
            }) for _ in range(num_experts)
        ])
        
        # 不确定性感知的校准层 - 标量
        self.calibration_scalar = nn.Linear(hidden_channels, hidden_channels)
        
        # 不确定性感知的校准层 - 向量
        self.calibration_vector = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
        # 全局不确定性估计器 - 估计最终预测的总体不确定性
        self.global_uncertainty = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # 特征分布分析器 - 检测输入是否为OOD (out-of-distribution)
        self.ood_detector = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出OOD概率(0-1)
        )
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, z=None):
        batch_size = scalar_short.size(0)
        
        # 组合短程和长程特征
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)
        vector_combined = torch.cat([vector_short, vector_long], dim=-1)
        
        # 1. 检测输入是否为分布外(OOD)数据
        ood_score = self.ood_detector(scalar_combined)  # [batch_size, 1]
        
        # 2. 基础专家权重
        initial_gates = self.gate_network(scalar_combined)  # [batch_size, num_experts]
        
        # 3. 标量特征：专家预测和不确定性估计
        scalar_outputs = []
        scalar_uncertainties = []
        
        for i in range(self.num_experts):
            # 专家预测
            expert_out = self.scalar_experts[i]['prediction'](scalar_combined)
            scalar_outputs.append(expert_out)
            
            # 专家不确定性估计
            uncertainty = self.scalar_experts[i]['uncertainty'](scalar_combined)
            # 对OOD样本增加不确定性
            uncertainty = uncertainty * (1.0 + ood_score)
            scalar_uncertainties.append(uncertainty)
        
        # 4. 向量特征：专家预测和不确定性估计
        vector_outputs = []
        vector_uncertainties = []
        
        for i in range(self.num_experts):
            # 专家预测
            expert_out = self.vector_experts[i]['prediction'](vector_combined)
            vector_outputs.append(expert_out)
            
            # 专家不确定性估计
            uncertainty = self.vector_experts[i]['uncertainty'](vector_combined)
            # 对OOD样本增加不确定性
            uncertainty = uncertainty * (1.0 + ood_score.unsqueeze(1))
            vector_uncertainties.append(uncertainty)
        
        # 5. 基于不确定性校准专家权重
        # 计算每个专家的平均不确定性
        scalar_mean_uncertainties = [u.mean(dim=1, keepdim=True) for u in scalar_uncertainties]
        
        # 不确定性反比重新加权 (不确定性越低，权重越高)
        adjusted_gates = []
        for i in range(self.num_experts):
            # 计算置信度 (不确定性的倒数)
            confidence = 1.0 / (scalar_mean_uncertainties[i] + 1e-6)
            # 应用不确定性权重因子
            confidence = confidence ** self.uncertainty_weight
            # 调整门控权重
            adjusted_weight = initial_gates[:, i:i+1] * confidence
            adjusted_gates.append(adjusted_weight)
        
        # 归一化调整后的权重
        sum_adjusted = sum(adjusted_gates)
        normalized_gates = [w / (sum_adjusted + 1e-6) for w in adjusted_gates]
        
        # 6. 应用校准后的权重
        # 标量特征
        weighted_scalar_outputs = [
            out * gate for out, gate in zip(scalar_outputs, normalized_gates)
        ]
        combined_scalar = sum(weighted_scalar_outputs)
        
        # 向量特征 (扩展门控维度)
        weighted_vector_outputs = [
            out * gate.unsqueeze(1) for out, gate in zip(vector_outputs, normalized_gates)
        ]
        combined_vector = sum(weighted_vector_outputs)
        
        # 7. 通过校准层
        scalar_result = self.calibration_scalar(combined_scalar)
        vector_result = self.calibration_vector(combined_vector)
        
        # 8. 估计全局不确定性
        global_uncertainty = self.global_uncertainty(scalar_result)
        
        # 9. 计算不确定性感知MOE损失
        # 各专家不确定性的多样性损失 - 鼓励专家在不同区域具有不同的确定性
        uncertainty_stacked = torch.stack(scalar_mean_uncertainties, dim=1).squeeze(-1)  # [batch_size, num_experts]
        
        # 计算每个专家的平均不确定性
        avg_uncertainties = uncertainty_stacked.mean(0)  # [num_experts]
        
        # 差异性损失 - 鼓励不同专家在不确定性上有差异
        diversity_loss = -torch.var(avg_uncertainties) * 0.1
        
        # OOD检测损失 - 使OOD检测更加敏感
        ood_loss = -torch.mean(ood_score * uncertainty_stacked.mean(1)) * 0.1
        
        # 专家使用均衡 - 轻微鼓励均衡使用
        gates_stacked = torch.stack([g.squeeze(1) for g in normalized_gates], dim=1)  # [batch_size, num_experts]
        usage_loss = F.mse_loss(
            gates_stacked.mean(0), torch.ones_like(gates_stacked[0]) / self.num_experts
        ) * 0.05
        
        # 总MOE损失
        moe_loss = diversity_loss + ood_loss + usage_loss
        
        return scalar_result, vector_result, moe_loss 