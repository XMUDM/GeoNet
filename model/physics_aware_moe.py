class DualGatingMoE(nn.Module):
    """修复版双门控混合专家模型
    
    为标量特征和向量特征分别实现独立的门控机制，
    并解决维度不匹配问题。
    同时保留物理环境分析功能。
    """
    def __init__(self, hidden_channels, num_experts=2, dropout=0.1, use_physics_context=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_experts = num_experts
        self.use_physics_context = use_physics_context
        
        # 标量门控 - 保持简单的结构
        self.scalar_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_experts)
        )
        
        # 向量门控 - 保持与标量门控相同的维度
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_experts)
        )
        
        # 仅当启用物理上下文时创建这些模块
        if use_physics_context:
            # 标量特征物理环境分析器 - 简化版本
            self.scalar_physics_analyzer = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
            
            # 向量特征物理环境分析器 - 简化版本
            self.vector_physics_analyzer = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
            
            # 标量特征调制器 - 更稳定的设计
            self.scalar_modulator = nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, num_experts),
                nn.Tanh()  # 输出-1到1之间的调制系数
            )
            
            # 向量特征调制器 - 更稳定的设计
            self.vector_modulator = nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, num_experts),
                nn.Tanh()  # 输出-1到1之间的调制系数
            )
        
        # 标量特征专家
        self.scalar_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels)
            ) for _ in range(num_experts)
        ])
        
        # 向量特征专家 - 确保输入输出维度一致
        self.vector_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            ) for _ in range(num_experts)
        ])
        
        print("[INFO] 使用带物理环境分析的维度修复版双门控MoE")
    
    def compute_gates(self, features, gate_network, physics_analyzer=None, modulator=None):
        """计算门控权重，可选地包含物理环境分析"""
        # 基础门控logits
        base_logits = gate_network(features)
        
        # 如果启用物理上下文且提供了分析器和调制器
        if self.use_physics_context and physics_analyzer is not None and modulator is not None:
            # 物理环境分析
            physics_ctx = physics_analyzer(features)
            
            # 基于物理环境生成调制系数
            mod_coeff = modulator(physics_ctx)
            
            # 调制基础logits (使用小系数限制调制影响，提高稳定性)
            adjusted_logits = base_logits + mod_coeff * 0.1
        else:
            adjusted_logits = base_logits
            physics_ctx = None
        
        # 应用softmax获取最终门控权重
        gates = F.softmax(adjusted_logits, dim=-1)
        
        return gates
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, z=None):
        batch_size = scalar_short.size(0)
        
        # 合并短程和长程特征
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)  # [batch, hidden*2]
        vector_combined = torch.cat([vector_short, vector_long], dim=-1)  # [batch, 3, hidden*2]
        
        # 为门控计算池化向量特征
        vector_pooled = torch.mean(vector_combined, dim=1)  # [batch, hidden*2]
        
        # 计算标量特征的门控权重
        scalar_gates = self.compute_gates(
            scalar_combined, 
            self.scalar_gate,
            self.scalar_physics_analyzer if self.use_physics_context else None,
            self.scalar_modulator if self.use_physics_context else None
        )
        
        # 计算向量特征的门控权重
        vector_gates = self.compute_gates(
            vector_pooled, 
            self.vector_gate,
            self.vector_physics_analyzer if self.use_physics_context else None,
            self.vector_modulator if self.use_physics_context else None
        )
        
        # 应用专家网络 - 标量特征
        scalar_outputs = []
        for i in range(self.num_experts):
            scalar_out = self.scalar_experts[i](scalar_combined)  # [batch, hidden]
            scalar_outputs.append(scalar_out * scalar_gates[:, i:i+1])  # [batch, hidden]
        
        # 应用专家网络 - 向量特征
        vector_outputs = []
        for i in range(self.num_experts):
            vector_out = self.vector_experts[i](vector_combined)  # [batch, 3, hidden]
            # 注意这里的维度扩展方式，确保与向量维度匹配
            gate = vector_gates[:, i:i+1].unsqueeze(1)  # [batch, 1, 1]
            vector_outputs.append(vector_out * gate)  # [batch, 3, hidden]
        
        # 组合专家输出
        scalar_result = sum(scalar_outputs)  # [batch, hidden]
        vector_result = sum(vector_outputs)  # [batch, 3, hidden]
        
        return scalar_result, vector_result
    
    def get_physics_analysis(self, scalar_short, scalar_long, vector_short, vector_long):
        """获取物理环境分析结果，用于可视化和分析"""
        if not self.use_physics_context:
            return None
            
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)
        vector_pooled = torch.mean(torch.cat([vector_short, vector_long], dim=-1), dim=1)
        
        scalar_physics_ctx = self.scalar_physics_analyzer(scalar_combined)
        vector_physics_ctx = self.vector_physics_analyzer(vector_pooled)
        
        # 计算标量门控
        scalar_gates, _ = self.compute_gates(
            scalar_combined, 
            self.scalar_gate,
            self.scalar_physics_analyzer,
            self.scalar_modulator
        )
        
        # 计算向量门控
        vector_gates, _ = self.compute_gates(
            vector_pooled, 
            self.vector_gate,
            self.vector_physics_analyzer,
            self.vector_modulator
        )
        
        return {
            'scalar_physics_context': scalar_physics_ctx,
            'vector_physics_context': vector_physics_ctx,
            'scalar_gates': scalar_gates,
            'vector_gates': vector_gates,
            'context_difference': (scalar_physics_ctx - vector_physics_ctx).abs()
        }