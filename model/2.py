class MixtureOfExperts(nn.Module):
    def __init__(self, hidden_channels, num_experts=4, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_experts = num_experts
        
        # 向量特征处理和聚合
        self.vector_pooling = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.SiLU()
        )
        
        # 标量特征处理
        self.scalar_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.SiLU()
        )
        
        # 标量特征的专家选择器
        self.scalar_gate = nn.Sequential(
            nn.Linear(hidden_channels // 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 向量特征的专家选择器
        # self.vector_gate = nn.Sequential(
        #     nn.Linear(hidden_channels // 2, 128),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(128, num_experts),
        #     nn.Softmax(dim=-1)
        # )
        
        # 专家网络 - 标量
        self.experts_scalar = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels)
            ) for _ in range(num_experts)
        ])
        
        # 专家网络 - 向量
        self.experts_vector = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            ) for _ in range(num_experts)
        ])
        
        # 最终投影层
        self.final_scalar = nn.Linear(hidden_channels, hidden_channels)
        self.final_vector = nn.Linear(hidden_channels, hidden_channels, bias=False)
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, z=None):
        batch_size = scalar_short.size(0)
        
        # 组合短程和长程特征
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)
        vector_combined = torch.cat([vector_short, vector_long], dim=-1)
        # 1. 特征处理和提取
        # 标量特征处理
        scalar_features = self.scalar_projection(scalar_combined)
        
        # 向量特征处理
        vector_pooled = torch.mean(vector_combined, dim=1)
        vector_features = self.vector_pooling(vector_pooled)
        # 2. 独立的专家选择
        # 标量专家权重
        scalar_expert_weights = self.scalar_gate(scalar_features)
        
        # 向量专家权重
        vector_expert_weights = self.scalar_gate(vector_features)
        
        # 3. 应用专家
        scalar_outputs = []
        vector_outputs = []
        
        for i in range(self.num_experts):
            # 获取专家输出
            scalar_out = self.experts_scalar[i](scalar_combined)
            vector_out = self.experts_vector[i](vector_combined)
            
            # 应用独立的专家权重
            scalar_outputs.append(scalar_out * scalar_expert_weights[:, i:i+1])
            vector_outputs.append(vector_out * vector_expert_weights[:, i:i+1].unsqueeze(1))
        
        # 4. 整合专家输出
        scalar_result = sum(scalar_outputs)
        vector_result = sum(vector_outputs)
        
        # 5. 最终投影
        scalar_result = self.final_scalar(scalar_result)
        vector_result = self.final_vector(vector_result)
        
        return scalar_result, vector_result