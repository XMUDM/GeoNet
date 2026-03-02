import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax
import math

class PhysicsInformedLongRangeModel(MessagePassing):
    """物理约束驱动的长程相互作用模型
    
    基于量子化学原理的注意力计算，包括：
    1. 轨道重叠积分
    2. 静电相互作用
    3. 范德华力
    4. 氢键相互作用
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff, 
                 norm=False, act="silu", num_heads=8, p=0.1, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.cutoff = cutoff
        self.norm = norm
        self.p = p
        
        # 物理相互作用组件
        self.orbital_overlap = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, num_heads),
            nn.Sigmoid()
        )
        
        self.electrostatic_interaction = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_channels, num_heads),
            nn.Tanh()  # 可正可负的静电相互作用
        )
        
        self.vdw_interaction = nn.Sequential(
            nn.Linear(1, hidden_channels // 4),  # distance-based
            nn.SiLU(),
            nn.Linear(hidden_channels // 4, num_heads),
            nn.Sigmoid()
        )
        
        self.hydrogen_bond = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 3, hidden_channels),  # +3 for position vectors
            nn.SiLU(),
            nn.Linear(hidden_channels, num_heads),
            nn.Sigmoid()
        )
        
        # 物理权重融合网络
        self.physics_fusion = nn.Sequential(
            nn.Linear(num_heads * 4, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, num_heads),
            nn.Softmax(dim=-1)
        )
        
        # 特征变换
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # 向量处理
        self.vector_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Dropout
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        
        # 层归一化
        if norm:
            self.layernorm_node = nn.LayerNorm(hidden_channels)
            self.layernorm_group = nn.LayerNorm(hidden_channels)
    
    def forward(self, edge_index, node_embedding, node_pos, node_vec, 
                group_embedding, group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            group_embedding = self.layernorm_group(group_embedding)
        
        if self.p > 0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        # 计算物理相互作用
        physics_attention = self.calculate_physics_attention(
            node_embedding, group_embedding, node_pos, group_pos,
            edge_index, edge_weight, edge_vec
        )
        
        # 消息传递
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight,
            physics_attn=physics_attention,
            mode='group_to_node'
        )
        
        # 最终特征更新
        v_node_1 = self.vector_proj(node_vec)
        v_node_2 = self.vector_proj(node_vec)
        
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * m_s_node + m_s_node
        dv_node = m_v_node + v_node_1
        
        return dx_node, dv_node
    
    def calculate_physics_attention(self, node_embedding, group_embedding, 
                                  node_pos, group_pos, edge_index, edge_weight, edge_vec):
        """基于物理原理计算注意力"""
        
        node_idx, group_idx = edge_index[0], edge_index[1]
        
        # 1. 轨道重叠积分 (Orbital Overlap)
        combined_features = torch.cat([
            node_embedding[node_idx], 
            group_embedding[group_idx]
        ], dim=-1)
        orbital_attn = self.orbital_overlap(combined_features)
        
        # 2. 静电相互作用 (Electrostatic Interaction)
        electro_features = torch.cat([
            combined_features,
            edge_weight.unsqueeze(-1)
        ], dim=-1)
        electro_attn = self.electrostatic_interaction(electro_features)
        
        # 3. 范德华力 (Van der Waals)
        # 使用r^-6依赖性
        vdw_distance = edge_weight.unsqueeze(-1)
        vdw_attn = self.vdw_interaction(vdw_distance)
        
        # 4. 氢键相互作用 (Hydrogen Bonding)
        # 考虑角度和距离依赖性
        position_diff = node_pos[node_idx] - group_pos[group_idx]
        hbond_features = torch.cat([
            combined_features,
            position_diff
        ], dim=-1)
        hbond_attn = self.hydrogen_bond(hbond_features)
        
        # 融合所有物理相互作用
        all_physics = torch.cat([
            orbital_attn, electro_attn, vdw_attn, hbond_attn
        ], dim=-1)
        
        physics_weights = self.physics_fusion(all_physics)
        
        return physics_weights
    
    def message(self, x_i, x_j, v, u_ij, d_ij, physics_attn, mode):
        """使用物理约束的消息传递"""
        
        # Query, Key, Value变换
        q = self.q_proj(x_i).view(-1, self.num_heads, self.attn_channels)
        k = self.k_proj(x_j).view(-1, self.num_heads, self.attn_channels)
        val = self.v_proj(x_j).view(-1, self.num_heads, self.attn_channels)
        
        # 传统注意力 + 物理约束
        classical_attn = (q * k).sum(dim=-1) / math.sqrt(self.attn_channels)
        classical_attn = torch.sigmoid(classical_attn)
        
        # 组合物理注意力和传统注意力
        combined_attn = classical_attn * physics_attn
        
        # 标量消息
        m_s_ij = (val * combined_attn.unsqueeze(-1)).view(-1, self.hidden_channels)
        
        # 向量消息（考虑物理方向性）
        m_v_ij = self.vector_proj(m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) + \
                 self.vector_proj(m_s_ij).unsqueeze(1) * v
        
        return m_s_ij, m_v_ij
    
    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class HierarchicalLongRangeModel(MessagePassing):
    """分层长程相互作用模型
    
    故事性：模拟生物系统中的多层次相互作用
    - 原子级相互作用
    - 残基级相互作用  
    - 结构域级相互作用
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff,
                 norm=False, act="silu", num_heads=8, num_layers=3, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cutoff = cutoff
        
        # 多层次相互作用模块
        self.hierarchical_layers = nn.ModuleList([
            HierarchicalLayer(hidden_channels, num_heads, scale=i+1)
            for i in range(num_layers)
        ])
        
        # 跨层融合
        self.cross_layer_fusion = nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 自适应权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, edge_index, node_embedding, node_pos, node_vec,
                group_embedding, group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        
        layer_outputs = []
        
        for i, layer in enumerate(self.hierarchical_layers):
            layer_out = layer(
                edge_index, node_embedding, group_embedding,
                edge_weight, edge_vec, scale_factor=2**i
            )
            layer_outputs.append(layer_out)
        
        # 加权融合
        weighted_sum = sum(w * out for w, out in zip(torch.softmax(self.layer_weights, dim=0), layer_outputs))
        
        # 跨层特征融合
        concatenated = torch.cat(layer_outputs, dim=-1)
        fused_features = self.cross_layer_fusion(concatenated)
        
        # 最终输出
        final_features = weighted_sum + fused_features
        
        return final_features, torch.zeros_like(node_vec)


class HierarchicalLayer(nn.Module):
    """单个分层模块"""
    
    def __init__(self, hidden_channels, num_heads, scale):
        super().__init__()
        self.scale = scale
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # 尺度特定的注意力
        self.scale_attention = nn.MultiheadAttention(
            hidden_channels, num_heads, batch_first=True
        )
        
        # 尺度特定的特征变换
        self.scale_transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
    def forward(self, edge_index, node_embedding, group_embedding, 
                edge_weight, edge_vec, scale_factor):
        
        # 根据尺度调整cutoff
        scaled_mask = edge_weight < (self.scale * scale_factor)
        filtered_edges = edge_index[:, scaled_mask]
        
        if filtered_edges.size(1) == 0:
            return torch.zeros_like(node_embedding)
        
        # 在该尺度下计算注意力
        node_features = node_embedding[filtered_edges[0]]
        group_features = group_embedding[filtered_edges[1]]
        
        # 简化的注意力计算
        combined = node_features + group_features
        transformed = self.scale_transform(combined)
        
        # 聚合回原始节点
        output = torch.zeros_like(node_embedding)
        output.index_add_(0, filtered_edges[0], transformed)
        
        return output


class FrequencyDomainLongRangeModel(MessagePassing):
    """频域长程相互作用模型
    
    故事性：利用信号处理中的频域变换来捕获长程周期性相互作用
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff,
                 norm=False, act="silu", num_heads=8, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # 频域变换网络
        self.frequency_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )
        
        # 频率滤波器
        self.frequency_filters = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(num_heads)
        ])
        
        # 逆变换网络
        self.frequency_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
    def forward(self, edge_index, node_embedding, node_pos, node_vec,
                group_embedding, group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        
        # 频域编码
        freq_node = self.frequency_encoder(node_embedding)
        freq_group = self.frequency_encoder(group_embedding)
        
        # 模拟FFT操作（简化版）
        freq_node_complex = torch.complex(freq_node, torch.zeros_like(freq_node))
        freq_group_complex = torch.complex(freq_group, torch.zeros_like(freq_group))
        
        # 频域滤波
        filtered_features = []
        for filter_net in self.frequency_filters:
            filtered = filter_net(freq_node.real)
            filtered_features.append(filtered)
        
        # 组合滤波结果
        combined_freq = torch.stack(filtered_features, dim=1).mean(dim=1)
        
        # 频域解码
        output_features = self.frequency_decoder(combined_freq)
        
        return output_features, torch.zeros_like(node_vec) 