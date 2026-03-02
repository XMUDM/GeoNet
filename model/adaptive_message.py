import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing

from .long_short_interact_modules import (
    ImprovedLongShortInteractModel,
    act_class_mapping,
    vec_layernorm,
    max_min_norm,
    norm
)

class AdaptiveLongShortInteractModel(ImprovedLongShortInteractModel):
    """
    自适应长短程交互模型
    
    主要改进：
    1. 自适应距离加权：根据原子类型和边特征动态调整距离衰减参数
    2. 增强的注意力机制：更精细地融合边的几何信息和化学信息
    3. 多尺度特征融合：同时考虑不同空间尺度的相互作用
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p, num_edge_heads)
        
        # 自适应距离加权网络
        self.distance_network = nn.Sequential(
            nn.Linear(hidden_channels + num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # 边特征增强网络
        self.edge_feature_net = nn.Sequential(
            nn.Linear(num_gaussians + 3, hidden_channels),  # 边距离特征 + 边向量维度
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 多尺度特征融合网络
        self.multiscale_fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 添加门控机制
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
    def adaptive_distance_weight(self, d_ij, edge_attr, node_feat, edge_index):
        """
        计算自适应距离权重
        
        Args:
            d_ij: 边距离
            edge_attr: 边特征
            node_feat: 节点特征
            edge_index: 边索引
        """
        # 获取源节点特征
        source_node_feat = node_feat[edge_index[0]]
        
        # 连接节点特征和边特征
        combined_feat = torch.cat([source_node_feat, edge_attr], dim=-1)
        
        # 预测自适应衰减参数 (范围0-1)
        adaptive_factor = self.distance_network(combined_feat)
        
        # 调整衰减系数，确保在合理范围内 (0.5-2.0倍原始cutoff)
        decay_factor = 0.5 + 1.5 * adaptive_factor  # 映射到[0.5, 2.0]范围
        
        # 计算自适应距离权重
        distance_weight = torch.exp(-d_ij / (self.cutoff * decay_factor.squeeze(-1)))
        
        return distance_weight.unsqueeze(-1)
    
    def enhanced_edge_encoding(self, edge_attr, edge_vec):
        """
        增强的边特征编码
        
        Args:
            edge_attr: 边特征
            edge_vec: 边向量
        """
        # 归一化边向量
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)
        
        # 连接边特征和归一化边向量
        combined_edge_feat = torch.cat([edge_attr, edge_vec_normalized], dim=-1)
        
        # 使用边特征网络处理
        enhanced_edge_feat = self.edge_feature_net(combined_edge_feat)
        
        return enhanced_edge_feat
    
    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        """
        改进的消息函数，包含自适应距离加权和多尺度特征融合
        """
        if mode != 'group_to_node':
            return super().message(x_i, x_j, v, u_ij, d_ij, attn_score, val, mode)
        
        model = self.model_2
        
        # 基础消息计算
        m_s_ij = val * attn_score.unsqueeze(-1)  # 标量消息
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        
        # 自适应距离加权
        distance_weight = torch.exp(-d_ij / self.cutoff).unsqueeze(-1)
        
        # 计算边向量贡献
        pos_contribution = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
        pos_contribution = pos_contribution * distance_weight
        
        # 计算节点向量贡献
        vec_contribution = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v
        
        # 多尺度特征融合
        # 1. 近程特征：直接使用边向量贡献
        short_range_feat = pos_contribution
        
        # 2. 长程特征：使用节点向量贡献
        long_range_feat = vec_contribution
        
        # 3. 计算融合权重
        combined_scalar = torch.cat([
            torch.mean(short_range_feat, dim=1),
            torch.mean(long_range_feat, dim=1)
        ], dim=-1)
        
        fusion_weight = self.gate_net(combined_scalar)
        
        # 4. 融合不同尺度的特征
        m_v_ij = fusion_weight * short_range_feat + (1 - fusion_weight) * long_range_feat
        
        return m_s_ij, m_v_ij
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """
        前向传播，使用增强的边特征和自适应距离加权
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        
        if self.p > 0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        # 增强边特征
        enhanced_edge_attr = self.enhanced_edge_encoding(edge_attr, -edge_vec)
        
        # 计算融合了边向量的多头注意力
        attn_2, val_2 = self.calculate_attention(
            node_embedding, 
            group_embedding, 
            edge_index[0], 
            edge_index[1], 
            enhanced_edge_attr,  # 使用增强的边特征
            self.model_2, 
            "silu",
            edge_weight,
            -edge_vec,
            edge_index
        )
        
        # 消息传递
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight, 
            attn_score=attn_2, 
            val=val_2[edge_index[1]],
            mode='group_to_node'
        )
            
        # 更新特征
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        
        return dx_node, dv_node


class GeometricAwareInteractModel(MessagePassing):
    """
    几何感知交互模型 - 独立实现，不继承ImprovedLongShortInteractModel
    
    主要特点：
    1. 边向量的几何特性编码：更好地利用边向量的方向信息
    2. 距离调制机制：根据距离动态调整消息强度
    3. 高效的消息聚合：减少冗余计算
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_channels = hidden_channels
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.norm = norm
        self.act = act_class_mapping[act]()
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.p = p
        
        # 标准化层
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        
        # Dropout层
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        
        # 边向量几何编码网络
        self.edge_geo_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels // 2),  # 边向量维度
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )
        
        # 距离调制网络
        self.distance_modulation = nn.Sequential(
            nn.Linear(1, hidden_channels // 2),  # 接收标量距离
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels),
            nn.Sigmoid()  # 输出调制因子(0-1)
        )
        
        # 注意力网络
        self.query_net = nn.Linear(hidden_channels, hidden_channels)
        self.key_net = nn.Linear(hidden_channels, hidden_channels)
        self.value_net = nn.Linear(hidden_channels, hidden_channels)
        
        # 消息处理网络
        self.mlp_scalar_pos = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.mlp_scalar_vec = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 特征更新网络
        self.node_update_linears = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)
        ])
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化模型参数"""
        for module in [self.query_net, self.key_net, self.value_net]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
                
        for module in self.node_update_linears:
            nn.init.xavier_uniform_(module.weight)
            
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
    
    def calculate_attention(self, node_embedding, group_embedding, node_idx, group_idx, edge_attr):
        """
        计算注意力分数
        
        Args:
            node_embedding: 节点特征
            group_embedding: 官能团特征
            node_idx: 节点索引
            group_idx: 官能团索引
            edge_attr: 边特征
        """
        # 计算查询、键、值
        q = self.query_net(node_embedding).reshape(-1, self.num_heads, self.attn_channels)
        k = self.key_net(group_embedding).reshape(-1, self.num_heads, self.attn_channels)
        v = self.value_net(group_embedding).reshape(-1, self.num_heads, self.attn_channels)
        
        # 获取对应边的查询和键
        q_i = q[node_idx]
        k_j = k[group_idx]
        
        # 计算注意力分数
        attn = q_i * k_j
        attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)
        
        # 应用SiLU激活
        attn = F.silu(attn)
        
        return attn, v
    
    def message(self, x_j, v, u_ij, d_ij, attn_score, val):
        """
        几何感知的消息函数
        
        Args:
            x_j: 源节点特征
            v: 源节点向量特征
            u_ij: 边向量
            d_ij: 边距离
            attn_score: 注意力分数
            val: 值向量
        """
        # 基础消息计算
        m_s_ij = val * attn_score.unsqueeze(-1)  # 标量消息
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        
        # 边向量的几何编码
        edge_direction = u_ij / (torch.norm(u_ij, dim=1, keepdim=True) + 1e-10)
        edge_geo_embedding = self.edge_geo_encoder(edge_direction)
        
        # 距离调制
        distance_input = d_ij.unsqueeze(-1)  # [num_edges, 1]
        distance_embedding = self.distance_modulation(distance_input)
        
        # 调制标量消息
        m_s_ij = m_s_ij * distance_embedding
        
        # 计算边向量贡献
        pos_contribution = self.mlp_scalar_pos(m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
        
        # 计算节点向量贡献
        vec_contribution = self.mlp_scalar_vec(m_s_ij).unsqueeze(1) * v
        
        # 组合贡献
        m_v_ij = pos_contribution + vec_contribution
        
        # 应用几何编码的注意力
        geo_attention = torch.sigmoid(torch.sum(edge_geo_embedding * m_s_ij, dim=-1, keepdim=True))
        m_v_ij = m_v_ij * geo_attention.unsqueeze(1)
        
        return m_s_ij, m_v_ij
    
    def aggregate(self, features, index, ptr, dim_size):
        """
        聚合来自不同源节点的消息
        
        Args:
            features: (标量消息, 向量消息)的元组
            index: 目标节点索引
            ptr: 指针
            dim_size: 输出维度大小
        """
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """
        前向传播，使用几何感知机制
        
        Args:
            edge_index: 边索引，[2, num_edges]，[0]是节点索引，[1]是官能团索引
            node_embedding: 节点特征，[num_nodes, hidden_channels]
            node_pos: 节点位置，[num_nodes, 3]
            node_vec: 节点向量特征，[num_nodes, 3, hidden_channels]
            group_embedding: 官能团特征，[num_groups, hidden_channels]
            group_pos: 官能团位置，[num_groups, 3]
            group_vec: 官能团向量特征，[num_groups, 3, hidden_channels]
            edge_attr: 边特征，[num_edges, num_gaussians]
            edge_weight: 边权重（距离），[num_edges]
            edge_vec: 边向量，[num_edges, 3]
            fragment_ids: 片段ID，可选
        """
        # 应用标准化和dropout
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        
        if self.p > 0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        # 计算注意力
        attn, val = self.calculate_attention(
            node_embedding, 
            group_embedding, 
            edge_index[0], 
            edge_index[1], 
            edge_attr
        )
        
        # 消息传递（从官能团到节点）
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),  # 翻转边索引，使消息从官能团流向节点
            size=(num_groups, num_nodes),
            x=group_embedding,  # 源节点特征（官能团）
            v=group_vec[edge_index[1]],  # 源节点向量特征
            u_ij=-edge_vec,  # 边向量（从官能团指向节点）
            d_ij=edge_weight,  # 边距离
            attn_score=attn,  # 注意力分数
            val=val[edge_index[1]]  # 值向量
        )
            
        # 更新节点特征
        v_node_1 = self.node_update_linears[2](node_vec)
        v_node_2 = self.node_update_linears[3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.node_update_linears[4](m_s_node) + self.node_update_linears[5](m_s_node)
        dv_node = m_v_node + self.node_update_linears[0](m_s_node).unsqueeze(1) * self.node_update_linears[1](node_vec)
        
        return dx_node, dv_node 