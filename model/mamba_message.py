import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax

from .long_short_interact_modules import ImprovedLongShortInteractModel
from .torchmdnet.models.utils import (
    CosineCutoff,
    act_class_mapping,
    vec_layernorm,
    max_min_norm,
    norm
)

class MambaBlock(nn.Module):
    """
    简化版的Mamba块，用于序列建模
    
    基于状态空间模型(SSM)的序列建模块，相比Transformer具有更高的效率和更好的长序列建模能力。
    这是一个简化实现，完整版本需要更复杂的S4D或Mamba实现。
    """
    def __init__(self, hidden_dim, state_dim=16, expand_factor=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.expanded_dim = hidden_dim * expand_factor
        
        # 投影层
        self.in_proj = nn.Linear(hidden_dim, self.expanded_dim)
        self.out_proj = nn.Linear(self.expanded_dim, hidden_dim)
        
        # SSM参数
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, 1))
        self.C = nn.Parameter(torch.randn(1, state_dim))
        
        # 将输入投影到状态空间
        self.in_ssm = nn.Linear(self.expanded_dim, state_dim)
        self.out_ssm = nn.Linear(state_dim, self.expanded_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        # 初始化SSM参数
        nn.init.normal_(self.A, mean=0.0, std=0.1)
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        nn.init.normal_(self.C, mean=0.0, std=0.1)
        
        # 使A具有稳定性
        self.A.data = -torch.abs(self.A.data)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            
        Returns:
            输出特征 [batch_size, seq_len, hidden_dim]
        """
        residual = x
        x = self.norm(x)
        
        # 投影到更高维度
        x = self.in_proj(x)
        x = F.silu(x)
        
        # 提取形状
        batch_size, seq_len, _ = x.shape
        
        # 将输入投影到状态空间
        u = self.in_ssm(x)  # [batch_size, seq_len, state_dim]
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.state_dim, device=x.device)
        
        # 按序列长度迭代
        outputs = []
        for t in range(seq_len):
            # 状态更新
            h = torch.matmul(h, self.A) + torch.matmul(u[:, t, :].unsqueeze(1), self.B).squeeze(1)
            # 输出计算
            y = torch.matmul(h.unsqueeze(1), self.C.t()).squeeze(1)
            outputs.append(y)
        
        # 堆叠输出
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, state_dim]
        
        # 投影回原始维度
        output = self.out_ssm(output)  # [batch_size, seq_len, expanded_dim]
        output = F.silu(output)
        output = self.dropout(output)
        output = self.out_proj(output)  # [batch_size, seq_len, hidden_dim]
        
        return output + residual


class MambaMessageModel(ImprovedLongShortInteractModel):
    """
    基于Mamba的长程力消息传递模型
    
    该模型使用状态空间模型(SSM)替代原有的注意力机制，通过Mamba架构
    更高效地处理分子中的长程相互作用。
    
    特点:
    1. 使用SSM建模序列依赖关系，更适合长程依赖
    2. 线性复杂度，更高效的计算
    3. 保留向量和标量特征的双通道处理
    4. 支持距离加权的消息传递
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p, num_edge_heads, **kwargs)
        
        # Mamba参数
        self.state_dim = 16  # 状态空间维度
        self.num_layers = 2  # Mamba层数
        
        # 距离编码器
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_channels // 4),
            nn.SiLU(),
            nn.Linear(hidden_channels // 4, hidden_channels)
        )
        
        # 方向编码器
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )
        
        # 节点到官能团的Mamba层
        self.node_to_group_mamba = nn.ModuleList([
            MambaBlock(hidden_channels, self.state_dim, expand_factor=2, dropout=p)
            for _ in range(self.num_layers)
        ])
        
        # 官能团到节点的Mamba层
        self.group_to_node_mamba = nn.ModuleList([
            MambaBlock(hidden_channels, self.state_dim, expand_factor=2, dropout=p)
            for _ in range(self.num_layers)
        ])
        
        # 向量特征转换网络
        self.vec_transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 初始化参数
        self._init_mamba_params()
    
    def _init_mamba_params(self):
        """初始化Mamba组件的参数"""
        for module in [self.distance_encoder, self.direction_encoder, self.vec_transform]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
    
    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        """
        基于Mamba的消息传递函数
        
        Args:
            x_i: 目标节点特征
            x_j: 源节点特征
            v: 源节点向量特征
            u_ij: 边向量
            d_ij: 边距离
            attn_score: 注意力分数
            val: 值向量
            mode: 传递模式
            
        Returns:
            m_s_ij: 标量消息
            m_v_ij: 向量消息
        """
        if mode == 'node_to_group':
            # 对于node_to_group模式，使用原有实现
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim=-1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
                    model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            # 对于group_to_node模式，使用Mamba处理
            model = self.model_2
            
            # 1. 基础标量消息计算
            m_s_ij_base = val * attn_score.unsqueeze(-1)  # 原始标量消息
            m_s_ij_base = m_s_ij_base.reshape(-1, self.num_heads * self.attn_channels)
            
            # 2. 融合距离和方向信息
            # 距离编码
            distance_input = d_ij.unsqueeze(-1)  # [num_edges, 1]
            distance_embedding = self.distance_encoder(distance_input)
            
            # 方向编码
            direction_embedding = self.direction_encoder(u_ij)
            
            # 融合特征
            m_s_ij = m_s_ij_base + 0.1 * distance_embedding + 0.1 * direction_embedding
            
            # 3. 向量消息计算
            pos_vec = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            feat_vec = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v
            
            # 添加距离调制
            distance_weight = torch.exp(-d_ij / self.cutoff).unsqueeze(-1).unsqueeze(-1)
            m_v_ij = distance_weight * pos_vec + (1 - distance_weight) * feat_vec
            
            return m_s_ij, m_v_ij
    
    def _apply_mamba_to_batch(self, features, mamba_layers):
        """应用Mamba层到特征批次"""
        # 添加批次维度
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        
        # 应用Mamba层
        for layer in mamba_layers:
            features = layer(features)
        
        # 移除批次维度
        if features.size(0) == 1:
            features = features.squeeze(0)
            
        return features
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """
        前向传播函数
        
        使用Mamba架构处理节点和官能团之间的交互
        
        Args:
            edge_index: 边索引
            node_embedding: 节点特征
            node_pos: 节点位置
            node_vec: 节点向量特征
            group_embedding: 官能团特征
            group_pos: 官能团位置
            group_vec: 官能团向量特征
            edge_attr: 边特征
            edge_weight: 边权重
            edge_vec: 边向量
            fragment_ids: 片段ID
            
        Returns:
            dx_node: 节点特征更新
            dv_node: 节点向量特征更新
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
        
        # 1. 应用Mamba到官能团特征
        enhanced_group_embedding = self._apply_mamba_to_batch(group_embedding, self.group_to_node_mamba)
        
        # 2. 计算注意力分数 - 保留原有的注意力计算，但使用增强的官能团特征
        attn_2, val_2 = self.calculate_attention(
            node_embedding, enhanced_group_embedding, 
            edge_index[0], edge_index[1], 
            edge_attr, self.model_2, "silu",
            edge_weight, -edge_vec, edge_index.flip(0)
        )
        
        # 3. 消息传递 - 使用我们的消息传递函数
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(enhanced_group_embedding, node_embedding), 
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec, 
            d_ij=edge_weight, 
            attn_score=attn_2, 
            val=val_2[edge_index[1]],
            mode='group_to_node'
        )
        
        # 4. 应用Mamba到节点消息
        m_s_node = self._apply_mamba_to_batch(m_s_node, self.node_to_group_mamba)
        
        # 5. 更新节点特征 - 保留原有的特征更新机制
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        
        return dx_node, dv_node 