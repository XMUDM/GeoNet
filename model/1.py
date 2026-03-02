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

class EnhancedMessageModel(ImprovedLongShortInteractModel):
    """
    增强型消息传递模型
    
    该模型继承自ImprovedLongShortInteractModel，修改m_s_ij的构造方式，
    通过融入group_embedding的信息来增强消息传递。
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p, num_edge_heads, **kwargs)
        
        # 添加group特征转换网络
        self.group_feature_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 添加特征融合网络
        self.feature_fusion_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 初始化参数
        self._init_enhanced_params()
    
    def _init_enhanced_params(self):
        """初始化增强型组件的参数"""
        for module in [self.group_feature_net, self.feature_fusion_net]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
    
    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        """
        增强型消息传递函数
        
        通过融入group_embedding的信息来修改m_s_ij的构造方式
        
        Args:
            x_i: 目标节点特征
            x_j: 源节点特征（在group_to_node模式下，这是group特征）
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
            # 对于node_to_group模式，保持原有实现不变
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim=-1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
                    model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            # 对于group_to_node模式，修改m_s_ij的构造方式
            model = self.model_2
            
            # 1. 基础标量消息计算
            m_s_ij_base = val * attn_score.unsqueeze(-1)  # 原始标量消息
            m_s_ij_base = m_s_ij_base.reshape(-1, self.num_heads * self.attn_channels)
            
            # 2. 增强m_s_ij的构造 - 融入group_embedding信息
            # 注意：在group_to_node模式下，x_j就是group_embedding
            
            # 2.1 转换group特征
            group_features = self.group_feature_net(x_j)
            
            # 2.2 融合基础消息和group特征
            combined_features = torch.cat([m_s_ij_base, group_features], dim=-1)
            m_s_ij_enhanced = self.feature_fusion_net(combined_features)
            
            # 2.3 残差连接
            m_s_ij = m_s_ij_base + m_s_ij_enhanced
            
            # 3. 向量消息计算 - 与原始实现完全相同
            m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
                   + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v
            
            return m_s_ij, m_v_ij

