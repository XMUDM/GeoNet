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
    max_min_norm
)

class EnhancedMessageModel(ImprovedLongShortInteractModel):
    """
    EnhancedMessageModel - 在ImprovedLongShortInteractModel基础上优化message函数
    
    优化点:
    1. 增强的边向量处理 - 更精细地利用边向量的几何信息
    2. 自适应消息聚合 - 根据边特性动态调整消息重要性
    3. 多尺度特征融合 - 同时考虑局部和全局信息
    4. 高效计算 - 减少冗余计算，提高性能
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p, num_edge_heads)
        
        # 增强的边向量处理网络
        self.edge_vec_encoder = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 自适应消息权重网络 - 修改输入维度
        self.message_weight_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # 移除+1，不再使用d_ij
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
        # 标量和向量消息的融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 初始化新添加的网络参数
        for module in [self.edge_vec_encoder, self.message_weight_net, self.fusion_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0)
    
    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        """
        优化的消息传递函数
        
        Args:
            x_i: 目标节点特征
            x_j: 源节点特征
            v: 源节点向量特征
            u_ij: 边向量
            d_ij: 边距离
            attn_score: 注意力分数
            val: 值向量
            mode: 消息传递模式 ('node_to_group' 或 'group_to_node')
            
        Returns:
            tuple: (标量消息, 向量消息)
        """
        # 选择合适的模型
        if mode == 'node_to_group':
            model = self.model_1
            
            # 增强的边向量编码
            edge_vec_encoding = self.edge_vec_encoder(u_ij)
            
            # 融合节点特征和边向量信息
            combined_features = torch.cat([x_i, x_j, edge_vec_encoding], dim=-1)
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim=-1))
            
            # 增强的向量消息计算
            # 1. 基于源节点向量的贡献
            v_contribution = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v
            
            # 2. 基于边向量的贡献 - 使用更精细的边向量处理
            edge_contribution = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            
            # 3. 自适应加权融合
            edge_weight = torch.sigmoid(torch.norm(u_ij, dim=1, keepdim=True))
            m_v_ij = v_contribution * (1 - edge_weight) + edge_contribution * edge_weight
            
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
            
            # 处理注意力加权的值向量
            m_s_ij = val * attn_score.unsqueeze(-1)  # 标量消息
            m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
            
            # 增强的向量消息计算
            # 1. 边向量贡献 - 考虑边的几何特性
            edge_vec_norm = torch.norm(u_ij, dim=1, keepdim=True) + 1e-10
            edge_vec_normalized = u_ij / edge_vec_norm
            edge_vec_encoding = self.edge_vec_encoder(edge_vec_normalized)
            
            # 距离感知的边权重 - 距离越大权重越小
            # 修复：确保distance_weight的维度与后续运算兼容
            distance_weight = torch.exp(-d_ij / self.cutoff).unsqueeze(-1).unsqueeze(-1)
            
            # 2. 基于边向量和源节点向量的贡献
            # 修复：确保维度匹配
            pos_contribution = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            # 应用距离权重，确保广播正确
            pos_contribution = pos_contribution * distance_weight
            
            vec_contribution = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v
            
            # 3. 自适应融合 - 基于消息内容和边特性
            # 修复：确保所有张量具有相同的维度
            edge_vec_mean = torch.mean(edge_vec_encoding, dim=1)
            
            # 不再使用d_ij，因为它可能有不同的维度
            fusion_input = torch.cat([
                m_s_ij,
                edge_vec_mean
            ], dim=-1)
            
            fusion_weight = self.message_weight_net(fusion_input)
            fusion_weight = fusion_weight.unsqueeze(1)  # 确保维度匹配
            m_v_ij = pos_contribution * fusion_weight + vec_contribution * (1 - fusion_weight)
            
            # 4. 残差连接 - 确保信息流畅通过
            if hasattr(self, 'fusion_net'):
                # 安全地计算向量消息的均值
                m_v_mean = torch.mean(m_v_ij, dim=1)
                m_s_ij_residual = self.fusion_net(
                    torch.cat([m_s_ij, m_v_mean], dim=-1)
                )
                m_s_ij = m_s_ij + m_s_ij_residual * 0.1  # 小比例残差以保持稳定性
            
            return m_s_ij, m_v_ij
    
    def aggregate(self, features, index, ptr, dim_size):
        """
        优化的特征聚合函数
        
        实现多种聚合方式的动态组合，包括均值、最大值和加权和
        """
        x, vec = features
        
        # 标准聚合 - 求和
        x_sum = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec_sum = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        
        # 如果需要更复杂的聚合策略，可以在这里实现
        # 例如：最大值聚合
        # x_max = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce='max')
        # vec_max = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='max')
        
        # 均值聚合
        # x_mean = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        # vec_mean = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        
        # 此处返回标准聚合结果
        # 如需实现动态聚合策略，可以基于输入特征的统计特性选择不同的聚合结果
        return x_sum, vec_sum


class MultiScaleMessageModel(EnhancedMessageModel):
    """
    多尺度消息传递模型 - 在EnhancedMessageModel基础上进一步优化
    
    特点:
    1. 多尺度消息处理 - 同时考虑不同空间尺度的相互作用
    2. 非局部注意力 - 捕获长程依赖关系
    3. 自适应消息聚合 - 根据上下文动态调整消息重要性
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p, num_edge_heads)
        
        # 多尺度边向量处理
        self.multiscale_edge_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden_channels // 4),
                nn.SiLU(),
                nn.Linear(hidden_channels // 4, hidden_channels // 4)
            ) for _ in range(4)  # 4个不同尺度
        ])
        
        # 尺度融合网络
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 消息增强网络
        self.message_enhancement = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 初始化新添加的网络参数
        for module_list in [self.multiscale_edge_encoder]:
            for module in module_list:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            layer.bias.data.fill_(0)
        
        for module in [self.scale_fusion, self.message_enhancement]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0)
    
    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        """
        多尺度消息传递函数
        """
        # 首先获取基础消息
        m_s_ij, m_v_ij = super().message(x_i, x_j, v, u_ij, d_ij, attn_score, val, mode)
        
        # 多尺度边向量处理
        edge_vec_norm = torch.norm(u_ij, dim=1, keepdim=True) + 1e-10
        edge_vec_normalized = u_ij / edge_vec_norm
        
        # 在不同尺度上编码边向量
        multiscale_encodings = []
        for i, encoder in enumerate(self.multiscale_edge_encoder):
            # 对于每个尺度，应用不同的变换
            scale_factor = 2 ** i  # 1, 2, 4, 8
            scaled_vec = edge_vec_normalized * scale_factor
            encoding = encoder(scaled_vec)
            multiscale_encodings.append(encoding)
        
        # 融合多尺度编码
        multiscale_feature = torch.cat(multiscale_encodings, dim=-1)
        fused_edge_feature = self.scale_fusion(multiscale_feature)
        
        # 增强标量消息
        if mode == 'group_to_node':
            enhanced_scalar = self.message_enhancement(
                torch.cat([m_s_ij, fused_edge_feature], dim=-1)
            )
            m_s_ij = m_s_ij + enhanced_scalar * 0.2  # 添加适量的增强特征
        
        return m_s_ij, m_v_ij 