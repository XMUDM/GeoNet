import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from lightnp.LSRM.models.long_short_interact_modules import LongShortIneractModel_dis_direct_vector2_drop, act_class_mapping, vec_layernorm, max_min_norm

class VectorEdgeAwareModel(LongShortIneractModel_dis_direct_vector2_drop):
    """
    分子模型类，在注意力计算中同时考虑边权重和边向量信息
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", num_heads=8, p=0.1, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p)
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
        self.cutoff = cutoff
        
        # 为向量特征添加额外的变换层
        self.vec_proj = nn.Linear(hidden_channels, self.num_heads)
        self.vec_importance = nn.Parameter(torch.ones(1))  # 可学习的向量重要性参数
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.zeros_(self.vec_proj.bias)
        nn.init.constant_(self.vec_importance, 0.5)  # 初始化为中等重要性

    def calculate_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type, edge_weight=None, edge_vec=None):
        """
        计算注意力值，同时考虑边权重和边向量信息
        
        Args:
            x_1: 目标节点特征
            x_2: 源节点特征
            x1_index: 目标节点索引
            x2_index: 源节点索引
            expanded_edge_weight: 扩展的边权重
            model: 模型字典
            attn_type: 注意力类型
            edge_weight: 边权重（距离）
            edge_vec: 边向量（方向）
        """
        __supported_attn__ = ['softmax', 'silu']
        
        # 1. 计算基础注意力
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels)
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels)
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        # 2. 处理边权重信息
        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        
        # 3. 计算基础注意力分数
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)
        
        # 4. 加入边权重信息
        if edge_weight is not None:
            # 将边权重转换为注意力权重
            edge_weight_attn = torch.exp(-edge_weight / self.cutoff)  # 使用指数衰减
            edge_weight_attn = edge_weight_attn.unsqueeze(-1).repeat(1, self.num_heads)
            
            # 将边权重注意力与基础注意力相乘
            attn = attn * edge_weight_attn
        
        # 5. 加入边向量信息
        if edge_vec is not None:
            # 计算边向量的方向信息
            edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
            edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)  # 归一化，避免除零
            
            # 将边向量投影到注意力头维度
            edge_vec_proj = self.vec_proj(edge_vec_normalized)  # [num_edges, num_heads]
            
            # 使用sigmoid激活，确保值在0-1之间
            edge_vec_attn = torch.sigmoid(edge_vec_proj)
            
            # 使用可学习参数调节向量特征的重要性
            vec_importance = torch.sigmoid(self.vec_importance)
            
            # 将边向量注意力与当前注意力结合
            attn = attn * (1.0 + vec_importance * (edge_vec_attn - 1.0))
        
        # 6. 应用注意力激活函数
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim=0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        
        return attn, val

    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """
        前向传播，同时考虑边权重和边向量信息
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
        
        # 计算注意力，同时加入边权重和边向量信息
        attn_2, val_2 = self.calculate_attention(
            node_embedding, 
            group_embedding, 
            edge_index[0], 
            edge_index[1], 
            edge_attr, 
            self.model_2, 
            "silu",
            edge_weight,
            edge_vec
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
