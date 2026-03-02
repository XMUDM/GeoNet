import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from lightnp.LSRM.models.long_short_interact_modules import (
    LongShortIneractModel_dis_direct_vector2_drop, 
    act_class_mapping, 
    vec_layernorm, 
    max_min_norm
)

class MultiAttentionVectorModel(LongShortIneractModel_dis_direct_vector2_drop):
    """
    基于边向量的多注意力机制模型
    
    该模型实现了一种创新的多注意力机制，将边向量信息直接融入到注意力计算中，
    而不是简单地通过线性投影调制已有的注意力分数。
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=4, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p)
        
        # 保存参数
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
        self.cutoff = cutoff
        self.num_edge_heads = num_edge_heads  # 边向量的注意力头数量
        
        # 创建边向量的多头注意力机制
        self.edge_q = nn.Linear(hidden_channels, hidden_channels * num_edge_heads)
        self.edge_k = nn.Linear(3, hidden_channels * num_edge_heads)  # 3维向量(xyz)
        
        # 注意力融合层
        self.attention_fusion = nn.Parameter(torch.ones(2))  # 用于融合两种注意力
        
        # 初始化参数
        self.reset_edge_parameters()
        
    def reset_edge_parameters(self):
        """初始化边向量注意力参数"""
        nn.init.xavier_uniform_(self.edge_q.weight)
        nn.init.zeros_(self.edge_q.bias)
        nn.init.xavier_uniform_(self.edge_k.weight)
        nn.init.zeros_(self.edge_k.bias)
        nn.init.constant_(self.attention_fusion, 0.5)

    def calculate_edge_attention(self, node_embedding, edge_vec, edge_index):
        """
        计算基于边向量的注意力
        
        Args:
            node_embedding: 节点特征 [num_nodes, hidden_channels]
            edge_vec: 边向量 [num_edges, 3]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            edge_attention: 边向量注意力 [num_edges, num_heads]
        """
        # 归一化边向量
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)
        
        # 获取源节点和目标节点的索引
        source_idx, target_idx = edge_index
        
        # 计算查询向量 (从节点特征)
        q = self.edge_q(node_embedding).view(-1, self.num_edge_heads, self.hidden_channels)  # [num_nodes, num_edge_heads, hidden_channels]
        q_i = q[source_idx]  # [num_edges, num_edge_heads, hidden_channels]
        
        # 计算键向量 (从边向量)
        k = self.edge_k(edge_vec_normalized).view(-1, self.num_edge_heads, self.hidden_channels)  # [num_edges, num_edge_heads, hidden_channels]
        
        # 计算注意力分数 (点积注意力)
        edge_attn = torch.sum(q_i * k, dim=2) / math.sqrt(self.hidden_channels)  # [num_edges, num_edge_heads]
        
        # 应用激活函数
        edge_attn = F.silu(edge_attn)
        
        # 如果num_edge_heads与num_heads不同，进行维度调整
        # if self.num_edge_heads != self.num_heads:
        #     # 使用平均池化或最大池化调整维度
        #     if self.num_edge_heads > self.num_heads:
        #         # 降维: 分组平均
        #         edge_attn = edge_attn.view(edge_attn.size(0), self.num_heads, -1).mean(dim=2)
        #     else:
        #         # 升维: 复制
        #         repeat_factor = self.num_heads // self.num_edge_heads
        #         edge_attn = edge_attn.repeat_interleave(repeat_factor, dim=1)
        #         # 如果不能整除，补充剩余的维度
        #         if self.num_heads % self.num_edge_heads != 0:
        #             remaining = self.num_heads - edge_attn.size(1)
        #             edge_attn = torch.cat([edge_attn, edge_attn[:, :remaining]], dim=1)
                    
        return edge_attn

    def calculate_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type, edge_weight=None, edge_vec=None, edge_index=None):
        """
        计算融合了边向量信息的多头注意力
        
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
            edge_index: 边索引 [2, num_edges]
        """
        __supported_attn__ = ['softmax', 'silu']
        
        # 1. 计算基础注意力 (与原模型相同)
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels)
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels)
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)  # [num_edges, num_heads]
        
        # 2. 加入边权重信息
        if edge_weight is not None:
            edge_weight_attn = torch.exp(-edge_weight / self.cutoff)
            edge_weight_attn = edge_weight_attn.unsqueeze(-1).repeat(1, self.num_heads)
            attn = attn * edge_weight_attn
        
        # 3. 计算边向量注意力 (新增的多头注意力机制)
        if edge_vec is not None and edge_index is not None:
            edge_attn = self.calculate_edge_attention(x_1, edge_vec, edge_index)
            
            # 4. 融合两种注意力
            # 使用softmax归一化融合权重
            fusion_weights = F.softmax(self.attention_fusion, dim=0)
            
            # 融合两种注意力 (加权平均)
            attn = fusion_weights[0] * attn + fusion_weights[1] * edge_attn
        
        # 5. 应用注意力激活函数
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
        前向传播，使用融合了边向量的多头注意力机制
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
        
        # 计算融合了边向量的多头注意力
        attn_2, val_2 = self.calculate_attention(
            node_embedding, 
            group_embedding, 
            edge_index[0], 
            edge_index[1], 
            edge_attr, 
            self.model_2, 
            "silu",
            edge_weight,
            edge_vec,
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


class HierarchicalAttentionVectorModel(MultiAttentionVectorModel):
    """
    层次化边向量注意力模型
    
    该模型实现了一种层次化的多头注意力机制，不仅考虑边向量信息，
    还考虑边向量在不同尺度上的特征，实现多尺度的注意力融合。
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=4, num_scales=3, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p, num_edge_heads)
        
        # 多尺度参数
        self.num_scales = num_scales
        
        # 创建多尺度边向量处理层
        self.scale_projections = nn.ModuleList([
            nn.Linear(3, hidden_channels) 
            for _ in range(num_scales)
        ])
        
        # 尺度注意力融合层
        self.scale_fusion = nn.Parameter(torch.ones(num_scales))
        
        # 初始化参数
        self.reset_scale_parameters()
        
    def reset_scale_parameters(self):
        """初始化多尺度参数"""
        for proj in self.scale_projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.constant_(self.scale_fusion, 1.0 / self.num_scales)

    def calculate_multiscale_edge_attention(self, node_embedding, edge_vec, edge_index):
        """
        计算多尺度边向量注意力
        
        Args:
            node_embedding: 节点特征 [num_nodes, hidden_channels]
            edge_vec: 边向量 [num_edges, 3]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            edge_attention: 边向量注意力 [num_edges, num_heads]
        """
        # 归一化边向量
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)
        
        # 获取源节点和目标节点的索引
        source_idx, target_idx = edge_index
        
        # 计算查询向量 (从节点特征)
        q = self.edge_q(node_embedding).view(-1, self.num_edge_heads, self.hidden_channels)
        q_i = q[source_idx]
        
        # 多尺度注意力计算
        scale_attentions = []
        
        for scale_idx, scale_proj in enumerate(self.scale_projections):
            # 在不同尺度上投影边向量
            scale_k = scale_proj(edge_vec_normalized).view(-1, 1, self.hidden_channels)
            scale_k = scale_k.expand(-1, self.num_edge_heads, -1)  # 扩展到所有注意力头
            
            # 计算该尺度的注意力分数
            scale_attn = torch.sum(q_i * scale_k, dim=2) / math.sqrt(self.hidden_channels)
            scale_attentions.append(scale_attn)
        
        # 融合不同尺度的注意力
        scale_weights = F.softmax(self.scale_fusion, dim=0)
        edge_attn = torch.zeros_like(scale_attentions[0])
        
        for i, scale_attn in enumerate(scale_attentions):
            edge_attn += scale_weights[i] * scale_attn
        
        # 应用激活函数
        edge_attn = F.silu(edge_attn)
        
        # 调整维度以匹配num_heads
        if self.num_edge_heads != self.num_heads:
            if self.num_edge_heads > self.num_heads:
                edge_attn = edge_attn.view(edge_attn.size(0), self.num_heads, -1).mean(dim=2)
            else:
                repeat_factor = self.num_heads // self.num_edge_heads
                edge_attn = edge_attn.repeat_interleave(repeat_factor, dim=1)
                if self.num_heads % self.num_edge_heads != 0:
                    remaining = self.num_heads - edge_attn.size(1)
                    edge_attn = torch.cat([edge_attn, edge_attn[:, :remaining]], dim=1)
                    
        return edge_attn

    def calculate_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type, edge_weight=None, edge_vec=None, edge_index=None):
        """
        计算层次化融合了边向量信息的多头注意力
        """
        __supported_attn__ = ['softmax', 'silu']
        
        # 1. 计算基础注意力 (与原模型相同)
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels)
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels)
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)
        
        # 2. 加入边权重信息
        if edge_weight is not None:
            edge_weight_attn = torch.exp(-edge_weight / self.cutoff)
            edge_weight_attn = edge_weight_attn.unsqueeze(-1).repeat(1, self.num_heads)
            attn = attn * edge_weight_attn
        
        # 3. 计算多尺度边向量注意力
        if edge_vec is not None and edge_index is not None:
            edge_attn = self.calculate_multiscale_edge_attention(x_1, edge_vec, edge_index)
            
            # 4. 融合两种注意力
            fusion_weights = F.softmax(self.attention_fusion, dim=0)
            attn = fusion_weights[0] * attn + fusion_weights[1] * edge_attn
        
        # 5. 应用注意力激活函数
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim=0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        
        return attn, val 