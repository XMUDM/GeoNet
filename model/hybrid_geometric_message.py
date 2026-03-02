import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_geometric.nn.models.schnet import GaussianSmearing

from .torchmdnet.models.utils import (
    act_class_mapping,
    vec_layernorm,
    max_min_norm
)


class HybridGeometricInteractModel(MessagePassing):
    """
    混合几何交互模型
    
    该模型融合了两种不同的几何感知注意力机制：
    1. 基于边向量的显式几何编码
    2. 基于距离和角度的多尺度几何感知
    
    特点:
    - 多通道几何编码：同时考虑边向量、距离和角度信息
    - 自适应注意力融合：通过可学习的门控机制融合不同的注意力分数
    - 高效消息聚合：使用优化的消息传递和聚合机制
    - 标量-向量耦合：实现标量特征和向量特征之间的相互影响
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        
        # 基本参数
        self.hidden_channels = hidden_channels
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.norm = norm
        self.p = p
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        
        # 激活函数
        self.act = act_class_mapping[act]()
        
        # 层归一化
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        
        # Dropout层
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        
        # 基础注意力模块
        self.base_attention = nn.ModuleDict({
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'v': nn.Linear(hidden_channels, hidden_channels),
        })
        
        # 边几何编码模块
        self.edge_geometry = nn.ModuleDict({
            # 边向量编码
            'edge_encoder': nn.Sequential(
                nn.Linear(3, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),
            ),
            
            # 边向量注意力
            'edge_q': nn.Linear(hidden_channels, hidden_channels * num_heads),
            'edge_k': nn.Linear(3, hidden_channels * num_heads),
            
            # 距离编码
            'distance_encoder': nn.Sequential(
                nn.Linear(1, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, num_heads),
            ),
            
            # 角度编码
            'angle_encoder': nn.Sequential(
                nn.Linear(3, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, num_heads),
            ),
        })
        
        # 几何特征融合
        self.geometry_fusion = nn.Sequential(
            nn.Linear(hidden_channels + 2 * num_heads, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, num_heads),
        )
        
        # 注意力融合门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(2 * num_heads, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, num_heads),
            nn.Sigmoid()
        )
        
        # 消息处理模块
        self.message_module = nn.ModuleDict({
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),
            ),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),
            ),
            'linears': nn.ModuleList([
                nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)
            ])
        })
        
        # 参考向量（用于角度计算）
        self.reference_direction = nn.Parameter(torch.randn(3))
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化模型参数"""
        # 初始化参考方向为单位向量
        with torch.no_grad():
            self.reference_direction.data = F.normalize(self.reference_direction.data, dim=0)
        
        # 初始化层归一化
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        
        # 初始化线性层和序列模块
        for module_dict in [self.base_attention, self.edge_geometry, self.message_module]:
            for key, module in module_dict.items():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        module.bias.data.fill_(0)
                elif isinstance(module, nn.Sequential):
                    for m in module.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                m.bias.data.fill_(0)
                elif isinstance(module, nn.ModuleList):
                    for m in module:
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
        
        # 初始化融合模块和门控网络
        for module in [self.geometry_fusion, self.gate_net]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
    
    def forward(self, edge_index, node_embedding, node_pos, node_vec, 
                group_embedding, group_pos, group_vec, edge_attr, 
                edge_weight, edge_vec, fragment_ids=None):
        """
        前向传播
        
        Args:
            edge_index: 边索引 [2, num_edges]，[0]是节点索引，[1]是官能团索引
            node_embedding: 节点特征 [num_nodes, hidden_channels]
            node_pos: 节点位置 [num_nodes, 3]
            node_vec: 节点向量特征 [num_nodes, 3, hidden_channels]
            group_embedding: 官能团特征 [num_groups, hidden_channels]
            group_pos: 官能团位置 [num_groups, 3]
            group_vec: 官能团向量特征 [num_groups, 3, hidden_channels]
            edge_attr: 边特征 [num_edges, num_gaussians]
            edge_weight: 边权重 [num_edges]
            edge_vec: 边向量 [num_edges, 3]
            fragment_ids: 节点所属的官能团ID
        """
        # 特征归一化
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        
        # Dropout正则化
        if self.p > 0:
            node_embedding = self.dropout_s(node_embedding)
            node_vec = self.dropout_v(node_vec)
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        # 获取节点和官能团数量
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        # 计算融合的几何注意力
        attn, val = self.compute_hybrid_attention(
            node_embedding, group_embedding,
            edge_index, edge_attr, edge_weight, edge_vec
        )
        
        # 消息传递（官能团 -> 节点）
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),  # 翻转边索引，使消息从官能团流向节点
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight,
            attn_score=attn,
            val=val
        )
        
        # 更新节点特征（标量-向量耦合）
        v_node_1 = self.message_module['linears'][2](node_vec)
        v_node_2 = self.message_module['linears'][3](node_vec)
        
        # 标量更新：向量内积调制 + 残差连接
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.message_module['linears'][4](m_s_node) + \
                  self.message_module['linears'][5](m_s_node)
        
        # 向量更新：标量调制 + 向量调制
        dv_node = m_v_node + self.message_module['linears'][0](m_s_node).unsqueeze(1) * \
                 self.message_module['linears'][1](node_vec)
        
        return dx_node, dv_node
    
    def compute_hybrid_attention(self, node_features, group_features, edge_index, edge_attr, edge_weight, edge_vec):
        """
        计算混合几何注意力
        
        融合两种不同的注意力机制：
        1. 基础特征注意力：基于节点和官能团特征的交互
        2. 几何感知注意力：基于边向量、距离和角度的几何编码
        """
        # 获取节点和官能团索引
        node_idx = edge_index[0]
        group_idx = edge_index[1]
        
        # 1. 计算基础特征注意力
        q = self.base_attention['q'](node_features).reshape(-1, self.num_heads, self.attn_channels)
        k = self.base_attention['k'](group_features).reshape(-1, self.num_heads, self.attn_channels)
        v = self.base_attention['v'](group_features).reshape(-1, self.num_heads, self.attn_channels)
        
        q_i = q[node_idx]
        k_j = k[group_idx]
        v_j = v[group_idx]
        
        # 计算基础注意力分数
        base_attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.attn_channels)
        
        # 2. 计算边向量几何注意力
        # 归一化边向量
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)
        
        # 边向量注意力
        q_edge = self.edge_geometry['edge_q'](node_features).view(-1, self.num_heads, self.hidden_channels)
        q_edge_i = q_edge[node_idx]
        
        k_edge = self.edge_geometry['edge_k'](edge_vec_normalized).view(-1, self.num_heads, self.hidden_channels)
        
        edge_attn = torch.sum(q_edge_i * k_edge, dim=2) / math.sqrt(self.hidden_channels)
        
        # 3. 应用距离衰减权重
        edge_weight_attn = torch.exp(-edge_weight / self.cutoff)
        edge_weight_attn = edge_weight_attn.unsqueeze(-1).repeat(1, self.num_heads)
        
        # 对基础注意力应用距离衰减
        base_attn = base_attn * edge_weight_attn
        
        # 4. 使用门控网络融合两种注意力
        # 拼接两种注意力作为门控网络的输入
        gate_input = torch.cat([base_attn, edge_attn], dim=-1)
        
        # 计算门控值
        gate = self.gate_net(gate_input)
        
        # 使用门控值融合两种注意力
        final_attn = gate * base_attn + (1 - gate) * edge_attn
        
        # 5. 应用激活函数
        final_attn = F.silu(final_attn)
        
        # 重塑值向量
        v_reshaped = v_j.reshape(-1, self.num_heads * self.attn_channels)
        
        return final_attn, v_reshaped
    
    def message(self, x_j, v, u_ij, d_ij, attn_score, val):
        """
        计算从节点j到节点i的消息
        
        Args:
            x_j: 源节点特征
            v: 源节点向量特征
            u_ij: 边向量
            d_ij: 边权重
            attn_score: 注意力分数
            val: 值向量
        """
        # 计算标量消息
        head_dim = self.hidden_channels // self.num_heads
        val_reshaped = val.view(-1, self.num_heads, head_dim)
        m_s_ij = val_reshaped * attn_score.unsqueeze(-1)
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * head_dim)
        
        # 计算向量消息
        m_v_ij = self.message_module['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) + \
                self.message_module['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v
        
        return m_s_ij, m_v_ij
    
    def aggregate(self, features, index, ptr, dim_size):
        """
        聚合来自邻居的消息
        
        Args:
            features: 要聚合的特征 (标量消息, 向量消息)
            index: 聚合索引
            ptr: 指针索引
            dim_size: 输出维度大小
        """
        x, vec = features
        
        # 使用scatter操作聚合消息
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        
        return x, vec 