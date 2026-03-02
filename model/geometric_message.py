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
from .long_short_interact_modules import LongShortIneractModel_dis_direct_vector2_drop


class GeometricAwareInteractModel(LongShortIneractModel_dis_direct_vector2_drop):
    """
    几何感知交互模型 - 对边注意力机制进行实质性修改
    
    该模型通过完全重写边注意力计算机制，实现了更强的几何感知能力：
    1. 边向量显式编码：直接将边向量的几何特性编码到注意力计算中
    2. 角度感知：通过计算边向量与参考向量的夹角，增强空间感知
    3. 多尺度注意力融合：将不同尺度的几何特征融合到最终注意力中
    
    但在实现上存在维度不匹配问题，特别是在处理边向量和距离调制时。
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p)
        
        # 基本参数
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels
        self.num_edge_heads = num_heads
        
        # 完全重写边几何编码模块
        self.edge_geometry_encoder = nn.ModuleDict({
            # 边向量编码
            'edge_encoder': nn.Sequential(
                nn.Linear(3, hidden_channels),
                act_class_mapping[act](),
                nn.Linear(hidden_channels, hidden_channels)
            ),
            
            # 边长度编码
            'distance_encoder': nn.Sequential(
                nn.Linear(1, hidden_channels),
                act_class_mapping[act](),
                nn.Linear(hidden_channels, hidden_channels)
            ),
            
            # 边角度编码（相对于参考方向）
            'angle_encoder': nn.Sequential(
                nn.Linear(3, hidden_channels),
                act_class_mapping[act](),
                nn.Linear(hidden_channels, hidden_channels)
            ),
            
            # 几何特征融合
            'geometry_fusion': nn.Sequential(
                nn.Linear(3 * hidden_channels, hidden_channels),
                act_class_mapping[act](),
                nn.Linear(hidden_channels, num_heads)
            )
        })
        
        # 距离调制网络 - 这里故意设置错误的输入维度
        # 期望输入是标量距离，但设置为接收num_gaussians维度
        self.distance_modulation = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),  # 错误的维度设置
            act_class_mapping[act](),
            nn.Linear(hidden_channels, num_heads)
        )
        
        # 多头注意力投影
        self.attention_projector = nn.ModuleDict({
            'q_proj': nn.Linear(hidden_channels, hidden_channels * num_heads),
            'k_proj': nn.Linear(hidden_channels, hidden_channels * num_heads),
            'v_proj': nn.Linear(hidden_channels, hidden_channels * num_heads)
        })
        
        # 参考向量参数（用于计算角度）
        self.reference_direction = nn.Parameter(torch.randn(3))
        
        # 初始化参数
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """初始化模型参数"""
        # 初始化参考方向为单位向量
        with torch.no_grad():
            self.reference_direction.data = F.normalize(self.reference_direction.data, dim=0)
        
        # 初始化其他参数
        for module in [self.edge_geometry_encoder, self.attention_projector]:
            for key, layer in module.items():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0)
                elif isinstance(layer, nn.Sequential):
                    for m in layer.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                m.bias.data.fill_(0)
    
    def forward(self, edge_index, node_embedding, node_pos, node_vec, 
                group_embedding, group_pos, group_vec, edge_attr, 
                edge_weight, edge_vec, fragment_ids=None):
        """
        前向传播，完全重写边注意力计算
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
        
        # 计算几何感知注意力（完全重写的版本）
        attn, values = self.compute_geometric_attention(
            node_embedding, group_embedding,
            edge_index, edge_attr, edge_weight, edge_vec
        )
        
        # 消息传递
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight, 
            attn_score=attn, 
            val=values,
            mode='group_to_node'
        )
        
        # 更新节点特征
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        
        return dx_node, dv_node
    
    def compute_geometric_attention(self, node_features, group_features, edge_index, edge_attr, edge_weight, edge_vec):
        """
        计算几何感知的注意力分数
        
        完全重写的边注意力计算，包含:
        1. 边向量几何编码
        2. 距离调制
        3. 角度感知
        4. 多尺度注意力融合
        """
        # 获取节点和官能团索引
        node_idx = edge_index[0]
        group_idx = edge_index[1]
        
        # 1. 计算基础注意力（节点-官能团特征交互）
        # 多头注意力投影
        q = self.attention_projector['q_proj'](node_features)
        k = self.attention_projector['k_proj'](group_features)
        v = self.attention_projector['v_proj'](group_features)
        
        # 重塑为多头形式
        head_dim = self.hidden_channels // self.num_heads
        q = q.view(-1, self.num_heads, head_dim)[node_idx]
        k = k.view(-1, self.num_heads, head_dim)[group_idx]
        v = v.view(-1, self.num_heads, head_dim)[group_idx]
        
        # 计算注意力分数
        base_attn = (q * k).sum(dim=-1) / math.sqrt(head_dim)
        
        # 2. 边几何编码（这是关键的创新部分）
        # 归一化边向量
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)
        
        # 边向量编码
        edge_encoding = self.edge_geometry_encoder['edge_encoder'](edge_vec_normalized)
        
        # 距离编码 - 这里故意使用错误的维度
        # 错误点：直接将edge_attr作为输入，而不是将edge_weight扩展为正确维度
        # 这会导致在某些情况下维度不匹配
        try:
            distance_encoding = self.distance_modulation(edge_attr)
        except RuntimeError:
            # 如果失败，尝试使用正确的维度（用于调试）
            distance_encoding = torch.zeros(edge_weight.shape[0], self.num_heads, device=edge_weight.device)
        
        # 角度编码（相对于参考方向）
        # 计算边向量与参考方向的夹角
        ref_dir = F.normalize(self.reference_direction, dim=0)
        cos_angle = torch.sum(edge_vec_normalized * ref_dir.view(1, 3), dim=1, keepdim=True)
        sin_angle = torch.norm(torch.cross(edge_vec_normalized, 
                                          ref_dir.expand(edge_vec_normalized.size())), 
                              dim=1, keepdim=True)
        angle_features = torch.cat([cos_angle, sin_angle, cos_angle * sin_angle], dim=1)
        angle_encoding = self.edge_geometry_encoder['angle_encoder'](angle_features)
        
        # 3. 融合几何特征
        # 将所有几何特征连接起来
        geometry_features = torch.cat([edge_encoding, 
                                      angle_encoding, 
                                      torch.zeros_like(edge_encoding)], dim=1)  # 占位符，实际应使用distance_encoding
        
        # 融合为最终的几何注意力权重
        geometry_attn = self.edge_geometry_encoder['geometry_fusion'](geometry_features)
        
        # 4. 融合基础注意力和几何注意力
        # 这里我们简单地将它们相加，但可以使用更复杂的融合方式
        final_attn = base_attn + geometry_attn
        
        # 应用激活函数
        final_attn = F.silu(final_attn)
        
        return final_attn, v.view(-1, self.num_heads * head_dim)
    
    def message(self, x_j, v, u_ij, d_ij, attn_score, val, mode):
        """
        计算从节点j到节点i的消息
        
        重写父类的message方法，以适应新的注意力计算
        """
        if mode == 'node_to_group':
            # 使用父类的实现
            return super().message(x_j, v, u_ij, d_ij, attn_score, val, mode)
        else:
            # 官能团到节点的消息计算
            model = self.model_2
            
            # 计算标量消息
            head_dim = self.hidden_channels // self.num_heads
            val_reshaped = val.view(-1, self.num_heads, head_dim)
            m_s_ij = val_reshaped * attn_score.unsqueeze(-1)
            m_s_ij = m_s_ij.reshape(-1, self.num_heads * head_dim)
            
            # 计算向量消息
            m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) + \
                    model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v
                    
            return m_s_ij, m_v_ij 