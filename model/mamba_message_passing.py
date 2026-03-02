import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax, remove_self_loops

from .long_short_interact_modules import ImprovedLongShortInteractModel
from .torchmdnet.models.utils import (
    CosineCutoff,
    act_class_mapping,
    vec_layernorm,
    max_min_norm,
    norm
)


class SequentialSSM(nn.Module):
    """
    序列化状态空间模型(SSM) - Mamba的核心机制
    
    Mamba的核心是状态空间模型，它通过以下方式工作：
    1. 状态更新：h(t+1) = Ah(t) + Bx(t)
    2. 输出计算：y(t) = Ch(t) + Dx(t)
    3. 选择性SSM：参数是输入的函数
    
    与Transformer的注意力机制不同：
    - 注意力：全局交互，O(n²)复杂度
    - SSM：序列化处理，O(n)复杂度
    """
    def __init__(self, hidden_dim, state_dim=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # S4D/Mamba参数化
        # 投影矩阵：将输入投影到状态空间
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 状态空间参数
        # ∆: 时间步长参数
        self.log_step = nn.Parameter(torch.zeros(hidden_dim))
        
        # A, B: 状态矩阵参数 (对角化表示)
        # 在Mamba中，A是对角矩阵，表示为向量
        self.A_log = nn.Parameter(torch.randn(hidden_dim))
        # B向量
        self.B = nn.Parameter(torch.randn(hidden_dim, state_dim))
        
        # C: 输出投影
        self.C = nn.Parameter(torch.randn(state_dim, hidden_dim))
        
        # 输入依赖参数 (S6/Mamba)
        # 这些参数使模型能够基于输入调整状态空间
        self.A_proj = nn.Linear(hidden_dim, hidden_dim)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)
        self.C_proj = nn.Linear(hidden_dim, hidden_dim)
        self.D_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 门控机制
        self.input_gate = nn.Linear(hidden_dim, state_dim)
        self.forget_gate = nn.Linear(hidden_dim, state_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化参数，确保稳定性"""
        # 初始化A为负值，确保稳定性
        with torch.no_grad():
            # 初始化为-0.5到-1.5之间的均匀分布
            nn.init.uniform_(self.A_log, -1.5, -0.5)
            
            # 初始化B和C
            nn.init.normal_(self.B, 0, 0.1)
            nn.init.normal_(self.C, 0, 0.1)
            
            # 初始化投影层
            for proj in [self.in_proj, self.A_proj, self.B_proj, self.C_proj, self.D_proj, self.out_proj]:
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
    
    def forward(self, x):
        """
        前向传播 - 实现Mamba的核心状态空间模型
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim] 或 [num_nodes, hidden_dim]
            
        Returns:
            输出特征，形状与输入相同
        """
        # 处理输入维度
        orig_shape = x.shape
        if x.dim() == 2:
            # 如果是2D张量，添加序列维度
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.in_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        # 序列化处理
        for t in range(seq_len):
            x_t = x[:, t]  # [batch_size, hidden_dim]
            
            # 计算输入依赖参数 (Mamba的关键创新)
            # 使状态空间参数成为输入的函数
            A_param = -torch.exp(self.A_log + self.A_proj(x_t))  # [batch_size, hidden_dim]
            B_param = self.B_proj(x_t)  # [batch_size, hidden_dim]
            C_param = self.C_proj(x_t).unsqueeze(1)   # [batch_size, 1, hidden_dim]
            D_param = self.D_proj(x_t)  # [batch_size, hidden_dim]
            
            # 计算时间步长
            delta = torch.exp(self.log_step)  # [hidden_dim]
            
            # 离散化状态更新 (使用对角A简化计算)
            # h_next = exp(A*∆) * h + B * x
            # 对于对角A，exp(A*∆)是元素级操作
            exp_A = torch.exp(A_param.unsqueeze(1) * delta.unsqueeze(0).unsqueeze(1))  # [batch_size, hidden_dim, 1]
            
            # 门控机制 - 现在输出state_dim维度
            input_gate = torch.sigmoid(self.input_gate(x_t))  # [batch_size, state_dim]
            forget_gate = torch.sigmoid(self.forget_gate(x_t))  # [batch_size, state_dim]
            
            # 状态更新，使用门控机制
            # 将h从[batch_size, state_dim]重塑为[batch_size, 1, state_dim]
            # 确保维度匹配
            h_reshaped = h.view(batch_size, 1, self.state_dim)  # [batch_size, 1, state_dim]
            
            # 状态更新
            # 修复维度不匹配问题
            # 使用矩阵乘法计算B_x
            B_x = torch.matmul(B_param, self.B).view(batch_size, self.state_dim)  # [batch_size, state_dim]
            h_next = forget_gate.unsqueeze(1) * h_reshaped + input_gate.unsqueeze(1) * B_x.unsqueeze(1)
            h = h_next.view(batch_size, self.state_dim)
            
            # 输出计算: y = C * h + D * x
            # 将h从[batch_size, state_dim]重塑为适合与C相乘的形状
            h_for_C = h.view(batch_size, self.state_dim)
            
            # 计算输出
            y = torch.matmul(h_for_C, self.C) + D_param
            
            # 应用非线性激活和残差连接
            y = torch.tanh(y)
            y = self.out_proj(y) + x_t
            
            # 层归一化和dropout
            y = self.layer_norm(y)
            y = self.dropout(y)
            
            outputs.append(y)
        
        # 堆叠输出
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # 处理输出维度
        if squeeze_output:
            output = output.squeeze(1)  # [batch_size, hidden_dim]
        
        return output


class MambaBlock(nn.Module):
    """
    Mamba块
    
    结合序列化SSM和多层感知机，实现类似Mamba的功能
    """
    def __init__(self, hidden_dim, state_dim=32, expand_factor=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expanded_dim = hidden_dim * expand_factor
        
        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 输入投影
        self.in_proj = nn.Linear(hidden_dim, self.expanded_dim * 2)
        
        # SSM模块
        self.ssm = SequentialSSM(self.expanded_dim, state_dim, dropout)
        
        # 激活函数
        self.act = nn.SiLU()
        
        # 输出投影
        self.out_proj = nn.Linear(self.expanded_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim] 或 [num_nodes, hidden_dim]
            
        Returns:
            输出特征，形状与输入相同
        """
        # 残差连接
        residual = x
        
        # 层归一化
        x = self.norm(x)
        
        # 输入投影和分割
        x_proj = self.in_proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        
        # 应用SSM到第一部分
        x1 = self.ssm(x1)
        
        # 门控机制
        x = x1 * self.act(x2)
        
        # 输出投影
        x = self.out_proj(x)
        x = self.dropout(x)
        
        # 残差连接
        return residual + x


class PureMambaModel(ImprovedLongShortInteractModel):
    """
    纯粹的Mamba实现 - 专注于状态空间模型的核心机制
    
    这个实现更接近Mamba的原始设计，专注于：
    1. 状态空间模型处理序列信息
    2. 选择性SSM：参数是输入的函数
    3. 线性复杂度处理长序列
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, 
                 state_dim=16, expand_factor=2, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, 
                        num_heads, p, num_edge_heads, **kwargs)
        
        # Mamba核心：状态空间模型
        self.ssm = SequentialSSM(hidden_channels, state_dim, p)
        
        # 注意力投影层
        self.attn_proj = nn.Linear(hidden_channels, num_heads)
        
        # 值投影层
        self.val_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # 初始化参数
        self._init_pure_mamba_parameters()
    
    def _init_pure_mamba_parameters(self):
        """初始化纯粹Mamba的参数"""
        for module in [self.attn_proj, self.val_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
    
    def pure_mamba_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight):
        """
        纯粹的Mamba注意力计算
        
        核心思想：使用状态空间模型处理序列信息，而不是传统的注意力机制
        
        为了让节点和群组特征之间有序列关系，我们将：
        1. 基于边的连接关系构建序列
        2. 对每个节点，将其连接的所有群组作为序列输入
        3. 使用Mamba处理这种序列依赖关系
        """
        # 获取边对应的节点特征
        x1_i = x_1[x1_index]  # [num_edges, hidden_dim]
        x2_j = x_2[x2_index]  # [num_edges, hidden_dim]
        
        # 构建序列关系
        # 1. 对每个节点，找出其连接的所有群组
        unique_nodes = torch.unique(x1_index)
        batch_size = len(unique_nodes)
        
        # 初始化结果容器
        all_attn = []
        all_vals = []
        
        # 为每个节点单独处理其连接的群组序列
        for node_idx in unique_nodes:
            # 找出当前节点连接的所有边
            node_mask = (x1_index == node_idx)
            if not node_mask.any():
                continue
                
            # 获取当前节点连接的群组特征
            connected_groups = x2_j[node_mask]  # [num_connected, hidden_dim]
            connected_edge_weights = expanded_edge_weight[node_mask]  # [num_connected, hidden_dim]
            
            # 如果只有一个连接，直接处理
            if connected_groups.shape[0] == 1:
                # 合并特征
                node_feat = x_1[node_idx].unsqueeze(0)  # [1, hidden_dim]
                combined = node_feat + connected_groups  # [1, hidden_dim]
                
                # 使用SSM处理
                ssm_output = self.ssm(combined)  # [1, hidden_dim]
                
                # 投影到注意力空间
                attn_logits = self.attn_proj(ssm_output)  # [1, num_heads]
                
                # 应用边权重
                edge_weight_mean = connected_edge_weights.mean(dim=-1, keepdim=True)
                attn = F.silu(attn_logits * edge_weight_mean)  # [1, num_heads]
                
                # 处理值向量
                val_base = self.val_proj(ssm_output)  # [1, hidden_dim]
                
                all_attn.append(attn)
                all_vals.append(val_base)
            else:
                # 多个连接，构建序列
                # 先按距离排序（使用边权重的平均值作为距离指标）
                edge_distances = connected_edge_weights.mean(dim=-1)
                sorted_indices = torch.argsort(edge_distances)
                
                # 按距离排序群组特征，构建序列
                sorted_groups = connected_groups[sorted_indices]  # [num_connected, hidden_dim]
                sorted_edge_weights = connected_edge_weights[sorted_indices]  # [num_connected, hidden_dim]
                
                # 将节点特征添加到序列开头
                node_feat = x_1[node_idx].unsqueeze(0)  # [1, hidden_dim]
                sequence = torch.cat([node_feat, sorted_groups], dim=0)  # [1+num_connected, hidden_dim]
                
                # 使用SSM处理序列
                ssm_output = self.ssm(sequence)  # [1+num_connected, hidden_dim]
                
                # 我们只关心序列中除第一个元素（节点自身）之外的输出
                group_ssm_output = ssm_output[1:]  # [num_connected, hidden_dim]
                
                # 投影到注意力空间
                attn_logits = self.attn_proj(group_ssm_output)  # [num_connected, num_heads]
                
                # 应用边权重
                edge_weight_mean = sorted_edge_weights.mean(dim=-1, keepdim=True)
                attn = F.silu(attn_logits * edge_weight_mean)  # [num_connected, num_heads]
                
                # 处理值向量
                val_base = self.val_proj(group_ssm_output)  # [num_connected, hidden_dim]
                
                # 将结果恢复到原始顺序
                restore_indices = torch.argsort(sorted_indices)
                attn = attn[restore_indices]
                val_base = val_base[restore_indices]
                
                all_attn.append(attn)
                all_vals.append(val_base)
        
        # 将所有节点的结果合并
        final_attn = torch.cat(all_attn, dim=0)  # [num_edges, num_heads]
        final_vals = torch.cat(all_vals, dim=0)  # [num_edges, hidden_dim]
        
        # 重塑为多头形式
        final_vals = final_vals.reshape(-1, self.num_heads, self.hidden_channels // self.num_heads)
        
        return final_attn, final_vals
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """
        纯粹Mamba的前向传播
        """
        # 预处理
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        
        if self.p > 0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        # 使用纯粹Mamba计算注意力
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        attn_2, val_2 = self.pure_mamba_attention(
            node_embedding,  # 直接使用原始特征
            group_embedding, # 直接使用原始特征
            edge_index[0], 
            edge_index[1], 
            edge_attr
        )
        
        # 消息传递
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),  # 使用原始特征
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight, 
            attn_score=attn_2, 
            val=val_2,  # 直接使用val_2，不需要索引
            mode='group_to_node'
        )
        
        # 特征更新
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        
        return dx_node, dv_node


class MambaLongShortInteractModel(ImprovedLongShortInteractModel):
    """
    基于Mamba的长短程交互模型
    
    继承自ImprovedLongShortInteractModel，添加Mamba架构来增强序列建模能力
    
    特点:
    1. 保持与原有模型的完全兼容性
    2. 使用Mamba架构增强长程依赖建模
    3. 支持向量和标量特征的双通道处理
    4. 保留原有的多头注意力机制
    5. 添加序列建模能力用于处理分子片段间的依赖关系
    """
    
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, 
                 mamba_layers=2, state_dim=32, expand_factor=2, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, 
                        num_heads, p, num_edge_heads, **kwargs)
        
        # Mamba参数
        self.mamba_layers = mamba_layers
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        
        # 节点特征的Mamba层
        self.node_mamba_layers = nn.ModuleList([
            MambaBlock(hidden_channels, state_dim, expand_factor, p)
            for _ in range(mamba_layers)
        ])
        
        # 官能团特征的Mamba层  
        self.group_mamba_layers = nn.ModuleList([
            MambaBlock(hidden_channels, state_dim, expand_factor, p)
            for _ in range(mamba_layers)
        ])
        
        # 向量特征的变换网络
        self.vec_transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 注意力计算的Mamba模块
        self.attention_mamba = MambaBlock(hidden_channels, state_dim, expand_factor, p)
        
        # 值变换的Mamba模块
        self.value_mamba = MambaBlock(hidden_channels, state_dim, expand_factor, p)
        
        # # 距离编码器
        # self.distance_encoder = nn.Sequential(
        #     nn.Linear(1, hidden_channels // 4),
        #     nn.SiLU(),
        #     nn.Linear(hidden_channels // 4, hidden_channels)
        # )
        
        # # 方向编码器
        # self.direction_encoder = nn.Sequential(
        #     nn.Linear(3, hidden_channels // 4),
        #     nn.SiLU(),
        #     nn.Linear(hidden_channels // 4, hidden_channels)
        # )
        
        # 注意力投影层
        self.attn_proj = nn.Linear(hidden_channels, num_heads)
        
        # 值投影层
        self.val_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # 初始化参数
        self._init_mamba_parameters()
    
    def _init_mamba_parameters(self):
        """初始化Mamba相关的参数"""
        for module in [self.vec_transform, self.attn_proj, self.val_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
    
    def apply_mamba_to_features(self, features, mamba_layers):
        """
        应用Mamba层到特征
        
        Args:
            features: 输入特征 [num_items, hidden_dim]
            mamba_layers: Mamba层列表
            
        Returns:
            enhanced_features: 增强后的特征 [num_items, hidden_dim]
        """
        # 逐层应用Mamba
        x = features
        for mamba_layer in mamba_layers:
            x = mamba_layer(x)
        
        return x
    
    def mamba_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, edge_weight=None, edge_vec=None, edge_index=None):
        """
        使用Mamba计算注意力和值
        
        基于序列关系的注意力计算，将每个节点连接的群组视为序列进行处理
        
        Args:
            x_1: 查询节点特征 [num_nodes, hidden_dim]
            x_2: 键值节点特征 [num_groups, hidden_dim]
            x1_index: 查询节点索引 [num_edges]
            x2_index: 键值节点索引 [num_edges]
            expanded_edge_weight: 边权重 [num_edges, hidden_dim]
            edge_weight: 边距离 [num_edges]
            edge_vec: 边向量 [num_edges, 3]
            
        Returns:
            attn: 注意力分数 [num_edges, num_heads]
            val: 值向量 [num_groups, num_heads, attn_channels]
        """
        # 构建序列关系
        # 1. 对每个节点，找出其连接的所有群组
        unique_nodes = torch.unique(x1_index)
        
        # 初始化结果容器
        all_attn = []
        all_node_indices = []  # 记录每个注意力值对应的原始边索引
        
        # 处理值向量 - 对所有群组应用Mamba变换
        val_features = self.value_mamba(x_2)  # [num_groups, hidden_dim]
        val = self.val_proj(val_features)  # [num_groups, hidden_dim]
        
        # 重塑为多头形式
        val = val.reshape(-1, self.num_heads, self.hidden_channels // self.num_heads)
        
        # 为每个节点单独处理其连接的群组序列
        for node_idx in unique_nodes:
            # 找出当前节点连接的所有边
            node_mask = (x1_index == node_idx)
            edge_indices = torch.where(node_mask)[0]
            
            if edge_indices.shape[0] == 0:
                continue
                
            # 获取当前节点连接的群组特征和索引
            connected_group_indices = x2_index[node_mask]  # [num_connected]
            connected_groups = x_2[connected_group_indices]  # [num_connected, hidden_dim]
            
            # 获取边属性
            if edge_weight is not None:
                connected_edge_weights = edge_weight[node_mask]  # [num_connected]
            else:
                connected_edge_weights = torch.ones(connected_groups.shape[0], device=x_1.device)
                
            if edge_vec is not None:
                connected_edge_vecs = edge_vec[node_mask]  # [num_connected, 3]
            
            # 如果只有一个连接，直接处理
            if connected_groups.shape[0] == 1:
                # 获取节点特征
                node_feat = x_1[node_idx].unsqueeze(0)  # [1, hidden_dim]
                
                # 合并特征
                combined_features = node_feat + connected_groups  # [1, hidden_dim]
                
                # 应用Mamba处理
                enhanced_features = self.attention_mamba(combined_features)  # [1, hidden_dim]
                
                # 投影到注意力空间
                attn_logits = self.attn_proj(enhanced_features)  # [1, num_heads]
                
                # 应用边权重
                if edge_weight is not None:
                    # 使用距离衰减
                    distance_weight = torch.exp(-connected_edge_weights / self.cutoff).unsqueeze(-1)
                    attn_logits = attn_logits * distance_weight
                
                # 应用激活函数
                attn = F.silu(attn_logits)
                
                all_attn.append(attn)
                all_node_indices.append(edge_indices)
            else:
                # 多个连接，构建序列
                
                # 获取节点特征
                node_feat = x_1[node_idx].unsqueeze(0)  # [1, hidden_dim]
                
                # 排序连接的群组（按距离）
                if edge_weight is not None:
                    # 使用边距离排序
                    sorted_indices = torch.argsort(connected_edge_weights)
                    
                    # 按距离排序群组特征和边属性
                    sorted_groups = connected_groups[sorted_indices]  # [num_connected, hidden_dim]
                    sorted_edge_weights = connected_edge_weights[sorted_indices]  # [num_connected]
                    sorted_edge_indices = edge_indices[sorted_indices]  # [num_connected]
                    
                    if edge_vec is not None:
                        sorted_edge_vecs = connected_edge_vecs[sorted_indices]  # [num_connected, 3]
                else:
                    # 不排序
                    sorted_groups = connected_groups
                    sorted_edge_indices = edge_indices
                    sorted_edge_weights = connected_edge_weights
                    
                    if edge_vec is not None:
                        sorted_edge_vecs = connected_edge_vecs
                
                # 构建序列：节点特征 + 排序后的群组特征
                sequence = torch.cat([node_feat, sorted_groups], dim=0)  # [1+num_connected, hidden_dim]
                
                # 应用Mamba处理序列
                enhanced_sequence = self.attention_mamba(sequence)  # [1+num_connected, hidden_dim]
                
                # 我们只关心序列中除第一个元素（节点自身）之外的输出
                enhanced_groups = enhanced_sequence[1:]  # [num_connected, hidden_dim]
                
                # 投影到注意力空间
                attn_logits = self.attn_proj(enhanced_groups)  # [num_connected, num_heads]
                
                # 应用边权重
                if edge_weight is not None:
                    # 使用距离衰减
                    distance_weight = torch.exp(-sorted_edge_weights / self.cutoff).unsqueeze(-1)
                    attn_logits = attn_logits * distance_weight
                
                # 应用激活函数
                attn = F.silu(attn_logits)
                
                # 如果之前排序了，需要恢复原始顺序
                if edge_weight is not None:
                    restore_indices = torch.argsort(sorted_indices)
                    attn = attn[restore_indices]
                    # 不需要恢复edge_indices的顺序，因为我们记录的是原始索引
                
                all_attn.append(attn)
                all_node_indices.append(sorted_edge_indices)
        
        # 将所有节点的结果按原始边的顺序合并
        final_attn = torch.cat(all_attn, dim=0)
        all_edge_indices = torch.cat(all_node_indices, dim=0)
        
        # 创建一个与边数相同大小的空张量
        num_edges = x1_index.shape[0]
        result_attn = torch.zeros((num_edges, self.num_heads), device=x_1.device)
        
        # 将计算的注意力值放回对应的位置
        result_attn[all_edge_indices] = final_attn
        
        return result_attn, val
    
    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        计算从节点j到节点i的消息
        
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
        '''
        
        if mode == 'node_to_group':
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim=-1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
                    model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
        
        # 标量消息计算
        m_s_ij = val * attn_score.unsqueeze(-1)  # [num_edges, num_heads, attn_channels]
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)  # [num_edges, hidden_dim]
        
        # 向量消息计算
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v  # [num_edges, 3, hidden_dim]
        
        return m_s_ij, m_v_ij
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """
        前向传播函数
        
        使用Mamba架构增强节点和官能团特征，然后进行消息传递
        
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
        # 1. 预处理：层归一化和dropout
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        
        if self.p > 0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        # 2. 使用Mamba增强特征表示
        # 增强节点特征
        enhanced_node_embedding = self.apply_mamba_to_features(
            node_embedding, self.node_mamba_layers
        )
        
        # 增强官能团特征
        enhanced_group_embedding = self.apply_mamba_to_features(
            group_embedding, self.group_mamba_layers  
        )
        
        # 3. 使用Mamba计算注意力
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        attn_2, val_2 = self.mamba_attention(
            enhanced_node_embedding,     # 使用增强的节点特征
            enhanced_group_embedding,    # 使用增强的官能团特征
            edge_index[0], 
            edge_index[1], 
            edge_attr, 
            edge_weight,
            -edge_vec,
            edge_index
        )
        
        # 4. 消息传递
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(enhanced_group_embedding, enhanced_node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight, 
            attn_score=attn_2, 
            val=val_2[edge_index[1]],
            mode='group_to_node'
        )
        
        # 5. 特征更新（保持原有逻辑，但使用增强的向量特征）
        enhanced_node_vec = self.vec_transform(node_vec.view(-1, self.hidden_channels)).view(node_vec.shape)
        
        v_node_1 = self.model_2['linears'][2](enhanced_node_vec)
        v_node_2 = self.model_2['linears'][3](enhanced_node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](enhanced_node_vec)
        
        return dx_node, dv_node
