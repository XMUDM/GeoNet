import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops
from .long_short_interact_modules import LongShortIneractModel_dis_direct_vector2_drop, ImprovedLongShortInteractModel
from .utils import get_distance
from .torchmdnet.models.torchmd_norm import  EquivariantMultiHeadAttention
from .torchmdnet.models.utils import ExpNormalSmearing,GaussianSmearing,NeighborEmbedding, vec_layernorm, max_min_norm, norm
from .output_net import OutputNet
from ..utils import conditional_grad
import torch.nn.functional as F
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from .enhanced_message import EnhancedMessageModel
from .hybrid_geometric_message import HybridGeometricInteractModel
import math
import torch
import torch.nn as nn
from .torchmdnet.models.utils import ExpNormalSmearing, GaussianSmearing, CosineCutoff
import torch
import torch.nn as nn
from .dimenet_angle_features import SimplifiedDimeNetAngleExtractor, DimeNetStyleAngleFeatureExtractor,  DimeNetVectorAngleExtractor

class Node_Edge_Fea_Init(nn.Module):
    def __init__(self,
                 max_z = 100,
                 rbf_type="expnorm",
                 num_rbf = 50,
                 trainable_rbf = True,
                 hidden_channels = 128,
                 cutoff_lower = 0,
                 cutoff_upper = 5,
                 neighbor_embedding = True):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels)
        
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder=rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        self.rbf_linear = nn.Linear(num_rbf,hidden_channels)
        if neighbor_embedding:
            self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z)
        else:
            self.neighbor_embedding = None
            
    def forward(self,z,pos,edge_index):
        #z means atoms-ID: H1, C6, N7, O8,
        ### this part is for node short term neighbor.
        node_embedding = self.embedding(z)
        node_vec = torch.zeros(node_embedding.size(0), 3, node_embedding.size(1), device=node_embedding.device)
        edge_index, edge_weight, edge_vec = get_distance(pos,pos,edge_index)
        edge_attr = self.distance_encoder(edge_weight)
        # mask = edge_index[0] != edge_index[1]
        # edge_vec[mask] = edge_vec[mask]  / (torch.norm(edge_vec[mask], dim=1).unsqueeze(1)+1e-5)
        edge_vec = edge_vec  / norm(edge_vec, keepdim=True) 
        if self.neighbor_embedding is not None:
            node_embedding = self.neighbor_embedding(z, node_embedding, edge_index, edge_weight, edge_attr)
        edge_attr = self.rbf_linear(edge_attr)
        return node_embedding, node_vec, edge_index, edge_weight, edge_attr, edge_vec

class Edge_Feat_Init(nn.Module):
    def __init__(self,
                rbf_type="expnorm",
                num_rbf = 50,
                trainable_rbf = True,
                hidden_channels = 128,
                cutoff_lower = 0,
                cutoff_upper = 5):
    
        super().__init__()
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder=rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        self.rbf_linear = nn.Linear(num_rbf,hidden_channels)

    def forward(self, pos, edge_index):
        edge_index, edge_weight, edge_vec = get_distance(pos,pos,edge_index)
        edge_attr = self.distance_encoder(edge_weight)
        edge_vec = edge_vec  / norm(edge_vec, keepdim=True)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec
    
class Bipartite_Edge_Feat_Init(nn.Module):
    def __init__(self,
                rbf_type="expnorm",
                num_rbf = 50,
                trainable_rbf = True,
                hidden_channels = 128,
                cutoff_lower = 0,
                cutoff_upper = 10):
    
        super().__init__()
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder=rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        self.rbf_linear = nn.Linear(num_rbf,hidden_channels)

    def forward(self, edge_index, node_pos, group_pos, *args, **kwargs):
        edge_vec = node_pos[edge_index[0]] - group_pos[edge_index[1]]
        edge_weight = norm(edge_vec, dim=1)
        edge_vec = edge_vec / edge_weight.unsqueeze(1)
        edge_attr = self.distance_encoder(edge_weight)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec   

class AngleRBF(nn.Module):
    """
    角度径向基函数编码器，基于ExpNormalSmearing设计，专门针对角度范围[0, π]
    """
    def __init__(self, num_rbf=25, trainable=True):
        super().__init__()
        self.cutoff_lower = 0.0  # 角度下界
        self.cutoff_upper = math.pi  # 角度上界 π
        self.num_rbf = num_rbf
        self.trainable = trainable

        # 使用ExpNormalSmearing的alpha缩放参数
        self.alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # 根据ExpNormalSmearing的初始化方式，适应角度范围
        # 参考PhysNet的默认值，但适应角度范围[0, π]
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, angles):
        """
        Args:
            angles: [N,] 角度值（弧度，范围0到π）
        Returns:
            rbf_features: [N, num_rbf] RBF编码特征
        """
        angles = angles.unsqueeze(-1)
        # 使用ExpNormalSmearing的指数-正态编码，角度范围[0, π]通常不需要cutoff
        return torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-angles + self.cutoff_lower)) - self.means) ** 2
        )

class SimpleAngleFeatureExtractor(nn.Module):
    """
    简化的夹角特征提取器，参考distance_encoder的设计思路
    只计算最重要的两个夹角：node-edge夹角和group-edge夹角
    """
    
    def __init__(self, hidden_channels, num_angle_rbf=25, trainable=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_angle_rbf = num_angle_rbf
        
        # 夹角RBF编码器 - 类似distance_encoder
        self.angle_rbf = AngleRBF(num_rbf=num_angle_rbf, trainable=trainable)
        
        # 简单的线性映射 - 类似distance_encoder的rbf_linear
        # 输入是2个角度 * num_angle_rbf
        self.angle_linear = nn.Linear(num_angle_rbf, hidden_channels)
    
    def extract_3d_vectors(self, vec_features):
        """从不同格式的向量特征中提取3D向量"""
        if vec_features.dim() == 2:
            return vec_features
        elif vec_features.dim() == 3:
            return vec_features.mean(dim=-1)  # 简单取平均
        else:
            return vec_features
    
    def compute_angle(self, vec1, vec2, eps=1e-8):
        """计算两个向量之间的夹角（弧度）"""
        # # 归一化向量
        # vec1_norm = torch.norm(vec1, dim=-1, keepdim=True) + eps
        # vec2_norm = torch.norm(vec2, dim=-1, keepdim=True) + eps
        
        # vec1_unit = vec1 / vec1_norm
        # vec2_unit = vec2 / vec2_norm
        
        # # 计算余弦值
        cos_angle = (vec1 * vec2).sum(dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        
        # 计算角度
        angle = torch.acos(cos_angle)
        return angle
    
    def forward(self, edge_index, node_vec, group_vec, edge_vec):
        """
        提取简化的夹角特征
        
        Args:
            edge_index: [2, num_edges] 边索引
            node_vec: [num_nodes, 3, hidden] 或 [num_nodes, 3] node向量特征
            group_vec: [num_groups, 3, hidden] 或 [num_groups, 3] group向量特征  
            edge_vec: [num_edges, 3] 边向量 (归一化)
            
        Returns:
            angle_features: [num_edges, hidden_channels] 编码后的夹角特征
        """
        node_idx = edge_index[0]  # [num_edges]
        group_idx = edge_index[1]  # [num_edges]
        
        # 提取3D向量特征
        node_vec_3d = self.extract_3d_vectors(node_vec)  # [num_nodes, 3]
        # group_vec_3d = self.extract_3d_vectors(group_vec)  # [num_groups, 3]
        
        # 获取边对应的向量
        node_vec_edge = node_vec_3d[node_idx]  # [num_edges, 3]
        # group_vec_edge = group_vec_3d[group_idx]  # [num_edges, 3]
        
        # 只计算两个核心夹角
        angle_node_edge = self.compute_angle(node_vec_edge, edge_vec)   # [num_edges]
        # angle_group_edge = self.compute_angle(group_vec_edge, edge_vec) # [num_edges]
        
        # 使用RBF编码角度特征 - 类似distance_encoder的处理方式
        node_angle_rbf = self.angle_rbf(angle_node_edge)    # [num_edges, num_angle_rbf]
        # group_angle_rbf = self.angle_rbf(angle_group_edge)  # [num_edges, num_angle_rbf]
        
        # 连接两个角度的RBF特征
        # combined_rbf = torch.cat([node_angle_rbf, group_angle_rbf], dim=-1)  # [num_edges, 2 * num_angle_rbf]
        
        # 线性映射到目标维度 - 类似distance_encoder的rbf_linear
        angle_features = self.angle_linear(node_angle_rbf)  # [num_edges, hidden_channels]
        
        return angle_features

class BipartiteEdgeWithAngleFeatures(nn.Module):
    """
    分离的双分图边特征初始化模块
    分别处理距离特征和夹角特征，返回两个独立的特征
    """
    def __init__(self,
                edge_index,
                rbf_type="expnorm",
                num_rbf = 50,
                trainable_rbf = True,
                hidden_channels = 128,
                cutoff_lower = 0,
                cutoff_upper = 10,
                use_angle_features = True,
                angle_feature_dim = 12):
    
        super().__init__()
        self.use_angle_features = use_angle_features
        
        # 距离编码器（原有功能）
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder = rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        
        # 距离特征映射器
        self.rbf_linear = nn.Linear(num_rbf, hidden_channels)
        

        if use_angle_features:
            # 使用DimeNet风格的角度特征提取器，基于现有三元组，避免重复计算距离RBF
            self.angle_extractor = DimeNetStyleAngleFeatureExtractor(
                edge_index=edge_index,
                hidden_channels=hidden_channels,
                num_spherical=7,
                num_radial=6  # 这个参数现在不用于距离，只是为了兼容
            )
            print("[INFO] 使用DimeNetStyleAngleFeatureExtractor，基于现有三元组避免重复距离RBF计算")

    def forward(self, edge_index, node_pos, group_pos, node_vec=None, group_vec=None, *args, **kwargs):
        """
        Args:
            edge_index: [2, num_edges] 边索引
            node_pos: [num_nodes, 3] node位置
            group_pos: [num_groups, 3] group位置  
            node_vec: [num_nodes, 3, hidden] 或 [num_nodes, 3] node向量特征
            group_vec: [num_groups, 3, hidden] 或 [num_groups, 3] group向量特征
            
        Returns:
            edge_index: [2, num_edges] 边索引
            edge_weight: [num_edges] 边权重
            edge_attr: [num_edges, hidden_channels] 距离特征
            edge_vec_normalized: [num_edges, 3] 归一化边向量
            angle_attr: [num_edges, hidden_channels] 夹角特征 (如果启用)
        """
        # 计算基础距离特征
        edge_vec = node_pos[edge_index[0]] - group_pos[edge_index[1]]
        edge_weight = norm(edge_vec, dim=1)
        edge_vec_normalized = edge_vec / edge_weight.unsqueeze(1)
        
        # 处理距离特征
        distance_features = self.distance_encoder(edge_weight)
        edge_attr = self.rbf_linear(distance_features)  # [num_edges, hidden_channels]
        
        # 处理夹角特征
        if self.use_angle_features and node_vec is not None and group_vec is not None:
            # 使用DimeNet风格的角度特征提取器，基于现有三元组(node_vec, edge_vec, group_vec)
            angle_attr = self.angle_extractor(
                edge_index, node_vec, group_vec, edge_vec_normalized
            )  # [num_edges, hidden_channels]
            
            return edge_index, edge_weight, edge_attr, edge_vec_normalized, angle_attr
        else:
            # 如果不使用夹角特征，返回None
            return edge_index, edge_weight, edge_attr, edge_vec_normalized, None

class DualGatingMoE(nn.Module):
    """修复版双门控混合专家模型
    
    为标量特征和向量特征分别实现独立的门控机制，
    并解决维度不匹配问题。
    同时保留物理环境分析功能。
    """
    def __init__(self, hidden_channels, num_experts=2, dropout=0.1, use_physics_context=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_experts = num_experts
        self.use_physics_context = use_physics_context
        
        # 标量门控 - 保持简单的结构
        self.scalar_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_experts)
        )
        
        # 向量门控 - 保持与标量门控相同的维度
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_experts)
        )
        
        # 仅当启用物理上下文时创建这些模块
        if use_physics_context:
            # 标量特征物理环境分析器 - 简化版本
            self.scalar_physics_analyzer = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
            
            # 向量特征物理环境分析器 - 简化版本
            self.vector_physics_analyzer = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
            )
            
            # 标量特征调制器 - 更稳定的设计
            self.scalar_modulator = nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, num_experts),
                nn.Tanh()  # 输出-1到1之间的调制系数
            )
            
            # 向量特征调制器 - 更稳定的设计
            self.vector_modulator = nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, num_experts),
                nn.Tanh()  # 输出-1到1之间的调制系数
            )
        
        # 标量特征专家
        self.scalar_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels)
            ) for _ in range(num_experts)
        ])
        
        # 向量特征专家 - 确保输入输出维度一致
        self.vector_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            ) for _ in range(num_experts)
        ])
        # self.modulation_strength = nn.Parameter(torch.tensor(0.01))

        print("[INFO] 使用带物理环境分析的维度修复版双门控MoE")
    
    def compute_gates(self, features, gate_network, physics_analyzer=None, modulator=None):
        """计算门控权重，可选地包含物理环境分析"""
        # 基础门控logits
        base_logits = gate_network(features)
        
        # 如果启用物理上下文且提供了分析器和调制器
        if self.use_physics_context and physics_analyzer is not None and modulator is not None:
            # 物理环境分析
            physics_ctx = physics_analyzer(features)
            
            # 基于物理环境生成调制系数
            mod_coeff = modulator(physics_ctx)
            
            # 调制基础logits (使用小系数限制调制影响，提高稳定性)
            # adjusted_logits = base_logits + mod_coeff * self.modulation_strength
            adjusted_logits = base_logits + mod_coeff * 0.1

        else:
            adjusted_logits = base_logits
            physics_ctx = None
        
        # 应用softmax获取最终门控权重
        gates = F.softmax(adjusted_logits, dim=-1)
        
        return gates
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, z=None):
        batch_size = scalar_short.size(0)
        
        # 合并短程和长程特征
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)  # [batch, hidden*2]
        vector_combined = torch.cat([vector_short, vector_long], dim=-1)  # [batch, 3, hidden*2]
        
        # 为门控计算池化向量特征
        vector_pooled = torch.mean(vector_combined, dim=1)  # [batch, hidden*2]
        
        # 计算标量特征的门控权重
        scalar_gates = self.compute_gates(
            scalar_combined, 
            self.scalar_gate,
            self.scalar_physics_analyzer if self.use_physics_context else None,
            self.scalar_modulator if self.use_physics_context else None
        )
        
        # 计算向量特征的门控权重
        vector_gates = self.compute_gates(
            vector_pooled, 
            self.scalar_gate,
            self.vector_physics_analyzer if self.use_physics_context else None,
            self.vector_modulator if self.use_physics_context else None
        )
        
        # 应用专家网络 - 标量特征
        scalar_outputs = []
        for i in range(self.num_experts):
            scalar_out = self.scalar_experts[i](scalar_combined)  # [batch, hidden]
            scalar_outputs.append(scalar_out * scalar_gates[:, i:i+1])  # [batch, hidden]
        
        # 应用专家网络 - 向量特征
        vector_outputs = []
        for i in range(self.num_experts):
            vector_out = self.vector_experts[i](vector_combined)  # [batch, 3, hidden]
            # 注意这里的维度扩展方式，确保与向量维度匹配
            gate = vector_gates[:, i:i+1].unsqueeze(1)  # [batch, 1, 1]
            vector_outputs.append(vector_out * gate)  # [batch, 3, hidden]
        
        # 组合专家输出
        scalar_result = sum(scalar_outputs)  # [batch, hidden]
        vector_result = sum(vector_outputs)  # [batch, 3, hidden]
        
        return scalar_result, vector_result
    
    def get_physics_analysis(self, scalar_short, scalar_long, vector_short, vector_long):
        """获取物理环境分析结果，用于可视化和分析"""
        if not self.use_physics_context:
            return None
            
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)
        vector_pooled = torch.mean(torch.cat([vector_short, vector_long], dim=-1), dim=1)
        
        scalar_physics_ctx = self.scalar_physics_analyzer(scalar_combined)
        vector_physics_ctx = self.vector_physics_analyzer(vector_pooled)
        
        # 计算标量门控
        scalar_gates, _ = self.compute_gates(
            scalar_combined, 
            self.scalar_gate,
            self.scalar_physics_analyzer,
            self.scalar_modulator
        )
        
        # 计算向量门控
        vector_gates, _ = self.compute_gates(
            vector_pooled, 
            self.vector_gate,
            self.vector_physics_analyzer,
            self.vector_modulator
        )
        
        return {
            'scalar_physics_context': scalar_physics_ctx,
            'vector_physics_context': vector_physics_ctx,
            'scalar_gates': scalar_gates,
            'vector_gates': vector_gates,
            'context_difference': (scalar_physics_ctx - vector_physics_ctx).abs()
        }

class MixtureOfExperts(nn.Module):
    def __init__(self, hidden_channels, num_experts=8, dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_experts = num_experts
        # # 向量特征处理和聚合
        # self.vector_projection = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)
        # self.vector_pooling = nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels // 2),
        #     nn.SiLU()
        # )
        
        # 标量特征处理
        self.scalar_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.SiLU()
        )
        
        # 标量特征的专家选择器
        self.scalar_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # # 向量特征的专家选择器
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_channels // 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 专家网络 - 标量
        self.experts_scalar = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels)
            ) for _ in range(num_experts)
        ])
        
        # 专家网络 - 向量
        self.experts_vector = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            ) for _ in range(num_experts)
        ])
        
        # 最终投影层
        self.final_scalar = nn.Linear(hidden_channels, hidden_channels)
        self.final_vector = nn.Linear(hidden_channels, hidden_channels, bias=False)
        print("[INFO] 原始MoE")
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, z=None):
        batch_size = scalar_short.size(0)
        
        # 组合短程和长程特征
        scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)
        vector_combined = torch.cat([vector_short, vector_long], dim=-1)
        
        # 1. 特征处理和提取
        # 标量特征处理
        # scalar_features = self.scalar_projection(scalar_combined)#注释
        
        # 向量特征处理
        vector_pooled = torch.mean(vector_combined, dim=1)
        # vector_features = self.vector_pooling(vector_pooled)#注释

        # 2. 独立的专家选择
        # 标量专家权重
        scalar_expert_weights = self.scalar_gate(scalar_combined)
        
        # # 向量专家权重
        vector_expert_weights = self.scalar_gate(vector_pooled)



        
        # 3. 应用专家
        scalar_outputs = []
        vector_outputs = []
        
        for i in range(self.num_experts):
            # 获取专家输出
            scalar_out = self.experts_scalar[i](scalar_combined)
            vector_out = self.experts_vector[i](vector_combined)
            # 应用独立的专家权重
            scalar_outputs.append(scalar_out * scalar_expert_weights[:, i:i+1])
            vector_outputs.append(vector_out * vector_expert_weights[:, i:i+1].unsqueeze(1))
        
        # 4. 整合专家输出
        scalar_result = sum(scalar_outputs)
        vector_result = sum(vector_outputs)
        
        # 5. 最终投影
        scalar_result = self.final_scalar(scalar_result)
        vector_result = self.final_vector(vector_result)
        
        return scalar_result, vector_result


# 添加构象一致性损失类
class EnhancedConformationalConsistencyLoss(nn.Module):
    """
    增强版构象一致性损失，同时考虑短程和长程信息，标量和向量特征
    """
    def __init__(self, hidden_channels, consistency_factor=0.1, vector_weight=0.5, 
                short_long_ratio=0.3, brics_consistency_strength=0.8):
        super().__init__()
        self.consistency_factor = consistency_factor
        self.vector_weight = vector_weight  # 向量特征权重
        self.short_long_ratio = short_long_ratio  # 短程信息权重
        self.brics_consistency_strength = brics_consistency_strength
        
        # 标量特征处理
        self.scalar_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 向量特征处理 - 注意无偏置设计
        self.vector_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels, bias=False)
        )
        
        
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, fragment_ids):
        """
        计算增强版构象一致性损失
        
        Args:
            scalar_short: 短程标量嵌入 [num_nodes, hidden_dim]
            scalar_long: 长程标量嵌入 [num_nodes, hidden_dim]
            vector_short: 短程向量嵌入 [num_nodes, 3, hidden_dim]
            vector_long: 长程向量嵌入 [num_nodes, 3, hidden_dim]
            fragment_ids: BRICS片段ID [num_nodes]
            
        Returns:
            consistency_loss: 一致性损失
        """
        # 必须提供fragment_ids
        if fragment_ids is None:
            print("[WARNING] 未提供分子片段ID，无法计算一致性损失")
            return torch.tensor(0.0, device=scalar_long.device)
        
        # 组合短程和长程信息
        scalar_combined = scalar_short * self.short_long_ratio + scalar_long * (1 - self.short_long_ratio)
        
        # 提取增强的不变特征
        scalar_invariant = self.scalar_encoder(scalar_combined)
        
        # 初始化损失计算
        total_loss = 0
        valid_count = 0
        
        # 获取唯一片段ID
        unique_fragments = torch.unique(fragment_ids)#是否不需要unique
        
        # 针对每个片段计算损失
        for frag_id in unique_fragments:
            # 获取当前片段的原子
            frag_mask = (fragment_ids == frag_id)
            if torch.sum(frag_mask) <= 1:
                continue
                
            # --- 标量特征一致性 ---
            # 提取片段标量特征
            frag_scalar = scalar_invariant[frag_mask]
            
            # 方差损失
            scalar_mean = torch.mean(frag_scalar, dim=0, keepdim=True)
            scalar_variance = torch.mean(torch.sum((frag_scalar - scalar_mean)**2, dim=1))
            
            # 相似度损失
            scalar_norm = F.normalize(frag_scalar, p=2, dim=1)
            scalar_sim = torch.matmul(scalar_norm, scalar_norm.transpose(0, 1))
            scalar_mask = torch.triu(torch.ones_like(scalar_sim), diagonal=1).bool()
            
            if torch.sum(scalar_mask) > 0:
                scalar_target = self.brics_consistency_strength * torch.ones_like(scalar_sim)[scalar_mask]
                scalar_sim_loss = F.mse_loss(scalar_sim[scalar_mask], scalar_target)
                scalar_loss_frag = scalar_variance + scalar_sim_loss
                
                # --- 向量特征一致性 ---
                if self.vector_weight > 0:
                    # 组合短程和长程向量信息
                    vector_combined = vector_short * self.short_long_ratio + vector_long * (1 - self.short_long_ratio)
                    vector_invariant = self.vector_encoder(vector_combined)
                    
                    # 提取片段向量特征
                    frag_vector = vector_invariant[frag_mask]
                    
                    # 向量方差损失 - 考虑3D维度
                    vector_mean = torch.mean(frag_vector, dim=0, keepdim=True)  # [1, 3, hidden_dim]
                    vector_variance = torch.mean(torch.sum(torch.sum((frag_vector - vector_mean)**2, dim=2), dim=1))
                    
                    # 向量方向一致性 - 对每个向量维度计算方向一致性
                    vector_dir_loss = 0
                    for dim in range(3):  # x, y, z维度
                        # 提取该维度的向量
                        dim_vector = frag_vector[:, dim, :]  # [frag_size, hidden_dim]
                        # 归一化
                        dim_norm = F.normalize(dim_vector, p=2, dim=1)
                        # 计算点积相似度矩阵
                        dim_sim = torch.matmul(dim_norm, dim_norm.transpose(0, 1))
                        # 仅考虑上三角矩阵(不含对角线)
                        dim_sim_loss = F.mse_loss(dim_sim[scalar_mask], scalar_target)
                        vector_dir_loss += dim_sim_loss
                    
                    vector_dir_loss = vector_dir_loss / 3  # 平均三个维度
                    vector_loss_frag = vector_variance + vector_dir_loss
                    
                    # 组合标量和向量损失
                    frag_loss = (1 - self.vector_weight) * scalar_loss_frag + self.vector_weight * vector_loss_frag
                else:
                    frag_loss = scalar_loss_frag
                    
                # 修复: 使用组合后的frag_loss而不是仅scalar_loss_frag
                total_loss += frag_loss
                valid_count += 1
        
        # 避免除零错误
        if valid_count > 0:
            final_loss = total_loss / valid_count
            return final_loss 
        else:
            return torch.tensor(0.0, device=scalar_long.device)


class BalancedConformationalConsistencyLoss(nn.Module):
    """
    平衡型构象一致性损失，保持片段整体特性的同时保留原子独特性
    """
    def __init__(self, hidden_channels, consistency_factor=0.1, vector_weight=0.2, 
                short_long_ratio=0.3, brics_consistency_strength=0.4,
                atom_identity_weight=0.6):
        super().__init__()
        self.consistency_factor = consistency_factor
        self.vector_weight = vector_weight  # 降低向量权重
        self.short_long_ratio = short_long_ratio
        self.brics_consistency_strength = brics_consistency_strength  # 降低目标相似度
        self.atom_identity_weight = atom_identity_weight  # 增加原子独特性权重
        
        # 标量特征处理
        self.scalar_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 向量特征处理
        self.vector_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels, bias=False)
        )
        
        # 原子特征分解器 - 分离共享部分和独特部分
        self.feature_decomposer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels*2, hidden_channels*2)
        )
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, fragment_ids, atom_types=None):
        """
        计算平衡型构象一致性损失
        
        Args:
            scalar_short: 短程标量嵌入 [num_nodes, hidden_dim]
            scalar_long: 长程标量嵌入 [num_nodes, hidden_dim]
            vector_short: 短程向量嵌入 [num_nodes, 3, hidden_dim]
            vector_long: 长程向量嵌入 [num_nodes, 3, hidden_dim]
            fragment_ids: BRICS片段ID [num_nodes]
            atom_types: 可选的原子类型 [num_nodes]
            
        Returns:
            consistency_loss: 一致性损失
        """
        # 必须提供fragment_ids
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_long.device)
        
        # 组合短程和长程信息
        scalar_combined = scalar_short * self.short_long_ratio + scalar_long * (1 - self.short_long_ratio)
        
        # 提取增强的不变特征
        scalar_invariant = self.scalar_encoder(scalar_combined)
        
        # 特征分解：将特征分解为片段共享特征和原子独特特征
        decomposed_features = self.feature_decomposer(scalar_invariant)
        shared_features, unique_features = torch.chunk(decomposed_features, 2, dim=1)
        
        # 添加L2正则化
        l2_reg = 0.01 * (torch.norm(shared_features) + torch.norm(unique_features))
        
        # 初始化损失计算
        total_loss = 0
        valid_count = 0
        
        # 获取唯一片段ID
        unique_fragments = torch.unique(fragment_ids)
        
        # 针对每个片段计算损失
        for frag_id in unique_fragments:
            # 获取当前片段的原子
            frag_mask = (fragment_ids == frag_id)
            if torch.sum(frag_mask) <= 1:
                continue
                
            # --- 共享特征一致性 ---
            # 提取片段共享特征
            frag_shared = shared_features[frag_mask]
            
            # 方差损失 - 对共享特征应用
            shared_mean = torch.mean(frag_shared, dim=0, keepdim=True)
            shared_variance = torch.mean(torch.sum((frag_shared - shared_mean)**2, dim=1))
            
            # 相似度损失 - 对共享特征应用，目标降低
            shared_norm = F.normalize(frag_shared, p=2, dim=1)
            shared_sim = torch.matmul(shared_norm, shared_norm.transpose(0, 1))
            sim_mask = torch.triu(torch.ones_like(shared_sim), diagonal=1).bool()
            
            # --- 独特特征保持 ---
            # 提取片段独特特征
            frag_unique = unique_features[frag_mask]
            
            # 独特性损失 - 确保原子表示保持一定差异
            unique_norm = F.normalize(frag_unique, p=2, dim=1)
            unique_sim = torch.matmul(unique_norm, unique_norm.transpose(0, 1))
            
            # 计算原子类型一致性掩码 (相同类型原子可以更相似)
            if atom_types is not None:
                frag_atom_types = atom_types[frag_mask]
                type_consistency = (frag_atom_types.unsqueeze(1) == frag_atom_types.unsqueeze(0)).float()
                # 对相同类型原子，允许中等相似度；不同类型原子应更不同
                atom_type_targets = 0.3 * type_consistency + 0.1 * (1 - type_consistency)
            else:
                # 没有原子类型信息时，使用低目标相似度
                atom_type_targets = 0.2 * torch.ones_like(unique_sim)
            
            if torch.sum(sim_mask) > 0:
                # 共享特征相似度损失
                shared_target = self.brics_consistency_strength * torch.ones_like(shared_sim)[sim_mask]
                shared_sim_loss = F.mse_loss(shared_sim[sim_mask], shared_target)
                
                # 独特特征差异性损失
                unique_target = atom_type_targets[sim_mask]
                unique_sim_loss = F.mse_loss(unique_sim[sim_mask], unique_target)
                
                # 组合共享特征一致性损失
                shared_loss = shared_variance + shared_sim_loss
                
                # --- 向量特征一致性 ---
                vector_loss_frag = 0
                if self.vector_weight > 0 and vector_short is not None and vector_long is not None:
                    # 组合短程和长程向量信息
                    vector_combined = vector_short * self.short_long_ratio + vector_long * (1 - self.short_long_ratio)
                    vector_invariant = self.vector_encoder(vector_combined)
                    
                    # 提取片段向量特征
                    frag_vector = vector_invariant[frag_mask]
                    
                    # 向量方差损失 - 但允许更大变化
                    vector_mean = torch.mean(frag_vector, dim=0, keepdim=True)
                    vector_variance = torch.mean(torch.sum(torch.sum((frag_vector - vector_mean)**2, dim=2), dim=1))
                    
                    # 向量方向一致性 - 降低目标相似度
                    vector_dir_loss = 0
                    for dim in range(3):
                        dim_vector = frag_vector[:, dim, :]
                        dim_norm = F.normalize(dim_vector, p=2, dim=1)
                        dim_sim = torch.matmul(dim_norm, dim_norm.transpose(0, 1))
                        
                        # 使用不同的目标相似度 (较低)
                        vector_target = (self.brics_consistency_strength - 0.2) * torch.ones_like(dim_sim)[sim_mask]
                        dim_sim_loss = F.mse_loss(dim_sim[sim_mask], vector_target)
                        vector_dir_loss += dim_sim_loss
                    
                    vector_dir_loss = vector_dir_loss / 3
                    vector_loss_frag = vector_variance + vector_dir_loss
                
                # 最终损失 - 平衡共享特征一致性和原子独特性
                frag_loss = (1 - self.atom_identity_weight) * unique_sim_loss + \
                            self.atom_identity_weight * shared_loss + \
                            self.vector_weight * vector_loss_frag + \
                            l2_reg  # 添加L2正则化
                
                total_loss += frag_loss
                valid_count += 1
        
        # 避免除零错误
        if valid_count > 0:
            final_loss = total_loss / valid_count
            return self.consistency_factor * final_loss
        else:
            return torch.tensor(0.0, device=scalar_long.device)


class BalancedConformationalConsistencyLoss_group(nn.Module):
    """
    平衡型构象一致性损失，使用group级别的特征处理提高效率
    """
    def __init__(self, hidden_channels, consistency_factor=0.1, vector_weight=0.2, 
                short_long_ratio=0.3, brics_consistency_strength=0.5,
                atom_identity_weight=0.5):
        super().__init__()
        self.consistency_factor = consistency_factor
        self.vector_weight = vector_weight
        self.short_long_ratio = short_long_ratio
        self.brics_consistency_strength = brics_consistency_strength
        self.atom_identity_weight = atom_identity_weight
        

        self.feature_decomposer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels*2, hidden_channels*2)
        )

        self.vector_decomposer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2, bias=False),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels*2, hidden_channels*2, bias=False)
        )

    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, fragment_ids, atom_types=None):
        """
        计算平衡型构象一致性损失，使用group级别的特征处理
        
        Args:
            scalar_short: 短程标量嵌入 [num_nodes, hidden_dim]
            scalar_long: 长程标量嵌入 [num_nodes, hidden_dim]
            vector_short: 短程向量嵌入 [num_nodes, 3, hidden_dim]
            vector_long: 长程向量嵌入 [num_nodes, 3, hidden_dim]
            fragment_ids: BRICS片段ID [num_nodes]
            atom_types: 可选的原子类型 [num_nodes]
            
        Returns:
            consistency_loss: 一致性损失
        """
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_long.device)
        scalar_combined = scalar_short * self.short_long_ratio + scalar_long * (1 - self.short_long_ratio)
        # scalar_combined = scalar_short + scalar_long

        # 统一为DualGatingMoE的连接方式
        # scalar_combined = torch.cat([scalar_short, scalar_long], dim=-1)  # [num_nodes, hidden*2]
        
        # 特征分解：将特征分解为片段共享特征和原子独特特征
        decomposed_features = self.feature_decomposer(scalar_combined)
        shared_features, unique_features = torch.chunk(decomposed_features, 2, dim=1)
        
        # 计算group级别的特征
        group_shared = scatter(shared_features, fragment_ids, dim=0, reduce='mean')
        group_unique = scatter(unique_features, fragment_ids, dim=0, reduce='mean')
        

        
        # 计算group级别的相似度矩阵
        group_shared_norm = F.normalize(group_shared, p=2, dim=1)
        group_shared_sim = torch.matmul(group_shared_norm, group_shared_norm.transpose(0, 1))
        group_mask = torch.triu(torch.ones_like(group_shared_sim), diagonal=1).bool()
        
        # Group级别的共享特征损失
        group_shared_target = self.brics_consistency_strength * torch.ones_like(group_shared_sim)[group_mask]
        group_shared_loss = F.mse_loss(group_shared_sim[group_mask], group_shared_target)
        
        # Group级别的独特性损失
        group_unique_norm = F.normalize(group_unique, p=2, dim=1)
        group_unique_sim = torch.matmul(group_unique_norm, group_unique_norm.transpose(0, 1))
        group_unique_target = 0.2* torch.ones_like(group_unique_sim)[group_mask]
        group_unique_loss = F.mse_loss(group_unique_sim[group_mask], group_unique_target)
        
        # Group级别的向量损失
        group_vector_loss = 0
        # 优化后的向量损失处理
        if self.vector_weight > 0:
            vector_combined = vector_short
            # vector_combined = vector_short + vector_long
            # 统一为DualGatingMoE的连接方式
            # vector_combined = torch.cat([vector_short, vector_long], dim=-1)  # [num_nodes, 3, hidden*2]
            batch_size, dims, hidden_dim = vector_combined.shape  # [num_nodes, 3, hidden*2]
            
            # 方法一：保持向量空间结构的处理
            # 单独处理每个空间维度，保持向量的结构信息
            vector_shared_all_dims = []
            vector_unique_all_dims = []
            
            for dim_idx in range(dims):
                # 提取当前维度 [num_nodes, hidden*2]
                current_dim_vector = vector_combined[:, dim_idx, :]
                
                # 对当前维度进行特征分解
                decomposed_vector = self.vector_decomposer(current_dim_vector)
                vector_shared_dim, vector_unique_dim = torch.chunk(decomposed_vector, 2, dim=1)
                
                # 计算group级别的向量特征
                group_vector_shared_dim = scatter(vector_shared_dim, fragment_ids, dim=0, reduce='mean')
                group_vector_unique_dim = scatter(vector_unique_dim, fragment_ids, dim=0, reduce='mean')
                
                vector_shared_all_dims.append(group_vector_shared_dim)
                vector_unique_all_dims.append(group_vector_unique_dim)
            
            # 堆叠所有空间维度的结果 [3, num_groups, hidden_dim]
            vector_shared_stacked = torch.stack(vector_shared_all_dims, dim=0)
            vector_unique_stacked = torch.stack(vector_unique_all_dims, dim=0)
            
            # 简单相加而不是加权 - 直接对三个维度求平均
            weighted_vector_shared = torch.mean(vector_shared_stacked, dim=0)  # [num_groups, hidden_dim]
            weighted_vector_unique = torch.mean(vector_unique_stacked, dim=0)  # [num_groups, hidden_dim]
            
            # 计算共享特征相似度
            vector_shared_norm = F.normalize(weighted_vector_shared, p=2, dim=1)
            vector_shared_sim = torch.matmul(vector_shared_norm, vector_shared_norm.transpose(0, 1))
            vector_shared_target = self.brics_consistency_strength * torch.ones_like(vector_shared_sim)[group_mask]
            vector_shared_loss = F.mse_loss(vector_shared_sim[group_mask], vector_shared_target)
            
            # 计算独特特征相似度
            vector_unique_norm = F.normalize(weighted_vector_unique, p=2, dim=1)
            vector_unique_sim = torch.matmul(vector_unique_norm, vector_unique_norm.transpose(0, 1))
            vector_unique_target = 0.1 * torch.ones_like(vector_unique_sim)[group_mask]
            vector_unique_loss = F.mse_loss(vector_unique_sim[group_mask], vector_unique_target)
            
            # 组合向量损失
            group_vector_loss = (1 - self.atom_identity_weight) * vector_unique_loss + \
                                self.atom_identity_weight * vector_shared_loss
        
        # 组合所有损失
        total_loss = (1 - self.atom_identity_weight) * group_unique_loss + \
                    self.atom_identity_weight * group_shared_loss + \
                    group_vector_loss

        return total_loss

# 在BalancedConformationalConsistencyLoss_group类后添加优化版本
class OptimizedConformationalConsistencyLoss(nn.Module):
    """
    优化版构象一致性损失：解决原版本的负效果问题
    
    主要改进：
    1. 渐进式目标设定，避免过度强制约束
    2. 自适应权重平衡，减少超参数敏感性  
    3. 更温和的特征分解策略
    4. 简化的向量损失计算
    5. 添加损失监控和诊断功能
    """
    def __init__(self, hidden_channels, 
                 consistency_factor=0.05,  # 降低整体强度
                 vector_weight=0.1,        # 降低向量权重
                 short_long_ratio=0.5,     # 更平衡的短长程组合
                 adaptive_targets=True,    # 启用自适应目标
                 temperature=1.0,          # 软化相似度计算
                 min_fragment_size=2):     # 最小片段大小
        super().__init__()
        self.consistency_factor = consistency_factor
        self.vector_weight = vector_weight
        self.short_long_ratio = short_long_ratio
        self.adaptive_targets = adaptive_targets
        self.temperature = temperature
        self.min_fragment_size = min_fragment_size
        
        # 更温和的特征投影
        self.feature_projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),  # 添加标准化
            nn.SiLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 简化的向量投影
        self.vector_projector = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
        # 自适应权重网络
        if adaptive_targets:
            self.target_predictor = nn.Sequential(
                nn.Linear(hidden_channels, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 输出0-1之间的目标相似度
            )
        
        # 损失统计
        self.loss_stats = {
            'scalar_loss': [],
            'vector_loss': [],
            'num_valid_fragments': [],
            'avg_fragment_size': []
        }

    def compute_adaptive_targets(self, group_features):
        """计算自适应的目标相似度"""
        if not self.adaptive_targets:
            return 0.5  # 默认中等相似度目标
            
        # 基于特征的复杂度预测目标相似度
        targets = self.target_predictor(group_features)  # [num_groups, 1]
        targets = torch.clamp(targets.squeeze(-1), 0.2, 0.8)  # [num_groups]
        
        # 为了处理group间的相似度，我们取所有group的平均目标相似度
        # 这样所有group对都使用相同的目标相似度
        avg_target = torch.mean(targets)
        return avg_target  # 返回标量而不是向量

    def soft_similarity_loss(self, features, targets, mask):
        """使用温度参数的软相似度损失"""
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, device=features.device)
            
        # 计算相似度矩阵
        norm_features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(norm_features, norm_features.transpose(0, 1))
        
        # 应用温度参数软化
        sim_matrix = sim_matrix / self.temperature
        
        # 提取上三角部分
        sim_values = sim_matrix[mask]
        
        # 处理目标值的维度问题
        if targets.dim() == 1:
            # 如果targets是1D张量，需要扩展为2D矩阵
            num_features = features.size(0)
            target_matrix = targets.unsqueeze(1).expand(num_features, num_features)
            target_values = target_matrix[mask]
        elif targets.dim() == 2:
            # 如果targets已经是2D张量
            target_values = targets[mask]
        else:
            # 如果targets是标量
            target_values = targets
        
        # 使用Huber损失而不是MSE，对异常值更鲁棒
        loss = F.huber_loss(sim_values, target_values, reduction='mean', delta=0.1)
        return loss

    def compute_intra_fragment_variance(self, features, fragment_ids):
        """计算片段内方差损失，鼓励片段内一致性"""
        total_variance = 0
        valid_count = 0
        
        unique_fragments = torch.unique(fragment_ids)
        for frag_id in unique_fragments:
            frag_mask = (fragment_ids == frag_id)
            if torch.sum(frag_mask) < self.min_fragment_size:
                continue
                
            frag_features = features[frag_mask]
            # 计算方差，但使用更温和的损失
            frag_mean = torch.mean(frag_features, dim=0, keepdim=True)
            variance = torch.mean(torch.sum((frag_features - frag_mean)**2, dim=1))
            
            # 使用平方根减少方差损失的强度
            total_variance += torch.sqrt(variance + 1e-8)
            valid_count += 1
            
        return total_variance / valid_count if valid_count > 0 else torch.tensor(0.0, device=features.device)

    def forward(self, scalar_short, scalar_long, vector_short, vector_long, fragment_ids, atom_types=None):
        """
        优化版构象一致性损失计算
        """
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_short.device)
        
        # 更平衡的特征组合
        scalar_combined = scalar_short * self.short_long_ratio + scalar_long * (1 - self.short_long_ratio)
        
        # 温和的特征投影
        scalar_projected = self.feature_projector(scalar_combined)
        
        # 计算group级别特征
        group_features = scatter(scalar_projected, fragment_ids, dim=0, reduce='mean')
        
        # 统计信息
        unique_fragments = torch.unique(fragment_ids)
        valid_fragments = 0
        total_fragment_size = 0
        
        # === 标量特征损失 ===
        scalar_loss = 0
        
        # 1. 片段内方差损失（鼓励片段内一致性）
        intra_variance = self.compute_intra_fragment_variance(scalar_projected, fragment_ids)
        
        # 2. Group级别的相似度损失
        if len(group_features) > 1:
            # 自适应目标
            if self.adaptive_targets:
                similarity_targets = self.compute_adaptive_targets(group_features)
                # similarity_targets 现在是 [num_groups] 的1D张量
                group_mask = torch.triu(torch.ones(len(group_features), len(group_features), device=group_features.device), diagonal=1).bool()
            else:
                similarity_targets = 0.5  # 固定中等目标
                group_mask = torch.triu(torch.ones(len(group_features), len(group_features), device=group_features.device), diagonal=1).bool()
            
            group_sim_loss = self.soft_similarity_loss(group_features, similarity_targets, group_mask)
            scalar_loss = 0.3 * intra_variance + 0.7 * group_sim_loss
        else:
            scalar_loss = intra_variance
            
        # === 向量特征损失（简化版本）===
        vector_loss = 0
        if self.vector_weight > 0 and vector_short is not None and vector_long is not None:
            vector_combined = vector_short * self.short_long_ratio + vector_long * (1 - self.short_long_ratio)
            # 简化：只计算向量幅度的一致性，忽略复杂的方向约束
            vector_norms = torch.norm(vector_combined, dim=2)  # [num_nodes, 3]
            vector_magnitude_features = torch.mean(vector_norms, dim=1)  # [num_nodes]
            # 投影到标量空间进行处理，修复维度问题
            hidden_channels = scalar_projected.size(-1)  # 获取实际的hidden_channels
            vector_expanded = vector_magnitude_features.unsqueeze(-1).expand(-1, hidden_channels)
            vector_projected = self.vector_projector(vector_expanded)
            
            # 计算向量特征的片段内方差
            vector_intra_variance = self.compute_intra_fragment_variance(vector_projected, fragment_ids)
            vector_loss = vector_intra_variance
        
        # === 组合损失 ===
        total_loss = scalar_loss + self.vector_weight * vector_loss
        
        # 记录统计信息用于调试
        if self.training:
            self.loss_stats['scalar_loss'].append(scalar_loss.item())
            self.loss_stats['vector_loss'].append(vector_loss.item() if isinstance(vector_loss, torch.Tensor) else vector_loss)
            self.loss_stats['num_valid_fragments'].append(len(unique_fragments))
            
            # 定期清理统计信息
            if len(self.loss_stats['scalar_loss']) > 1000:
                for key in self.loss_stats:
                    self.loss_stats[key] = self.loss_stats[key][-100:]
        
        return self.consistency_factor * total_loss
    
    def get_loss_statistics(self):
        """获取损失统计信息，用于分析和调试"""
        if not self.loss_stats['scalar_loss']:
            return None
            
        import numpy as np
        stats = {}
        for key, values in self.loss_stats.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'recent': values[-10:] if len(values) >= 10 else values
                }
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        for key in self.loss_stats:
            self.loss_stats[key] = []

# 添加新的兼容MoE的一致性损失类
class MoECompatibleConsistencyLoss(nn.Module):
    """
    专为配合DualGatingMoE设计的构象一致性损失
    
    主要特点：
    1. 轻量级约束，不干扰MoE的门控机制
    2. 逐步应用的渐进式约束
    3. 直接使用短程特征而非混合特征
    4. 简化的向量处理，只关注幅度
    5. 支持片段大小过滤
    6. 自动更新epoch，无需外部调用
    """
    def __init__(self, 
                 hidden_channels, 
                 consistency_factor=0.03,      # 更低的约束强度
                 vector_weight=0.05,           # 最小的向量权重
                 min_fragment_size=3,          # 最小片段大小
                 progressive_schedule=True,    # 启用渐进式约束
                 warmup_epochs=15,             # 更长的预热期
                 initial_strength=0.05):       # 更低的初始强度
        super().__init__()
        self.hidden_channels = hidden_channels
        self.consistency_factor = consistency_factor
        self.vector_weight = vector_weight
        self.min_fragment_size = min_fragment_size
        
        # 渐进式缩放相关参数
        self.progressive_schedule = progressive_schedule
        self.warmup_epochs = warmup_epochs
        self.initial_strength = initial_strength
        self.current_epoch = 0
        self._last_step = -1  # 用于追踪最后一次更新的步骤
        self.intermolecular_weight = 0.2
        # 轻量级特征投影 - 只处理短程特征
        self.feature_projector = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.05)  # 更低的dropout
        )
        
    
    def update_epoch(self, epoch):
        """更新当前训练轮次 - 仅供外部手动调用"""
        self.current_epoch = epoch
        
    def _auto_update_epoch(self):
        """自动更新epoch - 基于训练步骤"""
        # 获取当前全局步骤
        if not hasattr(self, '_training_step'):
            self._training_step = 0
        
        # 增加步骤计数器
        self._training_step += 1
        
        # 每100步检查一次是否需要更新epoch
        if self._training_step % 100 == 0:
            # 估算当前epoch (根据典型的每个epoch步数)
            estimated_epoch = self._training_step // 1000  # 假设每个epoch约1000步
            if estimated_epoch > self.current_epoch:
                self.current_epoch = estimated_epoch
                # print(f"[自动更新] 构象一致性损失轮次更新为: {self.current_epoch}")
        
    def get_scaling_factor(self):
        """计算渐进式缩放因子"""
        # 自动更新当前epoch
        if self.training:
            self._auto_update_epoch()
            
        if not self.progressive_schedule:
            return 1.0
            
        # 从initial_strength到1.0的渐进增长
        scale = self.initial_strength + (1.0 - self.initial_strength) * min(1.0, self.current_epoch / self.warmup_epochs)
        return scale
    
    def compute_fragment_consistency(self, features, fragment_ids, valid_mask=None):
        """计算片段内特征的一致性损失"""
        if valid_mask is not None:
            # 过滤特征和片段ID
            features = features[valid_mask]
            fragment_ids = fragment_ids[valid_mask]
        
        # 计算片段级特征 (聚合)
        group_features = scatter(features, fragment_ids, dim=0, reduce='mean')
        
        # 计算每个原子相对于其片段平均特征的偏差
        atom_deviations = []
        fragment_sizes = []
        unique_fragments = torch.unique(fragment_ids)
        
        for frag_id in unique_fragments:
            # 获取当前片段的原子
            frag_mask = (fragment_ids == frag_id)
            frag_size = torch.sum(frag_mask)
            
            if frag_size < self.min_fragment_size:
                continue
                
            # 获取片段原子特征和片段平均特征
            frag_atoms = features[frag_mask]
            frag_mean = group_features[frag_id]
            
            # 计算L2归一化后的余弦相似度
            frag_atoms_norm = F.normalize(frag_atoms, p=2, dim=1)
            frag_mean_norm = F.normalize(frag_mean.unsqueeze(0), p=2, dim=1)
            
            # 计算相似度 (值越高越相似)
            similarity = torch.matmul(frag_atoms_norm, frag_mean_norm.transpose(0, 1))
            
            # 转换为偏差 (值越低越相似)
            deviation = 1.0 - similarity
            
            # 收集结果
            atom_deviations.append(deviation.mean())
            fragment_sizes.append(frag_size)
        
        if not atom_deviations:
            return torch.tensor(0.0, device=features.device)
            
        # 计算加权平均偏差，权重与片段大小成正比
        weights = torch.tensor(fragment_sizes, device=features.device).float()
        weights = weights / weights.sum()
        
        weighted_loss = torch.sum(torch.stack(atom_deviations) * weights)
        return weighted_loss
    def compute_interfragment_differentiation(self, scalar_features, vector_features, fragment_ids):
        """计算不同片段之间的特征差异性损失 - 减小不同片段间的相似度
        同时使用短程标量和向量特征建模不同BRICS片段之间的作用力
        
        Args:
            scalar_features: 短程标量特征 [num_atoms, hidden_dim]
            vector_features: 短程向量特征的幅度 [num_atoms, 1]
            fragment_ids: BRICS片段ID [num_atoms]
        """
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_features.device)
            
        # 计算标量片段级特征
        scalar_fragment_features = scatter(scalar_features, fragment_ids, dim=0, reduce='mean')
        
        # 计算向量片段级特征（如果提供）
        if vector_features is not None:
            vector_fragment_features = scatter(vector_features, fragment_ids, dim=0, reduce='mean')
            # 将向量特征与标量特征连接起来，形成组合特征
            combined_fragment_features = torch.cat([scalar_fragment_features, vector_fragment_features], dim=1)
        else:
            combined_fragment_features = scalar_fragment_features
        
        # 计算片段间特征的相似度矩阵
        fragment_features_norm = F.normalize(combined_fragment_features, p=2, dim=1)
        similarity_matrix = torch.matmul(fragment_features_norm, fragment_features_norm.transpose(0, 1))
        
        # 只关注不同片段之间的相似度 (非对角元素)
        mask = torch.ones_like(similarity_matrix) - torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        
        # 计算损失：让不同片段之间的相似度接近0（减小相似度）
        interfragment_loss = torch.sum(similarity_matrix * mask) / (torch.sum(mask) + 1e-6)
        
        return interfragment_loss
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, fragment_ids, atom_types=None):
        """计算MoE兼容的构象一致性损失"""
        # 参数检查
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_short.device)
        
        # 获取渐进式缩放因子
        scaling_factor = self.get_scaling_factor()
        if scaling_factor < 0.001:  # 如果几乎为零，直接返回
            return torch.tensor(0.0, device=scalar_short.device)
        
        # 创建有效片段掩码
        valid_mask = None
        if self.min_fragment_size > 1:
            valid_fragments = []
            for frag_id in torch.unique(fragment_ids):
                frag_size = torch.sum(fragment_ids == frag_id)
                if frag_size >= self.min_fragment_size:
                    valid_fragments.append(frag_id.item())
            
            if not valid_fragments:
                return torch.tensor(0.0, device=scalar_short.device)
                
            valid_mask = torch.zeros_like(fragment_ids, dtype=torch.bool)
            for frag_id in valid_fragments:
                valid_mask |= (fragment_ids == frag_id)
                
            if torch.sum(valid_mask) < self.min_fragment_size:
                return torch.tensor(0.0, device=scalar_short.device)
        
        # 重要: 只使用短程特征，不进行混合，避免干扰MoE
        scalar_features = self.feature_projector(scalar_short)
        
        # 计算标量特征的一致性
        scalar_loss = self.compute_fragment_consistency(scalar_features, fragment_ids, valid_mask)
        
        # 计算向量特征的一致性 (如果需要)
        vector_loss = 0.0
        vector_magnitudes = None
        if self.vector_weight > 0 and vector_short is not None:
            # 简化向量处理: 只计算向量大小，忽略方向
            vector_magnitudes = torch.norm(vector_short, dim=2).mean(dim=1)  # [batch_size]
            vector_loss = self.compute_fragment_consistency(
                vector_magnitudes.unsqueeze(1), fragment_ids, valid_mask
            )
        
        # 计算最终损失
        # total_loss = scalar_loss + self.vector_weight * vector_loss



        # 3. 计算片段间差异性损失 - 同时使用短程标量和向量特征
        interfragment_loss = 0.0
        if self.intermolecular_weight > 0:
            # 同时使用标量和向量特征计算片段间差异性
            interfragment_loss = self.compute_interfragment_differentiation(
                scalar_features, 
                vector_magnitudes,  # 可能为None
                fragment_ids
            )
        
        # 组合所有损失
        # 片段内部特征应当相似 (低intramolecular_loss)，片段之间特征应当不同 (高interfragment_loss)
        total_loss = (scalar_loss + self.vector_weight * vector_loss + 
                      self.intermolecular_weight * interfragment_loss)


        
        # 应用约束强度和渐进缩放
        return self.consistency_factor * total_loss * scaling_factor

class Visnorm_shared_LSRMNorm2_2branchSerial(nn.Module):
    def __init__(self,regress_forces = True,
                 hidden_channels=128,
                 num_layers=6,
                 num_rbf=50,
                 rbf_type="expnorm",
                 trainable_rbf=True,
                 neighbor_embedding=True,
                 short_cutoff_upper=10,
                 long_cutoff_upper=10,
                 mean = None,
                 std = None,
                 atom_ref = None,
                 max_z=100,
                 group_center='center_of_mass',
                 tf_writer = None,
                 **kwargs):
        super().__init__()
        # self.embedding_long = nn.Embedding(max_z, hidden_channels)
        # self.logger = {f'{i}': {'dx':[], 'vec': } for i in range(1, num_layers + 1)}
        self.hidden_channels = hidden_channels
        self.regress_forces = regress_forces
        self.num_layers = num_layers
        self.group_center = group_center
        self.tf_writer = tf_writer
        self.t = 0
        
        # 添加构象一致性损失模块
        self.use_conformer_loss = kwargs["config"].get("use_conformer_loss", False)
        if self.use_conformer_loss:
            # 根据配置选择使用的一致性损失类型
            
            # 使用短程力建模分子间作用力一致性损失
            # self.conformer_loss = ShortRangeIntermolecularConsistencyLoss(
            #     hidden_channels=hidden_channels,
            #     consistency_factor=kwargs["config"].get("consistency_factor", 0.03),
            #     vector_weight=kwargs["config"].get("vector_weight", 0.1),
            #     min_fragment_size=kwargs["config"].get("min_fragment_size", 3),
            #     progressive_schedule=kwargs["config"].get("progressive_schedule", True),
            #     warmup_epochs=kwargs["config"].get("warmup_epochs", 15),
            #     initial_strength=kwargs["config"].get("initial_strength", 0.05),
            #     intermolecular_weight=kwargs["config"].get("intermolecular_weight", 0.2)
            # )
            # print("[INFO] 启用短程力分子间作用力一致性损失:", 
            #     "约束强度:", kwargs["config"].get("consistency_factor", 0.03),
            #     "向量权重:", kwargs["config"].get("vector_weight", 0.1),
            #     "分子间权重:", kwargs["config"].get("intermolecular_weight", 0.2),
            #     )
            #     使用与MoE兼容的一致性损失
            self.conformer_loss = MoECompatibleConsistencyLoss(
                hidden_channels=hidden_channels,
                consistency_factor=kwargs["config"].get("consistency_factor", 0.03),
                vector_weight=kwargs["config"].get("vector_weight", 0.05),
                min_fragment_size=kwargs["config"].get("min_fragment_size", 3),
                progressive_schedule=kwargs["config"].get("progressive_schedule", True),
                warmup_epochs=kwargs["config"].get("warmup_epochs", 15),
                initial_strength=kwargs["config"].get("initial_strength", 0.05)
            )
            print("[INFO] 启用MoE兼容的构象一致性损失:", 
                "约束强度:", kwargs["config"].get("consistency_factor", 0.03),
                "向量权重:", kwargs["config"].get("vector_weight", 0.05),
                "最小片段大小:", kwargs["config"].get("min_fragment_size", 3),
                "渐进式约束:", kwargs["config"].get("progressive_schedule", True),
                "预热轮次:", kwargs["config"].get("warmup_epochs", 15))

        # Add Mixture of Experts module
        self.use_moe = True
        if self.use_moe:
            self.moe = DualGatingMoE(
                hidden_channels, 
                num_experts=2,
                dropout=0.1
            )
        print("use_moe")
        print(self.use_moe)

        self.node_fea_init = Node_Edge_Fea_Init(
                                    max_z = max_z,
                                    rbf_type=rbf_type,
                                    num_rbf = num_rbf,
                                    trainable_rbf = trainable_rbf,
                                    hidden_channels = hidden_channels,
                                    cutoff_lower = 0,
                                    cutoff_upper = short_cutoff_upper,
                                    neighbor_embedding = neighbor_embedding)
        self.mlp_node_fea = nn.Linear(hidden_channels,2*hidden_channels)
        self.mlp_node_vec_fea = nn.Linear(hidden_channels,2*hidden_channels, bias = False)
        
        self.edge_fea_init = Edge_Feat_Init(rbf_type = rbf_type,
                num_rbf = num_rbf,
                trainable_rbf = trainable_rbf,
                hidden_channels = hidden_channels,
                cutoff_lower = 0,
                cutoff_upper = short_cutoff_upper)
        


        # 检查是否启用夹角特征
        self.use_angle_features = True
        angle_feature_dim = kwargs["config"].get("angle_feature_dim", 12)
        
        if self.use_angle_features:
            self.bipartite_edge_fea_init = BipartiteEdgeWithAngleFeatures(
                edge_index=edge_index,
                rbf_type = rbf_type,
                num_rbf = num_rbf,
                trainable_rbf = trainable_rbf,
                hidden_channels = hidden_channels,
                cutoff_lower = 0,
                cutoff_upper = long_cutoff_upper,
                use_angle_features = True,
                angle_feature_dim = angle_feature_dim
            )
            print(f"[INFO] 启用边夹角特征提取，特征维度: {angle_feature_dim}")
        else:
            self.bipartite_edge_fea_init = Bipartite_Edge_Feat_Init(
                rbf_type = rbf_type,
                num_rbf = num_rbf,
                trainable_rbf = trainable_rbf,
                hidden_channels = hidden_channels,
                cutoff_lower = 0,
                cutoff_upper = long_cutoff_upper
            )

        self.long_cutoff_upper = long_cutoff_upper
        
        self.visnet_att0 = nn.ModuleList()
        self.longshortinteract_models = nn.ModuleList()
        for _ in range(self.num_layers):
            self.visnet_att0.append(EquivariantMultiHeadAttention(
                                        hidden_channels,
                                        distance_influence = "both",
                                        num_heads = 8,
                                        activation = "silu",
                                        attn_activation = "silu",
                                        cutoff_lower = 0,
                                        cutoff_upper = short_cutoff_upper,
                                        last_layer=False,
                                    ))

        config = kwargs["config"]
        self.long_num_layers = config["long_num_layers"]
        for i in range(self.long_num_layers):
            # self.longshortinteract_models.append(AdaptiveLongShortInteractModel(
            #     hidden_channels, 
            #     num_gaussians=50,
            #     cutoff=self.long_cutoff_upper,
            #     norm=True,
            #     act="silu",
            #     num_heads=8,
            #     p = config["dropout"]
            # ))


            # self.longshortinteract_models.append(LongShortIneractModel_dis_direct_vector2_drop(hidden_channels, num_gaussians=50, 
            #                                                                                  cutoff=self.long_cutoff_upper,norm=True, max_group_num = 3,act = "silu",num_heads=8,
            #                                                                                  p = config["dropout"]))
            self.longshortinteract_models.append(ImprovedLongShortInteractModel(hidden_channels, num_gaussians=50, 
                                                                                             cutoff=self.long_cutoff_upper,norm=True, max_group_num = 3,act = "silu",num_heads=8,
                                                                                             p = config["dropout"]))
            

            # self.group_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = None))
            # self.node_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = None))
            # self.node_vec_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = 1))
            # self.edge_embed_abn.append(All_Batch_Norm(normalized_shape = hidden_channels,L2_norm_dim = None))
        self.out_norm1 = nn.LayerNorm(hidden_channels)
        self.out_norm2 = nn.LayerNorm(hidden_channels)
        self.out_energy = OutputNet(hidden_channels*2, act = 'silu', dipole = False, mean = mean, std = std, atomref = atom_ref, scale = None)
        
    @conditional_grad(torch.enable_grad())
    def forward(self,
                data,
                *args,
                **kwargs
                ):
        
        '''
        data.grouping_graph # Grouping graph (intra group complete graph; inter group disconnected)
        data.interaction_graph #Bipartite graph, [0] node, [1] group
        '''
        # if self.debug:
        # torch.autograd.set_detect_anomaly(True)
        if self.regress_forces:
            data.pos.requires_grad_(True)
        data.edge_index =  remove_self_loops(data.edge_index)[0]
        # data.grouping_graph = remove_self_loops(data.grouping_graph)[0]
        z = data.atomic_numbers.long()
        pos = data.pos
        labels = data.labels
        atomic_numbers = data.atomic_numbers

        # node related feature, node-node distance
        if z.dim() == 2:  # if z of shape num_atoms x 1
            z = z.squeeze()  # squeeze to num_atoms
        device = pos.device
        #group related feature
        if self.group_center == 'geometric':
            group_pos = scatter(pos, data.labels, reduce='mean', dim=0)
        elif self.group_center == 'center_of_mass':
            group_pos = scatter(pos * atomic_numbers, labels, reduce='sum', dim=0) /scatter(atomic_numbers,labels, reduce='sum', dim=0)
        else:
            assert(False)
        node_id,group_id = data.interaction_graph[0],data.interaction_graph[1]
        node_group_dis = torch.sqrt(torch.sum((pos[node_id]-group_pos[group_id])**2,dim = 1))
        data.interaction_graph = data.interaction_graph[:,node_group_dis<=self.long_cutoff_upper]
        group_embedding = None
        group_vec = torch.zeros((group_pos.shape[0],3,self.hidden_channels),device = device)
        node_embedding, node_vec, edge_index_short, edge_weight_short, edge_attr_short, edge_vec_short = self.node_fea_init(z,pos,data.edge_index)

        node_embedding_short,node_embedding_long= torch.split(self.mlp_node_fea(node_embedding), self.hidden_channels, dim=-1)
        node_vec_short, node_vec_long= torch.split(self.mlp_node_vec_fea(node_vec), self.hidden_channels, dim=-1)
        
        # 为夹角特征计算预先准备group_vec
        if self.use_angle_features:
            # 使用初始的node_vec来计算group_vec
            initial_group_vec = scatter(node_vec, labels, dim=0, reduce='mean')
            # 重要：使用过滤后的data.interaction_graph来计算夹角特征
            edge_index_bipartite, edge_weight_bipartite, edge_attr_bipartite, edge_vec_bipartite, angle_attr_bipartite = self.bipartite_edge_fea_init(
                data.interaction_graph, pos, group_pos, node_vec, initial_group_vec
            )
        else:
            # 原有的距离特征初始化
            edge_index_bipartite, edge_weight_bipartite, edge_attr_bipartite, edge_vec_bipartite = self.bipartite_edge_fea_init(
                data.interaction_graph, pos, group_pos
            )
            angle_attr_bipartite = None
        for idx  in range(self.num_layers):
            # short term local neighbor
            delta_node_embedding_short, delta_node_vec_short, dedge_attr_short = self.visnet_att0[idx](node_embedding_short, node_vec_short, edge_index_short, edge_weight_short, edge_attr_short, edge_vec_short)
            node_embedding_short = node_embedding_short + delta_node_embedding_short
            node_vec_short = node_vec_short + delta_node_vec_short
            edge_attr_short = edge_attr_short + dedge_attr_short
        if self.long_num_layers!=0:
            node_embedding_long = node_embedding_short
            node_vec_long = node_vec_short
        else:
            node_embedding_long = node_embedding_long*0
            node_vec_long = node_vec_long*0
        for idx  in range(self.long_num_layers):        
            group_embedding = scatter(node_embedding_long, labels, dim=0, reduce = 'mean')
            # group_embedding = self.group_embed_ln[idx](group_embedding)
            group_vec = scatter(node_vec_long, labels, dim=0, reduce = 'mean')
            # Vector Scalar Interaction
            # node group interaction 
            # node_embedding0, group_embedding
            if self.use_angle_features and angle_attr_bipartite is not None:
                # 传递分离的边特征和夹角特征
                delta_node_embedding_long, delta_node_vec_long = self.longshortinteract_models[idx](
                    edge_index = edge_index_bipartite, 
                    node_embedding = node_embedding_long, node_pos = pos, node_vec = node_vec_long,
                    group_embedding = group_embedding, group_pos = group_pos,
                    group_vec = group_vec, edge_attr = edge_attr_bipartite,
                    edge_weight = edge_weight_bipartite, edge_vec = edge_vec_bipartite, 
                    fragment_ids = labels, angle_attr = angle_attr_bipartite
                )
            else:
                # 原有调用方式
                delta_node_embedding_long, delta_node_vec_long = self.longshortinteract_models[idx](
                    edge_index = edge_index_bipartite, 
                    node_embedding = node_embedding_long, node_pos = pos, node_vec = node_vec_long,
                    group_embedding = group_embedding, group_pos = group_pos,
                    group_vec = group_vec, edge_attr = edge_attr_bipartite,
                    edge_weight = edge_weight_bipartite, edge_vec = edge_vec_bipartite, 
                    fragment_ids = labels
                )
            
            # 移除之前的维度检查，因为新模型实现已确保输出维度正确
            node_embedding_long = node_embedding_long + delta_node_embedding_long
            node_vec_long = node_vec_long + delta_node_vec_long
            
        node_embedding_short = self.out_norm1(node_embedding_short)
        node_vec_short = vec_layernorm(node_vec_short, max_min_norm)
        node_embedding_long = self.out_norm2(node_embedding_long)
        node_vec_long = vec_layernorm(node_vec_long, max_min_norm)
        


        # 保存节点嵌入用于计算构象一致性损失
        self._last_node_embedding_long = node_embedding_long
        
        # Apply Mixture of Experts to combine short and long range interactions
        if self.use_moe:
            # Use MoE to dynamically combine short and long range interactions for each atom
            moe_embedding, moe_vec = self.moe(node_embedding_short, node_embedding_long, node_vec_short, node_vec_long, z)
            # node_embedding_combined = moe_embedding
            # node_vec_combined = moe_vec
            # Concatenate the MoE outputs with the original embeddings for richer representation
            node_embedding_combined = torch.cat([moe_embedding, node_embedding_short, node_embedding_long], dim=-1)
            node_vec_combined = torch.cat([moe_vec, node_vec_short, node_vec_long], dim=-1)
        else:
            # Original simple concatenation approach
            node_embedding_combined = torch.cat([node_embedding_short, node_embedding_long], dim=-1)
            node_vec_combined = torch.cat([node_vec_short, node_vec_long], dim=-1)

        # Use the combined embeddings for energy and force prediction
        energy = self.out_energy(node_embedding_combined, node_vec_combined, data)
            
        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True if self.training else False,
                    retain_graph=True if self.training else False
                )[0]
            )
            if torch.any(torch.isnan(energy)):
                assert(False)
            if torch.any(torch.isnan(forces)):
                assert(False)
            
            result = {"energy": energy, "forces": forces}
        else:
            result = {'energy': energy}

        # 计算构象一致性损失（如果启用）
        if self.use_conformer_loss and self.training:
            # 检查使用的是哪种一致性损失类型
            if isinstance(self.conformer_loss, ShortRangeIntermolecularConsistencyLoss):
                # 使用短程力建模分子间作用力一致性损失
                molecule_ids = data.get('molecule_ids', None)
                if molecule_ids is None and hasattr(data, 'batch'):
                    # 如果没有明确的分子ID，可以使用batch信息作为替代
                    molecule_ids = data.batch
                
                consistency_loss = self.conformer_loss(
                    node_embedding_short,     # 短程标量
                    node_embedding_long,      # 长程标量
                    node_vec_short,           # 短程向量
                    node_vec_long,            # 长程向量
                    fragment_ids=labels,      # BRICS片段标签
                    pos=pos,                  # 原子坐标
                    molecule_ids=molecule_ids # 分子ID
                )
            elif isinstance(self.conformer_loss, IntermolecularConsistencyLoss):
                # 使用分子间作用力一致性损失
                # 需要额外的分子ID信息
                molecule_ids = data.get('molecule_ids', None)
                if molecule_ids is None and hasattr(data, 'batch'):
                    # 如果没有明确的分子ID，可以使用batch信息作为替代
                    molecule_ids = data.batch
                
                consistency_loss = self.conformer_loss(
                    node_embedding_short,     # 短程标量
                    node_embedding_long,      # 长程标量
                    node_vec_short,           # 短程向量
                    node_vec_long,            # 长程向量
                    fragment_ids=labels,      # BRICS片段标签
                    pos=pos,                  # 原子坐标
                    molecule_ids=molecule_ids # 分子ID
                )
            elif isinstance(self.conformer_loss, MoECompatibleConsistencyLoss):
                # MoE兼容版本仅使用短程特征，避免干扰MoE功能
                consistency_loss = self.conformer_loss(
                    node_embedding_short,  # 短程标量
                    node_embedding_long,   # 长程标量 (不会使用)
                    node_vec_short,        # 短程向量
                    node_vec_long,         # 长程向量 (不会使用)
                    fragment_ids=labels    # BRICS片段标签
                )
            else:
                # 使用其他一致性损失
                consistency_loss = self.conformer_loss(
                    node_embedding_short,  # 短程标量
                    node_embedding_long,   # 长程标量
                    node_vec_short,        # 短程向量
                    node_vec_long,         # 长程向量
                    fragment_ids=labels    # BRICS片段标签
                )
            result["conformer_loss"] = consistency_loss
                
        return result
        
    def get_node_embeddings(self, data):
        """返回最近计算的节点嵌入，用于外部计算构象一致性"""
        # 确保已经运行过前向传播
        if not hasattr(self, '_last_node_embedding_long'):
            self.forward(data)
            
        return self._last_node_embedding_long

class IntermolecularConsistencyLoss(nn.Module):
    """
    简化版分子间作用力一致性损失
    
    基于MoECompatibleConsistencyLoss，添加对分子间长程作用力的建模
    主要特点：
    1. 保留分子片段内部特征相似性（提高内部一致性）
    2. 添加不同片段之间的特征差异性（减小不同片段间的相似度）
    3. 使用BRICS分割的片段标识建模分子间作用力
    """
    def __init__(self, 
                 hidden_channels, 
                 consistency_factor=0.03,      # 基础约束强度
                 vector_weight=0.05,           # 向量特征权重
                 min_fragment_size=3,          # 最小片段大小
                 progressive_schedule=True,    # 启用渐进式约束
                 warmup_epochs=15,             # 预热轮次
                 initial_strength=0.05,        # 初始强度
                 intermolecular_weight=0.2):   # 片段间差异性权重
        super().__init__()
        self.hidden_channels = hidden_channels
        self.consistency_factor = consistency_factor
        self.vector_weight = vector_weight
        self.min_fragment_size = min_fragment_size
        self.intermolecular_weight = intermolecular_weight
        
        # 渐进式缩放相关参数
        self.progressive_schedule = progressive_schedule
        self.warmup_epochs = warmup_epochs
        self.initial_strength = initial_strength
        self.current_epoch = 0
        
        # 特征投影 - 分别处理短程和长程特征
        self.short_feature_projector = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.05)
        )
        
        self.long_feature_projector = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.05)
        )
    
    def update_epoch(self, epoch):
        """更新当前训练轮次 - 仅供外部手动调用"""
        self.current_epoch = epoch
        
    def _auto_update_epoch(self):
        """自动更新epoch - 基于训练步骤"""
        if not hasattr(self, '_training_step'):
            self._training_step = 0
        
        self._training_step += 1
        
        if self._training_step % 100 == 0:
            estimated_epoch = self._training_step // 1000
            if estimated_epoch > self.current_epoch:
                self.current_epoch = estimated_epoch
        
    def get_scaling_factor(self):
        """计算渐进式缩放因子"""
        if self.training:
            self._auto_update_epoch()
            
        if not self.progressive_schedule:
            return 1.0
            
        scale = self.initial_strength + (1.0 - self.initial_strength) * min(1.0, self.current_epoch / self.warmup_epochs)
        return scale
    
    def compute_fragment_consistency(self, features, fragment_ids, valid_mask=None):
        """计算分子片段内部特征的一致性损失 - 增加内部相似度
        与MoECompatibleConsistencyLoss一致，让分子片段内部特征更相似
        """
        if valid_mask is not None:
            features = features[valid_mask]
            fragment_ids = fragment_ids[valid_mask]
        
        # 计算片段级特征 (聚合)
        group_features = scatter(features, fragment_ids, dim=0, reduce='mean')
        
        # 计算每个原子相对于其片段平均特征的偏差
        atom_deviations = []
        fragment_sizes = []
        unique_fragments = torch.unique(fragment_ids)
        
        for frag_id in unique_fragments:
            frag_mask = (fragment_ids == frag_id)
            frag_size = torch.sum(frag_mask)
            
            if frag_size < self.min_fragment_size:
                continue
                
            # 获取片段原子特征和片段平均特征
            frag_atoms = features[frag_mask]
            frag_mean = group_features[frag_id]
            
            # 计算L2归一化后的余弦相似度
            frag_atoms_norm = F.normalize(frag_atoms, p=2, dim=1)
            frag_mean_norm = F.normalize(frag_mean.unsqueeze(0), p=2, dim=1)
            
            # 计算相似度 (值越高越相似)
            similarity = torch.matmul(frag_atoms_norm, frag_mean_norm.transpose(0, 1))
            
            # 转换为偏差 (值越低越相似)
            deviation = 1.0 - similarity
            
            # 收集结果
            atom_deviations.append(deviation.mean())
            fragment_sizes.append(frag_size)
        
        if not atom_deviations:
            return torch.tensor(0.0, device=features.device)
            
        # 计算加权平均偏差，权重与片段大小成正比
        weights = torch.tensor(fragment_sizes, device=features.device).float()
        weights = weights / weights.sum()
        
        weighted_loss = torch.sum(torch.stack(atom_deviations) * weights)
        return weighted_loss
    
    def compute_interfragment_differentiation(self, features, fragment_ids):
        """计算不同片段之间的特征差异性损失 - 减小不同片段间的相似度
        使用长程特征建模不同BRICS片段之间的作用力
        """
        if fragment_ids is None:
            return torch.tensor(0.0, device=features.device)
            
        # 计算片段级特征
        fragment_features = scatter(features, fragment_ids, dim=0, reduce='mean')
        
        # 计算片段间特征的相似度矩阵
        fragment_features_norm = F.normalize(fragment_features, p=2, dim=1)
        similarity_matrix = torch.matmul(fragment_features_norm, fragment_features_norm.transpose(0, 1))
        
        # 只关注不同片段之间的相似度 (非对角元素)
        mask = torch.ones_like(similarity_matrix) - torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        
        # 计算损失：让不同片段之间的相似度接近0（减小相似度）
        interfragment_loss = torch.sum(similarity_matrix * mask) / (torch.sum(mask) + 1e-6)
        
        return interfragment_loss
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, fragment_ids, 
                pos=None, molecule_ids=None, atom_types=None):
        """计算分子内一致性 + 片段间差异性损失
        
        Args:
            scalar_short: 短程标量特征 [num_atoms, hidden_dim]
            scalar_long: 长程标量特征 [num_atoms, hidden_dim]
            vector_short: 短程向量特征 [num_atoms, 3, hidden_dim]
            vector_long: 长程向量特征 [num_atoms, 3, hidden_dim]
            fragment_ids: BRICS片段ID [num_atoms]
            pos: 原子坐标 [num_atoms, 3]，可选
            molecule_ids: 忽略，保留参数以兼容接口
            atom_types: 原子类型 [num_atoms]，可选
            
        Returns:
            total_loss: 总损失
        """
        # 参数检查
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_short.device)
        
        # 获取渐进式缩放因子
        scaling_factor = self.get_scaling_factor()
        if scaling_factor < 0.001:
            return torch.tensor(0.0, device=scalar_short.device)
        
        # 创建有效片段掩码
        valid_mask = None
        if self.min_fragment_size > 1:
            valid_fragments = []
            for frag_id in torch.unique(fragment_ids):
                frag_size = torch.sum(fragment_ids == frag_id)
                if frag_size >= self.min_fragment_size:
                    valid_fragments.append(frag_id.item())
            
            if not valid_fragments:
                return torch.tensor(0.0, device=scalar_short.device)
                
            valid_mask = torch.zeros_like(fragment_ids, dtype=torch.bool)
            for frag_id in valid_fragments:
                valid_mask |= (fragment_ids == frag_id)
                
            if torch.sum(valid_mask) < self.min_fragment_size:
                return torch.tensor(0.0, device=scalar_short.device)
        
        # 处理短程和长程特征
        short_features = self.short_feature_projector(scalar_short)
        long_features = self.long_feature_projector(scalar_long)
        
        # 1. 计算分子内构象一致性损失 (基于短程特征) - 与MoECompatibleConsistencyLoss一致
        intramolecular_loss = self.compute_fragment_consistency(short_features, fragment_ids, valid_mask)
        
        # 2. 计算向量特征的一致性损失 - 与MoECompatibleConsistencyLoss一致
        vector_loss = 0.0
        if self.vector_weight > 0 and vector_short is not None:
            vector_magnitudes = torch.norm(vector_short, dim=2).mean(dim=1)
            vector_loss = self.compute_fragment_consistency(
                vector_magnitudes.unsqueeze(1), fragment_ids, valid_mask
            )
        
        # 3. 计算片段间差异性损失 (使用长程特征) - 让不同片段之间的特征更不相似
        interfragment_loss = 0.0
        if self.intermolecular_weight > 0:
            # 使用长程特征计算片段间差异性（分子间作用力）
            interfragment_loss = self.compute_interfragment_differentiation(long_features, fragment_ids)
        
        # 组合所有损失
        # 片段内部特征应当相似 (低intramolecular_loss)，片段之间特征应当不同 (高interfragment_loss)
        total_loss = (intramolecular_loss + 
                      self.vector_weight * vector_loss + 
                      self.intermolecular_weight * interfragment_loss)
        
        # 应用约束强度和渐进缩放
        return self.consistency_factor * total_loss * scaling_factor

class ShortRangeIntermolecularConsistencyLoss(nn.Module):
    """
    使用短程力建模分子片段间差异性的一致性损失
    
    主要特点：
    1. 保留分子片段内部特征相似性（提高内部一致性）
    2. 添加不同片段之间的特征差异性（减小不同片段间的相似度）
    3. 使用短程力而非长程力来建模分子间作用力
    4. 渐进式应用约束，避免训练初期过度干扰
    """
    def __init__(self, 
                 hidden_channels, 
                 consistency_factor=0.03,      # 基础约束强度
                 vector_weight=0.05,           # 向量特征权重
                 min_fragment_size=3,          # 最小片段大小
                 progressive_schedule=True,    # 启用渐进式约束
                 warmup_epochs=15,             # 预热轮次
                 initial_strength=0.05,        # 初始强度
                 intermolecular_weight=0.2):   # 片段间差异性权重
        super().__init__()
        self.hidden_channels = hidden_channels
        self.consistency_factor = consistency_factor
        self.vector_weight = vector_weight
        self.min_fragment_size = min_fragment_size
        self.intermolecular_weight = intermolecular_weight
        
        # 渐进式缩放相关参数
        self.progressive_schedule = progressive_schedule
        self.warmup_epochs = warmup_epochs
        self.initial_strength = initial_strength
        self.current_epoch = 0
        
        # 特征投影 - 处理短程标量特征
        self.scalar_projector = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(0.05)
        )
        
        # 特征投影 - 处理短程向量特征
        self.vector_projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Dropout(0.05)
        )
    
    def update_epoch(self, epoch):
        """更新当前训练轮次 - 仅供外部手动调用"""
        self.current_epoch = epoch
        
    def _auto_update_epoch(self):
        """自动更新epoch - 基于训练步骤"""
        if not hasattr(self, '_training_step'):
            self._training_step = 0
        
        self._training_step += 1
        
        if self._training_step % 100 == 0:
            estimated_epoch = self._training_step // 1000
            if estimated_epoch > self.current_epoch:
                self.current_epoch = estimated_epoch
        
    def get_scaling_factor(self):
        """计算渐进式缩放因子"""
        if self.training:
            self._auto_update_epoch()
            
        if not self.progressive_schedule:
            return 1.0
            
        scale = self.initial_strength + (1.0 - self.initial_strength) * min(1.0, self.current_epoch / self.warmup_epochs)
        return scale
    
    def compute_fragment_consistency(self, features, fragment_ids, valid_mask=None):
        """计算分子片段内部特征的一致性损失 - 增加内部相似度
        与MoECompatibleConsistencyLoss一致，让分子片段内部特征更相似
        """
        if valid_mask is not None:
            features = features[valid_mask]
            fragment_ids = fragment_ids[valid_mask]
        
        # 计算片段级特征 (聚合)
        group_features = scatter(features, fragment_ids, dim=0, reduce='mean')
        
        # 计算每个原子相对于其片段平均特征的偏差
        atom_deviations = []
        fragment_sizes = []
        unique_fragments = torch.unique(fragment_ids)
        
        for frag_id in unique_fragments:
            frag_mask = (fragment_ids == frag_id)
            frag_size = torch.sum(frag_mask)
            
            if frag_size < self.min_fragment_size:
                continue
                
            # 获取片段原子特征和片段平均特征
            frag_atoms = features[frag_mask]
            frag_mean = group_features[frag_id]
            
            # 计算L2归一化后的余弦相似度
            frag_atoms_norm = F.normalize(frag_atoms, p=2, dim=1)
            frag_mean_norm = F.normalize(frag_mean.unsqueeze(0), p=2, dim=1)
            
            # 计算相似度 (值越高越相似)
            similarity = torch.matmul(frag_atoms_norm, frag_mean_norm.transpose(0, 1))
            
            # 转换为偏差 (值越低越相似)
            deviation = 1.0 - similarity
            
            # 收集结果
            atom_deviations.append(deviation.mean())
            fragment_sizes.append(frag_size)
        
        if not atom_deviations:
            return torch.tensor(0.0, device=features.device)
            
        # 计算加权平均偏差，权重与片段大小成正比
        weights = torch.tensor(fragment_sizes, device=features.device).float()
        weights = weights / weights.sum()
        
        weighted_loss = torch.sum(torch.stack(atom_deviations) * weights)
        return weighted_loss
    
    def compute_interfragment_differentiation(self, scalar_features, vector_features, fragment_ids):
        """计算不同片段之间的特征差异性损失 - 减小不同片段间的相似度
        同时使用短程标量和向量特征建模不同BRICS片段之间的作用力
        
        Args:
            scalar_features: 短程标量特征 [num_atoms, hidden_dim]
            vector_features: 短程向量特征的幅度 [num_atoms, 1]
            fragment_ids: BRICS片段ID [num_atoms]
        """
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_features.device)
            
        # 计算标量片段级特征
        scalar_fragment_features = scatter(scalar_features, fragment_ids, dim=0, reduce='mean')
        
        # 计算向量片段级特征（如果提供）
        if vector_features is not None:
            vector_fragment_features = scatter(vector_features, fragment_ids, dim=0, reduce='mean')
            # 将向量特征与标量特征连接起来，形成组合特征
            combined_fragment_features = torch.cat([scalar_fragment_features, vector_fragment_features], dim=1)
        else:
            combined_fragment_features = scalar_fragment_features
        
        # 计算片段间特征的相似度矩阵
        fragment_features_norm = F.normalize(combined_fragment_features, p=2, dim=1)
        similarity_matrix = torch.matmul(fragment_features_norm, fragment_features_norm.transpose(0, 1))
        
        # 只关注不同片段之间的相似度 (非对角元素)
        mask = torch.ones_like(similarity_matrix) - torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
        
        # 计算损失：让不同片段之间的相似度接近0（减小相似度）
        interfragment_loss = torch.sum(similarity_matrix * mask) / (torch.sum(mask) + 1e-6)
        
        return interfragment_loss
    
    def forward(self, scalar_short, scalar_long, vector_short, vector_long, fragment_ids, 
                pos=None, molecule_ids=None, atom_types=None):
        """计算分子内一致性 + 片段间差异性损失
        
        Args:
            scalar_short: 短程标量特征 [num_atoms, hidden_dim]
            scalar_long: 长程标量特征 [num_atoms, hidden_dim]
            vector_short: 短程向量特征 [num_atoms, 3, hidden_dim]
            vector_long: 长程向量特征 [num_atoms, 3, hidden_dim]
            fragment_ids: BRICS片段ID [num_atoms]
            pos: 原子坐标 [num_atoms, 3]，可选
            molecule_ids: 分子ID [num_atoms]，可选
            atom_types: 原子类型 [num_atoms]，可选
            
        Returns:
            total_loss: 总损失
        """
        # 参数检查
        if fragment_ids is None:
            return torch.tensor(0.0, device=scalar_short.device)
        
        # 获取渐进式缩放因子
        scaling_factor = self.get_scaling_factor()
        if scaling_factor < 0.001:
            return torch.tensor(0.0, device=scalar_short.device)
        
        # 创建有效片段掩码
        valid_mask = None
        if self.min_fragment_size > 1:
            valid_fragments = []
            for frag_id in torch.unique(fragment_ids):
                frag_size = torch.sum(fragment_ids == frag_id)
                if frag_size >= self.min_fragment_size:
                    valid_fragments.append(frag_id.item())
            
            if not valid_fragments:
                return torch.tensor(0.0, device=scalar_short.device)
                
            valid_mask = torch.zeros_like(fragment_ids, dtype=torch.bool)
            for frag_id in valid_fragments:
                valid_mask |= (fragment_ids == frag_id)
                
            if torch.sum(valid_mask) < self.min_fragment_size:
                return torch.tensor(0.0, device=scalar_short.device)
        
        # 处理短程标量特征
        scalar_features = self.scalar_projector(scalar_short)
        
        # 1. 计算分子内构象一致性损失 (基于短程特征)
        intramolecular_loss = self.compute_fragment_consistency(scalar_features, fragment_ids, valid_mask)
        
        # 2. 计算向量特征的一致性损失
        vector_loss = 0.0
        vector_magnitudes = None
        if self.vector_weight > 0 and vector_short is not None:
            # 处理短程向量特征 - 先计算向量的范数
            vector_norms = torch.norm(vector_short, dim=2)  # [num_atoms, 3]
            vector_magnitudes = torch.mean(vector_norms, dim=1, keepdim=True)  # [num_atoms, 1]
            
            # 投影向量特征
            batch_size, dim_size = vector_magnitudes.shape
            vector_features = self.vector_projector(
                vector_magnitudes.expand(-1, self.hidden_channels)
            )[:, :1]  # 只保留第一个维度 [num_atoms, 1]
            
            vector_loss = self.compute_fragment_consistency(
                vector_features, fragment_ids, valid_mask
            )
        
        # 3. 计算片段间差异性损失 - 同时使用短程标量和向量特征
        interfragment_loss = 0.0
        if self.intermolecular_weight > 0:
            # 同时使用标量和向量特征计算片段间差异性
            interfragment_loss = self.compute_interfragment_differentiation(
                scalar_features, 
                vector_magnitudes,  # 可能为None
                fragment_ids
            )
        
        # 组合所有损失
        # 片段内部特征应当相似 (低intramolecular_loss)，片段之间特征应当不同 (高interfragment_loss)
        total_loss = (intramolecular_loss + vector_loss + 
                      self.intermolecular_weight * interfragment_loss)
        
        # 应用渐进缩放
        return self.consistency_factor * total_loss * scaling_factor

