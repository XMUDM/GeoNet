import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sym
import numpy as np
from torch_scatter import scatter
from .torchmdnet.models.utils import ExpNormalSmearing, GaussianSmearing
import math


class BesselBasisLayer(nn.Module):
    """
    贝塞尔基函数层
    完全仿照DimeNet的BesselBasisLayer实现
    """
    def __init__(self, num_radial, cutoff, envelope_exponent):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        
        # 初始化频率 (仿照DimeNet)
        frequencies = torch.arange(1, num_radial + 1, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, distances):
        """
        计算贝塞尔基函数，仿照DimeNet的实现
        
        Args:
            distances: [N] 距离张量
            
        Returns:
            rbf: [N, num_radial] 径向基函数特征
        """
        # 距离归一化
        d_scaled = distances / self.cutoff  # [N]
        d_scaled = d_scaled.unsqueeze(-1)   # [N, 1]
        
        # 计算envelope函数 (余弦截止函数)
        # envelope = 0.5 * (cos(π * d_scaled) + 1) for d_scaled <= 1, else 0
        envelope = torch.where(
            d_scaled <= 1.0,
            0.5 * (torch.cos(math.pi * d_scaled) + 1.0),
            torch.zeros_like(d_scaled)
        )
        
        # 计算贝塞尔函数值
        # rbf = envelope * sin(freq * d_scaled) / d_scaled (with numerical stability)
        freq_d = self.frequencies * d_scaled  # [N, num_radial]
        
        # 避免除零：当d_scaled接近0时，sin(x)/x → 1
        eps = 1e-6
        safe_d_scaled = torch.clamp(d_scaled, min=eps)
        
        rbf = envelope * torch.sin(freq_d) / safe_d_scaled  # [N, num_radial]
        
        return rbf


class SphericalBasisLayer(nn.Module):
    """
    球面基函数层，用于角度编码
    完全仿照DimeNet的SphericalBasisLayer实现
    """
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        
        # 初始化贝塞尔基函数层
        self.bessel_layer = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        
        # 预计算球谐函数系数（基于勒让德多项式的简化实现）
        self._init_spherical_harmonics()
        
    def _init_spherical_harmonics(self):
        """
        初始化球谐函数系数
        基于勒让德多项式的简化实现，仿照DimeNet
        """
        # 预计算归一化常数
        coeffs = []
        for l in range(self.num_spherical):
            if l == 0:
                # P_0(cos θ) = 1, Y_0^0 = 1/(2√π)
                coeffs.append(1.0 / (2.0 * math.sqrt(math.pi)))
            elif l == 1:
                # P_1(cos θ) = cos θ, Y_1^0 = √(3/4π) cos θ
                coeffs.append(math.sqrt(3.0 / (4.0 * math.pi)))
            elif l == 2:
                # P_2(cos θ) = (3cos²θ - 1)/2, Y_2^0 = √(5/4π) P_2(cos θ)
                coeffs.append(math.sqrt(5.0 / (4.0 * math.pi)))
            elif l == 3:
                # P_3(cos θ) = (5cos³θ - 3cosθ)/2, Y_3^0 = √(7/4π) P_3(cos θ)
                coeffs.append(math.sqrt(7.0 / (4.0 * math.pi)))
            elif l == 4:
                # P_4(cos θ) = (35cos⁴θ - 30cos²θ + 3)/8
                coeffs.append(math.sqrt(9.0 / (4.0 * math.pi)))
            elif l == 5:
                # P_5(cos θ) = (63cos⁵θ - 70cos³θ + 15cosθ)/8
                coeffs.append(math.sqrt(11.0 / (4.0 * math.pi)))
            else:
                # 更高阶的简化处理
                coeffs.append(math.sqrt((2.0 * l + 1.0) / (4.0 * math.pi)))
        
        # 注册为缓冲区
        for i, coeff in enumerate(coeffs):
            self.register_buffer(f'sph_coeff_{i}', torch.tensor(coeff, dtype=torch.float32))
    
    def _compute_spherical_harmonics(self, angles):
        """
        计算球谐函数值，基于勒让德多项式
        
        Args:
            angles: [N] 角度张量 (弧度)
            
        Returns:
            cbf: [N, num_spherical] 球谐函数特征
        """
        cos_angles = torch.cos(angles)  # cos(θ)
        sph_features = []
        
        for l in range(self.num_spherical):
            coeff = getattr(self, f'sph_coeff_{l}')
            
            if l == 0:
                # P_0(x) = 1
                legendre_val = torch.ones_like(cos_angles)
            elif l == 1:
                # P_1(x) = x
                legendre_val = cos_angles
            elif l == 2:
                # P_2(x) = (3x² - 1)/2
                legendre_val = (3.0 * cos_angles**2 - 1.0) / 2.0
            elif l == 3:
                # P_3(x) = (5x³ - 3x)/2
                legendre_val = (5.0 * cos_angles**3 - 3.0 * cos_angles) / 2.0
            elif l == 4:
                # P_4(x) = (35x⁴ - 30x² + 3)/8
                x2 = cos_angles**2
                legendre_val = (35.0 * x2**2 - 30.0 * x2 + 3.0) / 8.0
            elif l == 5:
                # P_5(x) = (63x⁵ - 70x³ + 15x)/8
                x2 = cos_angles**2
                x3 = cos_angles**3
                legendre_val = (63.0 * x3 * x2 - 70.0 * x3 + 15.0 * cos_angles) / 8.0
            else:
                # 更高阶的切比雪夫多项式近似
                legendre_val = torch.cos(l * angles)
            
            # 应用归一化系数
            sph_val = coeff * legendre_val
            sph_features.append(sph_val)
        
        return torch.stack(sph_features, dim=1)  # [N, num_spherical]
    
    def forward(self, distances, angles, id_expand_kj):
        """
        计算球面基函数，完全仿照DimeNet的实现
        
        Args:
            distances: [N_pairs] 距离张量
            angles: [N_triplets] 角度张量
            id_expand_kj: [N_triplets] 将 pair 索引扩展到 triplet 的映射
        Returns:
            sbf_features: [N_triplets, num_spherical * num_radial] 球面基函数特征
        """
        # 1. 计算径向基函数 (RBF)
        rbf = self.bessel_layer(distances)  # [N_pairs, num_radial]
        
        # 2. 验证id_expand_kj的有效性并扩展rbf到triplet级别
        if len(id_expand_kj) > 0:
            max_idx = torch.max(id_expand_kj).item()
            if max_idx >= rbf.size(0):
                print(f"[ERROR] SphericalBasisLayer: id_expand_kj包含无效索引 {max_idx} >= {rbf.size(0)}")
                # 将超出范围的索引裁剪到有效范围
                safe_id_expand_kj = torch.clamp(id_expand_kj, 0, rbf.size(0) - 1)
                rbf_expanded = rbf[safe_id_expand_kj]  # [N_triplets, num_radial]
            else:
                rbf_expanded = rbf[id_expand_kj]  # [N_triplets, num_radial]
        else:
            # 如果没有三元组，返回空张量
            device = rbf.device
            return torch.empty(0, self.num_spherical * self.num_radial, device=device)
        
        # 3. 计算球谐函数 (CBF)
        cbf = self._compute_spherical_harmonics(angles)  # [N_triplets, num_spherical]
        
        # 4. 组合RBF和CBF，仿照DimeNet的实现
        # 每个CBF特征与所有RBF特征相乘
        cbf_expanded = cbf.repeat_interleave(self.num_radial, dim=1)  # [N_triplets, num_spherical * num_radial]
        
        # 重复RBF以匹配CBF的spherical维度
        rbf_repeated = rbf_expanded.repeat(1, self.num_spherical)  # [N_triplets, num_spherical * num_radial]
        
        # 5. 最终的SBF特征 = RBF * CBF (element-wise multiplication)
        sbf = rbf_repeated * cbf_expanded  # [N_triplets, num_spherical * num_radial]
        
        return sbf


class AngleTripletGenerator(nn.Module):
    """
    角度三元组生成器
    参考 DimeNet 的三元组生成逻辑
    """
    def __init__(self, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
    
    def generate_triplets(self, pos, edge_index, batch=None):
        """
        生成角度三元组 (i, j, k)，其中 i 是中心原子，j 和 k 是其邻居
        
        Args:
            pos: [N, 3] 原子坐标
            edge_index: [2, E] 边索引
            batch: [N] 批次信息，可选
            
        Returns:
            id3_i: [T] 三元组中心原子索引
            id3_j: [T] 三元组第一个邻居索引
            id3_k: [T] 三元组第二个邻居索引
            distances_jk: [T] j-k 距离（用于 id_expand_kj）
            angles: [T] 角度值
        """
        row, col = edge_index  # row: source, col: target
        
        # 构建邻接列表
        num_nodes = pos.size(0)
        neighbors = [[] for _ in range(num_nodes)]
        edge_distances = torch.norm(pos[row] - pos[col], dim=1)
        
        # 只保留距离在 cutoff 内的边
        valid_edges = edge_distances <= self.cutoff
        valid_row = row[valid_edges]
        valid_col = col[valid_edges]
        
        for i, j in zip(valid_row.tolist(), valid_col.tolist()):
            neighbors[i].append(j)
        
        # 生成三元组
        id3_i, id3_j, id3_k = [], [], []
        
        for center in range(num_nodes):
            center_neighbors = neighbors[center]
            if len(center_neighbors) < 2:
                continue
                
            # 对于每个中心原子，生成所有可能的邻居对
            for idx_j, j in enumerate(center_neighbors):
                for idx_k, k in enumerate(center_neighbors):
                    if idx_j != idx_k:  # 确保 j != k
                        id3_i.append(center)
                        id3_j.append(j)
                        id3_k.append(k)
        
        if not id3_i:
            # 如果没有找到任何三元组，返回空张量
            device = pos.device
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, device=device),
                    torch.empty(0, device=device))
        
        id3_i = torch.tensor(id3_i, dtype=torch.long, device=pos.device)
        id3_j = torch.tensor(id3_j, dtype=torch.long, device=pos.device)
        id3_k = torch.tensor(id3_k, dtype=torch.long, device=pos.device)
        
        # 计算角度
        angles = self._calculate_angles(pos, id3_i, id3_j, id3_k)
        
        # 计算 j-k 距离（用于后续的 id_expand_kj 映射）
        distances_jk = torch.norm(pos[id3_j] - pos[id3_k], dim=1)
        
        return id3_i, id3_j, id3_k, distances_jk, angles
    
    def _calculate_angles(self, pos, id3_i, id3_j, id3_k):
        """
        计算三元组的角度 ∠jik
        参考 DimeNet 的 calculate_neighbor_angles 函数
        """
        Ri = pos[id3_i]  # 中心原子坐标
        Rj = pos[id3_j]  # 第一个邻居坐标
        Rk = pos[id3_k]  # 第二个邻居坐标
        
        # 计算向量
        R1 = Rj - Ri  # i -> j
        R2 = Rk - Ri  # i -> k
        
        # 计算点积和叉积
        dot_product = torch.sum(R1 * R2, dim=-1)
        cross_product = torch.cross(R1, R2, dim=-1)
        cross_norm = torch.norm(cross_product, dim=-1)
        
        # 使用 atan2 计算角度
        angles = torch.atan2(cross_norm, dot_product)
        
        return angles


class DimeNetStyleAngleFeatureExtractor(nn.Module):
    """
    DimeNet风格的角度特征提取器
    
    完全遵循DimeNet的实现流程：
    1. 计算距离和RBF特征
    2. 计算角度和CBF特征
    3. 组合成SBF特征 (RBF * CBF)
    
    三元组基于 (node, edge, group) 结构，其中 edge 位置为 node 和 group 的中点
    """
    
    def __init__(self, edge_index, hidden_channels=128, num_spherical=7, num_radial=6, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        
        # 基础参数
        self.hidden_channels = hidden_channels
        self.num_spherical = num_spherical  
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        
        # 保存边索引用于计算距离和RBF
        self.register_buffer('edge_i', edge_index[0])
        self.register_buffer('edge_j', edge_index[1])
        
        # 添加警告计数器
        self.identical_indices_warning_count = 0
        self.distance_filtering_warning_count = 0
        self.warning_limit = 3  # 只显示前3次警告
        
        # 仿照DimeNet初始化基函数层
        self.rbf_layer = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        self.final_rbf_linear = nn.Linear(num_radial, hidden_channels)
        # 最终投影层，将SBF特征投影到hidden_channels
        self.sbf_projection = nn.Linear(num_spherical * num_radial, hidden_channels, bias=False)
        
        print(f"[INFO] DimeNet风格角度特征提取器初始化:")
        print(f"  - 径向基函数: {num_radial}")
        print(f"  - 球谐函数阶数: {num_spherical}")
        print(f"  - SBF特征维度: {num_spherical * num_radial}")
        print(f"  - 输出特征维度: {hidden_channels}")
        print(f"  - 截止距离: {cutoff} Å")
        print(f"  - 包络指数: {envelope_exponent}")
        print(f"  - 完整DimeNet流程:")
        print(f"    1. 计算原子间距离")
        print(f"    2. 计算三元组角度")
        print(f"    3. 构造id_expand_kj映射")
        print(f"    4. SBF层: 距离+角度 -> RBF+CBF -> SBF特征")
        print(f"    5. 投影到目标维度 -> 最终角度特征")

    def calculate_interatomic_distances(self, pos, idx_i, idx_j):
        """
        计算原子间距离，完全仿照DimeNet的实现
        """
        Ri = pos[idx_i]  # [num_pairs, 3]
        Rj = pos[idx_j]  # [num_pairs, 3]
        
        # 计算距离，添加数值稳定性
        distances = torch.norm(Ri - Rj, dim=-1)  # [num_pairs]
        distances = torch.clamp(distances, min=1e-6)  # 防止除零
        
        return distances

    def calculate_neighbor_angles(self, pos, id3_i, id3_j, id3_k):
        """
        计算三元组角度，完全仿照 DimeNet 的 calculate_neighbor_angles
        数值稳定版本，特别处理向量长度接近零的情况
        
        Args:
            pos: [N, 3] 位置坐标 (可以是 node_pos 或 group_pos 的组合)
            id3_i: [T] 中心原子索引
            id3_j: [T] 第一个邻居索引  
            id3_k: [T] 第二个邻居索引
            
        Returns:
            angles: [T] 角度值 ∠jik (以i为中心，j和k的夹角)
        """
        
        eps = 1e-6  # 增加 epsilon 值以提高数值稳定性
        
        # 添加索引有效性检查
        max_idx = pos.size(0) - 1
        if torch.any(id3_i > max_idx) or torch.any(id3_j > max_idx) or torch.any(id3_k > max_idx):
            print(f"WARNING: Index out of bounds in calculate_neighbor_angles")
            print(f"pos.shape: {pos.shape}, max indices: i={id3_i.max()}, j={id3_j.max()}, k={id3_k.max()}")
            # 返回默认角度
            return torch.full_like(id3_i, math.pi / 2, dtype=torch.float32, device=pos.device)
        
        # 仿照 DimeNet 的实现
        Ri = pos[id3_i]  # [T, 3] 中心原子坐标
        Rj = pos[id3_j]  # [T, 3] 第一个邻居坐标
        Rk = pos[id3_k]  # [T, 3] 第二个邻居坐标
        
        # 计算向量 (与 DimeNet 完全一致)
        R1 = Rj - Ri  # [T, 3] i -> j
        R2 = Rk - Ri  # [T, 3] i -> k  (DimeNet 原版的实现)
        
        # 计算向量长度，添加数值稳定性
        R1_norm = torch.norm(R1, dim=-1, keepdim=True) + eps  # [T, 1]
        R2_norm = torch.norm(R2, dim=-1, keepdim=True) + eps  # [T, 1]
        
        # 检查是否有向量长度过小的情况
        min_distance = 1e-4  # 最小距离阈值
        too_close_mask = (R1_norm.squeeze() < min_distance) | (R2_norm.squeeze() < min_distance)
        
        # 归一化向量
        R1_normalized = R1 / R1_norm  # [T, 3]
        R2_normalized = R2 / R2_norm  # [T, 3]
        
        # 计算点积和叉积（使用归一化向量）
        x = torch.sum(R1_normalized * R2_normalized, dim=-1)  # [T] 点积
        y = torch.cross(R1_normalized, R2_normalized, dim=-1)  # [T, 3] 叉积
        y_norm = torch.norm(y, dim=-1)  # [T] 叉积的模长
        
        # 裁剪x到有效范围，避免数值误差
        x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
        
        # 使用 atan2 计算角度 (与 DimeNet 完全一致)
        angles = torch.atan2(y_norm, x)  # [T]
        
        # 对于向量过于接近的情况，使用默认角度
        if torch.any(too_close_mask):
            angles[too_close_mask] = math.pi / 2  # 90度作为默认值
        
        # 检查角度是否有NaN或无穷大
        if torch.any(torch.isnan(angles)) or torch.any(torch.isinf(angles)):
            print(f"WARNING: NaN or Inf detected in angles calculation, using fallback")
            # 用安全值替换异常值
            safe_angles = torch.full_like(angles, math.pi / 2, dtype=torch.float32, device=angles.device)
            nan_mask = torch.isnan(angles) | torch.isinf(angles)
            angles = torch.where(nan_mask, safe_angles, angles)
        
        # 最终检查，确保角度在有效范围内
        angles = torch.clamp(angles, eps, math.pi - eps)
        
        return angles
    
    def forward(self, node_pos, group_pos, edge_attr, triplet_i, triplet_j, triplet_k, id_expand_kj, id_reduce_ji):
        """
        使用预计算的三元组和DimeNet索引计算几何特征
        
        完全基于数据预处理阶段生成的标准DimeNet索引：
        1. 计算距离 -> RBF特征
        2. 计算角度 -> CBF特征  
        3. 使用预计算的id_expand_kj组合 -> SBF = RBF * CBF
        4. 使用预计算的id_reduce_ji聚合到边级别
        
        Args:
            node_pos: [num_nodes, 3] node位置坐标
            group_pos: [num_groups, 3] group位置坐标
            edge_attr: [num_edges, hidden_channels] 原始边特征(仅用于获取形状和设备信息)
            triplet_i: [num_triplets] 中心group索引
            triplet_j: [num_triplets] node1索引  
            triplet_k: [num_triplets] node2索引
            id_expand_kj: [num_triplets] DimeNet标准的pair到triplet映射
            id_reduce_ji: [num_triplets] DimeNet标准的triplet到pair映射
            
        Returns:
            edge_rbf_features: [num_edges, num_radial] 边级别的RBF特征
            edge_sbf_features: [num_edges, hidden_channels] 基于DimeNet SBF的边特征
        """
        device = edge_attr.device
        num_edges = edge_attr.size(0)
        
        if len(triplet_i) == 0:
            # 没有三元组，返回零特征 (因为我们现在完全基于角度信息)
            print("WARNING: No triplets provided, returning zero features")
            return torch.zeros_like(edge_attr)

        try:
            # 修正索引问题：node索引需要偏移num_groups
            num_groups = group_pos.size(0)
            offset_tensor = torch.tensor(num_groups, dtype=torch.long, device=triplet_j.device)
            corrected_triplet_j = triplet_j + offset_tensor  # node1索引偏移
            corrected_triplet_k = triplet_k + offset_tensor  # node2索引偏移
            
            # 组合位置坐标：[group_pos; node_pos]
            combined_pos = torch.cat([group_pos, node_pos], dim=0)
            
            # 过滤有问题的三元组
            identical_mask = (triplet_i == corrected_triplet_j) | (triplet_i == corrected_triplet_k) | (corrected_triplet_j == corrected_triplet_k)
            
            if torch.any(identical_mask):
                if self.identical_indices_warning_count < self.warning_limit:
                    identical_count = torch.sum(identical_mask).item()
                    print(f"WARNING: Found {identical_count} triplets with identical indices, filtering them out")
                    self.identical_indices_warning_count += 1
                        
                # 过滤掉有问题的三元组
                valid_mask = ~identical_mask
                if torch.any(valid_mask):
                    triplet_i = triplet_i[valid_mask]
                    corrected_triplet_j = corrected_triplet_j[valid_mask]
                    corrected_triplet_k = corrected_triplet_k[valid_mask]
                    # 同步过滤DimeNet索引
                    id_expand_kj = id_expand_kj[valid_mask]
                    id_reduce_ji = id_reduce_ji[valid_mask]
                else:
                    return torch.zeros_like(edge_attr), torch.zeros_like(edge_attr)
            
            # 距离过滤
            if len(triplet_i) > 0:
                pos_i = combined_pos[triplet_i]
                pos_j = combined_pos[corrected_triplet_j] 
                pos_k = combined_pos[corrected_triplet_k]
                
                min_distance = 0.01
                dist_ij = torch.norm(pos_i - pos_j, dim=1)
                dist_ik = torch.norm(pos_i - pos_k, dim=1)
                dist_jk = torch.norm(pos_j - pos_k, dim=1)
                
                distance_mask = (dist_ij > min_distance) & (dist_ik > min_distance) & (dist_jk > min_distance)
                
                if not torch.all(distance_mask):
                    filtered_count = torch.sum(~distance_mask).item()
                    if self.distance_filtering_warning_count < self.warning_limit:
                        print(f"WARNING: Filtering {filtered_count} triplets due to too-close distances")
                        self.distance_filtering_warning_count += 1
                    
                    triplet_i = triplet_i[distance_mask]
                    corrected_triplet_j = corrected_triplet_j[distance_mask]
                    corrected_triplet_k = corrected_triplet_k[distance_mask]
                    # 同步过滤DimeNet索引
                    id_expand_kj = id_expand_kj[distance_mask]
                    id_reduce_ji = id_reduce_ji[distance_mask]
            
            if triplet_i.size(0) == 0:
                return torch.zeros_like(edge_attr), torch.zeros_like(edge_attr)
                
            # === DimeNet标准流程 ===
            
            # 1. 计算边距离 (基于原始edge_index)
            edge_distances = self.calculate_interatomic_distances(
                combined_pos, self.edge_i, self.edge_j
            )  # [num_edges]
            print(f"edge_distances.shape: {edge_distances.shape}")
            # 2. 显式计算边的RBF特征
            edge_rbf_features = self.rbf_layer(edge_distances)  # [num_edges, num_radial]
            print(f"edge_rbf_features.shape: {edge_rbf_features.shape}")
            edge_rbf_features = self.final_rbf_linear(edge_rbf_features)  # [num_edges, hidden_channels]
            print(f"edge_rbf_features.shape: {edge_rbf_features.shape}")
            # 3. 计算角度 (基于三元组)
            angles = self.calculate_neighbor_angles(
                combined_pos,
                triplet_i,           # group索引（中心）
                corrected_triplet_j, # node1索引
                corrected_triplet_k  # node2索引
            )  # [num_triplets]
            
            # 4. 使用预计算的id_expand_kj并验证有效性
            num_triplets = len(triplet_i)
            
            # 验证索引的有效性
            max_expand_idx = torch.max(id_expand_kj).item() if len(id_expand_kj) > 0 else -1
            if max_expand_idx >= num_edges:
                print(f"[ERROR] id_expand_kj包含无效索引 {max_expand_idx} >= {num_edges}，裁剪到有效范围")
                # 将超出范围的索引裁剪到有效范围
                final_id_expand_kj = torch.clamp(id_expand_kj, 0, num_edges - 1).to(device)
            else:
                final_id_expand_kj = id_expand_kj.to(device)
                if not hasattr(self, '_expand_kj_info_logged'):
                    print(f"[INFO] 使用预计算的DimeNet标准id_expand_kj，形状: {final_id_expand_kj.shape}")
                    self._expand_kj_info_logged = True
            
            # 5. 使用DimeNet的标准SBF层计算特征
            # edge_distances: [num_edges], angles: [num_triplets], final_id_expand_kj: [num_triplets]
            sbf_features = self.sbf_layer(edge_distances, angles, final_id_expand_kj)  # [num_triplets, num_spherical * num_radial]
            
            # 6. 投影到目标维度
            sbf_projected = self.sbf_projection(sbf_features)  # [num_triplets, hidden_channels]
            
            # 7. 聚合到边级别 (使用id_reduce_ji)
            from torch_scatter import scatter_add
            
            # 使用预计算的DimeNet标准聚合索引并验证有效性
            max_reduce_idx = torch.max(id_reduce_ji).item() if len(id_reduce_ji) > 0 else -1
            if max_reduce_idx >= num_edges:
                print(f"[ERROR] id_reduce_ji包含无效索引 {max_reduce_idx} >= {num_edges}，裁剪到有效范围")
                # 将超出范围的索引裁剪到有效范围
                valid_reduce_mapping = torch.clamp(id_reduce_ji, 0, num_edges - 1).to(device)
                valid_sbf_projected = sbf_projected
            else:
                if not hasattr(self, '_reduce_ji_info_logged'):
                    print(f"[INFO] 使用预计算的id_reduce_ji进行聚合")
                    self._reduce_ji_info_logged = True
                valid_reduce_mapping = id_reduce_ji.to(device)
                valid_sbf_projected = sbf_projected
            
            if len(valid_sbf_projected) == 0:
                return torch.zeros_like(edge_attr), torch.zeros_like(edge_attr)
            
            # 聚合SBF特征到边级别
            edge_sbf_features = scatter_add(
                valid_sbf_projected, 
                valid_reduce_mapping, 
                dim=0, 
                dim_size=num_edges
            )
            
            # 计算每条边对应的三元组数量
            edge_counts = scatter_add(
                torch.ones(len(valid_sbf_projected), device=device),
                valid_reduce_mapping,
                dim=0,
                dim_size=num_edges
            )
            
            # 计算平均特征
            valid_mask = edge_counts > 0
            edge_sbf_features[valid_mask] = edge_sbf_features[valid_mask] / edge_counts[valid_mask].unsqueeze(-1)
            
            # 8. 使用SBF特征作为最终输出 (替代原始edge_attr)
            # SBF特征已经包含了丰富的距离和角度信息，比原始边特征更有表达力
            final_features = edge_sbf_features
            
            # # 对于没有角度信息的边，使用零特征或小的随机特征
            # no_angle_mask = ~valid_mask
            # if torch.any(no_angle_mask):
            #     # 使用小的随机特征替代，避免完全的零特征
            #     final_features[no_angle_mask] = torch.randn_like(final_features[no_angle_mask]) * 0.01
            
            # # 数值稳定性检查
            # if torch.any(torch.isnan(final_features)) or torch.any(torch.isinf(final_features)):
            #     print(f"WARNING: Final features contain NaN/Inf, falling back to zero features")
            #     final_features = torch.zeros_like(edge_attr)
            
            # # 最终裁剪
            # final_features = torch.clamp(final_features, min=-10.0, max=10.0)
            
            return edge_rbf_features, edge_sbf_features
            
        except Exception as e:
            print(f"ERROR in DimeNetStyleAngleFeatureExtractor: {e}")
            print(f"Returning zero features to avoid crash")
            return torch.zeros_like(edge_attr), torch.zeros_like(edge_attr)


class SimplifiedDimeNetAngleExtractor(nn.Module):
    """
    简化版 DimeNet 角度特征提取器
    专门针对双分图 node-group 边进行角度特征计算
    只使用 node_vec 和 edge_vec 的信息，无需三元组生成
    直接替代 SimpleAngleFeatureExtractor
    """
    def __init__(self, 
                 hidden_channels, 
                 num_angle_rbf=25,
                 cutoff=5.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_angle_rbf = num_angle_rbf
        self.cutoff = cutoff
        
        # 角度径向基函数（类似 DimeNet 的贝塞尔基函数）
        self.angle_rbf = self._create_angle_rbf()
        
        # 特征投影层 - 只用一个角度
        self.feature_projection = nn.Sequential(
            nn.Linear(num_angle_rbf, hidden_channels),  
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        print(f"[INFO] 简化版DimeNet角度提取器初始化:")
        print(f"  - 角度RBF数量: {num_angle_rbf}")
        print(f"  - 截断距离: {cutoff}")
        print(f"  - 输出特征维度: {hidden_channels}")
    
    def _create_angle_rbf(self):
        """创建角度径向基函数"""
        # 使用正弦基函数，类似 DimeNet 的贝塞尔基函数
        frequencies = torch.arange(1, self.num_angle_rbf + 1, dtype=torch.float32) * np.pi
        return nn.Parameter(frequencies, requires_grad=False)
    
    def _compute_angle_rbf(self, angles):
        """计算角度的径向基函数特征"""
        # angles: [N] 范围在 [0, π]
        # 使用正弦基函数：sin(n * angle) 其中 n = 1, 2, ..., num_angle_rbf
        
        # 归一化角度到 [0, 1]
        normalized_angles = angles / np.pi  # [N]
        
        # 计算基函数
        rbf_features = torch.sin(self.angle_rbf[None, :] * normalized_angles[:, None])  # [N, num_angle_rbf]
        
        return rbf_features
    
    def _compute_angle(self, vec1, vec2, eps=1e-8):
        """计算两个向量之间的夹角（弧度）"""
        # 向量已经预处理过，直接计算余弦值
        cos_angle = (vec1 * vec2).sum(dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        
        # 计算角度
        angle = torch.acos(cos_angle)
        return angle
    
    def extract_3d_vectors(self, vec_features):
        """从不同格式的向量特征中提取3D向量"""
        if vec_features.dim() == 2:
            return vec_features
        elif vec_features.dim() == 3:
            return vec_features.mean(dim=-1)  # 简单取平均
        else:
            return vec_features
    
    def forward(self, edge_index, node_vec, group_vec, edge_vec):
        """
        提取简化版的 DimeNet 风格角度特征
        
        Args:
            edge_index: [2, num_edges] 边索引（node-group）
            node_vec: [num_nodes, 3, hidden] 或 [num_nodes, 3] node向量特征
            group_vec: [num_groups, 3, hidden] 或 [num_groups, 3] group向量特征  
            edge_vec: [num_edges, 3] 边向量 (归一化)
            
        Returns:
            angle_features: [num_edges, hidden_channels] 编码后的角度特征
        """
        device = edge_vec.device
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return torch.zeros(num_edges, self.hidden_channels, device=device)
        
        node_idx = edge_index[0]  # [num_edges]
        group_idx = edge_index[1]  # [num_edges]
        
        # 提取3D向量特征
        node_vec_3d = self.extract_3d_vectors(node_vec)  # [num_nodes, 3]
        group_vec_3d = self.extract_3d_vectors(group_vec)  # [num_groups, 3]
        
        # 获取边对应的向量
        node_vec_edge = node_vec_3d[node_idx]  # [num_edges, 3]
        group_vec_edge = group_vec_3d[group_idx]  # [num_edges, 3]
        
        # 计算核心角度：node向量与边向量的夹角
        angle_node_edge = self._compute_angle(node_vec_edge, edge_vec)   # [num_edges]
        
        # 使用DimeNet风格的RBF编码角度特征
        node_angle_rbf = self._compute_angle_rbf(angle_node_edge)    # [num_edges, num_angle_rbf]
        
        # 线性映射到目标维度
        angle_features = self.feature_projection(node_angle_rbf)  # [num_edges, hidden_channels]
        
        return angle_features 


class DimeNetVectorAngleExtractor(nn.Module):
    """
    基于 DimeNet 设计的向量角度特征提取器
    使用 DimeNet 的核心组件：贝塞尔基函数 + 球谐函数
    但只基于 node_vec 和 edge_vec，无需三元组
    
    DimeNet 的核心思想：
    - 使用贝塞尔基函数编码径向信息（这里用向量长度）
    - 使用球谐函数编码角度信息
    - 两者相乘得到球面基函数(SBF)
    """
    def __init__(self, 
                 hidden_channels,
                 num_spherical=7, 
                 num_radial=6,
                 cutoff=5.0, 
                 envelope_exponent=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # DimeNet 的贝塞尔基函数层
        self.bessel_layer = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        
        # DimeNet 的球谐函数系数
        self._init_spherical_harmonics()
        
        # 球面基函数特征维度
        sbf_dim = num_spherical * num_radial
        
        # 特征投影网络（保持DimeNet风格）
        self.sbf_projection = nn.Sequential(
            nn.Linear(sbf_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        print(f"[INFO] DimeNet风格向量角度提取器初始化:")
        print(f"  - 球谐函数阶数: {num_spherical}")
        print(f"  - 贝塞尔基函数数量: {num_radial}")
        print(f"  - SBF特征维度: {sbf_dim}")
        print(f"  - 输出维度: {hidden_channels}")
    
    def _init_spherical_harmonics(self):
        """初始化球谐函数系数（与DimeNet一致）"""
        coeffs = []
        for l in range(self.num_spherical):
            if l == 0:
                # Y_0^0 = 1/sqrt(4π)
                coeffs.append([1.0 / math.sqrt(4 * math.pi)])
            elif l == 1:
                # Y_1^0 = sqrt(3/4π) * cos(θ)
                coeffs.append([math.sqrt(3.0 / (4 * math.pi))])
            elif l == 2:
                # Y_2^0 = sqrt(5/4π) * (3*cos²(θ) - 1)/2
                coeffs.append([math.sqrt(5.0 / (4 * math.pi)) * 0.5])
            elif l == 3:
                # Y_3^0 = sqrt(7/4π) * (5*cos³(θ) - 3*cos(θ))/2
                coeffs.append([math.sqrt(7.0 / (4 * math.pi)) * 0.5])
            else:
                # 更高阶使用简化形式
                coeffs.append([1.0 / math.sqrt(4 * math.pi)])
        
        # 注册为缓冲区
        for i, coeff in enumerate(coeffs):
            self.register_buffer(f'sph_coeff_{i}', torch.tensor(coeff, dtype=torch.float32))
    
    def _compute_spherical_harmonics(self, angles):
        """计算球谐函数值（与DimeNet保持一致）"""
        cos_angles = torch.cos(angles)
        sph_features = []
        
        for l in range(self.num_spherical):
            coeff = getattr(self, f'sph_coeff_{l}')
            
            if l == 0:
                # Y_0^0 = constant
                sph_val = coeff[0] * torch.ones_like(angles)
            elif l == 1:
                # Y_1^0 ∝ cos(θ)
                sph_val = coeff[0] * cos_angles
            elif l == 2:
                # Y_2^0 ∝ (3*cos²(θ) - 1)
                sph_val = coeff[0] * (3 * cos_angles**2 - 1)
            elif l == 3:
                # Y_3^0 ∝ (5*cos³(θ) - 3*cos(θ))
                sph_val = coeff[0] * (5 * cos_angles**3 - 3 * cos_angles)
            else:
                # 更高阶使用 Chebyshev 多项式近似
                sph_val = coeff[0] * torch.cos(l * angles)
            
            sph_features.append(sph_val)
        
        return torch.stack(sph_features, dim=1)  # [N, num_spherical]
    
    def _compute_angle(self, vec1, vec2, eps=1e-8):
        """计算两个向量之间的夹角"""
        # # 归一化向量
        # vec1_norm = F.normalize(vec1, p=2, dim=-1)
        # vec2_norm = F.normalize(vec2, p=2, dim=-1)
        
        # 计算余弦值
        cos_angle = (vec1 * vec2).sum(dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
        
        # 计算角度
        angle = torch.acos(cos_angle)
        return angle
    
    def _extract_3d_vector(self, vec_features):
        """从向量特征中提取3D向量"""
        if vec_features.dim() == 2:
            return vec_features
        elif vec_features.dim() == 3:
            # 对于 [N, 3, hidden] 格式，取平均
            return vec_features.mean(dim=-1)
        else:
            raise ValueError(f"不支持的向量特征维度: {vec_features.shape}")
    
    def forward(self, edge_index, node_vec, edge_vec):
        """
        基于 DimeNet 设计提取角度特征
        
        Args:
            edge_index: [2, num_edges] 边索引
            node_vec: [num_nodes, 3] 或 [num_nodes, 3, hidden] 节点向量
            edge_vec: [num_edges, 3] 边向量（归一化）
            
        Returns:
            sbf_features: [num_edges, hidden_channels] DimeNet风格的球面基函数特征
        """
        device = edge_vec.device
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return torch.zeros(num_edges, self.hidden_channels, device=device)
        
        # 提取3D向量
        node_vec_3d = self._extract_3d_vector(node_vec)  # [num_nodes, 3]
        
        # 获取边对应的节点向量
        node_idx = edge_index[0]  # [num_edges]
        node_vec_edge = node_vec_3d[node_idx]  # [num_edges, 3]
        
        # === DimeNet 的核心计算 ===
        
        # 1. 计算角度（球谐函数的输入）
        angles = self._compute_angle(node_vec_edge, edge_vec)  # [num_edges]
        
        # 2. 计算球谐函数特征（DimeNet的角度编码）
        cbf = self._compute_spherical_harmonics(angles)  # [num_edges, num_spherical]
        
        # 3. 计算"径向"信息（这里使用节点向量的长度）
        node_vec_lengths = torch.norm(node_vec_edge, p=2, dim=-1)  # [num_edges]
        
        # 将长度限制在 cutoff 范围内，模拟 DimeNet 的距离处理
        node_vec_lengths = torch.clamp(node_vec_lengths, 0.0, self.cutoff)
        
        # 4. 使用贝塞尔基函数编码"径向"信息
        rbf = self.bessel_layer(node_vec_lengths)  # [num_edges, num_radial]
        
        # 5. 计算球面基函数 (SBF) = RBF ⊗ CBF
        # 这是 DimeNet 的核心：径向基函数与球谐函数的张量积
        cbf_expanded = cbf.repeat_interleave(self.num_radial, dim=1)  # [num_edges, num_spherical * num_radial]
        rbf_expanded = rbf.repeat(1, self.num_spherical)  # [num_edges, num_spherical * num_radial]
        
        sbf = rbf_expanded * cbf_expanded  # [num_edges, num_spherical * num_radial]
        
        # 6. 投影到目标维度（保持 DimeNet 的网络结构）
        features = self.sbf_projection(sbf)  # [num_edges, hidden_channels]
        
        return features 


if __name__ == "__main__":
    """
    简单测试重构后的DimeNetStyleAngleFeatureExtractor
    """
    print("=== 测试重构后的DimeNet风格角度特征提取器 ===")
    
    import torch
    torch.manual_seed(42)
    
    # 模拟数据
    num_nodes = 10
    num_groups = 5
    num_edges = 15
    hidden_channels = 64
    
    # 模拟边索引
    edge_index = torch.randint(0, num_nodes + num_groups, (2, num_edges))
    
    # 初始化特征提取器
    extractor = DimeNetStyleAngleFeatureExtractor(
        edge_index=edge_index,
        hidden_channels=hidden_channels,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5
    )
    
    print(f"特征提取器初始化成功")
    print(f"参数数量: {sum(p.numel() for p in extractor.parameters())}")
    
    # 模拟输入数据
    node_pos = torch.randn(num_nodes, 3) * 2.0
    group_pos = torch.randn(num_groups, 3) * 2.0
    edge_attr = torch.randn(num_edges, hidden_channels)
    
    # 模拟三元组（简单起见，只创建一些有效的三元组）
    num_triplets = 20
    triplet_i = torch.randint(0, num_groups, (num_triplets,))  # group索引
    triplet_j = torch.randint(0, num_nodes, (num_triplets,))  # node1索引
    triplet_k = torch.randint(0, num_nodes, (num_triplets,))  # node2索引
    id_expand_kj = torch.randint(0, num_edges, (num_triplets,))  # DimeNet索引
    id_reduce_ji = torch.randint(0, num_edges, (num_triplets,))  # DimeNet索引
    
    print(f"输入数据:")
    print(f"  - node_pos: {node_pos.shape}")
    print(f"  - group_pos: {group_pos.shape}")
    print(f"  - edge_attr: {edge_attr.shape} (仅用于形状参考)")
    print(f"  - triplets: {num_triplets}")
    print(f"  - DimeNet索引: id_expand_kj={id_expand_kj.shape}, id_reduce_ji={id_reduce_ji.shape}")
    
    # 前向传播
    try:
        print(f"开始前向传播...")
        edge_rbf_features, edge_sbf_features = extractor(node_pos, group_pos, edge_attr, triplet_i, triplet_j, triplet_k, id_expand_kj, id_reduce_ji)
        print(f"前向传播成功!")
        print(f"RBF特征形状: {edge_rbf_features.shape}")
        print(f"SBF特征形状: {edge_sbf_features.shape}")
        print(f"SBF特征范围: [{edge_sbf_features.min().item():.4f}, {edge_sbf_features.max().item():.4f}]")
        print(f"SBF特征均值: {edge_sbf_features.mean().item():.4f}")
        print(f"SBF特征标准差: {edge_sbf_features.std().item():.4f}")
        
        # 检查是否完全为零 (这表明可能没有有效的三元组)
        if torch.allclose(edge_sbf_features, torch.zeros_like(edge_sbf_features)):
            print("注意: 输出特征全为零，可能没有有效的三元组信息")
        
        # 检查是否有NaN或Inf
        if torch.any(torch.isnan(edge_sbf_features)) or torch.any(torch.isinf(edge_sbf_features)):
            print("警告: 输出包含NaN或Inf值!")
        else:
            print("✓ 输出数值稳定")
            
        print(f"\n注意: 现在输出的是基于DimeNet SBF的全新边特征，")
        print(f"包含丰富的几何信息，可以直接替代原始边特征使用。")
            
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== 测试完成 ===") 