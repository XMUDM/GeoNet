import torch
import numpy as np
from torch_geometric.utils import coalesce, to_undirected
from torch_scatter import scatter

def build_hypergraph_incidence_matrix(labels):
    """
    构建超图的关联矩阵 (incidence matrix)
    在超图中，关联矩阵 H 的每一列代表一个超边，每一行代表一个节点
    如果节点 i 在超边 j 中，则 H[i,j] = 1，否则为 0
    
    @param labels: 节点的标签张量，表示每个节点属于哪个组(超边)
    @return: 超图的关联矩阵和超边的数量
    """
    num_nodes = labels.shape[0]
    num_hyperedges = torch.unique(labels).shape[0]
    
    # 创建关联矩阵
    H = torch.zeros((num_nodes, num_hyperedges))
    
    # 填充关联矩阵
    for i in range(num_hyperedges):
        nodes_in_edge = torch.where(labels == i)[0]
        H[nodes_in_edge, i] = 1.0
    
    return H, num_hyperedges

def hypergraph_laplacian(H, W=None):
    """
    计算超图的拉普拉斯矩阵
    L = I - D_v^(-1/2) H W D_e^(-1) H^T D_v^(-1/2)
    
    @param H: 超图的关联矩阵
    @param W: 超边的权重矩阵，默认为单位矩阵
    @return: 超图的拉普拉斯矩阵
    """
    num_nodes, num_edges = H.shape
    
    # 如果没有给定超边权重，默认所有超边权重为1
    if W is None:
        W = torch.eye(num_edges)
    elif isinstance(W, torch.Tensor) and W.dim() == 1:
        W = torch.diag(W)
    
    # 节点度矩阵：每个节点所连接的超边权重之和
    D_v = torch.sum(H @ W @ H.t(), dim=1)
    D_v_invsqrt = torch.diag(torch.pow(D_v, -0.5))
    
    # 超边度矩阵：每个超边所包含的节点数
    D_e = torch.sum(H, dim=0)
    D_e_inv = torch.diag(torch.pow(D_e, -1.0))
    
    # 超图拉普拉斯矩阵
    L = torch.eye(num_nodes) - D_v_invsqrt @ H @ W @ D_e_inv @ H.t() @ D_v_invsqrt
    
    return L

def build_hypergraph(g):
    """
    基于节点标签构建超图结构
    
    @param g: PyG图对象，包含节点标签
    @return: 更新后的g，添加了超图相关的属性
    """
    labels = g.labels
    
    # 构建超图的关联矩阵
    H, num_hyperedges = build_hypergraph_incidence_matrix(labels)
    
    # 保存超图的关联矩阵
    g.hypergraph_H = H
    g.num_hyperedges = num_hyperedges
    
    # 创建超边到节点的映射（每个超边包含哪些节点）
    hyperedge_to_nodes = {}
    for i in range(num_hyperedges):
        hyperedge_to_nodes[i] = torch.where(labels == i)[0]
    
    g.hyperedge_to_nodes = hyperedge_to_nodes
    
    # 构建超图边索引表示（用于与PyG兼容）
    # 为每个超边创建一个虚拟节点，连接到所有在该超边中的节点
    edge_index_list = []
    
    for edge_id, nodes in hyperedge_to_nodes.items():
        # 创建从超边虚拟节点到其所有成员节点的边
        source = torch.ones_like(nodes) * (edge_id + g.num_nodes)  # 超边ID + 原始节点数作为虚拟节点ID
        edges = torch.stack([source, nodes], dim=0)
        edge_index_list.append(edges)
    
    # 合并所有边列表
    if edge_index_list:
        hyperedge_index = torch.cat(edge_index_list, dim=1)
        # 添加反向边，使图变为无向图
        hyperedge_index = to_undirected(hyperedge_index)
        g.hyperedge_index = hyperedge_index
    
    return g

def build_hypergraph_neighborhood(g):
    """
    构建超图的邻域结构，连接不同超边间的交互
    
    @param g: PyG图对象，包含超图结构
    @return: 更新后的g，添加了超图邻域结构
    """
    labels = g.labels
    H = g.hypergraph_H
    num_nodes = g.num_nodes
    num_hyperedges = g.num_hyperedges
    
    # 计算节点间的超边共享关系
    # 如果两个节点共享至少一个超边，它们之间就有连接
    node_similarity = H @ H.t()
    # 移除自环
    node_similarity.fill_diagonal_(0)
    
    # 构建节点间的交互边
    interaction_edges = torch.where(node_similarity > 0)
    interaction_edges = torch.stack(interaction_edges, dim=0)
    
    # 获取原始边
    original_edges = g.edge_index
    
    # 合并原始边和交互边
    combined_edges = torch.cat([original_edges, interaction_edges], dim=1)
    combined_edges = coalesce(combined_edges)  # 去除重复边
    
    # 更新图的边
    g.edge_index = combined_edges
    
    # 构建超边间的交互关系
    # 如果两个超边共享至少一个节点，它们之间就有连接
    hyperedge_similarity = H.t() @ H
    hyperedge_similarity.fill_diagonal_(0)
    
    # 构建超边间的交互矩阵
    hyperedge_interaction = torch.where(hyperedge_similarity > 0)
    g.hyperedge_interaction_graph = torch.stack(hyperedge_interaction, dim=0)
    
    return g 