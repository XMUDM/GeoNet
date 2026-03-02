import torch
from torch_geometric.utils import subgraph,remove_self_loops

def label_to_graph(g, labels):
    r'''
    Convert a graph g with clustering label to a disjoint graph where the components are connected by labels
    @parms g: a PyG graph
    @parms label: a Tensor of the clustering label 
    Return: The edge list of the induced label grouping graph
    '''
    num_labels = torch.unique(labels).shape[0]
    sub_g_list = []
    for i in range(num_labels):
        curr = torch.where(labels == i)
        sub_g, _ = subgraph(curr[0], g.edge_index)
        sub_g_list.append(sub_g)
    edge_list = torch.cat(sub_g_list, dim = 1)
    return edge_list




def create_complete_graph(nodes):
    r'''
    Given a list of nodes, return the complete graph induced by those nodes
    @params curr: a numpy array of nodes
    Return: the edge index of the induced graph
    '''
    num_nodes = len(nodes)
    edges = torch.zeros(2, num_nodes, num_nodes)
    edges_src = nodes.unsqueeze(1).expand(num_nodes, num_nodes)
    edges_target = nodes.unsqueeze(0).expand(num_nodes, num_nodes)
    edges[0] = edges_src
    edges[1] = edges_target
    # ret = remove_self_loops(edges.reshape(2, -1))[0]
    # assert ret.shape[1] == (num_nodes * (num_nodes - 1)), f'Error, wrong # of graph edges. Want {num_nodes * (num_nodes - 1)}, but get {ret.shape[1]}'
    return edges.reshape(2, -1)

def label_to_complete_graph(labels):
    r'''
    Convert a graph g with clustering label to a disjoint graph where the components are connected by labels
    @parms g: a PyG graph
    @parms label: a Tensor of the clustering label 
    Return: The edge list of the induced label grouping graph
    '''
    num_labels = torch.unique(labels).shape[0]
    sub_g_list = []
    for i in range(num_labels):
        curr = torch.where(labels == i)
        sub_g  = create_complete_graph(curr[0])
        sub_g_list.append(sub_g)
    edge_list = torch.cat(sub_g_list, dim = 1)
    return edge_list

def build_grouping_graph(g):
    r'''
    Build the grouping graph of a graph g
    @params g: a PyG graph
    Return: The edge list of the grouping graph
    '''
    labels = g.labels
    g.grouping_graph = label_to_complete_graph(labels)

if __name__ == '__main__':
    print(label_to_complete_graph(labels= torch.LongTensor([1, 2 ,3 ,0 ,2 ,3, 1])))

def build_break_edge(g, cutoff=3.0):
    """
    Build break edge, i.e., the edges that connect different groupings together

    @params g: a PyG graph with grouping information
    @params cutoff: cutoff radius
    Return: Add the g.break_edge which is a list of edges.
    """
    labels = g.labels
    num_nodes = g.pos.shape[0]
    pos = g.pos

    edge_index = g.edge_index
    edge_attr = g.edge_attr

    cut_edge = []
    break_edge_attr = []
    
    for i in range(edge_index.shape[1]):
        start, end = edge_index[:, i]
        # If across groupings
        if labels[start] != labels[end]:
            cut_edge.append(edge_index[:, i].unsqueeze(-1))
            if hasattr(g, "edge_attr") and g.edge_attr is not None:
                break_edge_attr.append(edge_attr[i].unsqueeze(0))
    
    if cut_edge:
        break_edge = torch.cat(cut_edge, dim = 1)
        if break_edge_attr:
            g.break_edge_attr = torch.cat(break_edge_attr, dim = 0)
        g.break_edge = break_edge
    else:
        g.break_edge = None
        g.break_edge_attr = None


def build_group_edge(g):
    """Build the grouping edge index, which connects each group node to every other group node

    Args:
        g (PyG graph): The full graph with grouping information
    """
    unique_labels = torch.unique(g.labels)
    g.group_edge_index = create_complete_graph(unique_labels)

def build_node_group_edge(g, cutoff = 3.0):
    """Build the node group edge index, which connects each node to its respective group node
    
    Args:
        g (PyG graph): The full graph with grouping information
        cutoff (float): Cutoff radius for neighbor determination
    """
    num_nodes = g.pos.shape[0]
    num_groups = torch.unique(g.labels).shape[0]
    
    node_group_src = []
    node_group_tgt = []
    
    for i in range(num_nodes):
        node_group_src.append(i)
        node_group_tgt.append(g.labels[i])
    
    node_group_src = torch.LongTensor(node_group_src)
    node_group_tgt = torch.LongTensor(node_group_tgt)
    
    node_group_edge = torch.stack([node_group_src, node_group_tgt])
    g.node_group_edge_index = node_group_edge


def build_weighted_node_group_edge(g, cutoff=3.0):
    """
    Build weighted node-group edges for multi-group membership atoms based on electron density grouping
    
    Args:
        g (PyG graph): The full graph with grouping information and group_weights
        cutoff (float): Cutoff radius for neighbor determination
    """
    if not hasattr(g, 'group_weights') or g.group_weights is None:
        # 如果没有权重信息，使用普通的node_group_edge
        build_node_group_edge(g, cutoff)
        return
    
    num_nodes = g.pos.shape[0]
    num_groups = g.group_weights.shape[1]  # 从权重矩阵获取组数
    
    node_group_src = []
    node_group_tgt = []
    edge_weights = []
    
    # 为每个原子创建到每个组的边，权重基于group_weights
    for i in range(num_nodes):
        for g_idx in range(num_groups):
            weight = g.group_weights[i, g_idx]
            if weight > 0.01:  # 仅考虑权重大于阈值的连接
                node_group_src.append(i)
                node_group_tgt.append(g_idx)
                edge_weights.append(weight)
    
    if len(node_group_src) > 0:
        node_group_src = torch.LongTensor(node_group_src)
        node_group_tgt = torch.LongTensor(node_group_tgt)
        edge_weights = torch.FloatTensor(edge_weights)
        
        node_group_edge = torch.stack([node_group_src, node_group_tgt])
        g.node_group_edge_index = node_group_edge
        g.node_group_edge_weight = edge_weights
    else:
        # 如果没有有效连接，使用普通方法
        build_node_group_edge(g, cutoff)