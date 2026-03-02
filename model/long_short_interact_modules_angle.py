import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus
import torch.nn as nn
from .torchmdnet.models.utils import (
    CosineCutoff,
    act_class_mapping,
    vec_layernorm,
    max_min_norm,
    norm
)
from .utils import *
from torch_geometric.utils import softmax
import copy
import math
import torch.nn.functional as F


class LongShortIneractModel_distance(MessagePassing):
    r'''
    This is the long term model to capture the relationship b/w center node and long term groups
    First, perform a pointTransformer within each group to obtain the representation of each group.
    Second, MP is performed on a bipartite graph of size #nodes * #groups.
    Below is an disection of the architecture of the model

    in_channels_node ---- (PointTransformer) ----- > in channels group

    in_channels_group --- linear --- num_filters -|    |-- > out_channels
                                                  | MP |
    in_channels_node  --- linear --- num_filters -| -> |-- > out_channels
                                                  |    
    num_gaussians     --- linear --- num_filters -|    

    @param in_channels_node: The size of node features
    @param in_chnnels_right: The size of group features
    @param out_channels: The size of output node features and group features (unified output size)
    @param num_filters: number of filters 
    @param num_gaussians: number of gaussians used to expand edge weight
    @param cutoff: long term cutoff distance
    ''' 
    def __init__(self, hidden_channels,num_gaussians, cutoff ,max_group_num = 3,act = "silu",**kwargs):
        super().__init__()

        self.act1 = nn.SiLU()
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.max_group_num  = max_group_num
        
        self.act = nn.SiLU()
        self.mlp_1 = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)
        # self.lin4 = nn.Linear(out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        pass
        # torch.nn.init.xavier_uniform_(self.lin.weight)
        # self.lin.bias.data.fill_(0)
        # Get the center position
        
        
        # if self.group_center == 'geometric':
        #     group_pos = scatter(node_pos, labels, reduce='mean', dim=0)
        # elif self.group_center == 'center_of_mass':
        #     group_pos = scatter(node_pos * data.atomic_numbers, labels, reduce='sum', dim=0) / scatter(data.atomic_numbers, labels, reduce='sum', dim=0)
        # else:
        #     raise NotImplementedError

    def forward(self, edge_index, node_embedding, node_pos,group_embedding,group_pos,**kwargs):
        '''
        grouping_graph_edge_idx = data.grouping_graph # Grouping graph (intra group complete graph; inter group disconnected)
        edge_idx = data.interaction_graph # Bipartite graph
        
        '''

        nodes = edge_index[0]
        groups = edge_index[1]

        # Distance Expansion
        edge_weight = norm((node_pos[nodes] - group_pos[groups]), dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.cutoff) + 1.0) #cutoff function
        W = self.mlp_1(edge_attr) * C.view(-1, 1)

        node_embedding = self.lin1(node_embedding)
        group_embedding = self.lin2(group_embedding)
        # Message flow from group to node
        node_embedding = self.propagate(edge_index.flip(0), size = (num_groups, num_nodes), x=(group_embedding, node_embedding), W=W)/self.max_group_num
        node_embedding = self.lin3(node_embedding)
        
        return node_embedding,None
        
    def message(self, x_j, W):
        return x_j * W
    
    

# class CFVectorConvBipartite(MessagePassing):
class LongShortIneractModel_dis_direct(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            # 'mlp_edge_attr': nn.Sequential(
            #     nn.Linear(num_gaussians, hidden_channels),
            #     self.act,
            #     nn.Linear(hidden_channels, hidden_channels),),
            'edge_q': nn.Linear(hidden_channels, hidden_channels * num_heads),
            'edge_k': nn.Linear(3, hidden_channels * num_heads),  # 3维向量(xyz)
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)])    
        })
        self.model_1 = None
        # self.model_1 = copy.deepcopy(self.model_2)
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        # self.attn_layers.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            group_embedding = self.layernorm_group(group_embedding)
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        
        # vec to scalar, ||dot((W1*v), (W2*w))||
        # if self.select == 3:
        #     v_node_1 = self.model_2['linears'][2](node_vec)
        #     v_node_2 = self.model_2['linears'][3](node_vec)
        #     dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node)
        #     dv_node = m_v_node
        # elif self.select == 2:
        #     v_node_1 = self.model_2['linears'][2](node_vec)
        #     v_node_2 = self.model_2['linears'][3](node_vec)
        #     dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node)
        #     dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        # elif self.select == 1:
        #     v_node_1 = self.model_2['linears'][2](node_vec)
        #     v_node_2 = self.model_2['linears'][3](node_vec)
        #     dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        #     dv_node = m_v_node
        # elif self.select == 0:
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self,x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type):
        r'''
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        '''
        __supported_attn__ = ['softmax', 'silu']
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels) # num_groups x num_heads x attn_channels
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels) # num_nodes x num_heads x attn_channels
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 
        
        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim = -1) / math.sqrt(self.attn_channels)
        # attn = attn.sum(dim = -1) / self.attn_channels
        # attn = attn.sum(dim = -1)

            
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim = 0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        return attn, val      

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        # distance gated attention, larger the distance, smaller the attention should be.
        # There is the intuition, but we gave the model flexibility to learn on its own.
        if mode == 'node_to_group':
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim = -1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
            model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
        

        # a_ij = self.act(q_i * k_j * edge_attr.reshape(-1, self.num_heads, self.attn_channels) / (self.attn_channels ** 0.5))
        # a_ij = a_ij.sum(dim = -1, keepdim = True)
        # num_nodes, num_heads, attn_channels = q_i.shape
        
        # a_ij = q_i * k_j *  / (self.attn_channels ** 0.5)
        # a_ij = a_ij.(dim = -1, keepdim = True)
        
        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        # x, vec, w = features
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        # w = scatter(w, index, dim=self.node_dim, dim_size=dim_size)
        # return x, vec, w
        return x, vec    


class LongShortIneractModel_dis_direct_vector(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node




class LongShortIneractModel_dis_direct_vector2(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)

        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    



    
        
class LongShortIneractModel_dis_direct_vector3(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            # group_embedding = self.layernorm_group(group_embedding)
            # group_vec = vec_layernorm(group_vec, max_min_norm)

        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]

        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    





class LongShortIneractModel_dis_direct_two_way(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            # 'mlp_edge_attr': nn.Sequential(
            #     nn.Linear(num_gaussians, hidden_channels),
            #     self.act,
            #     nn.Linear(hidden_channels, hidden_channels),),
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)])    
        })
        # self.model_1 = None
        self.model_1 = nn.ModuleDict({
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar': nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
        })
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        # self.attn_layers.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]
        # attn_1, val_1 = self.calculate_attention(
        #     group_embedding, node_embedding, 
        #     edge_index[1], edge_index[0], 
        #     edge_attr, self.model_1, "silu"
        # )
        m_s_group, m_v_group = self.propagate(edge_index,
                                size = (num_nodes, num_groups,),
                                x =(node_embedding, group_embedding, ), 
                                v = node_vec[edge_index[0]],
                                u_ij = edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = None, 
                                val = None,
                                mode = 'node_to_group')
        
        # 更新官能团特征（仿照节点更新的复杂机制）
        # 官能团标量-向量耦合更新
        v_group_1 = self.model_2['linears'][2](group_vec)
        v_group_2 = self.model_2['linears'][3](group_vec)
        dx_group = (v_group_1 * v_group_2).sum(dim=1) * self.model_2['linears'][4](m_s_group) + self.model_2['linears'][5](m_s_group)
        dv_group = m_v_group + self.model_2['linears'][0](m_s_group).unsqueeze(1) * self.model_2['linears'][1](group_vec)
        
        # 更新官能团特征
        group_embedding = dx_group
        group_vec = dv_group
        
        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')       
        
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, angle_attr, model, attn_type, edge_weight=None, edge_vec=None, edge_index=None):
        """
        计算融合了边向量信息的多头注意力
        """
        __supported_attn__ = ['softmax', 'silu']
        
        # 1. 计算基础注意力 (与原模型相同)
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels)
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels)
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 
        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        
        # 2. 处理angle_attr的维度匹配
        if angle_attr is not None:
            # 确保angle_attr具有正确的维度 [num_edges, num_heads, attn_channels]
            if angle_attr.dim() == 2:  # [num_edges, hidden_channels]
                angle_attr = angle_attr.reshape(-1, self.num_heads, self.attn_channels)
            
            # 检查维度是否匹配
            assert angle_attr.shape == expanded_edge_weight.shape, \
                f"Dimension mismatch: angle_attr.shape {angle_attr.shape} vs expanded_edge_weight.shape {expanded_edge_weight.shape}"
            
            attn = q_i * k_j * expanded_edge_weight * angle_attr
        else:
            attn = q_i * k_j * expanded_edge_weight
            
        attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)  # [num_edges, num_heads]
        
        # 3. 加入边权重信息
        if edge_weight is not None:
            edge_weight_attn = torch.exp(-edge_weight / self.cutoff)
            edge_weight_attn = edge_weight_attn.unsqueeze(-1).repeat(1, self.num_heads)
        else:
            edge_weight_attn = torch.ones_like(attn)
        
        # 4. 应用边权重
        attn = attn * edge_weight_attn
        
        # 5. 应用注意力激活函数
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim=0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        
        return attn, val

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        # distance gated attention, larger the distance, smaller the attention should be.
        # There is the intuition, but we gave the model flexibility to learn on its own.
        if mode == 'node_to_group':
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim = -1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
            model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
        

        # a_ij = self.act(q_i * k_j * edge_attr.reshape(-1, self.num_heads, self.attn_channels) / (self.attn_channels ** 0.5))
        # a_ij = a_ij.sum(dim = -1, keepdim = True)
        # num_nodes, num_heads, attn_channels = q_i.shape
        
        # a_ij = q_i * k_j *  / (self.attn_channels ** 0.5)
        # a_ij = a_ij.(dim = -1, keepdim = True)
        
        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        # x, vec, w = features
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        # w = scatter(w, index, dim=self.node_dim, dim_size=dim_size)
        # return x, vec, w
        return x, vec
    
    



class LongShortIneractModel_dis_direct_vector2_drop(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8, p =0.1,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        if self.p>0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        # v_node = self.layernorm_node_vec(v_node)
        # v_group = self.layernorm_group_vec(v_group)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        # Message flow from node to group
        # relative_vec_embed = node_vec[edge_index[0]] - group_vec[edge_index[1]]
        
        # Message flow from group to node
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')       
        
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) 
        + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    

class ImprovedLongShortInteractModel(LongShortIneractModel_dis_direct_vector2_drop):
    """
    基于边向量的多注意力机制模型
    
    该模型实现了一种创新的多注意力机制，将边向量信息直接融入到注意力计算中，
    而不是简单地通过线性投影调制已有的注意力分数。
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p)
        
        # 保存参数
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
        self.cutoff = cutoff
        self.num_edge_heads = num_edge_heads  # 边向量的注意力头数量
        self.hidden_channels = hidden_channels
        
        # 重新定义边注意力网络层，使用正确的头数量
        self.model_2['edge_q'] = nn.Linear(hidden_channels, hidden_channels)
        self.model_2['edge_k'] = nn.Linear(3, hidden_channels)  # 3维边向量
        
        # 初始化新添加的层
        torch.nn.init.xavier_uniform_(self.model_2['edge_q'].weight)
        self.model_2['edge_q'].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.model_2['edge_k'].weight)
        self.model_2['edge_k'].bias.data.fill_(0)

    def calculate_edge_attention(self, node_embedding, edge_vec, edge_index, angle_attr):
        """
        计算基于边向量和夹角特征的注意力
        实现与calculate_attention类似的点乘操作
        """
        # 归一化边向量
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-10)      
        # 获取源节点和目标节点的索引
        source_idx, target_idx = edge_index
        
        # 1. 计算查询向量和边向量键向量
        q = self.model_2['edge_q'](node_embedding).view(-1, self.num_edge_heads, self.hidden_channels // self.num_edge_heads)
        q_i = q[source_idx]  # [num_edges, num_edge_heads, head_dim]
        
        # 对于每条边计算边向量键向量
        k_edge = self.model_2['edge_k'](edge_vec_normalized).view(-1, self.num_edge_heads, self.hidden_channels // self.num_edge_heads)
        # 注意：k_edge的形状已经是 [num_edges, num_edge_heads, head_dim]
        
        # 2. 融合夹角特征到键向量中 (类似expanded_edge_weight的作用)
        if angle_attr is not None:
            # 确保angle_attr的形状正确
            if angle_attr.shape[0] != k_edge.shape[0]:
                print(f"DEBUG: angle_attr shape: {angle_attr.shape}, k_edge shape: {k_edge.shape}")
                print(f"DEBUG: edge_vec shape: {edge_vec.shape}, q_i shape: {q_i.shape}")
                raise ValueError(f"Dimension mismatch: angle_attr shape {angle_attr.shape} vs k_edge shape {k_edge.shape}")
            
            # 检查angle_attr的维度是否可以正确重塑
            expected_size = self.num_edge_heads * (self.hidden_channels // self.num_edge_heads)
            if angle_attr.shape[1] != expected_size:
                print(f"DEBUG: angle_attr dim 1: {angle_attr.shape[1]}, expected: {expected_size}")
                print(f"DEBUG: hidden_channels: {self.hidden_channels}, num_edge_heads: {self.num_edge_heads}")
            
            # 将夹角特征转换为与边向量键向量相同的形状
            angle_k = angle_attr.view(-1, self.num_edge_heads, self.hidden_channels // self.num_edge_heads)
            # angle_k = angle_k * 0.1
            # 融合边几何特征和夹角特征
            k_combined = k_edge * angle_k  # [num_edges, num_edge_heads, head_dim]
        else:
            k_combined = k_edge
        
        # 3. 计算注意力分数 (类似calculate_attention中的点乘操作)
        edge_attn = q_i * k_combined  # [num_edges, num_edge_heads, head_dim]
        edge_attn = edge_attn.sum(dim=-1) / math.sqrt(self.hidden_channels // self.num_edge_heads)  # [num_edges, num_edge_heads]
        
        # 4. 应用激活函数 (与calculate_attention保持一致)
        edge_attn = F.silu(edge_attn)
        
        return edge_attn

    def calculate_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, angle_attr, model, attn_type, edge_weight=None, edge_vec=None, edge_index=None):
        """
        计算融合了边向量信息的多头注意力
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

        
        # # 2. 处理angle_attr的维度匹配
        # if angle_attr is not None:
        #     # 确保angle_attr具有正确的维度 [num_edges, num_heads, attn_channels]
        #     if angle_attr.dim() == 2:  # [num_edges, hidden_channels]
        #         angle_attr = angle_attr.reshape(-1, self.num_heads, self.attn_channels)
        #     # 检查维度是否匹配
        #     assert angle_attr.shape == expanded_edge_weight.shape, \
        #         f"Dimension mismatch: angle_attr.shape {angle_attr.shape} vs expanded_edge_weight.shape {expanded_edge_weight.shape}"
            
        #     attn = q_i * k_j * expanded_edge_weight * angle_attr
        # else:
        #     attn = q_i * k_j * expanded_edge_weight
            
        # attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)  # [num_edges, num_heads]
        
        # 3. 加入边权重信息
        if edge_weight is not None:
            edge_weight_attn = torch.exp(-edge_weight / self.cutoff)
            edge_weight_attn = edge_weight_attn.unsqueeze(-1).repeat(1, self.num_heads)
        else:
            edge_weight_attn = torch.ones_like(attn)

                
        # 3. 计算边向量注意力 (新增的多头注意力机制)
        edge_attn = None
        if edge_vec is not None and edge_index is not None:
            edge_attn = self.calculate_edge_attention(x_1, -edge_vec, edge_index, angle_attr)
            
            # # 调试信息：检查维度匹配
            # print(f"DEBUG: attn shape: {attn.shape}, edge_weight_attn shape: {edge_weight_attn.shape}, edge_attn shape: {edge_attn.shape}")
            # print(f"DEBUG: num_heads: {self.num_heads}, num_edge_heads: {self.num_edge_heads}")
            
            # 确保所有张量的维度匹配
            if attn.shape != edge_attn.shape:
                print(f"ERROR: Dimension mismatch - attn: {attn.shape} vs edge_attn: {edge_attn.shape}")
                # 应急处理：只使用基础注意力
                attn = attn * edge_weight_attn
            else:
                attn = attn * edge_weight_attn + edge_attn
        else:
            # 如果没有边向量信息，只使用基础注意力
            attn = attn * edge_weight_attn
        
        # # 4. 应用边权重
        # attn = attn * edge_weight_attn
        
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
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None, angle_attr=None):
        """
        前向传播，使用门控机制融合注意力
        支持分离的边特征和夹角特征
        
        Args:
            angle_attr: [num_edges, hidden_channels] 夹角特征，可选
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
        
        # 计算融合了角度信息的多头注意力
        attn_2, val_2 = self.calculate_attention(
                node_embedding, 
                group_embedding, 
                edge_index[0], 
                edge_index[1], 
                edge_attr,
                angle_attr,  # 直接传递angle_attr，让方法内部处理
                self.model_2, 
                "silu",
                edge_weight,
                -edge_vec,
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


