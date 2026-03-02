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
            'edge_k': nn.Linear(hidden_channels, hidden_channels * num_heads),  # 几何投影特征
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
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None, return_contrastive_loss=False):
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
    
    



class LongShortIneractModel_dis_direct_vector2_drop(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8, p =0.1,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, fragment_ids=None, return_contrastive_loss=False):
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
        return dx_node, dv_node, 0
    

class ImprovedLongShortInteractModel(LongShortIneractModel_dis_direct_vector2_drop):
    """
    基于边向量的多注意力机制模型
    
    该模型实现了一种创新的多注意力机制，将边向量信息直接融入到注意力计算中，
    而不是简单地通过线性投影调制已有的注意力分数。
    
    新增功能：
    - 对比学习模块，用于学习更鲁棒的分子表示
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", 
                 num_heads=8, p=0.1, num_edge_heads=8, enable_contrastive=True, 
                 contrastive_temp=0.25, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads, p)
        
        # 保存参数
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
        self.cutoff = cutoff
        self.num_edge_heads = num_edge_heads  # 边向量的注意力头数量
        self.hidden_channels = hidden_channels
        # 对比学习相关参数
        self.enable_contrastive = enable_contrastive
        self.contrastive_temp = contrastive_temp
        self.contrastive_weight = 1.0  # 对比损失权重，可以通过调参优化
        
        # 重新定义边注意力网络层，使用正确的头数量
        self.model_2['edge_q'] = nn.Linear(hidden_channels, hidden_channels)
        self.model_2['edge_k'] = nn.Linear(hidden_channels, hidden_channels)  # 几何投影特征
        
        # 对比学习模块
        if self.enable_contrastive:
            self.contrastive_projector = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, hidden_channels // 4)
            )
            
            # 数据增强参数
            self.noise_std = 0.1  # 噪声标准差
            self.dropout_ratio = 0.1  # 边/节点dropout比例
            
        # 初始化新添加的层
        torch.nn.init.xavier_uniform_(self.model_2['edge_q'].weight)
        self.model_2['edge_q'].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.model_2['edge_k'].weight)
        self.model_2['edge_k'].bias.data.fill_(0)
        
        # 初始化对比学习模块
        if self.enable_contrastive:
            self._init_contrastive_module()
    
    def _init_contrastive_module(self):
        """初始化对比学习模块"""
        for layer in self.contrastive_projector:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
    
    def create_augmented_view(self, node_embedding, node_vec, group_embedding, group_vec, 
                            edge_index, edge_vec, edge_weight):
        """
        创建分子的增强视图用于对比学习
        
        增强策略：
        1. 添加高斯噪声到节点/组特征
        2. 随机dropout部分边
        3. 添加噪声到边向量
        """
        # 1. 节点特征增强：添加高斯噪声
        if self.training:
            node_noise = torch.randn_like(node_embedding) * self.noise_std
            node_embedding_aug = node_embedding + node_noise
            
            node_vec_noise = torch.randn_like(node_vec) * self.noise_std
            node_vec_aug = node_vec + node_vec_noise
            
            # 组特征增强
            group_noise = torch.randn_like(group_embedding) * self.noise_std
            group_embedding_aug = group_embedding + group_noise
            
            group_vec_noise = torch.randn_like(group_vec) * self.noise_std
            group_vec_aug = group_vec + group_vec_noise
            
            # 2. 边增强：随机dropout边 + 添加噪声到边向量
            num_edges = edge_index.shape[1]
            edge_mask = torch.rand(num_edges, device=edge_index.device) > self.dropout_ratio
            
            edge_index_aug = edge_index[:, edge_mask]
            edge_vec_aug = edge_vec[edge_mask]
            edge_weight_aug = edge_weight[edge_mask]
            
            # 为保留的边添加噪声
            edge_vec_noise = torch.randn_like(edge_vec_aug) * self.noise_std * 0.5  # 边向量噪声较小
            edge_vec_aug = edge_vec_aug + edge_vec_noise
            
        else:
            # 推理时不做增强
            node_embedding_aug = node_embedding
            node_vec_aug = node_vec
            group_embedding_aug = group_embedding
            group_vec_aug = group_vec
            edge_index_aug = edge_index
            edge_vec_aug = edge_vec
            edge_weight_aug = edge_weight
            
        return (node_embedding_aug, node_vec_aug, group_embedding_aug, group_vec_aug, 
                edge_index_aug, edge_vec_aug, edge_weight_aug)
    
    def compute_contrastive_loss(self, repr1, repr2, temperature=None):
        """
        计算对比学习损失 (InfoNCE Loss)
        
        Args:
            repr1: 第一个视图的表示 [batch_size, hidden_dim]
            repr2: 第二个视图的表示 [batch_size, hidden_dim]
            temperature: 温度参数
        """
        if temperature is None:
            temperature = self.contrastive_temp
            
        # L2归一化
        repr1 = F.normalize(repr1, dim=-1)
        repr2 = F.normalize(repr2, dim=-1)
        
        batch_size = repr1.shape[0]

        # 标准InfoNCE损失
        # 计算相似度矩阵
        sim_matrix = torch.matmul(repr1, repr2.T) / temperature
        # print(f"相似度矩阵形状: {sim_matrix.shape}")
        # print(f"相似度矩阵对角线 (正样本): {torch.diag(sim_matrix).mean().item():.4f}")
        # print(f"相似度矩阵非对角线 (负样本): {(sim_matrix.sum() - torch.diag(sim_matrix).sum()).item()/(batch_size*(batch_size-1)):.4f}")
        
        # 提取正样本（对角线元素）nohup bash train.sh > 3m_ceshi_new_con_0.01_t0.3.log 2>&1 &
        positive_samples = torch.diag(sim_matrix)
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(sim_matrix)
        exp_sim_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim)  # 排除正样本
        
        # 避免log(0)
        exp_sim_sum = torch.clamp(exp_sim_sum, min=1e-8)
        
        # 计算正样本概率 (这是关键的监控指标)
        positive_exp = torch.diag(exp_sim)
        total_exp = positive_exp + exp_sim_sum
        positive_probs = positive_exp / total_exp
        
        loss = -positive_samples + torch.log(torch.diag(exp_sim) + exp_sim_sum)
        
        # 监控关键指标
        avg_positive = positive_samples.mean().item()
        avg_negative = (exp_sim_sum / (batch_size - 1)).mean().item() if batch_size > 1 else 0
        separation = avg_positive - torch.log(torch.tensor(avg_negative + 1e-8)).item()
        
        # 正样本概率统计
        avg_positive_prob = positive_probs.mean().item()
        min_positive_prob = positive_probs.min().item()
        max_positive_prob = positive_probs.max().item()
        std_positive_prob = positive_probs.std().item()
        
        # 负样本概率计算
        negative_probs = exp_sim_sum / total_exp  # 所有负样本的概率和
        avg_negative_prob = negative_probs.mean().item()
        
        # print(f"正样本概率: 平均={avg_positive_prob:.4f}, 范围=[{min_positive_prob:.4f}, {max_positive_prob:.4f}], 标准差={std_positive_prob:.4f}")
        # print(f"负样本概率和: 平均={avg_negative_prob:.4f}")
        
        
        final_loss = loss.mean()

        
        return final_loss

    def aggregate_molecular_representation(self, node_features, group_features, node_vec_features=None, batch_idx=None):
        """
        聚合分子级别的表示，支持向量特征
        
        Args:
            node_features: 节点标量特征 [num_nodes, hidden_dim]
            group_features: 组标量特征 [num_groups, hidden_dim]
            node_vec_features: 节点向量特征 [num_nodes, 3, hidden_dim]（可选）
            batch_idx: 批次索引
        """
        if batch_idx is not None:
            # 按分子聚合节点特征
            node_repr = scatter_mean(node_features, batch_idx, dim=0)
            # 假设group和node有相同的batch索引分布
            group_repr = scatter_mean(group_features, batch_idx, dim=0)
            
            # 聚合向量特征
            if node_vec_features is not None:
                node_vec_repr = scatter_mean(node_vec_features, batch_idx, dim=0)  # [batch_size, 3, hidden_dim]
                # 计算向量特征的模长并在hidden_dim维度上求平均
                vec_norm = torch.norm(node_vec_repr, dim=1)  # [batch_size, hidden_dim]
                vec_norm_scalar = vec_norm.mean(dim=-1, keepdim=True)  # [batch_size, 1]
                # 扩展到与node_repr相同的维度
                vec_norm_expanded = vec_norm_scalar.expand(-1, node_repr.shape[-1])  # [batch_size, hidden_dim]
                node_repr = node_repr + vec_norm_expanded  # 融合向量信息到标量表示
        else:
            # 全局聚合 (单分子情况)
            node_repr = node_features.mean(dim=0, keepdim=True)
            group_repr = group_features.mean(dim=0, keepdim=True)
            
            # 聚合向量特征
            if node_vec_features is not None:
                node_vec_repr = node_vec_features.mean(dim=0, keepdim=True)  # [1, 3, hidden_dim]
                vec_norm = torch.norm(node_vec_repr, dim=1)  # [1, hidden_dim]
                vec_norm_scalar = vec_norm.mean(dim=-1, keepdim=True)  # [1, 1]
                vec_norm_expanded = vec_norm_scalar.expand(-1, node_repr.shape[-1])  # [1, hidden_dim]
                node_repr = node_repr + vec_norm_expanded
        
        # 融合节点和组表示
        mol_repr = node_repr + group_repr
        return mol_repr

    def calculate_edge_attention(self, node_embedding, node_vec, edge_vec, edge_index, angle_attr):
        """
        计算基于边向量和夹角特征的注意力
        使用向量特征和几何投影，保持SE(3)等变性
        """
        # 获取源节点和目标节点的索引
        source_idx, target_idx = edge_index
        
        # 归一化边向量 (方向向量)
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        u_ij = edge_vec / torch.clamp(edge_vec_norm, min=1e-8)  # [num_edges, 3]
        
        # 1. 从node_vec计算几何投影特征（保持等变性）
        node_vec_i = node_vec[source_idx]  # [num_edges, 3, hidden_channels]
        
        # 计算向量投影：(node_vec · u_ij) -> 标量几何特征
        # node_vec_i: [num_edges, 3, hidden_channels], u_ij: [num_edges, 3]
        geometric_proj = torch.einsum('eic,ei->ec', node_vec_i, u_ij)  # [num_edges, hidden_channels]
        
        # 2. 计算查询向量（基于标量特征）
        q = self.model_2['edge_q'](node_embedding)  # [num_nodes, hidden_channels]
        q = q.view(-1, self.num_heads, self.attn_channels)  # [num_nodes, num_heads, attn_channels]
        q_i = q[source_idx]  # [num_edges, num_heads, attn_channels]
        
        # 3. 计算键向量（基于几何投影特征）
        k_geom = self.model_2['edge_k'](geometric_proj)  # [num_edges, hidden_channels]
        k_geom = k_geom.view(-1, self.num_heads, self.attn_channels)  # [num_edges, num_heads, attn_channels]
        
        # 4. 融合夹角特征到键向量中 (如果需要的话)
        if angle_attr is not None:
            # 确保angle_attr的形状正确
            if angle_attr.shape[0] != k_geom.shape[0]:
                print(f"DEBUG: angle_attr shape: {angle_attr.shape}, k_geom shape: {k_geom.shape}")
                print(f"DEBUG: edge_vec shape: {edge_vec.shape}, q_i shape: {q_i.shape}")
                raise ValueError(f"Dimension mismatch: angle_attr shape {angle_attr.shape} vs k_geom shape {k_geom.shape}")
            
            # 检查angle_attr的维度并重塑
            if angle_attr.shape[1] != self.hidden_channels:
                # 如果angle_attr维度不匹配，需要投影到正确维度
                if not hasattr(self.model_2, 'angle_proj'):
                    self.model_2['angle_proj'] = nn.Linear(angle_attr.shape[1], self.hidden_channels).to(angle_attr.device)
                angle_attr = self.model_2['angle_proj'](angle_attr)
            
            # 将夹角特征转换为与几何键向量相同的形状
            angle_k = angle_attr.view(-1, self.num_heads, self.attn_channels)
            # 融合几何特征和夹角特征
            k_combined = k_geom + angle_k  # [num_edges, num_edge_heads, head_dim]
        else:
            k_combined = k_geom
        
        # 5. 计算注意力分数 (类似calculate_attention中的点乘操作)
        edge_attn = q_i * k_combined  # [num_edges, num_heads, attn_channels]
        edge_attn = edge_attn.sum(dim=-1) / math.sqrt(self.attn_channels)  # [num_edges, num_heads]
        
        # 6. 应用激活函数 (与calculate_attention保持一致)
        edge_attn = F.silu(edge_attn)
        
        return edge_attn

    def calculate_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, angle_attr, model, attn_type, edge_weight=None, edge_vec=None, edge_index=None, node_vec=None):
        """
        计算融合了边向量信息的多头注意力
        Args:
            node_vec: 节点向量特征，用于计算边几何投影 [num_nodes, 3, hidden_channels]
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
        else:
            edge_weight_attn = torch.ones_like(attn)
        
        # 3. 计算边向量注意力 (新增的多头注意力机制)
        edge_attn = None
        if edge_vec is not None and edge_index is not None and node_vec is not None:
            # 使用向量特征计算基于几何投影的边注意力
            edge_attn = self.calculate_edge_attention(x_1, node_vec, -edge_vec, edge_index, angle_attr)
            
            attn = attn * edge_weight_attn + edge_attn
        else:
            # 如果没有边向量信息，只使用基础注意力
            attn = attn * edge_weight_attn
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
                group_pos, group_vec, edge_attr, edge_weight, edge_vec, 
                fragment_ids=None, angle_attr=None, batch_idx=None, 
                return_contrastive_loss=True):
        """
        前向传播，支持对比学习
        
        Args:
            return_contrastive_loss: 是否返回对比学习损失
            batch_idx: 批次索引，用于对比学习
        """
        # 标准前向传播
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
            angle_attr,
            self.model_2, 
            "silu",
            edge_weight,
            -edge_vec,
            edge_index,
            node_vec
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

        # 对比学习
        contrastive_loss = None
        if self.enable_contrastive and return_contrastive_loss and self.training:
            # 创建增强视图
            # print("enable_contrastive")
            (node_emb_aug, node_vec_aug, group_emb_aug, group_vec_aug, 
             edge_idx_aug, edge_vec_aug, edge_weight_aug) = self.create_augmented_view(
                node_embedding, node_vec, group_embedding, group_vec,
                edge_index, edge_vec, edge_weight
            )
            
            # 确保增强后还有边
            if edge_idx_aug.shape[1] > 0:
                # 根据增强后的边索引选择对应的边属性
                num_edges_orig = edge_index.shape[1]
                num_edges_aug = edge_idx_aug.shape[1]
                
                # 创建边属性映射 - 简化处理，重复使用原始边属性
                if edge_attr is not None:
                    if num_edges_aug <= num_edges_orig:
                        edge_attr_aug = edge_attr[:num_edges_aug]  # 取前N个
                    else:
                        # 如果增强后边数增加了，重复使用边属性
                        repeat_times = (num_edges_aug + num_edges_orig - 1) // num_edges_orig
                        edge_attr_aug = edge_attr.repeat(repeat_times, 1)[:num_edges_aug]
                else:
                    edge_attr_aug = None
                
                # 对增强视图进行完整的前向传播
                attn_2_aug, val_2_aug = self.calculate_attention(
                    node_emb_aug, group_emb_aug, edge_idx_aug[0], edge_idx_aug[1], 
                    edge_attr_aug, angle_attr, self.model_2, "silu", 
                    edge_weight_aug, -edge_vec_aug, edge_idx_aug, node_vec
                )
                
                # 重要：使用计算出的 attn_2_aug 和 val_2_aug 进行消息传递
                m_s_node_aug, m_v_node_aug = self.propagate(
                    edge_idx_aug.flip(0),
                    size=(group_emb_aug.shape[0], node_emb_aug.shape[0]),
                    x=(group_emb_aug, node_emb_aug),
                    v=group_vec_aug[edge_idx_aug[1]],
                    u_ij=-edge_vec_aug,
                    d_ij=edge_weight_aug, 
                    attn_score=attn_2_aug,  # 使用计算出的增强注意力
                    val=val_2_aug[edge_idx_aug[1]],  # 使用计算出的增强值
                    mode='group_to_node'
                )
                
                # 计算增强视图的特征更新
                v_node_1_aug = self.model_2['linears'][2](node_vec_aug)
                v_node_2_aug = self.model_2['linears'][3](node_vec_aug)
                dx_node_aug = (v_node_1_aug * v_node_2_aug).sum(dim=1) * self.model_2['linears'][4](m_s_node_aug) + self.model_2['linears'][5](m_s_node_aug)
                # 重要：计算增强视图的向量特征更新
                dv_node_aug = m_v_node_aug + self.model_2['linears'][0](m_s_node_aug).unsqueeze(1) * self.model_2['linears'][1](node_vec_aug)                
                # 直接使用节点特征进行对比学习，不进行分子级别聚合
                # 这样可以获得更多的对比样本，避免批次大小为1的问题
                
                # 融合标量和向量特征：将向量特征的模长作为额外信息加入标量特征
                # 原始视图：标量 + 向量特征 (移除detach以保持梯度传播)
                vec_norm1 = torch.norm(dv_node, dim=1)  # [num_nodes, hidden_dim] - 移除了detach()
                vec_scalar1 = vec_norm1.mean(dim=-1, keepdim=True)  # [num_nodes, 1]
                node_repr1 = dx_node + vec_scalar1  # [num_nodes, hidden_dim] - 移除了detach()
                
                # 增强视图：标量 + 向量特征  
                vec_norm2 = torch.norm(dv_node_aug, dim=1)  # [num_nodes_aug, hidden_dim]
                vec_scalar2 = vec_norm2.mean(dim=-1, keepdim=True)  # [num_nodes_aug, 1]
                node_repr2 = dx_node_aug + vec_scalar2  # [num_nodes_aug, hidden_dim]
                
                # 处理节点数量不匹配的情况（由于边dropout）
                min_nodes = min(node_repr1.shape[0], node_repr2.shape[0])
                node_repr1 = node_repr1[:min_nodes]  # 截取相同数量的节点
                node_repr2 = node_repr2[:min_nodes]
                
                # 投影到对比学习空间
                proj_repr1 = self.contrastive_projector(node_repr1)
                proj_repr2 = self.contrastive_projector(node_repr2)
                
                # 计算对比损失
                contrastive_loss = self.compute_contrastive_loss(node_repr1, node_repr2)
                
                # 添加对比损失权重调节

                contrastive_loss = 0.01 * contrastive_loss
                # print(f"权重化对比损失 (有边): {contrastive_loss.item():.6f}")
            else:
                # 如果增强后没有边，使用原始特征和增强的节点嵌入作为对比
                # 原始视图：标量 + 向量特征
                vec_norm1 = torch.norm(dv_node, dim=1)  # [num_nodes, hidden_dim] - 移除了detach()
                vec_scalar1 = vec_norm1.mean(dim=-1, keepdim=True)  # [num_nodes, 1]
                node_repr1 = dx_node + vec_scalar1  # [num_nodes, hidden_dim] - 移除了detach()
                
                # 增强视图：只有标量特征（因为没有边传播，没有向量更新）
                node_repr2 = node_emb_aug         # [num_nodes, hidden_dim] - 增强的节点嵌入
                
                print(f"node_repr1 shape (no edges, 包含向量信息): {node_repr1.shape}")
                print(f"node_repr2 shape (no edges, 仅标量): {node_repr2.shape}")
                
                # 投影到对比学习空间
                proj_repr1 = self.contrastive_projector(node_repr1)
                proj_repr2 = self.contrastive_projector(node_repr2)
                
                # 计算对比损失
                contrastive_loss = self.compute_contrastive_loss(proj_repr1, proj_repr2)
                
                # 添加对比损失权重调节
                contrastive_weight = getattr(self, 'contrastive_weight', 1.0)
                contrastive_loss = contrastive_weight * contrastive_loss
                
                print(f"权重化对比损失 (无边): {contrastive_loss.item():.6f}")
        if return_contrastive_loss:
            return dx_node, dv_node, contrastive_loss
        else:
            return dx_node, dv_node


class LongShortIneractModel_graph_aware(LongShortIneractModel_dis_direct):
    """
    图结构感知的长短程交互模型
    使用方案4：图结构感知的Attention，考虑邻居信息并使用RBF核函数
    """
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", num_heads=8, **kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads)
        
        # 保存hidden_channels用于后续使用
        self.hidden_channels = hidden_channels
        
        # 图结构感知所需的模块
        # 1. 邻居聚合模块（用于node和group）
        self.model_2['neighbor_agg_node'] = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.model_2['neighbor_agg_group'] = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 2. RBF核参数（可学习的gamma参数）- 注册为模块参数，不能放在ModuleDict中
        self.register_parameter('rbf_kernel', nn.Parameter(torch.ones(num_heads)))
        
        # 3. 邻居融合权重（控制自身特征和邻居特征的融合比例）- 注册为模块参数
        self.register_parameter('neighbor_fusion_weight', nn.Parameter(torch.tensor(0.5)))
        
        # 4. 边注意力模块（用于计算基于边向量的注意力）
        self.model_2['edge_q'] = nn.Linear(hidden_channels, hidden_channels)
        self.model_2['edge_k'] = nn.Linear(hidden_channels, hidden_channels)  # 几何投影特征
        
        # 5. 边注意力RBF核参数（独立的gamma参数，用于边注意力）
        self.register_parameter('edge_rbf_kernel', nn.Parameter(torch.ones(num_heads)))
        
        # 6. 边注意力融合权重（控制边注意力对总注意力的影响）
        self.register_parameter('edge_attn_weight', nn.Parameter(torch.tensor(0.5)))
        
        # 初始化新添加的参数
        torch.nn.init.xavier_uniform_(self.model_2['neighbor_agg_node'][0].weight)
        torch.nn.init.xavier_uniform_(self.model_2['neighbor_agg_node'][2].weight)
        torch.nn.init.xavier_uniform_(self.model_2['neighbor_agg_group'][0].weight)
        torch.nn.init.xavier_uniform_(self.model_2['neighbor_agg_group'][2].weight)
        torch.nn.init.xavier_uniform_(self.model_2['edge_q'].weight)
        self.model_2['edge_q'].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.model_2['edge_k'].weight)
        self.model_2['edge_k'].bias.data.fill_(0)
    
    def calculate_edge_attention(self, node_embedding, node_vec, edge_vec, edge_index, angle_attr=None,
                                node_edge_index=None):
        """
        计算基于边向量和几何投影的注意力（使用RBF核函数）
        使用向量特征和几何投影，保持SE(3)等变性
        
        Args:
            node_embedding: 节点标量特征 [num_nodes, hidden_channels]
            node_vec: 节点向量特征 [num_nodes, 3, hidden_channels]
            edge_vec: 边向量 [num_edges, 3]
            edge_index: 边索引 [2, num_edges]，edge_index[0]是源节点，edge_index[1]是目标节点
            angle_attr: 夹角特征（可选） [num_edges, hidden_channels]
            node_edge_index: node之间的边索引 [2, num_edges_node]，用于聚合node的邻居
            
        Returns:
            edge_attn: 基于RBF的边注意力分数 [num_edges, num_heads]
        """
        # 获取源节点和目标节点的索引
        source_idx, target_idx = edge_index
        
        # 归一化边向量 (方向向量)
        edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
        u_ij = edge_vec / torch.clamp(edge_vec_norm, min=1e-8)  # [num_edges, 3]
        
        # 1. 从node_vec计算几何投影特征（保持等变性）
        node_vec_i = node_vec[source_idx]  # [num_edges, 3, hidden_channels]
        
        # 计算向量投影：(node_vec · u_ij) -> 标量几何特征
        # node_vec_i: [num_edges, 3, hidden_channels], u_ij: [num_edges, 3]
        geometric_proj = torch.einsum('eic,ei->ec', node_vec_i, u_ij)  # [num_edges, hidden_channels]
        
        # ========== 聚合邻居信息 ==========
        # 2. 计算查询向量（基于标量特征，考虑邻居信息）
        q = self.model_2['edge_q'](node_embedding)  # [num_nodes, hidden_channels]
        q = q.view(-1, self.num_heads, self.attn_channels)  # [num_nodes, num_heads, attn_channels]
        q_i = q[source_idx]  # [num_edges, num_heads, attn_channels]
        
        # 2.1 聚合node的邻居信息（用于query）
        if node_edge_index is not None and node_embedding is not None:
            num_nodes = node_embedding.shape[0]
            
            # 聚合邻居特征（mean pooling）
            neighbor_features = scatter(
                node_embedding[node_edge_index[0]], 
                node_edge_index[1], 
                dim=0, 
                dim_size=num_nodes,
                reduce='mean'
            )  # [num_nodes, hidden_channels]
            
            # 通过MLP处理邻居特征
            neighbor_context = self.model_2['neighbor_agg_node'](neighbor_features)
            neighbor_context_q = neighbor_context.view(-1, self.num_heads, self.attn_channels)
            
            # 融合自身特征和邻居特征
            fusion_weight = torch.clamp(self.neighbor_fusion_weight, 0.0, 1.0)
            q_i = (1 - fusion_weight) * q_i + fusion_weight * neighbor_context_q[source_idx]
        
        # 3. 计算键向量（基于几何投影特征，考虑邻居信息）
        k_geom = self.model_2['edge_k'](geometric_proj)  # [num_edges, hidden_channels]
        k_geom = k_geom.view(-1, self.num_heads, self.attn_channels)  # [num_edges, num_heads, attn_channels]
        
        # 3.1 聚合几何投影特征的邻居信息（用于key）
        if node_edge_index is not None:
            num_nodes = node_embedding.shape[0]
            
            # 聚合几何投影特征的邻居信息
            # 首先需要将geometric_proj映射回节点空间（通过source_idx）
            # 然后聚合邻居，再映射回边空间
            geometric_proj_node = scatter(
                geometric_proj,
                source_idx,
                dim=0,
                dim_size=num_nodes,
                reduce='mean'
            )  # [num_nodes, hidden_channels] - 每个节点的平均几何投影
            
            # 聚合邻居的几何投影特征
            neighbor_geometric_proj = scatter(
                geometric_proj_node[node_edge_index[0]],
                node_edge_index[1],
                dim=0,
                dim_size=num_nodes,
                reduce='mean'
            )  # [num_nodes, hidden_channels]
            
            # 通过MLP处理邻居几何投影特征
            neighbor_geometric_context = self.model_2['neighbor_agg_node'](neighbor_geometric_proj)
            neighbor_geometric_k = neighbor_geometric_context.view(-1, self.num_heads, self.attn_channels)
            
            # 融合自身几何特征和邻居几何特征
            fusion_weight = torch.clamp(self.neighbor_fusion_weight, 0.0, 1.0)
            k_geom = (1 - fusion_weight) * k_geom + fusion_weight * neighbor_geometric_k[source_idx]
        
        # 4. 融合夹角特征到键向量中 (如果需要的话)
        if angle_attr is not None:
            # 确保angle_attr的形状正确
            if angle_attr.shape[0] != k_geom.shape[0]:
                raise ValueError(f"Dimension mismatch: angle_attr shape {angle_attr.shape} vs k_geom shape {k_geom.shape}")
            
            # 检查angle_attr的维度并重塑
            if angle_attr.shape[1] != self.hidden_channels:
                # 如果angle_attr维度不匹配，需要投影到正确维度
                if not hasattr(self.model_2, 'angle_proj'):
                    self.model_2['angle_proj'] = nn.Linear(angle_attr.shape[1], self.hidden_channels).to(angle_attr.device)
                    torch.nn.init.xavier_uniform_(self.model_2['angle_proj'].weight)
                    self.model_2['angle_proj'].bias.data.fill_(0)
                angle_attr = self.model_2['angle_proj'](angle_attr)
            
            # 将夹角特征转换为与几何键向量相同的形状
            angle_k = angle_attr.view(-1, self.num_heads, self.attn_channels)
            # 融合几何特征和夹角特征
            k_combined = k_geom + angle_k  # [num_edges, num_heads, attn_channels]
        else:
            k_combined = k_geom
        
        # 5. 使用RBF核函数计算注意力分数（替代点积）
        # RBF核：exp(-gamma * ||q - k||^2)
        q_k_diff = q_i - k_combined  # [num_edges, num_heads, attn_channels]
        squared_dist = (q_k_diff ** 2).sum(dim=-1)  # [num_edges, num_heads]
        
        # 获取可学习的gamma参数（确保为正）
        gamma_edge = torch.abs(self.edge_rbf_kernel) + 0.1  # [num_heads]
        
        # 计算RBF核相似度
        edge_attn = torch.exp(-gamma_edge.unsqueeze(0) * squared_dist)  # [num_edges, num_heads]
        
        # 6. 应用激活函数 (与calculate_attention保持一致)
        edge_attn = act_class_mapping['silu']()(edge_attn)
        
        return edge_attn
    
    def calculate_attention(self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type,
                           edge_index=None, labels=None, node_embedding=None, group_embedding=None,
                           node_vec=None, edge_vec=None, angle_attr=None, node_edge_index=None):
        """
        图结构感知的Attention计算（包含边向量注意力）
        
        Args:
            x_1: query特征（node embedding）
            x_2: key特征（group embedding）
            x1_index: node索引 [num_edges]
            x2_index: group索引 [num_edges]
            expanded_edge_weight: 扩展的边权重特征
            model: 模型参数字典
            attn_type: attention类型 ('softmax' 或 'silu')
            edge_index: node-group交互图的边索引 [2, num_edges]，用于边注意力计算
            node_edge_index: node之间的边索引 [2, num_edges_node]，用于聚合node的邻居
            labels: 节点到group的标签映射 [num_nodes]，用于聚合group的邻居
            node_embedding: 原始node embedding（用于邻居聚合和边注意力）
            group_embedding: 原始group embedding（用于邻居聚合）
            node_vec: 节点向量特征 [num_nodes, 3, hidden_channels]，用于边注意力计算
            edge_vec: 边向量 [num_edges, 3]，用于边注意力计算
            angle_attr: 夹角特征（可选） [num_edges, hidden_channels]，用于边注意力计算
        """
        __supported_attn__ = ['softmax', 'silu']
        
        # 获取基础Q、K、V
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels)
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels)
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels)
        
        # ========== 步骤1：图结构感知 - 聚合邻居信息 ==========
        q_i = q[x1_index]  # [num_edges, num_heads, attn_channels]
        k_j = k[x2_index]  # [num_edges, num_heads, attn_channels]        # 1.1 聚合node的邻居信息（使用node_edge_index，即node之间的边）
        if node_edge_index is not None and node_embedding is not None:
            # 使用node_edge_index聚合每个node的邻居
            # node_edge_index[0]是源节点，node_edge_index[1]是目标节点
            # 我们聚合所有指向node_i的邻居
            num_nodes = node_embedding.shape[0]
            
            # 聚合邻居特征（mean pooling）
            neighbor_features = scatter(
                node_embedding[node_edge_index[0]], 
                node_edge_index[1], 
                dim=0, 
                dim_size=num_nodes,
                reduce='mean'
            )  # [num_nodes, hidden_channels]
            
            # 通过MLP处理邻居特征
            neighbor_context = model['neighbor_agg_node'](neighbor_features)
            neighbor_context_q = neighbor_context.reshape(-1, self.num_heads, self.attn_channels)
            
            # 融合自身特征和邻居特征
            fusion_weight = torch.clamp(self.neighbor_fusion_weight, 0.0, 1.0)
            q_i = (1 - fusion_weight) * q_i + fusion_weight * neighbor_context_q[x1_index]
        else:
            # 如果没有提供edge_index，使用原始特征
            pass
        
        # 1.2 聚合group的邻居信息（通过labels）
        if labels is not None and group_embedding is not None:
            num_groups = group_embedding.shape[0]
            
            # 聚合属于同一group的node特征（作为group的"邻居"）
            # 这里我们将属于同一group的nodes视为group的邻居
            group_neighbor_features = scatter(
                node_embedding,
                labels,
                dim=0,
                dim_size=num_groups,
                reduce='mean'
            )  # [num_groups, hidden_channels]
            
            # 通过MLP处理group的邻居特征
            group_neighbor_context = model['neighbor_agg_group'](group_neighbor_features)
            group_neighbor_context_k = group_neighbor_context.reshape(-1, self.num_heads, self.attn_channels)
            
            # 融合自身特征和邻居特征
            fusion_weight = torch.clamp(self.neighbor_fusion_weight, 0.0, 1.0)
            k_j = (1 - fusion_weight) * k_j + fusion_weight * group_neighbor_context_k[x2_index]
        else:
            # 如果没有提供labels，使用原始特征
            pass
        
        # ========== 步骤2：使用RBF核函数计算相似度 ==========
        # RBF核：exp(-gamma * ||q - k||^2)
        # 计算q和k之间的差异
        q_k_diff = q_i - k_j  # [num_edges, num_heads, attn_channels]
        
        # 计算L2距离的平方
        squared_dist = (q_k_diff ** 2).sum(dim=-1)  # [num_edges, num_heads]
        
        # 获取可学习的gamma参数（确保为正）
        gamma = torch.abs(self.rbf_kernel) + 0.1  # [num_heads]
        
        # 计算RBF核相似度（node-group注意力）
        attn_node_group = torch.exp(-gamma.unsqueeze(0) * squared_dist)  # [num_edges, num_heads]
        
        # ========== 步骤3：融合边向量注意力（基于RBF） ==========
        if node_vec is not None and edge_vec is not None and edge_index is not None:
            # 计算基于边向量的注意力（使用RBF核函数，包含邻居聚合）
            edge_attn = self.calculate_edge_attention(
                node_embedding, 
                node_vec, 
                edge_vec, 
                edge_index, 
                angle_attr,
                node_edge_index=node_edge_index  # 传递node_edge_index用于邻居聚合
            )  # [num_edges, num_heads]
            
            # 融合node-group注意力和边注意力
            edge_weight_val = torch.clamp(self.edge_attn_weight, 0.0, 1.0)
            attn = (1.0 - edge_weight_val) * attn_node_group + edge_weight_val * edge_attn
        else:
            attn = attn_node_group
        
        # ========== 步骤4：融合edge_weight作为重要性权重 ==========
        # edge_weight作为门控机制，控制attention的强度
        if expanded_edge_weight.dim() == 3:
            # 如果expanded_edge_weight是3D的 [num_edges, num_heads, attn_channels]
            edge_importance = expanded_edge_weight.mean(dim=-1)  # [num_edges, num_heads]
        else:
            # 如果是2D的 [num_edges, num_gaussians]
            edge_importance = expanded_edge_weight.mean(dim=-1, keepdim=True)  # [num_edges, 1]
            edge_importance = edge_importance.expand(-1, self.num_heads)  # [num_edges, num_heads]
        
        # 使用sigmoid将edge_weight转换为[0, 1]范围的重要性权重
        # edge_gate = torch.sigmoid(edge_importance)
        
        # # 应用edge权重门控
        # attn = attn * edge_gate
        
        # ========== 步骤5：应用激活函数 ==========
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim=0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        
        return attn, val
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec,
                fragment_ids=None, **kwargs):
        """
        前向传播，支持图结构感知的attention
        
        Args:
            edge_index: node-group交互图的边索引 [2, num_edges]
            node_embedding: 节点标量特征 [num_nodes, hidden_channels]
            node_pos: 节点位置 [num_nodes, 3]
            node_vec: 节点向量特征 [num_nodes, 3, hidden_channels]
            group_embedding: 组标量特征 [num_groups, hidden_channels]
            group_pos: 组位置 [num_groups, 3]
            group_vec: 组向量特征 [num_groups, 3, hidden_channels]
            edge_attr: 边属性 [num_edges, num_gaussians]
            edge_weight: 边权重（距离） [num_edges]
            edge_vec: 边向量 [num_edges, 3]
            fragment_ids: 片段ID（用于获取node的邻居信息） [num_nodes]
            **kwargs: 其他参数，可能包含node间的edge_index用于邻居聚合
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        # 获取node间的edge_index（用于邻居聚合）
        # 如果kwargs中没有提供，尝试从其他来源获取
        node_edge_index = kwargs.get('node_edge_index', None)
        
        # 计算图结构感知的attention（包含边向量注意力）
        attn_2, val_2 = self.calculate_attention(
            node_embedding, 
            group_embedding, 
            edge_index[0],  # x1_index: node索引
            edge_index[1],  # x2_index: group索引
            edge_attr, 
            self.model_2, 
            "silu",
            edge_index=edge_index,  # 用于边注意力计算（node-group交互图的边索引）
            node_edge_index=node_edge_index,  # 用于node邻居聚合（node之间的边索引）
            labels=fragment_ids,  # 用于group邻居聚合
            node_embedding=node_embedding,  # 原始node embedding
            group_embedding=group_embedding,  # 原始group embedding
            node_vec=node_vec,  # 用于边注意力计算
            edge_vec=edge_vec,  # 用于边注意力计算
            angle_attr=kwargs.get('angle_attr', None)  # 夹角特征（可选）
        )
        
        # 消息传递（与原模型相同）
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
        
        # 更新特征（与原模型相同）
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        
        return dx_node, dv_node, 0


