from calendar import c
from graph import rdkit_grouping, ed_rdkit_grouping
from physics_guided_grouping import physics_guided_grouping
from schnetpack.datasets import MD17
import os
import numpy as np
# import hashlib
from collections import defaultdict
from tqdm import trange
import torch

class Atom:
    def __init__(self, x) -> None:
        self.x = x
    
    def __lt__(self, other):
        idx = {6: 0, 1: 1, 8: 2, 7: 3, 16: 4, 15: 5, 9: 6, 17: 7, 14: 8, 18: 9,}
        idx1 = idx[self.x]
        idx2 = idx[other.x]
        # if self.x == 8 and other.x == 1:
        #     return True
        # elif  self.x == 1 and other.x == 8:
        #     return False
        return idx1 < idx2
        
def count_atom_types(atomic_numbers):
    atom_types = defaultdict(int)
    for atom in atomic_numbers:
        atom_types[atom] += 1
    return atom_types

def dict2str(d):
    dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    return ''.join(['{}{}'.format(dict[int(k)], v if v >=2 else '') for k, v in sorted(d.items(), key = lambda x : Atom(x[0]))])


def grouping2label(groups, group_weights=None) -> torch.Tensor:
    '''
    Take the groups and implement the label associated with the grouping
    
    参数:
        groups: 每个原子所属的组
        group_weights: 每个原子对每个组的归属权重(可选)
    
    返回:
        label: 每个原子的组标签
    '''
    num_atoms = sum([len(group) for group in groups])
    label = torch.zeros(num_atoms).long()
    
    if group_weights is not None:
        # 如果提供了权重，使用权重最大的组作为标签
        # 重塑权重以匹配原子索引
        weight_dict = {}
        for i, group in enumerate(groups):
            for atom_idx in group:
                if atom_idx not in weight_dict:
                    weight_dict[atom_idx] = {}
                weight_dict[atom_idx][i] = group_weights[i].get(atom_idx, 0.0)
        
        # 为每个原子分配最大权重的组
        for atom_idx in range(num_atoms):
            if atom_idx in weight_dict:
                weights = weight_dict[atom_idx]
                max_group = max(weights.items(), key=lambda x: x[1])[0]
                label[atom_idx] = max_group
    else:
        # 传统方式：每个原子只属于一个组
        for i, group in enumerate(groups):
            for atom_idx in group:
                label[atom_idx] = i    
    
    return label

def rdkit_label_builder(g, min_group_size, charge=0, use_ed_grouping=False, use_physics_guided=False) -> None:
    '''
    Given a graph g, build the label for the associated graph using different grouping methods
    
    参数:
        g: 图对象
        min_group_size: 最小组大小
        charge: 分子电荷
        use_ed_grouping: 是否使用电子密度分组
        use_physics_guided: 是否使用物理引导的动态分组
    '''
    if use_physics_guided:
        # 使用物理引导的动态分组方法
        grouping, break_bonds = physics_guided_grouping(
            g.atomic_numbers.squeeze().numpy(), 
            g.pos.numpy(), 
            min_group_size=min_group_size, 
            charge=charge
        )
        
        # 存储分组信息和断键信息
        g.grouping = grouping
        g.break_bonds = break_bonds
        
        # 计算标签
        g.labels = grouping2label(grouping)
        g.num_labels = len(grouping)
        
    elif use_ed_grouping:
        # 使用电子密度驱动的分组方法
        grouping, break_bonds = ed_rdkit_grouping(
            g.atomic_numbers.squeeze().numpy(), 
            g.pos.numpy(), 
            min_group_size=min_group_size, 
            charge=charge
        )
        
        # 存储分组信息和断键信息
        g.grouping = grouping
        g.break_bonds = break_bonds
        
        # 计算标签
        g.labels = grouping2label(grouping)
        g.num_labels = len(grouping)
        
        # 存储额外信息
        if hasattr(g, '_extra_data') and g._extra_data is not None and 'group_weights' in g._extra_data:
            g.group_weights = g._extra_data['group_weights']
    else:
        # 使用原始的BRICS分组方法
        grouping, break_bonds = rdkit_grouping(
            g.atomic_numbers.squeeze().numpy(), 
            g.pos.numpy(), 
            min_group_size=min_group_size, 
            charge=charge
        )
        
        g.grouping = grouping
        g.break_bonds = break_bonds
        g.labels = grouping2label(grouping)
        g.num_labels = len(grouping)


if __name__ == '__main__':
    ################### TESTING #################
    datapath = "/home/zhangjia/v-yunyangli/dataset"
    dataset = MD17(os.path.join(datapath, f"md17_aspirin.db"),
                    molecule = 'aspirin', load_only =["energy","forces"])

    output = defaultdict(int)
    # problem_idx = [1000]
    for i in trange(0, len(dataset)):
        atom_dict = {'C':0, 'O': 0, 'H': 0}
        example = dataset[i]
        
        # 测试两种方法
        print(f"Testing molecule {i}")
        
        # BRICS方法
        brics_grouping, brics_break_bonds = rdkit_grouping(example['_atomic_numbers'].numpy(), example['_positions'].numpy(), min_group_size=4)
        brics_labels = grouping2label(brics_grouping)
        
        # 物理引导动态分组方法
        physics_grouping, physics_break_bonds = physics_guided_grouping(example['_atomic_numbers'].numpy(), example['_positions'].numpy(), min_group_size=4)
        physics_labels = grouping2label(physics_grouping)
        
        # 电子密度方法
        ed_grouping_result, ed_break_bonds = ed_rdkit_grouping(example['_atomic_numbers'].numpy(), example['_positions'].numpy(), min_group_size=4)
        ed_labels = grouping2label(ed_grouping_result)
        
        print(f"BRICS groups: {len(brics_grouping)}, Physics groups: {len(physics_grouping)}, ED groups: {len(ed_grouping_result)}")
        
        # 比较分组
        atom_ids = [example['_atomic_numbers'].numpy()[np.array(group)] for group in brics_grouping]
        atom_types = [dict2str(count_atom_types(atoms)) for atoms in atom_ids]
        
        hash_str = ';'.join(atom_types)
        output[hash_str] += 1
        
    print(output)

# print(problem_idx)
# for i in problem_idx:
#     dict = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
#     for number, pos in zip(dataset[i]['_atomic_numbers'].numpy(), dataset[i]['_positions'].numpy()):
#         print(f'{dict[number]}    {pos[0]:9.5f}    {pos[1]:9.5f}    {pos[2]:9.5f}')
#     print('\n')



# print(output)






    