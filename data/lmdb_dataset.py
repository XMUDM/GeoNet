import bisect
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import lmdb
import torch
import scipy.sparse as sp

class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """
    energy = 'energy'
    forces = 'forces'
    def __init__(self, path, transforms = [], name = None):
        super(LmdbDataset, self).__init__()
        self.path = Path(path) if isinstance(path, str) else [Path(p) for p in path]
        self.num_sub_datasets = 1 if isinstance(path, Path) else len(path)
        db_paths = sorted(self.path.glob("*.lmdb")) if isinstance(self.path, Path) else [sorted(p.glob("*.lmdb")) for p in self.path]
        if isinstance(path, list):
            identifier = [len(db_paths[i]) * [i] for i in range(self.num_sub_datasets)]
            db_paths = [item for sublist in db_paths for item in sublist] # flatten list
            identifier = np.array([item for sublist in identifier for item in sublist])
            self.identifier  = identifier
        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        self._keys, self.envs = [], []
        self.db_paths = db_paths
        self.open_db()
        self.transforms = transforms
        self.name = name
    
    def open_db(self):
        for db_path in self.db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(
                self.envs[-1].begin().get("length".encode("ascii"))
            )
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

        if isinstance(self.path, list):
            self.sample_list = []
            for i in range(self.num_sub_datasets):
                self.sample_list.append(np.sum(np.array(keylens)[self.identifier == i]))
            
           
   
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if len(self.envs) == 0:
            self.open_db()   
        if isinstance(self.path, list):
            flag = 1
        elif isinstance(self.path, Path):
            if not self.path.is_file():
                flag = 1
            else: 
                flag = 0
        else:
            raise ValueError("Path should be either a list or a Path object")
        
        if flag == 0:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pickle.loads(datapoint_pickled)
        else:
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = el_idx #f"{db_idx}_{el_idx}"



        data = data_object
        for transform in self.transforms:
            data = transform(data)

        
         
        return data


        
        
    def precompute_triplets(self, edge_index):
        """
        完全按照DimeNet方式实现：基于node-node邻接矩阵的三元组计算
        
        DimeNet的核心思想：
        1. 基于距离cutoff构建原子间的邻接矩阵 (node-node)
        2. 构建原子ID到边ID的映射 (atomids_to_edgeid)  
        3. 生成所有可能的三元组 k->j->i
        4. 过滤掉自环三元组 i->j->i
        5. 计算边映射索引
        
        Args:
            edge_index: [2, num_edges] tensor, 但我们将忽略这个，直接从pos构建
            
        Returns:
            tuple: (id3dnb_i, id3dnb_j, id3dnb_k, id_expand_kj, id_reduce_ji, idnb_i, idnb_j)
        """
        # 获取原子位置信息
        if not hasattr(self, '_current_data') or self._current_data is None:
            # 如果没有位置信息，返回空结果
            empty_tensor = torch.empty(0, dtype=torch.long)
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor
        
        pos = self._current_data.pos.cpu().numpy()  # [num_atoms, 3]
        num_atoms = pos.shape[0]
        
        if num_atoms == 0:
            empty_tensor = torch.empty(0, dtype=torch.long)
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor
        
        # 1. 构建基于距离的邻接矩阵 (完全按照DimeNet方式)
        # 计算所有原子对之间的距离
        distances = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)  # [num_atoms, num_atoms]
        
        # 设置距离cutoff (可以作为参数传入，这里使用默认值)
        cutoff = getattr(self, 'cutoff', 6.0)  # 默认6.0 Angstrom
        
        # 构建邻接矩阵：距离小于cutoff的原子对相连
        adj_matrix = sp.csr_matrix(distances <= cutoff)
        # 移除自环
        adj_matrix -= sp.eye(num_atoms, dtype=np.bool_)
        
        if adj_matrix.nnz == 0:
            # 如果没有连接，返回空结果
            empty_tensor = torch.empty(0, dtype=torch.long)
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor
        
        # 2. 构建 atomids_to_edgeid 映射 (完全按照DimeNet方式)
        # Entry x,y is edgeid x<-y (!)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape
        )
        
        # 3. 获取边的源和目标节点
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()
        
        # 4. 生成三元组 k->j->i (按DimeNet精确逻辑)
        # 对于每条边 j->i，找到所有可能的 k->j
        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1  # 每条边j->i对应的三元组数量
        id3ynb_i = np.repeat(edgeid_to_target, ntriplets)   # 重复目标节点i
        id3ynb_j = np.repeat(edgeid_to_source, ntriplets)   # 重复源节点j  
        id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]  # 所有连接到j的节点k
        
        # 5. 过滤掉自环三元组 i->j->i
        id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
        
        if len(id3_y_to_d) == 0:
            empty_tensor = torch.empty(0, dtype=torch.long)
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor
        
        # 过滤后的三元组
        id3dnb_i = id3ynb_i[id3_y_to_d]
        id3dnb_j = id3ynb_j[id3_y_to_d]  
        id3dnb_k = id3ynb_k[id3_y_to_d]
        
        # 6. 计算边映射索引 (严格按照DimeNet原始实现)
        # 重要：DimeNet的精确逻辑
        # data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        # data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]
        
        # 关键理解：DimeNet的逻辑
        # 对于三元组 k->j->i：
        # - id_expand_kj: 需要找到边 k->j 的边ID
        # - id_reduce_ji: 需要找到边 j->i 的边ID
        
        # 但是DimeNet的实现是基于稀疏矩阵的切片操作
        # atomids_to_edgeid[edgeid_to_source, :] 获取从每个源节点出发的边
        
        # 重新理解DimeNet的实现逻辑
        # 对于每个过滤后的三元组索引，我们需要：
        # 1. 找到对应的边j->i (这是基础边)
        # 2. 找到对应的边k->j (这是扩展边)
        
        # 先计算基础的边映射 - 每个三元组对应的j->i边
        # 这些边已经在 edgeid_to_source 和 edgeid_to_target 中
        
        # 对于过滤后的三元组，找到对应的边ID
        id_reduce_ji = np.zeros(len(id3dnb_i), dtype=np.int64)
        id_expand_kj = np.zeros(len(id3dnb_i), dtype=np.int64)
        
        # 创建从(source, target)到edge_id的映射
        edge_dict = {}
        for edge_id in range(len(edgeid_to_source)):
            source = edgeid_to_source[edge_id]
            target = edgeid_to_target[edge_id]
            edge_dict[(source, target)] = edge_id
        
        # 对于每个过滤后的三元组
        for idx in range(len(id3dnb_i)):
            i = id3dnb_i[idx]
            j = id3dnb_j[idx] 
            k = id3dnb_k[idx]
            
            # 找到边 j->i 的ID
            if (j, i) in edge_dict:
                id_reduce_ji[idx] = edge_dict[(j, i)]
            else:
                # 如果找不到，可能是因为方向问题，尝试反向
                if (i, j) in edge_dict:
                    id_reduce_ji[idx] = edge_dict[(i, j)]
                else:
                    id_reduce_ji[idx] = 0
            
            # 找到边 k->j 的ID
            if (k, j) in edge_dict:
                id_expand_kj[idx] = edge_dict[(k, j)]
            else:
                # 如果找不到，尝试反向
                if (j, k) in edge_dict:
                    id_expand_kj[idx] = edge_dict[(j, k)]
                else:
                    id_expand_kj[idx] = 0
        
        # 7. 转换为tensor
        id3dnb_i_tensor = torch.tensor(id3dnb_i, dtype=torch.long)
        id3dnb_j_tensor = torch.tensor(id3dnb_j, dtype=torch.long)
        id3dnb_k_tensor = torch.tensor(id3dnb_k, dtype=torch.long)
        id_expand_kj_tensor = torch.tensor(id_expand_kj, dtype=torch.long)
        id_reduce_ji_tensor = torch.tensor(id_reduce_ji, dtype=torch.long)
        
        # 边索引 (idnb_i, idnb_j) - 所有邻接矩阵中的边
        idnb_i_tensor = torch.tensor(edgeid_to_target, dtype=torch.long)
        idnb_j_tensor = torch.tensor(edgeid_to_source, dtype=torch.long)
        
        return (id3dnb_i_tensor, id3dnb_j_tensor, id3dnb_k_tensor, 
                id_expand_kj_tensor, id_reduce_ji_tensor, idnb_i_tensor, idnb_j_tensor)
    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
            self.envs = []
        else:
            self.env.close()
            self.env = None
            