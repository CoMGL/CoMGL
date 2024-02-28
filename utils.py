import os
import os.path as osp
import os
import random
import time
import subprocess
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP, OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import AddMetaPaths, ToUndirected, RandomLinkSplit
from torch_geometric.utils import bipartite_subgraph, degree, negative_sampling
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
import torch_cluster
from torch import Tensor


def assign_free_gpus(threshold_vram_usage=10000, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = np.array([
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ])  # Remove garbage
        rank = np.argsort(gpu_info)
        # Keep gpus under threshold only
        free_gpus = [i for i in rank if gpu_info[i] < threshold_vram_usage]

        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join([str(i) for i in free_gpus])
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")
    return gpus_to_use

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and torch.cuda.is_available(): 
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_idx)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def load_dataset(args):
    if args.dataset == 'mag':
        dataset = OGB_MAG(root=args.data_path, preprocess='metapath2vec')
        data = dataset[0]
        del data[('paper', 'cites', 'paper')]

        args.node_class_num = data['paper'].y.unique().max() + 1  # 349

        data['author'].nid = torch.arange(data['author'].x.size(0)).reshape(-1, 1)

        view_1 = [('paper', 'writes_by', 'author'), ('author', 'writes', 'paper')]  # ('paper', 'cites', 'paper')
        view_2_ = [('institution', 'affiliated_with_by', 'author'), ('author', 'affiliated_with', 'institution')]  # Initial graph's edge type
        view_3 = [('field_of_study', 'has_topic_by', 'paper'), ('paper', 'has_topic', 'field_of_study')]
        view_dict = [view_1, view_2_, view_3]
        for view in view_dict:
            data[view[0]].edge_index = torch.flipud(data.edge_index_dict[view[1]]) # directed -> undirected
        
        if os.path.exists('/data/charity/mag_processed.pt'):
            print('load data successfully !')
            data = torch.load('/data/charity/mag_processed.pt')
        else:
            metapaths = [[view_1[0], view_2_[1]], [view_2_[0], view_1[1]]]
            data = AddMetaPaths(metapaths)(data)
            del data[view_2_[0]]
            del data[view_2_[1]]
            torch.save(data, '/data/charity/mag_processed.pt')

        view_2 = [('institution', 'metapath_1', 'paper'), ('paper', 'metapath_0', 'institution')]  # Generate new edge type around paper
        args.view_dict = view_dict = [view_1, view_2, view_3]
        args.u_type = data.u_type = 'paper'
        args.v_type = data.v_type = 'author'   # main view 的两种节点类型
        v_emb = data['author'].x.clone()    # record authors'embedding during training for inductive settings

    elif args.dataset == 'dblp':
        dataset = DBLP(root=args.data_path)
        data = dataset[0]

        args.u_type = data.u_type = 'paper'
        args.v_type = data.v_type = 'author'   # main view 的两种节点类型
        args.node_class_num = data['author'].y.unique().max() + 1  # 4

        data["conference"].x = torch.ones(data["conference"].num_nodes, 1)

        view_1 = [('paper', 'to', 'author'), ('author', 'to', 'paper')]
        view_2 = [('paper', 'to', 'conference'), ('conference', 'to', 'paper')]
        view_3 = [('paper', 'to', 'term'), ('term', 'to', 'paper')]
        args.view_dict = view_dict = [view_1, view_2, view_3]
    
    else:
        data = torch.load(osp.join(args.data_path, 'process/hetero_data.pt'))
        # Add reverse edge type for each view
        data = ToUndirected()(data)

        print('load Heterograph successfully !')

        # view_1 = [('openid', 'to', 'project'), ('project', 'to', 'openid'), ('project', 'to', 'institution'), ('institution', 'to', 'project')]
        view_1 = [('openid', 'to', 'project'), ('project', 'rev_to', 'openid')]
        view_2 = [('openid', 'to', 'qimei36'), ('qimei36', 'rev_to', 'openid')]
        view_3 = [('openid', 'to', 'uin'), ('uin', 'rev_to', 'openid')]
        args.view_dict = view_dict = [view_1, view_2, view_3]

        args.u_type = data.u_type = 'openid'
        args.v_type = data.v_type = 'project'   # main view 核心任务的两种节点类型
        args.node_class_num = 2   # main view 节点预测类别数
        
        # data[args.v_type].nid = torch.arange(data[args.v_type].x.size(0)).reshape(-1, 1)
        # v_emb = data[args.v_type].x.clone()  # record v_node's embedding during training for inductive settings

    # main view's edge_type
    args.edge_type = edge_type = view_1[0]
    args.rev_edge_type = rev_edge_type = view_1[1]
    args.feature_dim = {}
    for node_type in data.metadata()[0]:
        args.feature_dim[node_type] = data[node_type].x.size(1)
    args.metadata = data.metadata()

    print(data)
    train_data, val_data, test_data = get_data_split(data, edge_type, rev_edge_type, view_dict)

    print('split dataset successfully !')

    # train_data = add_delete_edges(train_data, view_dict=view_dict, noise_ratio=args.noise_ratio)

    if args.train_on_subgraph:
        train_loader = NeighborLoader(
            train_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [15] * 2 for key in train_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=args.batch_size,
            input_nodes=(args.u_type, train_data[args.u_type].mask),
            num_workers=4
        )

        val_loader = NeighborLoader(
            val_data,
            num_neighbors={key: [15] * 2 for key in val_data.edge_types},
            batch_size=args.batch_size,
            input_nodes=(args.u_type, val_data[args.u_type].mask),
        )

        test_loader = NeighborLoader(
            test_data,
            num_neighbors={key: [15] * 2 for key in test_data.edge_types},
            batch_size=args.batch_size,
            input_nodes=(args.u_type, test_data[args.u_type].mask),
        )
        return train_loader, val_loader, test_loader

    return train_data, val_data, test_data

# def negative_sampling(data, edge_index_dict, edge_type, rev_edge_type, num_nodes, ratio):
#     edge_index = edge_index_dict[edge_type]
#     rev_edge_index = edge_index_dict[rev_edge_type]
#     num_edges = edge_index.size(-1)
#     num_pos_edges = int(num_edges * ratio)
#     if ratio < 1:
#         mask = torch.ones(num_edges, dtype=torch.bool)
#         perm = torch.randperm(num_edges)[:num_pos_edges]
#         mask[perm] = False
#         pos_edges = edge_index[:, ~mask]
#     else:
#         pos_edges = edge_index

#     if isinstance(num_nodes[0], int):
#         u_set, v_set = np.arange(num_nodes[0]), np.arange(num_nodes[1])
#     else:
#         u_set, v_set = num_nodes[0], num_nodes[1]
#     row = np.random.choice(u_set, size=num_pos_edges).reshape(1, -1)
#     col = np.random.choice(v_set, size=num_pos_edges).reshape(1, -1)
#     neg_edge_index = torch.cat([torch.tensor(row), torch.tensor(col)], dim=0).to(edge_index)
#     split = {'pos_edge': pos_edges, 'neg_edge': neg_edge_index}
#     # edge_label_index = torch.cat([pos_edges, neg_edge_index], dim=1)
#     # edge_label = torch.cat([torch.ones(num_pos_edges), torch.zeros(num_pos_edges)], dim=0)

#     if ratio < 1:
#         edge_index_dict[edge_type] = edge_index[:, mask]
#         edge_index_dict[rev_edge_type] = torch.flipud(edge_index_dict[edge_type])

#     data[edge_type].split = split

#     # rev_num_edges = rev_edge_index.size(-1)
#     # rev_mask = (pos_edges.repeat(rev_num_edges, 1).reshape(num_pos_edges, rev_num_edges, 2) == rev_edge_index).sum(dim=2).sum(dim=0) == 0
#     # data[rev_edge_type].edge_index = data[rev_edge_type].edge_index[rev_mask]
#     return data

def global_negative_sampling(args, data, ratio=[1, 0.2]):
    u_type, v_type = data.u_type, data.v_type
    aux_view_1 = args.view_dict[1]
    aux_view_2 = args.view_dict[2]
    edge_type, rev_edge_type = args.edge_type, args.rev_edge_type

    # # main view进行负采样
    # edge_index_dict = data.edge_index_dict
    # num_nodes = [data.x_dict[u_type].size(0), data.x_dict[v_type].size(0)]
    # data = negative_sampling(data, edge_index_dict, edge_type, rev_edge_type, num_nodes=num_nodes, ratio=ratio[0])

    # # auxiliary views' negative sampling
    # view_dict = args.view_dict
    
    # for idx, view in enumerate(view_dict[1:]):  
    #     if args.dataset == 'mag':
    #         rev_edge_type, edge_type = view[0], view[1]
    #     else:
    #         edge_type, rev_edge_type = view[0], view[1]
    #     v_type_aux = edge_type[2]
    #     num_nodes = [data.x_dict[u_type].size(0), data.x_dict[v_type_aux].size(0)]
    #     data = negative_sampling(data, edge_index_dict, edge_type, rev_edge_type, num_nodes=num_nodes, ratio=ratio[1]) 

    # Generate the construction target
    train_edge, val_edge, test_edge = RandomLinkSplit(
                    num_val=0.2,
                    num_test=0.0,
                    neg_sampling_ratio=1,
                    edge_types=[aux_view_1[0], aux_view_2[0]],
                    rev_edge_types=[aux_view_1[1], aux_view_2[1]],
                )(data)
    
    # Main view negative sampling
    edge_index = data[edge_type].edge_index
    neg_edge_index = negative_sampling(edge_index, num_nodes=(data[u_type].x.shape[0], data[v_type].x.shape[0]))
    data[edge_type].neg_edge_index = neg_edge_index

    # auxiliary views' negative sampling
    for idx, view in enumerate(args.view_dict[1:]):
        aux_edge_type, rev_aux_edge_type = view[0], view[1]
        aux_v_type = edge_type[2]
        aux_edge_index = data[aux_edge_type].edge_index
        neg_edge_index = negative_sampling(aux_edge_index, num_nodes=(data[u_type].x.shape[0], data[aux_v_type].x.shape[0]))
        data[edge_type].neg_edge_index = neg_edge_index

    return data, train_edge.edge_index_dict, val_edge.edge_index_dict, test_edge.edge_index_dict     # return   

def add_delete_edges(data, view_dict, noise_ratio):
    if noise_ratio > 0:
        print(f'noise type: add edges, ratio: {noise_ratio}')
    elif noise_ratio < 0:
        print(f'noise type: delete edges, ratio: {noise_ratio}')

    for view in view_dict:
        for edge_type in view:
            edge_index = data.edge_index_dict[edge_type]
            num_edges = edge_index.size(-1)
            if noise_ratio <= 0:   # delete edges
                noise_ratio = np.abs(noise_ratio)
                perm = torch.randperm(num_edges)[:int(num_edges * noise_ratio)]
                data.edge_index_dict[edge_type] = edge_index[:, perm]
            
            else:
                loc = 0 if edge_type[0]==data.u_type else 1
                u_set = torch.unique(edge_index[loc]).numpy()
                v_type = edge_type[2] if loc==0 else edge_type[0]
                v_set = np.arange(data.x_dict[v_type].size(0))
                edge_index, _ = negative_sampling(edge_index, [u_set, v_set], size=noise_ratio)
                data.edge_index_dict[edge_type] = edge_index

    return data

def get_subgraph(data, mask, view_dict):
    u_type = data.u_type  
    
    for view in view_dict:
        for edge_type in view:
            if edge_type[0] == u_type:
                v_set = torch.arange(data.x_dict[edge_type[2]].size(0))  # 只对paper/user 节点进行分割，其他节点全部保留
                subset = (mask.nonzero().squeeze(), v_set)
            elif edge_type[2] == u_type:
                u_set = torch.arange(data.x_dict[edge_type[0]].size(0))
                subset = (u_set, mask.nonzero().squeeze())
            else:
                continue
            edge_index, _ = bipartite_subgraph(subset, data.edge_index_dict[edge_type], relabel_nodes=True)
            data[edge_type].edge_index = edge_index

    data[u_type].x = data.x_dict[u_type][mask]
    data[u_type].y = data[u_type].y[mask]
    del data[u_type].train_mask 
    del data[u_type].test_mask 
    del data[u_type].val_mask
    data[u_type].mask = torch.ones(len(data[u_type].x), dtype=torch.bool)   # reindex the mask
    return data

def get_data_split(data, edge_type, rev_edge_type, view_dict, num_train=0.8, num_val=0.1, num_test=0.1):
    u_type, v_type = data.u_type, data.v_type
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
 
    num_u = data[u_type].x.size(0)
    num_v = data[v_type].x.size(0)
    
    if not hasattr(data[u_type], 'train_mask'):
        print('Manual division ')
        num_train = int(num_train * num_u)
        num_val = int(num_val * num_u)
        
        perm = torch.randperm(num_u)
        train_mask = torch.zeros(num_u, dtype=torch.bool)
        val_mask = torch.zeros(num_u, dtype=torch.bool)
        test_mask = torch.zeros(num_u, dtype=torch.bool)
        train_mask[perm[:num_train]] = True       
        val_mask[perm[num_train:num_train+num_val]] = True
        test_mask[perm[num_train+num_val:]] = True

        train_data[u_type].train_mask = train_mask
        val_data[u_type].val_mask = val_mask
        test_data[u_type].test_mask = test_mask

    # Eliminate redundant edges to satisfy "inductive" and "strict cold start" scenarios, only for view 1.
    print(f'the num of train data edges: {train_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    train_data = get_subgraph(train_data, train_data[u_type].train_mask, view_dict)
    print(f'{train_data[edge_type].edge_index.size(-1)}')

    print(f'the num of val data edges: {val_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    val_data = get_subgraph(val_data, val_data[u_type].val_mask, view_dict)
    print(f'{val_data[edge_type].edge_index.size(-1)}')

    print(f'the num of test data edges: {test_data[edge_type].edge_index.size(-1)}  -->', end="  ")
    test_data = get_subgraph(test_data, test_data[u_type].test_mask, view_dict)
    print(f'{test_data[edge_type].edge_index.size(-1)}')

    return train_data, val_data, test_data


# Define the complementarity loss function
def complementarity_loss(rep1, rep2):
    view1_norm = rep1 / rep1.norm(dim=1, keepdim=True)
    view2_norm = rep2 / rep2.norm(dim=1, keepdim=True)
    cosine_similarity = torch.matmul(view1_norm, view2_norm.t())
    loss = -torch.mean(torch.abs(cosine_similarity))
    return loss