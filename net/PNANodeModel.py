import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.nn import  DenseGraphConv, dense_diff_pool, PNAConv, BatchNorm, DenseSAGEConv, GraphSizeNorm
import math
from torch.nn import BatchNorm1d, ModuleList
from torch_sparse import spspmm
import numpy as np
from net.braingraphconv import MyNNConv
from net.graphisographconv import MyGINConv
from torch_scatter import scatter_mean
from einops import rearrange, repeat

class PNANodeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, activation, run_cfg):
        super(PNANodeModel, self).__init__()

        if run_cfg['nodemodel_aggr'] == 'all':
            aggregators = ['mean', 'min', 'max', 'std', 'sum']
        else:
            aggregators = [run_cfg['nodemodel_aggr']]

        if run_cfg['nodemodel_scalers'] == 'all':
            scalers = ['identity', 'amplification', 'attenuation']
        else:
            scalers = ['identity']

        print(f'--> PNANodeModel going with aggregators={aggregators}, scalers={scalers}')

        self.activation = activation
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.pools=ModuleList()
        for _ in range(3):
            conv = PNAConv(in_channels=num_node_features, out_channels=num_node_features,
                           aggregators=aggregators, scalers=scalers, deg=run_cfg['dataset_indegree'],
                           edge_dim=num_edge_features, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            pool = TopKPooling(num_node_features, ratio=0.5, multiplier=1, nonlinearity=torch.sigmoid)
            self.convs.append(conv)
            self.pools.append(pool)
            self.batch_norms.append(BatchNorm(num_node_features))
        

    def forward(self, x, edge_index, edge_attr, u=None, batch=None, pos=None):
        all_outputs_minmax=[]
        all_outputs_sero=[]
        for conv, batch_norm, pools in zip(self.convs, self.batch_norms, self.pools):
            # x = self.activation(batch_norm(conv(x, edge_index, edge_attr)))
            # all_outputs_minmax.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            # new_attention_features, attentiongraph = self.sero(x)
            # all_outputs_sero.append(new_attention_features)

            pool_output, edge_index, edge_attr, batch, perm, score1 = pools(conv(x, edge_index, edge_attr), edge_index, edge_attr, batch)
            x = self.activation(batch_norm(pool_output))
            all_outputs_minmax.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            
        return x, torch.cat(all_outputs_minmax, dim=1)