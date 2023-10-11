import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, sort_edge_index, softmax
from net.braingraphconv import MyNNConv
from torch.nn import Parameter
from net.brainmsgpassing import MyMessagePassing
from torch_geometric.utils import add_remaining_self_loops,softmax
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)

from torch_geometric.typing import (OptTensor)
from net.inits import uniform
import numpy as np
import torch
from torch.nn import Module, MultiheadAttention
from torch_geometric.utils import to_dense_batch


class GMPool(Module):
    def __init__(self, in_channels, out_channels, num_heads=1, dropout=0.0):
        super(GMPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.fc_g = torch.nn.Linear(in_channels, out_channels * num_heads)
        self.fc_v = torch.nn.Linear(in_channels, out_channels * num_heads)
        self.fc_a = torch.nn.Linear(in_channels, num_heads)
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.softmax_layer = torch.nn.Softmax(dim=-1)

    def forward(self, x, batch):
        # x.shape: [num_nodes, in_channels]
        # batch.shape: [num_nodes]

        g = self.fc_g(x).view(-1, self.num_heads, self.out_channels)
        v = self.fc_v(x).view(-1, self.num_heads, self.out_channels)
        a = self.fc_a(x).view(-1, self.num_heads)

        alpha = self.dropout_layer(self.softmax_layer(a))

        x_pooled = torch.sum(alpha.unsqueeze(-1) * v, dim=-2)

        return x_pooled