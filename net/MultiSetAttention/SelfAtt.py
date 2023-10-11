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
import math


class SelfAtt(torch.nn.Module):
    def __init__(self, dim, num_heads):
        '''
        :param dim: (int) input feature dimension
        :param num_heads: (int) number of attention heads
        '''
        super(SelfAtt, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q = torch.nn.Linear(dim, dim * num_heads, bias=False)
        self.k = torch.nn.Linear(dim, dim * num_heads, bias=False)
        self.v = torch.nn.Linear(dim, dim * num_heads, bias=False)
        self.out = torch.nn.Linear(dim * num_heads, dim)

    def forward(self, x, batch):
        x, batch = to_dense_batch(x, batch)
        num_nodes = x.size(1)

        # Compute queries, keys, and values for each head
        q = self.q(x).view(-1, self.num_heads, num_nodes, self.dim).transpose(1, 2)
        k = self.k(x).view(-1, self.num_heads, num_nodes, self.dim).permute(0, 1, 3, 2)
        v = self.v(x).view(-1, self.num_heads, num_nodes, self.dim).transpose(1, 2)

        # Compute attention scores and apply softmax
        scores = torch.matmul(q, k) / math.sqrt(self.dim)
        scores = scores.softmax(dim=-1)

        # Apply attention to values and concatenate across heads
        x = torch.matmul(scores, v).transpose(1, 2).reshape(-1, num_nodes, self.dim * self.num_heads)
        x = self.out(x)

        # Aggregate over nodes in each graph in the batch
        x = global_mean_pool(x, batch)
        return x