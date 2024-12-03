import math
import torch
import torch.nn as nn
from pytorch_apis import spmm


class GraphConvolution(nn.Module):
    """
    Simple GCN layer using custom spmm function.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # Optional bias term
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights and biases using uniform distribution.
        """
        stdv = 1.0 / math.sqrt(self.in_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, row_ptr, col_ind, values, adj_shape, device):
        """
        Forward pass of the GCN layer.

        Args:
        - input: Dense input feature matrix (N x in_features).
        - adj: Adjacency matrix (optional, for debugging or fallback).
        - row_ptr, col_ind, values: CSR representation of the adjacency matrix.
        - adj_shape: Shape of the adjacency matrix (N, N).
        - device: Device on which to perform computations (e.g., 'cuda:0').
        """
        # Multiply input by weight matrix
        support = torch.mm(input, self.weight)

        # Prepare dimensions for spmm
        dim_0 = adj_shape[0]
        dim_1 = support.shape[1]

        # Get the current CUDA stream (if available)
        current_stream = torch.cuda.current_stream(device=device)
        stream_ptr = current_stream.cuda_stream

        # Perform sparse-dense matrix multiplication
        output = spmm(row_ptr, col_ind, values, support, dim_0, dim_1, device, stream_ptr)

        # Add bias (if applicable)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """
        String representation of the layer.
        """
        return '{} ({} -> {})'.format(self.__class__.__name__,
                                      self.in_features,
                                      self.out_features)