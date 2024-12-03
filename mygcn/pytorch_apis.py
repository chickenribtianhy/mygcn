import torch as th
import gp_apis

class spmm_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, input3, input4, dim_0, dim_1, device0):
        # Get the current CUDA stream
        current_stream = th.cuda.current_stream(device=device0)
        stream_ptr = current_stream.cuda_stream

        # Call the spmm API with the stream
        res = gp_apis.gp_spmm(input1, input2, input3, input4, dim_0, dim_1, device0, stream_ptr)
        
        # Save tensors for backward computation
        ctx.save_for_backward(input1, input2, input3, input4)
        ctx.dim_0 = dim_0
        ctx.dim_1 = dim_1
        ctx.device0 = device0
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2, input3, input4 = ctx.saved_tensors
        dim_0 = ctx.dim_0
        dim_1 = ctx.dim_1
        device0 = ctx.device0

        # Construct sparse matrix S from input1 and input2
        indices = th.stack([input1.long(), input2.long()], dim=0)  # Assuming input1, input2 provide indices
        values = input3  # Assuming input3 contains the sparse matrix values
        S = th.sparse.FloatTensor(indices, values, th.Size([dim_0, input4.shape[0]])).to(device0)

        # Compute gradients
        grad_dense_matrix = th.sparse.mm(S.t(), grad_output)  # Gradient wrt dense matrix

        row_indices = input1.long()
        col_indices = input2.long()

        grad_output_rows = grad_output[row_indices, :]  # Shape: (nnz, dim_1)
        dense_matrix_rows = input4[col_indices, :]      # Shape: (nnz, dim_1)

        grad_values = th.sum(grad_output_rows * dense_matrix_rows, dim=1)  # Gradient wrt sparse values

        # Set gradients for input1 and input2 (if needed)
        grad_input1 = None  # Assuming input1 is not trainable
        grad_input2 = None  # Assuming input2 is not trainable

        # Return gradients in the same order as inputs to the forward method
        return grad_input1, grad_input2, grad_values, grad_dense_matrix, None, None, None


def spmm(input1, input2, input3, input4, dim_0, dim_1, device0):        
    # Call the custom autograd function
    return spmm_impl.apply(input1, input2, input3, input4, dim_0, dim_1, device0)