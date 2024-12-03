import torch as th
import torch.utils.dlpack
import graphpy as gpk

def gp_spmm(input1, input2, input3, input4, dim1_0, dim1_1, device0):
    # Get the current PyTorch stream for the specified device
    current_stream = th.cuda.current_stream(device=device0)
    current_stream_ptr = current_stream.cuda_stream  # Get the stream pointer
    
    # Convert tensors to DLPack format
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    input3_dl = th.utils.dlpack.to_dlpack(input3)
    input4_dl = th.utils.dlpack.to_dlpack(input4)
    
    # Allocate output tensor
    res1 = th.zeros(dim1_0, dim1_1, device=device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    
    # Launch the kernel with the stream
    gpk.spmm(input1_dl, input2_dl, input3_dl, input4_dl, res_dl1, current_stream_ptr)
    
    return res1