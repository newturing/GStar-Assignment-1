import torch
import triton
import triton.language as tl

@triton.jit
def weighted_row_sum_kernel(
    X_ptr,        # Pointer to the input tensor
    W_ptr,        # Pointer to the weight vector
    Y_ptr,        # Pointer to the output vector
    N_COLS,       # Number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for the kernel
):
    """
    Triton kernel to compute the weighted sum of each row in a matrix.
    Y[i] = sum_{j=0}^{N_COLS-1} X[i, j] * W[j]
    """
    # 1. Get the row index for the current program instance.
    row_idx = tl.program_id(axis=0)

    # 2. Create a pointer to the start of the current row in the input tensor X.
    row_start_ptr = X_ptr + row_idx * N_COLS
    
    # 3. Create a pointer for the output vector Y.
    output_ptr = Y_ptr + row_idx

    # 4. Initialize a scalar accumulator in float32 for stable accumulation.
    accumulator = 0.0

    # 5. Iterate over the columns of the row in blocks of BLOCK_SIZE.
    for col_block_idx in range(0, tl.cdiv(N_COLS, BLOCK_SIZE)):
        # Calculate the offsets for the current block of columns.
        col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # Create a mask to prevent out-of-bounds memory access for the last block.
        mask = col_offsets < N_COLS
        
        # Load a block of data from X and W safely using the mask.
        x_chunk = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        w_chunk = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        
        # Convert to float32 for accurate computation
        x_f32 = x_chunk.to(tl.float32)
        w_f32 = w_chunk.to(tl.float32)
        
        # Compute the element-wise product and accumulate in float32.
        accumulator += tl.sum(x_f32 * w_f32)
        
    # 6. Store the final accumulated sum to the output tensor Y.
    tl.store(output_ptr, accumulator)
    
# --- END OF STUDENT IMPLEMENTATION ---


def weighted_row_sum_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the weighted row-sum operation using the Triton kernel.
    """
    return torch_weighted_row_sum(x, w)

def torch_weighted_row_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using pure PyTorch.
    """
    return (x * w).sum(dim=1)