import torch
import torch.nn as nn
import math

class FlashAttention2Function(torch.autograd.Function):
    """
    A pure PyTorch implementation of the FlashAttention-2 forward pass.
    This version is a template for student implementation.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Get dimensions from input tensors following the (B, H, N, D) convention
        B, H, N_Q, D_H = Q.shape
        _, _, N_K, _ = K.shape

        # Define tile sizes
        Q_TILE_SIZE = 128
        K_TILE_SIZE = 128
        
        N_Q_tiles = math.ceil(N_Q / Q_TILE_SIZE)
        N_K_tiles = math.ceil(N_K / K_TILE_SIZE)

        # Initialize final output tensors
        O_final = torch.zeros_like(Q, dtype=Q.dtype)
        L_final = torch.zeros((B, H, N_Q), device=Q.device, dtype=torch.float32)
        
        scale = 1.0 / math.sqrt(D_H)

        # Main loops: Iterate over each batch and head
        for b in range(B):
            for h in range(H):
                Q_bh = Q[b, h, :, :]
                K_bh = K[b, h, :, :]
                V_bh = V[b, h, :, :]

                # Loop over query tiles
                for i in range(N_Q_tiles):
                    q_start = i * Q_TILE_SIZE
                    q_end = min((i + 1) * Q_TILE_SIZE, N_Q)
                    Q_tile = Q_bh[q_start:q_end, :]

                    # Initialize accumulators for this query tile
                    o_i = torch.zeros_like(Q_tile, dtype=Q.dtype)
                    l_i = torch.zeros(q_end - q_start, device=Q.device, dtype=torch.float32)
                    m_i = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32)

                    # Inner loop over key/value tiles
                    for j in range(N_K_tiles):
                        k_start = j * K_TILE_SIZE
                        k_end = min((j + 1) * K_TILE_SIZE, N_K)

                        K_tile = K_bh[k_start:k_end, :]
                        V_tile = V_bh[k_start:k_end, :]
                        
                        S_ij = (Q_tile @ K_tile.transpose(-1, -2)) * scale
                        
                        # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
                        # 1. Apply causal masking if is_causal is True.
                        if is_causal:
                            # Create causal mask for the current tile
                            # We need to consider the absolute positions in the sequence
                            row_indices = torch.arange(q_start, q_end, device=Q.device).unsqueeze(-1)
                            col_indices = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                            # Create mask where row < col (future positions)
                            causal_mask = row_indices < col_indices
                            # Apply mask: set future positions to -inf
                            S_ij = S_ij.masked_fill(causal_mask, -float('inf'))
                        #
                        # 2. Compute the new running maximum
                        m_ij, _ = torch.max(S_ij, dim=-1)
                        m_new = torch.maximum(m_i, m_ij)
                        #
                        # 3. Rescale the previous accumulators (o_i, l_i)
                        o_i = o_i * torch.exp(m_i - m_new).unsqueeze(-1)
                        l_i = l_i * torch.exp(m_i - m_new)
                        #
                        # 4. Compute the probabilities for the current tile, P_tilde_ij = exp(S_ij - m_new).
                        P_tilde_ij = torch.exp(S_ij - m_new.unsqueeze(-1)).to(Q.dtype)
                        #
                        # 5. Accumulate the current tile's contribution to the accumulators to update l_i and o_i
                        o_i = o_i + P_tilde_ij @ V_tile
                        l_i = l_i + torch.sum(P_tilde_ij, dim=-1)
                        #
                        # 6. Update the running max for the next iteration
                        m_i = m_new
                        
                        # --- END OF STUDENT IMPLEMENTATION ---

                    # After iterating through all key tiles, normalize the output
                    # This part is provided for you. It handles the final division safely.
                    l_i_reciprocal = torch.where(l_i > 0, 1.0 / l_i, 0)
                    o_i_normalized = o_i * l_i_reciprocal.unsqueeze(-1)
                    
                    L_tile = m_i + torch.log(l_i)
                    
                    # Write results for this tile back to the final output tensors
                    O_final[b, h, q_start:q_end, :] = o_i_normalized
                    L_final[b, h, q_start:q_end] = L_tile
        
        O_final = O_final.to(Q.dtype)

        ctx.save_for_backward(Q, K, V, O_final, L_final)
        ctx.is_causal = is_causal
 
        return O_final, L_final
    
    @staticmethod
    def backward(ctx, grad_out, grad_L):
        raise NotImplementedError("Backward pass not yet implemented for FlashAttention2Function")