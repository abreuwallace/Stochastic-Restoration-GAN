import torch
from src.models.generator import Generator

def profile_loss(gen, x, v, p_freq=1.3, p_rythm=1.6):
    """Calculate loss profile for prevention of mode collapsing.

    Args:
        gen (Generator): generator model
        x (torch.Tensor): input tensor [B, C, F, T]
        v (float): regularization strength
        p (float): norm order
    """
    if len(x.shape) < 4:
        AssertionError(f"Expect input shape len=4, received len={len(x.shape)}")
    z_i = torch.randn(1, 64, 1, 1)
    z_j = torch.randn(1, 64, 1, 1)
    P_i = torch.sqrt(torch.abs(gen(x, z_i)))
    P_j = torch.sqrt(torch.abs(gen(x, z_j)))
    
    L_freq = v * torch.linalg.norm(z_i - z_j, ord=2, dim=1) / torch.linalg.norm((torch.mean(P_i, dim=-2) - torch.mean(P_j, dim=-2)), ord=p_freq, dim=-1)
    L_rythm = v * torch.linalg.norm(z_i - z_j, ord=2, dim=1) / torch.linalg.norm((torch.mean(P_i, dim=-1) - torch.mean(P_j, dim=-1)), ord=p_rythm, dim=-1)

    return L_freq + L_rythm