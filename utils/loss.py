import torch
import torch.nn as nn
import torch.nn.functional as F

    
def compute_infonce_loss(H, z_pos, z_neg_list, temperature=0.07):
    """
    Compute the InfoNCE loss to pull H closer to z_pos and push H away from z_neg_list.

    Args:
        H: Tensor of shape (batch_size, feature_dim), the original representation.
        z_pos: Tensor of shape (batch_size, feature_dim), the positive sample.
        z_neg_list: List of Tensors, each of shape (batch_size, feature_dim), the negative samples.
        temperature: Float, the temperature scaling factor for the softmax.

    Returns:
        loss: Scalar Tensor, the computed InfoNCE loss.
    """
    batch_size = H.shape[0]
    H = F.normalize(H, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    neg = F.normalize(neg, dim=-1)
    z_neg_list = [F.normalize(z_neg, dim=1) for z_neg in z_neg_list]
    
    logits_pos = torch.sum(H * z_pos, dim=1, keepdim=True)
    logits_neg = torch.cat([torch.sum(H * z_neg, dim=1, keepdim=True) for z_neg in z_neg_list], dim=1)
    
    logits = torch.cat([logits_pos, logits_neg], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=H.device)
    
    logits = logits / temperature
    loss = F.cross_entropy(logits, labels)

    return loss
    