import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100, padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))
    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]
    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]
    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]


def temporal_masking(inputs, mask_prob=0.15, mask_length=5, device='cpu'):
    """
    Apply temporal masking to simulate variability in speaking rate and improve robustness.
    Randomly masks contiguous time steps by setting them to zero.
    
    Args:
        inputs (tensor : B x T x N): Neural features [batch, time, features]
        mask_prob (float): Probability of masking each time window (default: 0.15)
        mask_length (int): Length of each masked segment in time steps (default: 5)
        device (str): Device to use for computation
    
    Returns:
        masked (tensor : B x T x N): Masked neural features
    """
    B, T, N = inputs.shape
    masked = inputs.clone()
    
    for b in range(B):
        # Determine number of masks to apply
        num_masks = int((T // mask_length) * mask_prob)
        
        for _ in range(num_masks):
            # Random start position for mask
            start_idx = np.random.randint(0, max(1, T - mask_length))
            end_idx = min(start_idx + mask_length, T)
            
            # Apply mask (set to zero)
            masked[b, start_idx:end_idx, :] = 0
    
    return masked


def noise_injection(inputs, noise_std=0.1, device='cpu'):
    """
    Add Gaussian noise to neural features to improve model robustness.
    
    Args:
        inputs (tensor : B x T x N): Neural features [batch, time, features]
        noise_std (float): Standard deviation of Gaussian noise (default: 0.1)
        device (str): Device to use for computation
    
    Returns:
        noisy (tensor : B x T x N): Neural features with added noise
    """
    noise = torch.randn_like(inputs, device=device) * noise_std
    return inputs + noise


def feature_dropout(inputs, dropout_prob=0.1, device='cpu'):
    """
    Apply feature-wise dropout to encourage the model to not rely on specific features.
    
    Args:
        inputs (tensor : B x T x N): Neural features [batch, time, features]
        dropout_prob (float): Probability of dropping each feature (default: 0.1)
        device (str): Device to use for computation
    
    Returns:
        dropped (tensor : B x T x N): Neural features with dropout applied
    """
    if dropout_prob == 0:
        return inputs
    
    # Create dropout mask for features (same mask across time for each feature)
    B, T, N = inputs.shape
    mask = torch.rand(B, 1, N, device=device) > dropout_prob
    mask = mask.float() / (1 - dropout_prob)  # scale to maintain expected value
    
    return inputs * mask


def time_warping(inputs, warp_factor_range=(0.9, 1.1), device='cpu'):
    """
    Apply time warping to simulate variability in speaking rate.
    
    Args:
        inputs (tensor : B x T x N): Neural features [batch, time, features]
        warp_factor_range (tuple): Range of time warping factors (default: 0.9 to 1.1)
        device (str): Device to use for computation
    
    Returns:
        warped (tensor : B x T x N): Time-warped neural features
    """
    B, T, N = inputs.shape
    warped = torch.zeros_like(inputs)
    
    for b in range(B):
        # Random warp factor
        warp_factor = np.random.uniform(*warp_factor_range)
        new_T = int(T * warp_factor)
        
        # interpolate to new length
        sample = inputs[b:b+1].permute(0, 2, 1)  # [1, N, T]
        warped_sample = F.interpolate(sample, size=new_T, mode='linear', align_corners=False)
        
        # Crop or pad to original length
        if new_T > T:
            warped[b] = warped_sample[:, :, :T].permute(0, 2, 1)[0]
        else:
            warped[b, :new_T, :] = warped_sample.permute(0, 2, 1)[0]
    
    return warped


def session_normalize(inputs, session_stats=None, device='cpu'):
    """
    Normalize features based on session-specific statistics.
    
    Args:
        inputs (tensor : B x T x N): Neural features [batch, time, features]
        session_stats (dict): dictionary with 'mean' and 'std' tensors for normalization
        device (str): Device to use for computation
    
    Returns:
        normalized (tensor : B x T x N): Normalized neural features
    """
    if session_stats is None:
        # Compute statistics from input if not provided
        mean = inputs.mean(dim=(0, 1), keepdim=True)
        std = inputs.std(dim=(0, 1), keepdim=True) + 1e-8
    else:
        mean = session_stats['mean'].to(device)
        std = session_stats['std'].to(device)
    
    return (inputs - mean) / std


def apply_augmentations(
    inputs, 
    device='cpu',
    apply_temporal_mask=True,
    apply_noise=True,
    apply_dropout=True,
    apply_time_warp=False,
    temporal_mask_prob=0.15,
    temporal_mask_length=5,
    noise_std=0.05,
    dropout_prob=0.1,
    warp_range=(0.9, 1.1)
):
    """
    Apply a combination of augmentations to neural features during training.
    
    Args:
        inputs (tensor): Neural features
        device (str): Device for computation
        apply_* (bool): Flags to enable/disable specific augmentations
        *_params: Parameters for each augmentation
    
    Returns:
        augmented (tensor): Augmented neural features
    """
    augmented = inputs.clone()
    
    if apply_temporal_mask:
        augmented = temporal_masking(
            augmented, 
            mask_prob=temporal_mask_prob,
            mask_length=temporal_mask_length,
            device=device
        )
    
    if apply_noise:
        augmented = noise_injection(
            augmented,
            noise_std=noise_std,
            device=device
        )
    
    if apply_dropout:
        augmented = feature_dropout(
            augmented,
            dropout_prob=dropout_prob,
            device=device
        )
    
    if apply_time_warp:
        augmented = time_warping(
            augmented,
            warp_factor_range=warp_range,
            device=device
        )
    
    return augmented