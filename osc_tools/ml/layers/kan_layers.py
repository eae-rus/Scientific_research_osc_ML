import torch
import torch.nn as nn
import torch.nn.functional as F
from osc_tools.ml.kan_conv.KANLinear import KANLinear

class KANConv1d(nn.Module):
    """
    1D Convolutional Layer using KAN (Kolmogorov-Arnold Network) instead of linear weights.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=[-1, 1],
    ):
        super(KANConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # The input to KANLinear will be the flattened kernel patch
        self.kan_in_features = in_channels * kernel_size
        
        self.kan_layer = KANLinear(
            in_features=self.kan_in_features,
            out_features=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x):
        # x: (Batch, InChannels, Length)
        batch_size, in_channels, length = x.shape
        
        # Unfold: Extract sliding local blocks from a batched input tensor
        # Input for unfold must be 4D (Batch, Channel, Height, Width)
        # So we unsqueeze to (Batch, Channel, 1, Length)
        x_unsqueezed = x.unsqueeze(2)
        
        # unfold(dimension, size, step) is for tensor, but nn.Unfold is for 4D
        # Let's use F.unfold
        # kernel_size for unfold is (height, width) -> (1, kernel_size)
        # stride -> (1, stride)
        # padding -> (0, padding)
        # dilation -> (1, dilation)
        
        patches = F.unfold(
            x_unsqueezed, 
            kernel_size=(1, self.kernel_size),
            dilation=(1, self.dilation),
            padding=(0, self.padding),
            stride=(1, self.stride)
        )
        # patches shape: (Batch, InChannels * 1 * KernelSize, L_out)
        #              = (Batch, InChannels * KernelSize, L_out)
        
        # We need to transpose to apply KANLinear to the last dimension (features)
        # But KANLinear expects (..., in_features)
        # So we want (Batch, L_out, InChannels * KernelSize)
        
        patches = patches.transpose(1, 2) # (Batch, L_out, InChannels * KernelSize)
        
        # Flatten batch and length for processing
        b, l_out, feats = patches.shape
        patches_flat = patches.reshape(b * l_out, feats)
        
        # Apply KAN
        out_flat = self.kan_layer(patches_flat) # (B * L_out, OutChannels)
        
        # Reshape back
        out = out_flat.reshape(b, l_out, self.out_channels)
        
        # Transpose to (Batch, OutChannels, L_out) to match Conv1d output
        out = out.transpose(1, 2)
        
        return out

