import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer simulation.
    
    Instead of fixed activation functions on nodes (like ReLU), 
    KANs learn activation functions on edges. 
    
    We approximate this using a basis of learnable B-spline-like functions.
    Sum(w_i * activation_i(x))
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Base weight (like a standard linear layer residual)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Spline weights
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)
        
    def forward(self, x):
        """
        x: (batch, seq_len, in_features)
        """
        # Linear base
        base_output = F.linear(x, self.base_weight)
        
        # Spline approximation simulation
        # In a real KAN, we'd evaluate B-splines. 
        # Here we simulate the non-linear effect using a mix of sin/cos basis for visual complexity.
        
        x_expanded = x.unsqueeze(-1) # (batch, seq, in, 1)
        
        # Simulate basis functions (e.g., sin(x), sin(2x), ...)
        # This creates the "learnable function" effect
        basis_1 = torch.sin(x_expanded * torch.arange(1, 6, device=x.device).float())
        
        # Aggregate: (batch, seq, out)
        # We project the basis expansion to the output dim
        
        # (batch, seq, in, grid) * (out, in, grid) -> sum over in and grid
        # Using a simplified Einstein summation for clarity
        # We need to broadcast properly.
        
        # Let's simplify: map input features to a non-linear space, then combine
        spline_out = torch.einsum('bsig,oig->bso', basis_1, self.spline_weight)
        
        return base_output + spline_out
