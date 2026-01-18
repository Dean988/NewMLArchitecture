import torch
import torch.nn as nn
from src.layers.liquid_layer import LiquidLinear
from src.layers.mamba_block import MambaBlock
from src.layers.kan_layer import KANLayer

class LiquidMambaKAN(nn.Module):
    """
    LiquidMamba-KAN: A Hybrid State-of-the-Art Architecture.
    
    Combines:
    1. Liquid Neural Networks (LNN) for dynamic time-constant adaptation.
    2. Mamba (SSM) for efficient long-range sequence modeling.
    3. Kolmogorov-Arnold Networks (KAN) for high-precision, interpretable non-linearities.
    
    Architecture Flow:
    Input -> Liquid Projection -> [Mamba Block -> KAN Mixing] x N -> Head
    """
    def __init__(self, input_dim, d_model, num_layers=4, output_dim=10):
        super().__init__()
        
        # 1. Liquid Entry: Adapts to input dynamics immediately
        self.liquid_proj = LiquidLinear(input_dim, d_model, hidden_size=d_model)
        
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': MambaBlock(d_model),
                'kan': KANLayer(d_model, d_model)
            }))
            
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = KANLayer(d_model, output_dim) # KAN head for better decision boundaries

    def forward(self, x):
        """
        x: (batch, seq, input_dim)
        """
        # Liquid projection returns (output, final_hidden)
        # We use the sequence output
        x, _ = self.liquid_proj(x)
        
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            
            # Mamba Branch: Contextual processing
            x_mamba = layer['mamba'](x)
            
            # KAN Branch: Feature refinement
            # We treat the output of Mamba as input to KAN for refinement
            x_kan = layer['kan'](x_mamba)
            
            x = residual + x_kan
            
        x = self.final_norm(x)
        
        # Pooling (simple mean for now)
        x_pool = x.mean(dim=1)
        
        logits = self.classifier(x_pool)
        return logits
