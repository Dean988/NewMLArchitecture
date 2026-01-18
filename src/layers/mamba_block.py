import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    """
    A simplified simulation of a Mamba Block (Selective State Space Model).
    
    True Mamba uses hardware-optimized selective scans. Here we simulate the 
    architectural flow: Input -> Expand -> Conv1d -> SSM -> Project -> Output.
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 1D Convolution (acting as a local inductive bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            groups=self.d_inner,
            padding=3
        )
        
        # Simulated SSM parameters (A, B, C matrix approximations)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_model * 2)
        self.dt_proj = nn.Linear(d_state, self.d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Project inputs
        xz = self.in_proj(x) # (batch, seq, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Transpose for Conv1d
        x_proj = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_proj)[:, :, :seq_len] # Causal padding manually handled via slice
        x_conv = x_conv.transpose(1, 2)
        
        x_conv = self.act(x_conv)
        
        # Simulate SSM interaction (simplified as a gated transformation)
        # In a real Mamba, this is where the selective scan happens.
        # We simulate the mixing of information.
        
        ssm_out = x_conv * torch.sigmoid(self.x_proj(x_conv)[:, :, :self.d_inner])
        
        # Multiplicative gate
        y = ssm_out * self.act(z)
        
        out = self.out_proj(y)
        return out
