import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidLinear(nn.Module):
    """
    Simulates a Liquid Time-Constant (LTC) layer.
    
    The core idea is that the hidden state evolves according to a differential equation:
    dh/dt = -h/tau + S(x(t))
    
    Where 'tau' is a learnable time constant.
    """
    def __init__(self, in_features, out_features, hidden_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_map = nn.Linear(in_features, hidden_size)
        
        # Recurrent connection
        self.recurrent_map = nn.Linear(hidden_size, hidden_size)
        
        # Learnable time constant (tau)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # Output projection
        self.output_map = nn.Linear(hidden_size, out_features)

    def forward(self, x, h=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            h: Initial hidden state (optional)
        """
        batch_size, seq_len, _ = x.shape
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            # Compute underlying dynamics
            # I = Wx * x + Wh * h
            I = self.input_map(xt) + self.recurrent_map(h)
            
            # Non-linearity (Sigmoid is common in liquid networks to bound inputs)
            S = torch.tanh(I) 
            
            # Differential equation update (Euler integration step)
            # h_new = h + dt * (-h/tau + S)
            # Assuming dt=1 for simplicity in this discrete simulation
            
            # We use a softplus on tau to ensure it stays positive
            decay = torch.sigmoid(self.tau) 
            
            h_new = (1 - decay) * h + decay * S
            
            outputs.append(self.output_map(h_new))
            h = h_new
            
        return torch.stack(outputs, dim=1), h
