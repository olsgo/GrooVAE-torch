import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import random

class Encoder_base(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder_base, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mu_layer = nn.Linear(hidden_size, latent_dim)
        self.std_layer = nn.Linear(hidden_size, latent_dim)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]  # Take the last layer's hidden state
        mu = self.mu_layer(h_n)
        std = F.softplus(self.std_layer(h_n)) + 1e-8  # Ensure std > 0
        z = self.reparameterize(mu, std)
        return z, mu, std
    
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder_base(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder_base, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, z, seq_len):
        batch_size = z.size(0)
        h, c = self.init_hidden(batch_size)
        
        # Repeat z for each time step
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        out, _ = self.lstm(z_repeated, (h, c))
        out = self.fc(out)
        
        # Split into hits, velocities, and offsets
        output_hits, output_velocities, output_offsets = torch.chunk(out, 3, dim=-1)
        
        # Apply activations
        output_hits = torch.sigmoid(output_hits)
        output_velocities = torch.sigmoid(output_velocities)
        output_offsets = torch.tanh(output_offsets)
        
        # Combine outputs
        output = torch.cat([output_hits, output_velocities, output_offsets], dim=-1)
        
        return output, output_hits, output_velocities, output_offsets
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h, c
        
    def compute_loss(self, x_target, output_hits, output_velocities, output_offsets):
        # Split target into components
        target_hits, target_velocities, target_offsets = torch.chunk(x_target, 3, dim=-1)
        
        # Compute losses
        hits_loss = F.binary_cross_entropy_with_logits(output_hits, target_hits, reduction='none').sum(dim=-1)
        velocities_loss = F.mse_loss(output_velocities, target_velocities, reduction='none').sum(dim=-1)
        offsets_loss = F.mse_loss(output_offsets, target_offsets, reduction='none').sum(dim=-1)
        
        loss = hits_loss + velocities_loss + offsets_loss
        
        return loss.mean()

# -----------------------------
# AttentionPooling Module
# -----------------------------
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, x):  # x: (B, T, D)
        # Compute attention weights
        attn_weights = torch.tanh(self.attention(x))  # (B, T, hidden_dim)
        attn_weights = self.context_vector(attn_weights).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T)
        
        # Apply attention weights
        attended = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return attended

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = AttentionPooling(hidden_size * 2, hidden_size)
        self.mu_layer = nn.Linear(hidden_size * 2, latent_dim)
        self.std_layer = nn.Linear(hidden_size * 2, latent_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, 2*hidden_size)
        attended = self.attention(out)  # (B, 2*hidden_size)
        
        mu = self.mu_layer(attended)
        std = F.softplus(self.std_layer(attended)) + 1e-8
        z = self.reparameterize(mu, std)
        return z, mu, std
    
    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.input_proj = nn.Linear(output_size, input_size)
        
    def forward(self, z, seq_len, target=None, teacher_forcing_ratio=0.5, temperature=1.0, hit_threshold=0.5):
        batch_size = z.size(0)
        h, c = self.init_hidden(batch_size)
        
        input_step = z.unsqueeze(1)
        outputs = []
        logits = []
        
        for t in range(seq_len):
            out, (h, c) = self.lstm(input_step, (h, c))
            out = self.fc(out.squeeze(1))
            output_hits, output_velocities, output_offsets = torch.chunk(out, 3, dim=-1)
            
            # Apply temperature to hits
            probs = torch.sigmoid(output_hits / temperature)
            sampled_hits = (probs > hit_threshold).float()
            
            output_velocities = torch.sigmoid(output_velocities)
            output_offsets = torch.tanh(output_offsets)
            
            logits_step = torch.cat([output_hits, output_velocities, output_offsets], dim=-1)
            logits.append(logits_step.unsqueeze(1))
            output_step = torch.cat([sampled_hits, output_velocities, output_offsets], dim=-1)
            outputs.append(output_step.unsqueeze(1))
            
            if target is not None and random.random() < teacher_forcing_ratio:
                input_step = self.input_proj(target[:, t]).unsqueeze(1)
            else:
                input_step = self.input_proj(output_step).unsqueeze(1)
        
        outputs = torch.cat(outputs, dim=1)
        logits = torch.cat(logits, dim=1)
        output_hits, output_velocities, output_offsets = torch.chunk(logits, 3, dim=-1)
        return outputs, output_hits, output_velocities, output_offsets
    
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h, c

    def compute_loss(self, x_target, output_hits, output_velocities, output_offsets):
        target_hits, target_velocities, target_offsets = torch.chunk(x_target, 3, dim=-1)
        hits_loss = F.binary_cross_entropy_with_logits(output_hits, target_hits, reduction='none').sum(dim=-1)
        velocities_loss = F.mse_loss(output_velocities, target_velocities, reduction='none').sum(dim=-1)
        offsets_loss = F.mse_loss(output_offsets, target_offsets, reduction='none').sum(dim=-1)
        loss = hits_loss + velocities_loss + offsets_loss
        
        return loss.mean()