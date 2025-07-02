import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
import numpy as np
from time import time

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size * 2, latent_dim)  # Bidirectional이므로 *2
        self.fc_std = nn.Linear(hidden_size * 2, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n은 (num_layers * num_directions, batch, hidden_size)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # Bidirectional의 마지막 layer들 concat
        mu = self.fc_mu(h_n)
        std = self.fc_std(h_n)
        z = self.reparameterize(mu, std)
        return z, mu, std

    def reparameterize(self, mu, std):
        std = torch.exp(0.5 * std)  
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(27, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, z, seq_len, target=None, teacher_forcing_ratio=0.5, temperature=1.0, hit_threshold=0.5):
        batch_size = z.size(0)
        h, c = self.init_hidden(batch_size)

        input_step = z.unsqueeze(1).repeat(1, 1, 1) # (B, 1, latent_dim)
        outputs = []
        logits = []
        
        for t in range(seq_len):
            out, (h, c) = self.lstm(input_step, (h, c)) # (B, 1, H)
            out = self.fc(out.squeeze(1)) # (B, 27)
            output_hits, output_velocities, output_offsets = torch.chunk(out, 3, dim=-1) # 각 (B, 9)

            # Sampling hits
            probs = torch.sigmoid(output_hits / temperature)
            sampled_hits = (probs > hit_threshold).float()
            
            # bernoulli_dist = dist.Bernoulli(logits=output_hits / temperature)
            # sampled_hits = bernoulli_dist.sample()

            output_velocities = torch.sigmoid(output_velocities)
            output_offsets = torch.tanh(output_offsets)

            logits_step = torch.cat([output_hits, output_velocities, output_offsets], dim=-1)# (B, 27)
            logits.append(logits_step.unsqueeze(1)) # (B, 1, 27)
            output_step = torch.cat([sampled_hits, output_velocities, output_offsets], dim=-1) # (B, 27)
            outputs.append(output_step.unsqueeze(1)) # (B, 1, 27)

            if target is not None and random.random() < teacher_forcing_ratio:
                input_step = self.input_proj(target[:, t]).unsqueeze(1)
            else:
                input_step = self.input_proj(output_step).unsqueeze(1)
        
        outputs = torch.cat(outputs, dim=1) # (B, T, 27)
        logits = torch.cat(logits, dim=1) # (B, T, 27)
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

