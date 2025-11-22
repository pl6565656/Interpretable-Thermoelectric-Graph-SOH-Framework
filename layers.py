import torch
import torch.nn as nn
import numpy as np

class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.last_attention = None

    def forward(self, x):
        scores = self.attn(x)
        attn_weights = torch.softmax(scores, dim=1)
        self.last_attention = attn_weights
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled

    def get_last_attention_weights(self):
        if self.last_attention is None:
            return None
        return self.last_attention.squeeze(-1).detach().cpu()

class TrendBasis(nn.Module):
    def __init__(self, degree, input_size, output_size):
        super(TrendBasis, self).__init__()
        self.degree = degree
        self.input_size = input_size
        self.output_size = output_size
        self.backcast_time = torch.arange(input_size).float() / input_size
        self.forecast_time = torch.arange(output_size).float() / output_size

    def forward(self, theta_b, theta_f):
        batch_size = theta_b.size(0)
        backcast = torch.zeros(batch_size, self.input_size, device=theta_b.device)
        forecast = torch.zeros(batch_size, self.output_size, device=theta_b.device)
        for i in range(self.degree + 1):
            backcast = backcast + theta_b[:, i:i + 1] * (self.backcast_time.to(theta_b.device) ** i).unsqueeze(0)
            forecast = forecast + theta_f[:, i:i + 1] * (self.forecast_time.to(theta_b.device) ** i).unsqueeze(0)
        return backcast, forecast