import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import AttentivePooling

class NBEATSBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, n_layers, hidden_size, graph_feature_size=None, window_feature_size=None, window_length=None):
        super(NBEATSBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.graph_feature_size = graph_feature_size
        self.window_feature_size = window_feature_size
        self.window_length = window_length
        actual_input_size = window_feature_size * window_length

        if graph_feature_size is not None:
            self.graph_attention = nn.Sequential(
                nn.Linear(graph_feature_size, graph_feature_size // 2),
                nn.ReLU(),
                nn.Linear(graph_feature_size // 2, actual_input_size),
                nn.LayerNorm(actual_input_size),
                nn.Sigmoid()
            )
            self.attentive_pooling = AttentivePooling(graph_feature_size)

        self.fc_layers = nn.ModuleList([nn.Linear(actual_input_size, hidden_size)])
        for _ in range(n_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

    def forward(self, x, graph_features=None, window_features=None):
        batch_size, T, D = x.shape
        x = x.view(batch_size, -1)
        if graph_features is not None:
            if graph_features.dim() == 3:
                graph_features = self.attentive_pooling(graph_features)
            guide_vector = self.graph_attention(graph_features)
            x = x * guide_vector + x

        for layer in self.fc_layers:
            x = F.relu(layer(x))

        theta_b = self.theta_b(x)
        theta_f = self.theta_f(x)
        b, f = self.basis_function(theta_b, theta_f)
        b = b.view(batch_size, -1)
        return b, f, theta_b, theta_f