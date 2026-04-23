import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm

from src.models.common.modules.mlp import MLP


# this class is used for bloating the mlp layers in the encoder and decoder
# original T5 implementation only has one layer
class T5MultiLayerFF(nn.Module):
    def __init__(self, config: T5Config, num_layers: int):
        """
        Initialize the T5MultiLayerFF module.
        This module is a multi-layer feed-forward network (MLP) used in the T5 model.
        It consists of a series of linear layers with ReLU activation and dropout.
        And it also includes layer normalization and residual connections.
        Parameters:
            config (T5Config): The T5 configuration object.
            num_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        self.mlp = MLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dim_list=[config.d_ff for _ in range(num_layers)],
            activation=nn.ReLU,
            dropout=config.dropout_rate,
        )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the T5MultiLayerFF module.
        Parameters:
            hidden_states (torch.Tensor): The input hidden states for the MLP.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
