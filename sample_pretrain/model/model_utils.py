from typing import List, Optional

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from sample_pretrain.cfg.configurable import Configurable
from sample_pretrain.utils.typing import Config


def get_rnn_size(cfg, encoder_out_size):
    if cfg.use_rnn:
        size = cfg.rnn_size * cfg.rnn_num_layers
    else:
        size = 1

    if cfg.rnn_type == "lstm":
        size *= 2
    elif cfg.rnn_type == "mamba":
        size = (cfg.rnn_num_layers * cfg.mamba_model_size * cfg.mamba_expand) * (cfg.mamba_conv_size + cfg.mamba_state_size)

    if not cfg.actor_critic_share_weights:
        # actor and critic need separate states
        size *= 2

    if "kickstarting_loss_coeff" in cfg or "distillation_loss_coeff" in cfg:
        if cfg.kickstarting_loss_coeff != 0.0 or cfg.distillation_loss_coeff != 0.0:
            # teacher and student need separate states
            size *= 2

    return size


def nonlinearity(cfg: Config, inplace: bool = False) -> nn.Module:
    if cfg.nonlinearity == "elu":
        return nn.ELU(inplace=inplace)
    elif cfg.nonlinearity == "relu":
        return nn.ReLU(inplace=inplace)
    elif cfg.nonlinearity == "tanh":
        return nn.Tanh()
    else:
        raise Exception(f"Unknown {cfg.nonlinearity=}")


def fc_layer(in_features: int, out_features: int, bias=True, spec_norm=False) -> nn.Module:
    layer = nn.Linear(in_features, out_features, bias)
    if spec_norm:
        layer = spectral_norm(layer)

    return layer


def create_mlp(layer_sizes: List[int], input_size: int, activation: nn.Module) -> nn.Module:
    """Sequential fully connected layers."""
    layers = []
    for i, size in enumerate(layer_sizes):
        layers.extend([fc_layer(input_size, size), activation])
        input_size = size

    if len(layers) > 0:
        return nn.Sequential(*layers)
    else:
        return nn.Identity()


class ModelModule(nn.Module, Configurable):
    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        Configurable.__init__(self, cfg)

    def get_out_size(self):
        raise NotImplementedError()


def model_device(model: nn.Module) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None
