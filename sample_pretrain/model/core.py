from abc import ABC

import torch
from torch import nn

from sample_pretrain.model.model_utils import ModelModule
from sample_pretrain.utils.typing import Config

from mamba_ssm import Mamba as OrigMamba
from mamba_ssm.utils.generation import InferenceParams


class ModelCore(ModelModule, ABC):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.core_output_size = -1  # to be overridden in derived classes

    def get_out_size(self) -> int:
        return self.core_output_size


class ModelCoreRNN(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False
        self.is_mamba = False

        cfg.rnn_input_size = input_size

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(input_size, cfg.rnn_size, cfg.rnn_num_layers)
        elif cfg.rnn_type == "mamba":
            self.is_mamba = True
            assert cfg.rnn_num_layers == 1
            self.core = OrigMamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=input_size,  # Model dimension d_model
                d_state=cfg.rnn_size,   # SSM state expansion factor
                d_conv=cfg.mamba_conv_size,    # Local convolution width
                expand=cfg.mamba_expand,    # Block expansion factor
                layer_idx=0,
            )
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        if cfg.rnn_type == "lstm" or cfg.rnn_type == "gru":
            self.core_output_size = cfg.rnn_size
        else:
            # TODO: should we project it down with a linear layer?
            self.core_output_size = input_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        elif self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_mamba:
            print(head_output.shape)
            # TODO: think about max_seqlen and seqlen_offset
            inference_params = InferenceParams(max_seqlen=10,
                                               max_batch_size=rnn_states.shape[0],
                                               seqlen_offset=1)
            rnn_states = rnn_states.reshape(rnn_states.shape[0], -1, self.core.d_conv + self.core.d_state)
            rnn_states.requires_grad = False
            rnn_states = rnn_states.contiguous().clone()
            conv_state = rnn_states[..., :self.core.d_conv]
            rnn_state = rnn_states[..., self.core.d_conv:]

            conv_state.requires_grad = False
            rnn_state.requires_grad = False
            inference_params.key_value_memory_dict = {0: (conv_state, rnn_state)}
            x = self.core(head_output.permute(1, 0, 2), inference_params)
            x = x.permute(1, 0, 2)
        elif self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.rnn_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.is_mamba:
            # TODO: let's see if this works
            conv_state, rnn_state = inference_params.key_value_memory_dict[0]
            new_rnn_states = torch.cat((conv_state, rnn_state), dim=2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        elif self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


class ModelCoreIdentity(ModelCore):
    """A noop core (no recurrency)."""

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    # noinspection PyMethodMayBeStatic
    def forward(self, head_output, fake_rnn_states):
        return head_output, fake_rnn_states


def default_make_core_func(cfg: Config, core_input_size: int) -> ModelCore:
    if cfg.use_rnn:
        core = ModelCoreRNN(cfg, core_input_size)
    else:
        core = ModelCoreIdentity(cfg, core_input_size)

    return core
