from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sample_pretrain.utils.attr_dict import AttrDict
from sample_pretrain.utils.typing import Config


def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


def to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value


@torch.jit.script
def masked_select(x: torch.Tensor, mask: torch.Tensor, num_non_mask: int) -> torch.Tensor:
    if num_non_mask == 0:
        return x
    else:
        return torch.masked_select(x, mask)
