import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple

import numpy as np
import torch

from sample_pretrain.algo.learning.learner import Learner
from sample_pretrain.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_pretrain.algo.utils.optimizers import Lamb
from sample_pretrain.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_pretrain.algo.utils.tensor_dict import TensorDict, cat_tensordicts
from sample_pretrain.algo.utils.torch_utils import to_scalar
from sample_pretrain.cfg.arguments import maybe_load_from_checkpoint
from sample_pretrain.model.actor_critic import ActorCritic, create_actor_critic
from sample_pretrain.model.model_utils import get_rnn_size
from sample_pretrain.utils.attr_dict import AttrDict
from sample_pretrain.utils.timing import Timing
from sample_pretrain.utils.typing import Config
from sample_pretrain.utils.utils import log
from sp_examples.nethack.datasets.actions import ACTION_MAPPING
from sp_examples.nethack.datasets.render import render_screen_image


def run(cfg: Config, learner: Learner):
    if cfg.restart_behavior == "resume":
        # if we're resuming from checkpoint, we load all of the config parameters from the checkpoint
        # unless they're explicitly specified in the command line
        cfg = maybe_load_from_checkpoint(cfg)

    while learner.env_steps < cfg.train_for_env_steps:
        stats = learner.train()
        print(stats)
