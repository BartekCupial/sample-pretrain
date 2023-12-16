from __future__ import annotations

import json
import math
import shutil
import time
from collections import OrderedDict, deque
from os.path import isdir, join
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import wandb
from signal_slot.signal_slot import EventLoop, EventLoopObject, EventLoopStatus, Timer, process_name, signal
from tensorboardX import SummaryWriter

from sample_pretrain.algo.learning.learner import Learner
from sample_pretrain.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_pretrain.algo.utils.misc import (
    EPISODIC,
    LEARNER_ENV_STEPS,
    SAMPLES_COLLECTED,
    STATS_KEY,
    TIMING_STATS,
    TRAIN_STATS,
    ExperimentStatus,
)
from sample_pretrain.cfg.arguments import cfg_dict, cfg_str, preprocess_cfg
from sample_pretrain.cfg.configurable import Configurable
from sample_pretrain.utils.attr_dict import AttrDict
from sample_pretrain.utils.dicts import iterate_recursively
from sample_pretrain.utils.timing import Timing
from sample_pretrain.utils.typing import PolicyID, StatusCode
from sample_pretrain.utils.utils import (
    cfg_file,
    debug_log_every_n,
    ensure_dir_exists,
    experiment_dir,
    init_file_logger,
    log,
    memory_consumption_mb,
    save_git_diff,
    summaries_dir,
    videos_dir,
)

# from sample_pretrain.utils.wandb_utils import init_wandb


class Runner(Configurable):
    def __init__(self):
        self.env_info: Optional[EnvInfo] = None
