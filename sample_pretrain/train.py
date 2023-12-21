import os
from typing import Tuple

import torch.distributed as dist

from sample_pretrain.algo.runners.runner import Runner
from sample_pretrain.algo.utils.context import global_ddp_mode, sf_global_context
from sample_pretrain.algo.utils.misc import ExperimentStatus
from sample_pretrain.cfg.arguments import maybe_load_from_checkpoint
from sample_pretrain.utils.typing import Config


def setup():
    if "LOCAL_RANK" in os.environ:
        # setup_ddp function if running in a distributed setup
        sf_global_context().ddp_mode = True
        dist.init_process_group("nccl")


def cleanup():
    if global_ddp_mode():
        dist.destroy_process_group()


def make_runner(cfg: Config) -> Tuple[Config, Runner]:
    if cfg.restart_behavior == "resume":
        # if we're resuming from checkpoint, we load all of the config parameters from the checkpoint
        # unless they're explicitly specified in the command line
        cfg = maybe_load_from_checkpoint(cfg)

    runner = Runner(cfg)

    return cfg, runner


def run(cfg: Config):
    setup()

    cfg, runner = make_runner(cfg)

    # here we can register additional message or summary handlers
    # see sf_examples/dmlab/train_dmlab.py for example

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    cleanup()
    return status
