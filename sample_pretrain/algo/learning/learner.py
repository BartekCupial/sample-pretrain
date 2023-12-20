from __future__ import annotations

import glob
import os
import time
from abc import ABC, abstractmethod
from os.path import join
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from sample_pretrain.algo.utils.action_distributions import get_action_distribution, is_continuous_action_space
from sample_pretrain.algo.utils.env_info import EnvInfo
from sample_pretrain.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_pretrain.algo.utils.optimizers import Lamb
from sample_pretrain.algo.utils.rl_utils import gae_advantages, prepare_and_normalize_obs
from sample_pretrain.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_pretrain.algo.utils.torch_utils import masked_select, to_scalar
from sample_pretrain.cfg.configurable import Configurable
from sample_pretrain.model.actor_critic import ActorCritic, create_actor_critic
from sample_pretrain.utils.attr_dict import AttrDict
from sample_pretrain.utils.decay import LinearDecay
from sample_pretrain.utils.dicts import iterate_recursively
from sample_pretrain.utils.timing import Timing
from sample_pretrain.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_pretrain.utils.utils import ensure_dir_exists, experiment_dir, log


class LearningRateScheduler:
    def update(self, current_lr):
        return current_lr

    def invoke_after_each_minibatch(self):
        return False

    def invoke_after_each_epoch(self):
        return False


class LinearDecayScheduler(LearningRateScheduler):
    def __init__(self, cfg):
        num_updates = cfg.train_for_env_steps // cfg.batch_size * cfg.num_epochs
        self.linear_decay = LinearDecay([(0, cfg.learning_rate), (num_updates, 0)])
        self.step = 0

    def invoke_after_each_minibatch(self):
        return True

    def update(self, current_lr):
        self.step += 1
        lr = self.linear_decay.at(self.step)
        return lr


def get_lr_scheduler(cfg) -> LearningRateScheduler:
    if cfg.lr_schedule == "constant":
        return LearningRateScheduler()
    elif cfg.lr_schedule == "linear_decay":
        return LinearDecayScheduler(cfg)
    else:
        raise RuntimeError(f"Unknown scheduler {cfg.lr_schedule}")


class Learner(Configurable):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_id: PolicyID,
    ):
        Configurable.__init__(self, cfg)

        self.timing = Timing(name=f"Learner {policy_id} profile")

        self.policy_id = policy_id

        self.env_info = env_info

        self.device = None
        self.actor_critic: Optional[ActorCritic] = None

        self.optimizer = None

        self.curr_lr: Optional[float] = None
        self.lr_scheduler: Optional[LearningRateScheduler] = None

        self.train_step: int = 0  # total number of SGD steps
        self.env_steps: int = 0  # total number of environment steps consumed by the learner
        self.checkpoint_steps: int = -1  # used for saving the milestone every ith step

        self.best_performance = -1e9

        # for multi-policy learning (i.e. with PBT) when we need to load weights of another policy
        self.policy_to_load: Optional[PolicyID] = None

        # decay rate at which summaries are collected
        # save summaries every 5 seconds in the beginning, but decay to every 4 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self.summary_rate_decay_seconds = LinearDecay([(0, 2), (100000, 60), (1000000, 120)])
        self.last_summary_time = 0
        self.last_milestone_time = 0

        self.is_initialized = False

    def init(self) -> InitModelData:
        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # initialize device
        self.device = torch.device("cpu") if self.cfg.device == "cpu" else torch.device("cuda")

        log.debug("Initializing actor-critic model on device %s", self.device)

        # trainable torch module
        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self.actor_critic)
        self.actor_critic.model_to_device(self.device)
        self.actor_critic.train()

        params = list(self.actor_critic.parameters())

        optimizer_cls = dict(adam=torch.optim.Adam, lamb=Lamb)
        if self.cfg.optimizer not in optimizer_cls:
            raise RuntimeError(f"Unknown optimizer {self.cfg.optimizer}")

        optimizer_cls = optimizer_cls[self.cfg.optimizer]
        log.debug(f"Using optimizer {optimizer_cls}")

        optimizer_kwargs = dict(
            lr=self.cfg.learning_rate,  # use default lr only in ctor, then we use the one loaded from the checkpoint
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        if self.cfg.optimizer in ["adam", "lamb"]:
            optimizer_kwargs["eps"] = self.cfg.adam_eps

        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

        self.load_from_checkpoint(self.policy_id)

        self.lr_scheduler = get_lr_scheduler(self.cfg)
        self.curr_lr = self.cfg.learning_rate if self.curr_lr is None else self.curr_lr
        self._apply_lr(self.curr_lr)

        self.is_initialized = True

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f"checkpoint_p{policy_id}")
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern="checkpoint_*"):
        checkpoints = glob.glob(join(checkpoints_dir, pattern))
        return sorted(checkpoints)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning("No checkpoints found")
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                # noinspection PyBroadException
                try:
                    log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f"Could not load from checkpoint, attempt {attempt}")

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict["train_step"]
            self.env_steps = checkpoint_dict["env_steps"]
            self.best_performance = checkpoint_dict.get("best_performance", self.best_performance)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.curr_lr = checkpoint_dict.get("curr_lr", self.cfg.learning_rate)

        log.info(f"Loaded experiment state at {self.train_step=}, {self.env_steps=}")

    def load_from_checkpoint(self, policy_id: PolicyID, load_progress: bool = True) -> None:
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern=f"{name_prefix}_*")
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug("Did not load from checkpoint, starting from scratch!")
        else:
            log.debug("Loading model from checkpoint")

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def _should_save_summaries(self):
        summaries_every_seconds = self.summary_rate_decay_seconds.at(self.train_step)
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1

    def _get_checkpoint_dict(self):
        checkpoint = {
            "train_step": self.train_step,
            "env_steps": self.env_steps,
            "best_performance": self.best_performance,
            "model": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "curr_lr": self.curr_lr,
        }
        return checkpoint

    def _save_impl(self, name_prefix, name_suffix, keep_checkpoints, verbose=True) -> bool:
        if not self.is_initialized:
            return False

        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, f"{name_prefix}_temp")
        checkpoint_name = f"{name_prefix}_{self.train_step:09d}_{self.env_steps}{name_suffix}.pth"
        filepath = join(checkpoint_dir, checkpoint_name)
        if verbose:
            log.info("Saving %s...", filepath)

        # This should protect us from a rare case where something goes wrong mid-save and we end up with a corrupted
        # checkpoint file. It better be a corrupted temp file.
        torch.save(checkpoint, tmp_filepath)
        os.rename(tmp_filepath, filepath)

        while len(checkpoints := self.get_checkpoints(checkpoint_dir, f"{name_prefix}_*")) > keep_checkpoints:
            oldest_checkpoint = checkpoints[0]
            if os.path.isfile(oldest_checkpoint):
                if verbose:
                    log.debug("Removing %s", oldest_checkpoint)
                os.remove(oldest_checkpoint)

        return True

    def save(self) -> bool:
        return self._save_impl("checkpoint", "", self.cfg.keep_checkpoints)

    def save_milestone(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None
        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        checkpoint_name = f"checkpoint_{self.train_step:09d}_{self.env_steps}.pth"

        milestones_dir = ensure_dir_exists(join(checkpoint_dir, "milestones"))
        milestone_path = join(milestones_dir, f"{checkpoint_name}")
        log.info("Saving a milestone %s", milestone_path)
        torch.save(checkpoint, milestone_path)

    def save_best(self, policy_id, metric, metric_value) -> bool:
        if policy_id != self.policy_id:
            return False
        p = 3  # precision, number of significant digits
        if metric_value - self.best_performance > 1 / 10**p:
            log.info(f"Saving new best policy, {metric}={metric_value:.{p}f}!")
            self.best_performance = metric_value
            name_suffix = f"_{metric}_{metric_value:.{p}f}"
            return self._save_impl("best", name_suffix, 1, verbose=False)

        return False

    def set_new_cfg(self, new_cfg: Dict) -> None:
        self.new_cfg = new_cfg

    def set_policy_to_load(self, policy_to_load: PolicyID) -> None:
        self.policy_to_load = policy_to_load

    def _optimizer_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def _apply_lr(self, lr: float) -> None:
        """Change learning rate in the optimizer."""
        if lr != self._optimizer_lr():
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def _get_minibatch(self):
        raise NotImplementedError

    def _calculate_loss(self, mb: TensorDict):
        raise NotImplementedError

    def _calculate_metrics(self, mb: TensorDict, model_outputs: TensorDict):
        raise NotImplementedError

    def _train(self) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            stats_and_summaries: Optional[AttrDict] = None

            with_summaries = self._should_save_summaries()

            assert self.actor_critic.training

        with torch.no_grad(), timing.add_time("minibatch_init"):
            # current minibatch consisting of short trajectory segments with length == recurrence
            mb = self._get_minibatch()

        with timing.add_time("calculate_loss"):
            loss, model_outputs, loss_summaries = self._calculate_loss(mb)

        with torch.no_grad(), timing.add_time("calculate_metrics"):
            metric_summaries = self._calculate_metrics(mb, model_outputs)

        # update the weights
        with timing.add_time("update"):
            # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
            for p in self.actor_critic.parameters():
                p.grad = None

            loss.backward()

            if self.cfg.max_grad_norm > 0.0:
                with timing.add_time("clip"):
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

            self._apply_lr(self.curr_lr)

            self.optimizer.step()

        with torch.no_grad(), timing.add_time("after_optimizer"):
            self._after_optimizer_step()

            if self.lr_scheduler.invoke_after_each_minibatch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr)

            # collect and report summaries
            if with_summaries:
                # hacky way to collect all of the intermediate variables for summaries
                model_outputs = AttrDict(model_outputs)
                summary_vars = {**locals(), **loss_summaries}
                stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                del summary_vars

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        var = train_loop_vars

        self.last_summary_time = time.time()
        stats = AttrDict()

        stats.env_steps = self.env_steps
        stats.lr = self.curr_lr

        stats.update(self.actor_critic.summaries())

        grad_norm = (
            sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic.parameters() if p.grad is not None) ** 0.5
        )
        stats.grad_norm = grad_norm
        stats.loss = var.loss

        stats.act_min = var.model_outputs.actions.min()
        stats.act_max = var.model_outputs.actions.max()

        stats.max_abs_logprob = torch.abs(var.model_outputs.action_logits).max()

        # this caused numerical issues on some versions of PyTorch with second moment reaching infinity
        adam_max_second_moment = 0.0
        for key, tensor_state in self.optimizer.state.items():
            if "exp_avg_sq" in tensor_state:
                adam_max_second_moment = max(tensor_state["exp_avg_sq"].max().item(), adam_max_second_moment)
        stats.adam_max_second_moment = adam_max_second_moment

        for key, value in var.metric_summaries.items():
            stats[key] = value

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def train(self) -> Optional[Dict]:
        if self.cfg.save_milestones_ith > 0 and self.env_steps // self.cfg.save_milestones_ith > self.checkpoint_steps:
            self.save_milestone()
            self.checkpoint_steps = self.env_steps // self.cfg.save_milestones_ith

        with self.timing.add_time("train"):
            train_stats = self._train()

        self.env_steps += self.cfg.batch_size * self.cfg.rollout

        stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        if train_stats is not None:
            if train_stats is not None:
                stats[TRAIN_STATS] = train_stats
            stats[STATS_KEY] = memory_stats("learner", self.device)

        return stats
