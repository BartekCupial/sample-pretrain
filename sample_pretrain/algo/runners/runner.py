from __future__ import annotations

import json
import math
import shutil
import time
from collections import OrderedDict, deque
from os.path import isdir, join
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from tensorboardX import SummaryWriter

from sample_pretrain.algo.learning.learner import Learner
from sample_pretrain.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_pretrain.algo.utils.misc import (
    EPISODIC,
    LEARNER_ENV_STEPS,
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
    ensure_dir_exists,
    experiment_dir,
    init_file_logger,
    log,
    memory_consumption_mb,
    save_git_diff,
    summaries_dir,
)
from sample_pretrain.utils.wandb_utils import init_wandb

MsgHandler = Callable[["Runner", dict], None]
PolicyMsgHandler = Callable[["Runner", dict, PolicyID], None]


class Runner(Configurable):
    def __init__(self, cfg):
        Configurable.__init__(self, cfg)

        self.status: StatusCode = ExperimentStatus.SUCCESS
        self.stopped: bool = False

        self.env_info: Optional[EnvInfo] = None

        self.learner: Learner = None

        self.reward_shaping: List[Optional[Dict]] = [None for _ in range(self.cfg.num_policies)]

        self.timing = Timing("Runner profile")

        # env_steps counts total number of simulation steps per policy (including frameskipped)
        self.env_steps: Dict[PolicyID, int] = dict()

        self.total_env_steps_since_resume: Optional[int] = None
        self.start_time: float = time.time()

        # currently, this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()

        self.report_interval_sec = 5.0
        self.avg_stats_intervals = (2, 12, 60)  # by default: 10 seconds, 60 seconds, 5 minutes
        self.summaries_interval_sec = self.cfg.experiment_summaries_interval  # sec
        self.heartbeat_report_sec = self.cfg.heartbeat_reporting_interval
        self.update_training_info_every_sec = 5.0

        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))

        self.stats = dict()  # regular (non-averaged) stats
        self.avg_stats = dict()

        self.policy_avg_stats: Dict[str, List[Deque]] = dict()

        self._handle_restart()

        init_wandb(self.cfg)  # should be done before writers are initialized

        self.writers: Dict[int, SummaryWriter] = dict()
        for policy_id in range(self.cfg.num_policies):
            summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg)), str(policy_id))
            summary_dir = ensure_dir_exists(summary_dir)
            self.writers[policy_id] = SummaryWriter(summary_dir, flush_secs=cfg.flush_summaries_interval)

        # global msg handlers for messages from algo components
        self.msg_handlers: Dict[str, List[MsgHandler]] = {
            TIMING_STATS: [self._timing_msg_handler],
            STATS_KEY: [self._stats_msg_handler],
        }

        # handlers for policy-specific messages
        self.policy_msg_handlers: Dict[str, List[PolicyMsgHandler]] = {
            LEARNER_ENV_STEPS: [self._learner_steps_handler],
            TRAIN_STATS: [self._train_stats_handler],
        }

        class Timer:
            def __init__(self, cb, period):
                self.cb = cb
                self.period = period
                self.last_called = time.time()

            def __call__(self):
                if time.time() - self.last_called > self.period:
                    self.cb()
                    self.last_called = time.time()

        self.timers: List[Timer] = []

        self.component_profiles: Dict[str, Timing] = dict()

        def periodic(period, cb):
            t = Timer(cb, period)
            self.timers.append(t)

        periodic(self.report_interval_sec, self._update_stats_and_print_report)
        periodic(self.summaries_interval_sec, self._report_experiment_summaries)
        periodic(self.cfg.save_every_sec, self._save_policy)

    def _handle_restart(self):
        exp_dir = experiment_dir(self.cfg, mkdir=False)
        if isdir(exp_dir):
            log.debug(f"Experiment dir {exp_dir} already exists!")
            if self.cfg.restart_behavior == "resume":
                log.debug(f"Resuming existing experiment from {exp_dir}...")
            else:
                if self.cfg.restart_behavior == "restart":
                    attempt = 0
                    old_exp_dir = exp_dir
                    while isdir(old_exp_dir):
                        attempt += 1
                        old_exp_dir = f"{exp_dir}_old{attempt:04d}"

                    # move the existing experiment dir to a new one with a suffix
                    log.debug(f"Moving the existing experiment dir to {old_exp_dir}...")
                    shutil.move(exp_dir, old_exp_dir)
                elif self.cfg.restart_behavior == "overwrite":
                    log.debug(f"Overwriting the existing experiment dir {exp_dir}...")
                    shutil.rmtree(exp_dir)
                else:
                    raise ValueError(f"Unknown restart behavior {self.cfg.restart_behavior}")

                log.debug(f"Starting training in {exp_dir}...")

    def _process_msg(self, msgs):
        if isinstance(msgs, (dict, OrderedDict)):
            msgs = (msgs,)

        if not (isinstance(msgs, (List, Tuple)) and isinstance(msgs[0], (dict, OrderedDict))):
            log.error("While parsing a message: expected a dictionary or list/tuple of dictionaries, found %r", msgs)
            return

        for msg in msgs:
            # some messages are policy-specific
            policy_id = msg.get("policy_id", None)

            for key in msg:
                for handler in self.msg_handlers.get(key, ()):
                    handler(self, msg)
                if policy_id is not None:
                    for handler in self.policy_msg_handlers.get(key, ()):
                        handler(self, msg, policy_id)

    @staticmethod
    def _timing_msg_handler(runner, msg):
        for k, v in msg["timing"].items():
            if k not in runner.avg_stats:
                runner.avg_stats[k] = deque([], maxlen=50)
            runner.avg_stats[k].append(v)

    @staticmethod
    def _stats_msg_handler(runner, msg):
        runner.stats.update(msg["stats"])

    @staticmethod
    def _learner_steps_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        env_steps: int = msg[LEARNER_ENV_STEPS]
        if policy_id in runner.env_steps:
            delta = env_steps - runner.env_steps[policy_id]
            runner.total_env_steps_since_resume += delta
        elif runner.total_env_steps_since_resume is None:
            runner.total_env_steps_since_resume = 0

        runner.env_steps[policy_id] = env_steps

    @staticmethod
    def _train_stats_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        """We write the train summaries to disk right away instead of accumulating them."""
        train_stats = msg[TRAIN_STATS]

        for key, scalar in train_stats.items():
            runner.writers[policy_id].add_scalar(f"train/{key}", scalar, runner.env_steps[policy_id])

            if key not in runner.policy_avg_stats:
                runner.policy_avg_stats[key] = [
                    deque(maxlen=runner.cfg.stats_avg) for _ in range(runner.cfg.num_policies)
                ]

            runner.policy_avg_stats[key][policy_id].append(scalar)

    def _get_perf_stats(self):
        # total env steps simulated across all policies
        fps_stats = []
        for avg_interval in self.avg_stats_intervals:
            fps_for_interval = math.nan
            if len(self.fps_stats) > 1:
                t1, x1 = self.fps_stats[max(0, len(self.fps_stats) - 1 - avg_interval)]
                t2, x2 = self.fps_stats[-1]
                fps_for_interval = (x2 - x1) / (t2 - t1)

            fps_stats.append(fps_for_interval)

        return fps_stats

    def print_stats(self, fps, total_env_steps):
        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(f"{int(interval * self.report_interval_sec)} sec: {fps_value:.1f}")
        fps_str = f'({", ".join(fps_str)})'

        log.debug(
            "Fps is %s. Total num frames: %d.",
            fps_str,
            total_env_steps,
        )

        if "loss" in self.policy_avg_stats:
            policy_loss_stats = []
            for policy_id in range(self.cfg.num_policies):
                loss_stats = self.policy_avg_stats["loss"][policy_id]
                if len(loss_stats) > 0:
                    policy_loss_stats.append((policy_id, f"{np.mean(loss_stats):.3f}"))
            log.debug("Avg loss: %r", policy_loss_stats)

    def _update_stats_and_print_report(self):
        """
        Called periodically (every self.report_interval_sec seconds).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        # don't have enough statistic from the learners yet
        if len(self.env_steps) < self.cfg.num_policies:
            return

        if self.total_env_steps_since_resume is None:
            return

        now = time.time()
        self.fps_stats.append((now, self.total_env_steps_since_resume))

        fps_stats = self._get_perf_stats()
        total_env_steps = sum(self.env_steps.values())
        self.print_stats(fps_stats, total_env_steps)

    def _report_experiment_summaries(self):
        memory_mb = memory_consumption_mb()

        fps_stats = self._get_perf_stats()
        fps = fps_stats[0]

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            writer = self.writers[policy_id]
            if policy_id == default_policy:
                if not math.isnan(fps):
                    writer.add_scalar("perf/_fps", fps, env_steps)

                writer.add_scalar("stats/master_process_memory_mb", float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and self.total_train_seconds > 300):
                        writer.add_scalar(f"stats/{key}", np.mean(value), env_steps)

                for key, value in self.stats.items():
                    writer.add_scalar(f"stats/{key}", value, env_steps)

        for w in self.writers.values():
            w.flush()

    def _save_policy(self):
        self.learner.save()

    def _save_cfg(self):
        fname = cfg_file(self.cfg)
        with open(fname, "w") as json_file:
            log.debug(f"Saving configuration to {fname}...")
            json.dump(cfg_dict(self.cfg), json_file, indent=2)

    def _make_learner(self):
        from sample_pretrain.algo.utils.context import global_learner_cls

        learner_cls = global_learner_cls()
        learner = learner_cls(self.cfg, self.env_info)
        return learner

    def init(self) -> StatusCode:
        self.env_info = obtain_env_info_in_a_separate_process(self.cfg)

        # check for any incompatible arguments
        if not preprocess_cfg(self.cfg, self.env_info):
            return ExperimentStatus.FAILURE

        log.debug(f"Starting experiment with the following configuration:\n{cfg_str(self.cfg)}")

        init_file_logger(self.cfg)
        self._save_cfg()
        save_git_diff(experiment_dir(self.cfg))

        self.learner = self._make_learner()
        self.learner.init()

        return ExperimentStatus.SUCCESS

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds
        return end

    def _after_training_iteration(self):
        if self._should_end_training():
            self._stop_training()

    def _stop_training(self, failed: bool = False) -> None:
        if not self.stopped:
            self._save_policy()

            if failed:
                self.status = ExperimentStatus.FAILURE

            self.stopped = True

    def _new_training_batch(self):
        stats = self.learner.train()

        if stats is not None:
            self._process_msg(stats)

    def _check_periodics(self):
        [t() for t in self.timers]

    # noinspection PyBroadException
    def run(self) -> StatusCode:
        while not self.stopped:
            self._new_training_batch()
            self._after_training_iteration()
            self._check_periodics()

        if self.total_env_steps_since_resume is None:
            self.total_env_steps_since_resume = 0
        log.info("Collected %r", self.env_steps)
        return self.status
