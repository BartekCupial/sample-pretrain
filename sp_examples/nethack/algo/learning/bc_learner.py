from concurrent.futures import ThreadPoolExecutor

import nle.dataset as nld
import numpy as np
import torch
import torch.nn.functional as F

from sample_pretrain.algo.learning.learner import Learner
from sample_pretrain.algo.utils.env_info import EnvInfo
from sample_pretrain.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_pretrain.algo.utils.tensor_dict import TensorDict, clone_tensordict, stack_tensordicts
from sample_pretrain.model.model_utils import get_rnn_size
from sample_pretrain.utils.typing import Config, PolicyID
from sp_examples.nethack.datasets.actions import ACTION_MAPPING
from sp_examples.nethack.datasets.dataset import load_nld_aa_large_dataset
from sp_examples.nethack.datasets.render import render_screen_image
from sp_examples.nethack.datasets.roles import Alignment, Race, Role


class BCLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_id: PolicyID = 0,
    ):
        super().__init__(cfg, env_info, policy_id)

        self.dataset: nld.TtyrecDataset = None
        self.tp = None

        self.rnn_states = None
        self._iterators = None
        self._results = None

    def init(self):
        super().init()

        self.dataset = self._get_dataset()
        self.tp = ThreadPoolExecutor(max_workers=self.cfg.num_workers)

        def _make_sing_iter(dataset):
            dataset = iter(dataset)

            def _iter():
                prev_actions = np.zeros((self.cfg.batch_size, 1))
                prev_timestamps = np.ones((self.cfg.batch_size, 1)) * -1

                while True:
                    batch = next(dataset)

                    screen_image = render_screen_image(
                        tty_chars=batch["tty_chars"],
                        tty_colors=batch["tty_colors"],
                        tty_cursor=batch["tty_cursor"],
                        threadpool=self.tp,
                    )
                    batch["screen_image"] = screen_image
                    batch["actions"] = ACTION_MAPPING[batch["keypresses"]]
                    batch["prev_actions"] = np.concatenate([prev_actions, batch["actions"][:, :-1].copy()], axis=1)
                    prev_actions = np.expand_dims(batch["actions"][:, -1].copy(), -1)

                    # dones are broken in NLD-AA, so we just rewrite them with always done at last step
                    # see: https://github.com/facebookresearch/nle/issues/355
                    timestamp_diff = batch["timestamps"] - np.concatenate(
                        [prev_timestamps, batch["timestamps"][:, :-1].copy()], axis=1
                    )
                    batch["done"][np.where(timestamp_diff != 1)] = 1
                    prev_timestamps = np.expand_dims(batch["timestamps"][:, -1].copy(), -1)

                    # ensure that we don't overrite data
                    normalized_batch = prepare_and_normalize_obs(self.actor_critic, batch)
                    normalized_batch = clone_tensordict(TensorDict(normalized_batch))

                    yield normalized_batch

            return iter(_iter())

        self.rnn_states = [
            torch.zeros((self.cfg.batch_size, get_rnn_size(self.cfg)), dtype=torch.float32, device=self.device)
            for _ in range(self.cfg.worker_num_splits)
        ]
        self.idx = 0
        self.prev_idx = 0

        self._iterators = []
        self._results = []
        for _ in range(self.cfg.worker_num_splits):
            it = _make_sing_iter(self.dataset)
            self._iterators.append(it)
            self._results.append(self.tp.submit(next, it))

    def _get_dataset(self):
        if self.cfg.character == "@":
            role, race, align = None, None, None
        else:
            role, race, align = self.cfg.character.split("-")
            role, race, align = Role(role), Race(race), Alignment(align)

        dataset = load_nld_aa_large_dataset(
            dataset_name=self.cfg.dataset_name,
            data_path=self.cfg.data_path,
            db_path=self.cfg.db_path,
            seq_len=self.cfg.rollout,
            batch_size=self.cfg.batch_size,
            role=role,
            race=race,
            align=align,
        )

        return dataset

    def result(self):
        return self._results[self.idx].result()

    def step(self):
        fut = self.tp.submit(next, self._iterators[self.idx])
        self._results[self.idx] = fut
        self.prev_idx = self.idx
        self.idx = (self.idx + 1) % self.cfg.worker_num_splits

    def _get_minibatch(self) -> TensorDict:
        normalized_batch = self.result()
        self.step()
        return normalized_batch

    def _calculate_loss(self, mb: TensorDict):
        rnn_state = self.rnn_states[self.prev_idx]

        model_outputs = []
        seq_len = mb["actions"].shape[1]
        for i in range(seq_len):
            outputs = self.actor_critic(mb[:, i], rnn_state)
            not_done = (1.0 - mb["done"][:, i].float()).unsqueeze(-1)

            targets = mb[:, i]["actions"].unsqueeze(-1)
            observed_log_probs = self.actor_critic.last_action_distribution.log_prob(targets)
            outputs["observed_log_probs"] = observed_log_probs

            rnn_state = outputs["new_rnn_states"] * not_done
            model_outputs.append(outputs)

        # update prev_actions and rnn_states for next iteration
        self.rnn_states[self.prev_idx] = rnn_state.detach()

        model_outputs = stack_tensordicts(model_outputs, dim=1)
        loss = -model_outputs["observed_log_probs"].mean()

        return loss, model_outputs, {}

    def _calculate_metrics(self, mb: TensorDict, model_outputs: TensorDict):
        targets = mb["actions"].flatten(0, 1).long()
        outputs = model_outputs["action_logits"].flatten(0, 1)

        cross_entropy = F.cross_entropy(outputs, targets)
        metric_summaries = {"cross_entropy": cross_entropy}

        return metric_summaries
