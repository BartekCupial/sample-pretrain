from concurrent.futures import ThreadPoolExecutor

import nle.dataset as nld
import numpy as np
import torch

from sample_pretrain.algo.learning.learner import Learner
from sample_pretrain.algo.utils.env_info import EnvInfo
from sample_pretrain.algo.utils.rl_utils import gae_advantages, prepare_and_normalize_obs
from sample_pretrain.algo.utils.tensor_dict import TensorDict, cat_tensordicts, shallow_recursive_copy
from sample_pretrain.model.model_utils import get_rnn_size
from sample_pretrain.utils.attr_dict import AttrDict
from sample_pretrain.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
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

        self.rnn_state = None
        self.prev_actions = None

    def init(self):
        super().init()

        self.dataset = self._get_dataset()
        self.tp = ThreadPoolExecutor(max_workers=self.cfg.num_workers)

        self.rnn_state = torch.zeros(
            (self.cfg.batch_size, get_rnn_size(self.cfg)), dtype=torch.float32, device=self.device
        )
        self.prev_actions = np.zeros((self.cfg.batch_size, 1))

    def _get_dataset(self):
        if self.cfg.character == "@":
            role, race, align = None, None, None
        else:
            role, race, align = self.cfg.character.split("-")
            role, race, align = Role(role), Race(race), Alignment(align)

        dataset = load_nld_aa_large_dataset(
            data_path=self.cfg.data_path,
            db_path=self.cfg.db_path,
            seq_len=self.cfg.rollout,
            batch_size=self.cfg.batch_size,
            role=role,
            race=race,
            align=align,
        )

        return iter(dataset)

    def _get_minibatch(self) -> TensorDict:
        batch = next(self.dataset)
        screen_image = render_screen_image(
            tty_chars=batch["tty_chars"],
            tty_colors=batch["tty_colors"],
            tty_cursor=batch["tty_cursor"],
            threadpool=self.tp,
        )
        batch["screen_image"] = screen_image
        batch["actions"] = ACTION_MAPPING[batch["keypresses"]]
        batch["prev_actions"] = np.concatenate([self.prev_actions, batch["actions"][:, :-1]], axis=1)

        normalized_batch = prepare_and_normalize_obs(self.actor_critic, batch)
        normalized_batch = TensorDict(normalized_batch)

        return normalized_batch

    def _calculate_loss(self, mb: TensorDict):
        rnn_state = self.rnn_state

        results = []
        seq_len = mb["actions"].shape[1]
        for i in range(seq_len):
            policy_outputs = self.actor_critic(mb[:, i], rnn_state)
            not_done = (1.0 - mb["done"][:, i].float()).unsqueeze(-1)

            targets = mb[:, i]["actions"].unsqueeze(-1)
            observed_log_probs = self.actor_critic.last_action_distribution.log_prob(targets)
            policy_outputs["observed_log_probs"] = observed_log_probs

            rnn_state = policy_outputs["new_rnn_states"] * not_done
            results.append(policy_outputs)

        # update prev_actions and rnn_states for next iteration
        self.rnn_state = rnn_state.detach()
        self.prev_actions = mb["actions"][:, -1].unsqueeze(-1)

        results = AttrDict(cat_tensordicts(results))
        loss = -results["observed_log_probs"].mean()

        return loss, results, {}
