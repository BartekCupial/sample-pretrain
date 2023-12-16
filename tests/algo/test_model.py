import pytest
import torch

from sample_pretrain.algo.utils.make_env import make_env_func_batched
from sample_pretrain.model.actor_critic import create_actor_critic
from sample_pretrain.model.model_utils import get_rnn_size
from sample_pretrain.utils.timing import Timing
from sample_pretrain.utils.utils import log
from sp_examples.nethack.train_nethack import parse_nethack_args


class TestModel:
    @pytest.fixture(scope="class", autouse=True)
    def register_nethack_fixture(self):
        from sp_examples.nethack.train_nethack import register_nethack_components

        register_nethack_components()

    @staticmethod
    def forward_pass(device_type):
        env_name = "nethack_challenge"
        cfg = parse_nethack_args(argv=[f"--env={env_name}"])
        cfg.actor_critic_share_weights = True
        cfg.use_rnn = True

        env = make_env_func_batched(cfg, env_config=None)

        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        device = torch.device(device_type)
        actor_critic.to(device)

        timing = Timing()
        with timing.timeit("all"):
            batch = 128
            with timing.add_time("input"):
                # better avoid hardcoding here...
                observations = {
                    key: torch.rand([batch, *value.shape]).to(device) for key, value in env.observation_space.items()
                }
                rnn_states = torch.rand([batch, get_rnn_size(cfg)]).to(device)

            n = 100
            for i in range(n):
                with timing.add_time("forward"):
                    _ = actor_critic(observations, rnn_states)

                if i % 10 == 0:
                    log.debug("Progress %d/%d", i, n)

        log.debug("Timing: %s", timing)

    def test_forward_pass_cpu(self):
        self.forward_pass("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a GPU")
    def test_forward_pass_gpu(self):
        self.forward_pass("cuda")
