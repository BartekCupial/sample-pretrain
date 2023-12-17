import copy

import pytest

from sample_pretrain.algo.utils.env_info import extract_env_info
from sample_pretrain.algo.utils.make_env import make_env_func_batched
from sample_pretrain.cfg.arguments import verify_cfg
from sp_examples.nethack.algo.learning.bc_learner import BCLearner
from sp_examples.nethack.train_nethack import parse_nethack_args


class TestBCLearner:
    @pytest.fixture(scope="class", autouse=True)
    def register_nethack_fixture(self):
        from sp_examples.nethack.train_nethack import register_nethack_components

        register_nethack_components()

    @pytest.mark.parametrize("use_rnn", [False, True])
    def test_losses_match(self, use_rnn: bool):
        env_name = "nethack_challenge"
        cfg = parse_nethack_args(
            argv=[f"--env={env_name}", "--db_path=/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db"]
        )
        cfg.actor_critic_share_weights = True
        cfg.use_rnn = use_rnn
        cfg.num_workers = 2
        cfg.rollout = 8
        cfg.batch_size = 32
        cfg.device = "cpu"
        cfg.recurrence = cfg.rollout

        tmp_env = make_env_func_batched(cfg, env_config=None)
        env_info = extract_env_info(tmp_env, cfg)

        assert verify_cfg(cfg, env_info)

        learner: BCLearner = BCLearner(cfg, env_info)
        learner.init()
        assert learner.actor_critic is not None

        for _ in range(10):
            batch = learner._get_minibatch()
            learner._calculate_loss(batch)

        initial_rnn_state = learner.rnn_state.clone()
        initial_prev_actions = learner.prev_actions.clone()

        batch = learner._get_minibatch()
        og_batch = copy.deepcopy(batch)

        loss, *_ = learner._calculate_loss(batch)
        after_batch_rnn_state = learner.rnn_state.clone()
        after_batch_prev_actions = learner.prev_actions.clone()

        learner.rnn_state = initial_rnn_state
        learner.prev_actions = initial_prev_actions
        og_loss, *_ = learner._calculate_loss(og_batch)

        assert loss == og_loss
        assert (after_batch_rnn_state == learner.rnn_state).all()
        assert (after_batch_prev_actions == learner.prev_actions).all()
