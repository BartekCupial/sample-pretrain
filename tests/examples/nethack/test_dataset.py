import pytest
import torch

from sample_pretrain.algo.utils.make_env import make_env_func_batched
from sample_pretrain.model.actor_critic import create_actor_critic
from sample_pretrain.model.model_utils import get_rnn_size
from sample_pretrain.utils.timing import Timing
from sample_pretrain.utils.utils import log
from sp_examples.nethack.datasets.dataset import load_nld_aa_large_dataset
from sp_examples.nethack.datasets.roles import Alignment, Race, Role
from sp_examples.nethack.train_nethack import parse_nethack_args


class TestNLEDataset:
    @pytest.fixture(scope="class", autouse=True)
    def register_nethack_fixture(self):
        from sp_examples.nethack.train_nethack import register_nethack_components

        register_nethack_components()

    @pytest.mark.parametrize("batch_size", [128, 256])
    @pytest.mark.parametrize("seq_len", [16, 32])
    @pytest.mark.parametrize(
        "role, race, align",
        [
            (None, None, None),
            (Role("mon"), Race("hum"), Alignment("neu")),
            (Role("val"), Race("dwa"), Alignment("law")),
        ],
    )
    def test_sample_dataset(self, batch_size: int, seq_len: int, role: Role, race: Race, align: Alignment):
        timing = Timing()

        with timing.timeit("all"):
            with timing.timeit("create_ds"):
                dataset = load_nld_aa_large_dataset(
                    data_path="/home/bartek/Workspace/data/nle/nld-aa-taster/nle_data",
                    db_path="/home/bartek/Workspace/data/nethack/AA-taster/ttyrecs.db",
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_workers=8,
                    role=role,
                    race=race,
                    align=align,
                )

            with timing.timeit("create_iter"):
                ds = iter(dataset)

            with timing.timeit("sample_batch"):
                batch = next(ds)
                key = list(batch.keys())[0]
                assert batch[key].shape[0] == batch_size
                assert batch[key].shape[1] == seq_len

        log.debug("Timing: %s", timing)
