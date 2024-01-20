from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

# params for all exps
config = {
    "env": "nethack_challenge",
    "exp_tags": [name],
    "exp_point": "@-AA-BC",
    "train_for_env_steps": 2_000_000_000,
    "group": "@-AA-BC",
    "character": "@",
    "num_workers": 16,
    "worker_num_splits": 2,
    "rollout": 32,
    "batch_size": 16,
    "wandb_user": "rahid",
    "wandb_project": "sp_nethack",
    "wandb_group": "rahid",
    "with_wandb": False,
    "db_path": "/home/maciejwolczyk/Repos/ttyrecs.db",
    "data_path": "/home/maciejwolczyk/Pobrane/nld-aa-taster/nle_data",
    "dataset_name": "nld-aa-taster-v1",
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "rnn_size": 64,
    "policy_initialization": "torch_default",
    "rnn_type": "mamba",
}

# params different between exps
params_grid = [
    {
        "seed": list(range(5)),
    },
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="sp_nethack",
    with_neptune=False,
    script="ld-config /.singularity-libs/; python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
