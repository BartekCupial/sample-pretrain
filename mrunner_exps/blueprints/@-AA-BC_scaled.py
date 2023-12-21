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
    "batch_size": 512,
    "save_milestones_ith": 10_000_000,
    "wandb_user": "bartekcupial",
    "wandb_project": "sp_nethack",
    "wandb_group": "gmum",
    "with_wandb": False,
    "db_path": "/ttyrecs/ttyrecs.db",
    "data_path": "/nle/nld-aa-l/nle_data",
    "dataset_name": "autoascend",
    "use_prev_action": True,
    "model": "ScaledNet",
    "use_resnet": True,
    "rnn_size": 512,
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
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
