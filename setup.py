import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()
    descr_lines = long_description.split("\n")
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for dl in descr_lines:
        if not ("<img src=" in dl and "gif" in dl):
            descr_no_gifs.append(dl)

    long_description = "\n".join(descr_no_gifs)

_nethack_deps = [
    "pandas ~= 2.1",
    "matplotlib ~= 3.8",
    "seaborn ~= 0.12",
    "nle @ git+https://github.com/BartekCupial/nle.git",
]

_mrunner_deps = ["mrunner @ git+https://gitlab.com/awarelab/mrunner.git"]

_docs_deps = [
    "mkdocs-material",
    "mkdocs-minify-plugin",
    "mkdocs-redirects",
    "mkdocs-git-revision-date-localized-plugin",
    "mkdocs-git-committers-plugin-2",
    "mkdocs-git-authors-plugin",
]

setup(
    # Information
    name="sample-pretrain",
    description="Pretrain policies on offline data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    url="https://github.com/BartekCupial/sample-pretrain.git",
    author="Bartek Cupiał",
    license="MIT",
    keywords="pretrain reinforcement learning",
    project_urls={},
    install_requires=[
        "numpy>=1.18.1,<2.0",
        "torch>=1.9,<3.0,!=1.13.0",  # install with conda install -yq pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        "gymnasium>=0.27,<1.0",
        "pyglet",  # gym dependency
        "tensorboard>=1.15.0",
        "tensorboardx>=2.0",
        "psutil>=5.7.0",
        "threadpoolctl>=2.0.0",
        "colorlog",
        # "faster-fifo>=1.4.2,<2.0",  <-- installed by signal-slot-mp
        "signal-slot-mp>=1.0.3,<2.0",
        "filelock",
        "opencv-python",
        "wandb>=0.12.9",
        "huggingface-hub>=0.10.0,<1.0",
        "numba ~= 0.58",
        "scipy ~= 1.11",
        "shimmy",
        "tqdm ~= 4.66",
        "debugpy ~= 1.6",
        "nle @ git+https://github.com/BartekCupial/nle.git",
    ],
    extras_require={
        # some tests require Atari and Mujoco so let's make sure dev environment has that
        "dev": ["black", "isort>=5.12", "pytest<8.0", "flake8", "pre-commit", "twine"]
        + _docs_deps
        + _nethack_deps
        + _mrunner_deps,
        "nethack": _nethack_deps,
        "mrunner": _mrunner_deps,
    },
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./", include=["sample_pretrain*", "sp_examples*"]),
    include_package_data=True,
    python_requires=">=3.8",
)
