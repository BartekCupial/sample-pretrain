## Installation

### nle dependencies
```shell
apt-get install -yq autoconf libtool pkg-config libbz2-dev
```

### render utils
```shell
conda install -yq cmake flex bison lit
conda install -yq pybind11 -c conda-forge

cd sp_examples/nethack/render_utils
pip install .
cd ../../..
```

### install with nethack dependencies
```shell
pip install .[nethack]
```

### or if you want to develop
```shell
pip install -e .[dev]
```
