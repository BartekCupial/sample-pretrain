from sample_pretrain.utils.attr_dict import AttrDict


class Configurable:
    def __init__(self, cfg: AttrDict):
        self.cfg: AttrDict = cfg
