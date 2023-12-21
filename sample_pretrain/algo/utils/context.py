from typing import Dict

from sample_pretrain.model.model_factory import ModelFactory
from sample_pretrain.utils.typing import CreateEnvFunc


class SampleFactoryContext:
    def __init__(self):
        self.ddp_mode = False
        self.env_registry = dict()
        self.model_factory = ModelFactory()
        from sample_pretrain.algo.learning.learner import Learner

        self.learner_cls = Learner


GLOBAL_CONTEXT = None


def sf_global_context() -> SampleFactoryContext:
    global GLOBAL_CONTEXT
    if GLOBAL_CONTEXT is None:
        GLOBAL_CONTEXT = SampleFactoryContext()
    return GLOBAL_CONTEXT


def set_global_context(ctx: SampleFactoryContext):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = ctx


def reset_global_context():
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = SampleFactoryContext()


def global_ddp_mode() -> bool:
    """
    :return: global ddp mode
    :rtype: bool
    """
    return sf_global_context().ddp_mode


def global_env_registry() -> Dict[str, CreateEnvFunc]:
    """
    :return: global env registry
    :rtype: EnvRegistry
    """
    return sf_global_context().env_registry


def global_model_factory() -> ModelFactory:
    """
    :return: global model factory
    :rtype: ModelFactory
    """
    return sf_global_context().model_factory


def global_learner_cls() -> type:
    """
    :return: global learner class
    :rtype: type
    """
    return sf_global_context().learner_cls
