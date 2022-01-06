import os
import sys
import cupy as cp
import numpy as np
import tensorflow as tf
from ConfigTF import ConfigTF
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

class PipelineTF(object):

    def __init__(self, conf: DictConfig):

        self.conf = conf
        self._seed_everything(self.conf.seed)

    # ------------------------------------------------------------------
    # Main Public Methods
    # ------------------------------------------------------------------
    def preprocess():
        raise NotImplementedError
    
    def train():
        raise NotImplementedError

    def predict():
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main Public Methods
    # ------------------------------------------------------------------
    def _seed_everything(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        cp.random.seed(seed)


# -----------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------
if __name__ == "__main__":

    schema = OmegaConf.structured(ConfigTF)
    conf = OmegaConf.load("../config/config_clouds/vietnam_clouds.yaml")
    try:
        OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err=}")

    pipeline = PipelineTF(conf)
    sys.exit(pipeline)
