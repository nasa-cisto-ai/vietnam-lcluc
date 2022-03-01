# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from omegaconf import OmegaConf

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from vietnam_lcluc.model.config import Config
from vietnam_lcluc.model.cnn_base import CNNPipeline

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# main
#
# python view/cnn_pipeline.py -c config.yaml -s train
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to map LCLUC in Vietnam using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        choices=['preprocess', 'train', 'predict', 'all'])

    args = parser.parse_args()

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Configuration file intialization
    schema = OmegaConf.structured(Config)
    conf = OmegaConf.load(args.config_file)
    try:
        OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")

    # Semantic segmentation pipeline
    cnn_pipeline = CNNPipeline(conf)

    # execute pipeline step
    if args.pipeline_step in ['preprocess', 'all']:
        cnn_pipeline.preprocess()
    if args.pipeline_step in ['train', 'all']:
        cnn_pipeline.train()
    if args.pipeline_step in ['predict', 'all']:
        cnn_pipeline.predict()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
