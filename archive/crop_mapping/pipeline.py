# -*- coding: utf-8 -*-

import sys
import argparse
import logging
import TFVietnamCNN as CNNUtils

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# main
#
# python projects/crop-mapping/pipeline.py -c config.yaml -s train
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to map LCLUC in Vietnam using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='configFile',
                        help='Path to the configuration file')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        required=True,
                        dest='pipelineStep',
                        help='Pipeline step to perform',
                        choices=['preprocess', 'train', 'predict'])

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

    # semantic segmentation pipeline
    vietnam_cnn_pipeline = CNNUtils.TFVietnamCNN(args.configFile)

    # execute pipeline step
    if args.pipelineStep == 'preprocess':
        vietnam_cnn_pipeline.preprocess()
    elif args.pipelineStep == 'train':
        vietnam_cnn_pipeline.train()
    elif args.pipelineStep == 'predict':
        vietnam_cnn_pipeline.predict()


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
