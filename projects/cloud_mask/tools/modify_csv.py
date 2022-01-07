import os
import sys
import argparse
import logging
import pandas as pd

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def main():

    # Process command-line args.
    desc = 'Use this application to modify CSV file.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-i', '--input-csv', type=str, required=True,
        dest='input_csv', help='Input CSV to modify')

    parser.add_argument(
        '-o', '--output-csv', type=str, required=True,
        dest='output_csv', help='Output CSV to save')

    parser.add_argument(
        '-ic', '--input-columns', type=str, nargs='*', required=False,
        dest='input_columns', help='Columns from input CSV',
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow',
                 'Red', 'RedEdge', 'NIR1', 'NIR2', 'L'])

    parser.add_argument(
        '-oc', '--output-columns', type=str, nargs='*', required=False,
        dest='output_columns', help='Columns for output CSV',
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow',
                 'Red', 'RedEdge', 'NIR1', 'NIR2', 'L'])

    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # set logging
    # --------------------------------------------------------------------------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)  # set stdout handler
    ch.setLevel(logging.INFO)

    # set formatter and handlers
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --------------------------------------------------------------------------------
    # modify step
    # --------------------------------------------------------------------------------
    df = pd.read_csv(args.input_csv, names=args.input_columns)
    df = df[args.output_columns]

    # Save into file
    logging.info(df)
    df.to_csv(args.output_csv, index=False, header=False)

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
