import os
import sys
import glob
import argparse
import logging
import pandas as pd

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def main():

    # Process command-line args.
    desc = 'Use this application to merge CSV files.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-i', '--input-dir', type=str, required=True,
        dest='input_dir', help='Input directory where CSV are located')

    parser.add_argument(
        '-o', '--output-csv', type=str, required=True,
        dest='output_csv', help='Output CSV to save')

    parser.add_argument(
        '-ic', '--input-columns', type=str, nargs='*', required=False,
        dest='input_columns', help='Columns from input CSV',
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
    # merge step
    # --------------------------------------------------------------------------------
    df_list = list()
    input_columns = args.input_columns
    csv_files = glob.glob(os.path.join(args.input_dir, '*.csv'))

    for filename in csv_files:
        df_list.append(
            pd.read_csv(
                filename, index_col=None, header=None, names=input_columns)
        )

    # Concatenate all datasets
    full_df = pd.concat(df_list, axis=0, ignore_index=True)

    # Get the order of bands and move classes to last column
    input_columns.append(input_columns.pop(input_columns.index('L')))
    full_df = full_df[input_columns]

    # Save into file
    full_df.to_csv(args.output_csv, index=False, header=False)

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
