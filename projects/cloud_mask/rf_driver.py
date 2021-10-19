# -*- coding: utf-8 -*-
# Extract points from raster for RF training, train, and predict.

import os
import gc
import glob
import sys
import random
import argparse
import logging
import pandas as pd
import numpy as np
import xarray as xr
import joblib
from tqdm import tqdm
import rasterio as rio
import rasterio.features as riofeat

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as sklRFC
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_text

try:
    import cupy as cp
    import cudf as cf
    from cuml.ensemble import RandomForestClassifier as cumlRFC
    from cupyx.scipy.ndimage import median_filter
    cp.random.seed(seed=None)
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

HAS_GPU = False

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# rf_driver.py methods
# -----------------------------------------------------------------------------
def predict(data, model, ws=[5120, 5120]):
    """
    Predict from model.
    :param data: raster xarray object
    :param model: loaded model object
    :param ws: window size to predict on
    :return: prediction output in numpy format
    ----------
    Example
        raster.toraster(filename, raster_obj.prediction, outname)
    ----------
    """
    # open rasters and get both data and coordinates
    rast_shape = data[0, :, :].shape  # shape of the wider scene
    wsx, wsy = ws[0], ws[1]  # in memory sliding window predictions

    # if the window size is bigger than the image, predict full image
    if wsx > rast_shape[0]:
        wsx = rast_shape[0]
    if wsy > rast_shape[1]:
        wsy = rast_shape[1]

    prediction = np.zeros(rast_shape)  # crop out the window
    logging.info(f'wsize: {wsx}x{wsy}. Prediction shape: {prediction.shape}')

    for sx in tqdm(range(0, rast_shape[0], wsx)):  # iterate over x-axis
        for sy in range(0, rast_shape[1], wsy):  # iterate over y-axis
            x0, x1, y0, y1 = sx, sx + wsx, sy, sy + wsy  # assign window
            if x1 > rast_shape[0]:  # if selected x exceeds boundary
                x1 = rast_shape[0]  # assign boundary to x-window
            if y1 > rast_shape[1]:  # if selected y exceeds boundary
                y1 = rast_shape[1]  # assign boundary to y-window

            window = data[:, x0:x1, y0:y1]  # get window
            window = window.stack(z=('y', 'x'))  # stack y and x axis
            window = window.transpose("z", "band").values  # reshape

            # perform sliding window prediction
            prediction[x0:x1, y0:y1] = \
                model.predict(window).reshape((x1 - x0, y1 - y0))
    # save raster
    return prediction.astype('int16')  # type to int16


def toraster(
        rast: str, prediction: np.array, nodataval=[-9999],
        dtype: str = 'int16', output: str = 'rfmask.tif'):
    """
    Save tif file from numpy to disk.
    :param rast: raster name to get metadata from
    :param prediction: numpy array with prediction output
    :param dtype type to store mask on
    :param output: raster name to save on
    :return: None, tif file saved to disk
    ----------
    Example
        raster.toraster(filename, raster_obj.prediction, outname)
    ----------
    """
    # get meta features from raster
    with rio.open(rast) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')
    logging.info(meta)

    prediction = prediction.astype('int16')

    nodatavals[nodatavals == 0] = -9999
    prediction[nodatavals == -9999] = \
        nodatavals[nodatavals == -9999]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = dtype  # data type modification

    # write to a raster
    with rio.open(output, 'w', **out_meta) as dst:
        dst.write(prediction, 1)
    logging.info(f'Prediction saved at {output}')


# -----------------------------------------------------------------------------
# main
#
# python rf_driver.py options here
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to extract balanced points using WV data.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-om', '--output-pkl', type=str, required=True,
        dest='output_pkl', help='Path to the output PKL file')

    parser.add_argument(
        '-d', '--data-csv', type=str, required=False,
        dest='data_csv', help='Path to the data CSV configuration file')

    parser.add_argument(
        '-tc', '--train-csv', type=str, required=False,
        dest='train_csv', help='Path to the output CSV file')

    parser.add_argument(
        '-b', '--bands', type=str, nargs='*', required=False,
        dest='bands', help='Bands to store in CSV file',
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow',
                 'Red', 'RedEdge', 'NIR1', 'NIR2'])

    parser.add_argument(
        '-s', '--step', type=str, nargs='*', required=True,
        dest='pipeline_step', help='Pipeline step to perform',
        default=['prepare', 'train', 'predict', 'vis'],
        choices=['prepare', 'train', 'predict', 'vis'])

    parser.add_argument(
        '-se', '--seed', type=int, required=False, dest='seed',
        default=42, help='Random SEED value')

    parser.add_argument(
        '-ts', '--test-size', type=float, required=False,
        dest='test_size', default=0.20, help='Test size rate (e.g .30)')

    parser.add_argument(
        '-nt', '--n-trees', type=int, required=False,
        dest='n_trees', default=20, help='Number of trees (e.g 20)')

    parser.add_argument(
        '-mf', '--max-features', type=str, required=False,
        dest='max_feat', default='log2', help='Max features (e.g log2)')

    parser.add_argument(
        '-r', '--rasters', type=str, required=False, dest='rasters',
        default='*.tif', help='rasters to search for')

    parser.add_argument(
        '-ws', '--window-size', type=int, required=False,
        dest='ws', default=5120, help='Prediction window size (e.g 5120)')

    parser.add_argument(
        '-od', '--output-dir', type=str, required=False,
        dest='output_dir', default='', help='output directory')

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
    # prepare step
    # --------------------------------------------------------------------------------
    if "prepare" in args.pipeline_step:

        # read data csv file
        assert os.path.exists(args.data_csv), f'{args.data_csv} not found.'
        data_df = pd.read_csv(args.data_csv)
        assert not data_df.isnull().values.any(), f'NaN found: {args.data_csv}'

        # set empty dataframe to store point values
        df_points = pd.DataFrame(columns=args.bands + ['CLASS'])

        # start iterating over each file
        for di in data_df.index:

            # Get filename for output purposes, in the future, add to column
            filename = data_df['data'][di].split('/')[-1]
            logging.info(f'Processing {filename}')

            # Read imagery from disk and process both image and mask
            img = xr.open_rasterio(data_df['data'][di]).values
            mask = xr.open_rasterio(data_df['label'][di]).values

            # squeeze mask if needed, all non-1 values to 0
            mask = np.squeeze(mask) if len(mask.shape) != 2 else mask
            mask[mask != 1] = 0

            # crop ROI, from outside to inside based on pixel address
            ymin, ymax = data_df['ymin'][di], data_df['ymax'][di]
            xmin, xmax = data_df['xmin'][di], data_df['xmax'][di]
            img, mask = \
                img[:, ymin:ymax, xmin:xmax], mask[ymin:ymax, xmin:xmax]

            # crop ROI, from outside to inside based on pixel value
            img = np.clip(img, 0, 10000)

            # get N points from imagery
            points_per_class = data_df['ntiles'][di] // 2

            # extract values from imagery, two classes
            for cv in range(2):

                bbox = img.shape  # size of the image
                counter = 0  # counter for class balancing

                while counter < points_per_class:

                    # get indices and extract spectral and class value
                    y, x = random.randrange(bbox[1]), random.randrange(bbox[2])
                    sv, lv = img[:, y, x], int(mask[y, x])

                    if lv == cv:
                        # append to csv here
                        df_points = df_points.append(
                            pd.DataFrame(
                                [np.append(sv, [lv])],
                                columns=list(df_points.columns)),
                            ignore_index=True)
                        counter += 1

        # save file to disk
        df_points.to_csv(args.train_csv, index=False)
        logging.info(f'Saved dataset file {args.train_csv}')

    # --------------------------------------------------------------------------------
    # train step
    # --------------------------------------------------------------------------------
    if "train" in args.pipeline_step:

        # read data csv file
        assert os.path.exists(args.train_csv), f'{args.train_csv} not found.'
        data_df = pd.read_csv(args.train_csv, sep=',')
        assert not data_df.isnull().values.any(), f'Na found: {args.train_csv}'

        logging.info(f'Open {args.train_csv} dataset for training.')

        # shuffle dataset
        data_df = data_df.sample(frac=1).reset_index(drop=True)

        # split dataset, fix type
        x = data_df.iloc[:, :-1].astype(np.float32)
        y = data_df.iloc[:, -1].astype(np.int8)

        # split data into training and test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=args.test_size, random_state=args.seed)
        del data_df, x, y

        # logging some of the model information
        logging.info(f'X size: {x_train.shape}')  # shape of x data
        logging.info(f'Y size:  {y_train.shape}')  # shape of y data
        logging.info(f'ntrees={str(args.n_trees)}, maxfeat={args.max_feat}')

        # ------------------------------------------------------------------
        # 2. Instantiate RandomForest object
        # ------------------------------------------------------------------
        if HAS_GPU:  # run using RAPIDS library

            # initialize cudf data and log into GPU memory
            logging.info('Training model via RAPIDS.')
            x_train = cf.DataFrame.from_pandas(x_train)
            x_test = cf.DataFrame.from_pandas(x_test)
            y_train = cf.Series(y_train.values)

            rf_funct = cumlRFC  # RF Classifier

        else:
            logging.info('Training model via SKLearn.')
            rf_funct = sklRFC

        # Initialize model
        rf_model = rf_funct(
            n_estimators=args.n_trees, max_features=args.max_feat)

        # fit model to training data and predict for accuracy score
        rf_model.fit(x_train, y_train)

        if HAS_GPU:
            acc_score = accuracy_score(
                y_test, rf_model.predict(x_test).to_array())
            p_score = precision_score(
                y_test, rf_model.predict(x_test).to_array())
            r_score = recall_score(
                y_test, rf_model.predict(x_test).to_array())
            f_score = f1_score(
                y_test, rf_model.predict(x_test).to_array())
        else:
            acc_score = accuracy_score(y_test, rf_model.predict(x_test))
            p_score = precision_score(y_test, rf_model.predict(x_test))
            r_score = recall_score(y_test, rf_model.predict(x_test))
            f_score = f1_score(y_test, rf_model.predict(x_test))

        logging.info(f'Test Accuracy:  {acc_score}')
        logging.info(f'Test Precision: {p_score}')
        logging.info(f'Test Recall:    {r_score}')
        logging.info(f'Test F-Score:   {f_score}')

        # make output directory
        os.makedirs(
            os.path.dirname(os.path.realpath(args.output_pkl)), exist_ok=True)

        # export model to file
        try:
            joblib.dump(rf_model, args.output_pkl)
            logging.info(f'Model has been saved as {args.output_pkl}')
        except Exception as e:
            logging.error(f'ERROR: {e}')

    # --------------------------------------------------------------------------------
    # predict step
    # --------------------------------------------------------------------------------
    if "predict" in args.pipeline_step:

        assert os.path.exists(args.output_pkl), f'{args.output_pkl} not found.'
        model = joblib.load(args.output_pkl)  # loading pkl in parallel
        logging.info(f'Loaded model {args.output_pkl}.')

        # create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # 3b3. apply model and get predictions
        rasters = glob.glob(args.rasters)
        assert len(rasters) > 0, "No raster found"
        logging.info(f'Predicting {len(rasters)} files.')

        for rast in rasters:  # iterate over each raster

            filename = rast.split('/')[-1]

            gc.collect()  # clean garbage
            logging.info(f"Starting new prediction...{rast}")
            img = xr.open_rasterio(rast)

            # TODO: Add function for selecting bands
            img = img[:4, :, :]
            logging.info(f'Modified image: {img.shape}')

            # crop ROI, from outside to inside based on pixel value
            img = np.clip(img, 0, 10000)
            prediction = predict(img, model, ws=[args.ws, args.ws])

            # sieve
            riofeat.sieve(prediction, 800, prediction, None, 8)

            # median
            # with cp.cuda.Device(1):
            prediction = median_filter(cp.asarray(prediction), size=20)
            prediction = cp.asnumpy(prediction)

            output_filename = os.path.join(args.output_dir, filename)
            toraster(rast=rast, prediction=prediction, output=output_filename)
            prediction = None  # unload between each iteration

    # --------------------------------------------------------------------------------
    # predict step
    # --------------------------------------------------------------------------------
    if "vis" in args.pipeline_step:

        assert os.path.exists(args.output_pkl), f'{args.output_pkl} not found.'
        model = joblib.load(args.output_pkl)  # loading pkl in parallel
        logging.info(f'Loaded model {args.output_pkl}.')

        logging.info(dir(model))   # .estimators_[0])
        #    export_text(
        #        model.estimators_[0], spacing=3, decimals=3,
        #        feature_names=args.bands)
        #    )

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
