# -*- coding: utf-8 -*-

import os
import yaml
import sys
import math
import time
import logging
import numpy as np
import tensorflow as tf
from functools import wraps
import rasterio as rio
from sklearn.feature_extraction import image
from skimage import img_as_ubyte, exposure
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import cv2
from scipy.ndimage import median_filter, binary_fill_holes

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -----------------------------------------------------------------------------
# class ConfigYAML
# -----------------------------------------------------------------------------
class ConfigYAML(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, pathToFile):

        assert pathToFile, 'A fully-qualified path to a file must be specified.'
        assert os.path.exists(pathToFile), f'{str(pathToFile)} does not exist.'
        self._filePath = pathToFile  # configuration file
        
        # open configuration file
        with open(self._filePath) as f:
            hpr = yaml.load(f, Loader=yaml.SafeLoader)
        assert hpr != None, f'{self._filePath} is empty. Please add parameters.'

        for key in hpr:  # initialize attributes from configuration file
            setattr(self, key, hpr[key])

# -----------------------------------------------------------------------------
# class ToolBelt
# -----------------------------------------------------------------------------
class ToolBelt(object):

    # --------------------------------------------------------------------------
    # Decorators
    # --------------------------------------------------------------------------
    def timeit(my_func):
        @wraps(my_func)
        def timed(*args, **kw):
            tstart = time.time()
            output = my_func(*args, **kw)
            tend = time.time()
            logging.info(f'{my_func.__name__} took {(tend - tstart) * 1000: .3f} ms to execute')
            return output
        return timed

    # --------------------------------------------------------------------------
    # System Methods
    # --------------------------------------------------------------------------
    def _setup(self, strategy=None):
        """
        self._setup method: setup gpu and logging methods for the entire pipeline.
        """
        # verify GPU devices are available and ready
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_devices
        logging.info(f'TensorFlow version: {tf.__version__}')

        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
        else:  # Use the Default Strategy
            strategy = tf.distribute.get_strategy()
        logging.info(f'Using strategy: {strategy}')

        # enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        # tf.config.optimizer.set_jit(True)  # enable XLA acceleration
        return strategy
    
    def _createOutputDirs(self):
        """
        Create output directories for model output.
        """
        self.datasetDir = os.path.join(self.data_output_dir, 'dataset')        
        self.modelDir = os.path.join(self.data_output_dir, 'model')
        self.inferenceDir = os.path.join(self.inference_output_dir)
        self.imagesDir = os.path.join(self.datasetDir, 'images')
        self.labelsDir = os.path.join(self.datasetDir, 'labels')
        
        for d in [self.modelDir, self.inferenceDir, self.imagesDir, self.labelsDir]:
            os.makedirs(d, exist_ok=True)

    # --------------------------------------------------------------------------
    # Preprocessing Methods
    # --------------------------------------------------------------------------
    def _extractTiles(self, img, mask, tile_size=(256, 256), random_state=22, n_patches=100):
        """
        Extract small patches for final dataset
        Args:
            img (numoy array - c, y, x): imagery data
            tile_size (tuple): 2D dimensions of tile
            random_state (int): seed for reproducibility (match image and mask)
            n_patches (int): number of tiles to extract
        """
        img = image.extract_patches_2d(
            image=img, max_patches=n_patches,
            patch_size=tile_size, random_state=random_state
        )  
        mask = image.extract_patches_2d(
            image=mask, max_patches=n_patches,
            patch_size=tile_size, random_state=random_state
        )          
        return img, mask

    def _contrastStretchCalc(self, img, percentile=(0.5,99.5), out_range=(0,1)):
        """
        Stretch the image histogram for each channel independantly. The image histogram
        is streched such that the lower and upper percentile are saturated.
        ------------
        INPUT
            |---- img (3D numpy array) the image to stretch (H x W x B)
            |---- percentile (tuple) the two percentile value to saturate
            |---- out_range (tuple) the output range value
        OUTPUT
            |---- img_adj (3D numpy array) the streched image (H x W x B)
        """
        n_band = img.shape[-1]
        q = [tuple(np.percentile(img[:,:,i], [0,99.5])) for i in range(n_band)]
        return np.stack(
            [
                exposure.rescale_intensity(
                    img[:,:,i], in_range=q[i], out_range=out_range
                ) for i in range(n_band)
            ], 
            axis=-1
        )

    def _contrastStretch(
        self, img, lower=0.02, higher=0.98, min_value=0, max_value=1
    ):
        if len(img.shape) <= 3:
            return self._contrastStretchCalc(
                img, percentile=(2, 98), out_range=(0, 1)
            )
        else:
            for b in range(img.shape[0]):
                img[b, :, :, :] = self._contrastStretchCalc(
                    img[b, :, :, :], percentile=(2, 98), out_range=(0, 1)
                )
            return img

    def _getSTDInfo(self, images, axis=(0, 1)):
        means = images.mean(axis)
        stds = images.std(axis)
        np.savez(f"{self.experiment_name}_norm_data.npy", mean=means, std=stds)
        return means, stds

    def _standardizeCalc(self, img, mean=None, std=None, axis=(0, 1), c=1e-8):
        """
        Normalize to zero mean and unit standard deviation along the given axis
        Args:
            img (numpy or cupy): array (w, h, c)
            axis (integer tuple): into or tuple of width and height axis
            c (float): epsilon to bound given std value
        Return:
            Normalize single image
        ----------
        Example
        ----------
            image_normalize(arr, axis=(0, 1), c=1e-8)
        """
        if mean and std:
            return (img - mean) / (std + c)
        else:
            return (img - img.mean(axis)) / (img.std(axis) + c)

    def _standardizeLocalCalc(self, img, axis=(0, 1), c=1e-8):
        """
        Normalize to zero mean and unit standard deviation along the given axis
        Args:
            img (numpy or cupy): array (w, h, c)
            axis (integer tuple): into or tuple of width and height axis
            c (float): epsilon to bound given std value
        Return:
            Normalize single image
        ----------
        Example
        ----------
            image_normalize(arr, axis=(0, 1), c=1e-8)
        """
        for i in range(img.shape[-1]):  # for each channel in images
            img[:, :, i] = \
                (img[:, :, i] - self.means[i]) / (self.stds[i] + c)
        return img

    def _standardizeCalcTensor(self, img, mean=None, std=None, axis=(0, 1), c=1e-8):
        """
        Normalize to zero mean and unit standard deviation along the given axis
        Args:
            img (numpy or cupy): array (w, h, c)
            axis (integer tuple): into or tuple of width and height axis
            c (float): epsilon to bound given std value
        Return:
            Normalize single image
        ----------
        Example
        ----------
            image_normalize(arr, axis=(0, 1), c=1e-8)
        """
        if mean and std:
            return (img - mean) / (std + c)
        else:
            return (img - tf.reduce_mean(img, axis=axis)) / \
                (tf.math.reduce_std(img, axis=axis) + c)

    def _standardizeLocalCalcTensor(self, img, axis=(0, 1), c=1e-8):
        """
        Normalize to zero mean and unit standard deviation along the given axis
        Args:
            img (numpy or cupy): array (w, h, c)
            axis (integer tuple): into or tuple of width and height axis
            c (float): epsilon to bound given std value
        Return:
            Normalize single image
        ----------
        Example
        ----------
            image_normalize(arr, axis=(0, 1), c=1e-8)
        """
        for i in range(img.shape[-1]):  # for each channel in images
            img[:, :, i] = \
                (img[:, :, i] - self.means[i]) / (self.stds[i] + c)
        return img

    def _standardize(self, img, mean=None, std=None, axis=(0, 1), c=1e-8):
        """
        Normalize batch to zero mean and unit standard deviation.
        Args:
            img (numpy or cupy): array (n, w, h, c)
            axis (integer tuple): into or tuple of width and height axis
            c (float): epsilon to bound given std value
        Return:
            Normalize batch of images.
        ----------
        Example
        ----------
            batch_normalize(arr, axis=(0, 1), c=1e-8)
        """

        if len(img.shape) <= 3:
            return self._standardizeCalc(
                img, mean=mean, std=std, axis=axis, c=c
            )
        else:
            for b in range(img.shape[0]):
                img[b, :, :, :] = self._standardizeCalc(
                    img[b, :, :, :], mean=mean, std=std, axis=axis, c=c
                )
            return img

    # --------------------------------------------------------------------------
    # Loss Methods
    # --------------------------------------------------------------------------
    def bcedice_loss(y_true, y_pred):
        def dice_loss(y_true, y_pred):
            y_pred = tf.math.sigmoid(y_pred)
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)
            return 1 - numerator / denominator

        # y_true = tf.cast(y_true, tf.float16)
        o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
        return tf.reduce_mean(o)

    def dice_loss(y_true, y_pred, smooth=1):
        """
        Dice coefficient as a loss function.
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        dice_coef = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + \
            K.sum(K.square(y_pred),-1) + smooth)
        return 1 - dice_coef

    # --------------------------------------------------------------------------
    # Prediction Methods
    # --------------------------------------------------------------------------
    def _pred_mask(self, pr, threshold=0.50):
        '''Predicted mask according to threshold'''
        pr_cp = np.copy(pr)
        # logging.info(pr_cp.mean(), pr_cp.std())
        pr_cp[pr_cp < threshold] = 0
        pr_cp[pr_cp >= threshold] = 1
        return pr_cp

    def _denoise(self, mask, eps=30):
        """Removes noise from a mask.
        Args:
            mask: the mask to remove noise from.
            eps: the morphological operation's kernel size for noise removal, in pixel.
        Returns:
            The mask after applying denoising.
        """
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)

    def _grow(self, mask, eps=60):
        """Grows a mask to fill in small holes, e.g. to establish connectivity.
        Args:
            mask: the mask to grow.
            eps: the morphological operation's kernel size for growing, in pixel.
        Returns:
            The mask after filling in small holes.
        """
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct)

    def _binary_fill(self, mask):
        """Grows a mask to fill in small holes, e.g. to establish connectivity.
        Args:
            mask: the mask to grow.
            eps: the morphological operation's kernel size for growing, in pixel.
        Returns:
            The mask after filling in small holes.
        """
        #output = median_filter(output, size=20)
        #output = binary_fill_holes(output).astype(int)
        #mask = median_filter(mask, size=20)
        return binary_fill_holes(mask).astype(int)

    # --------------------------------------------------------------------------
    # Visualization Methods
    # --------------------------------------------------------------------------
    def display_sample(self, display_list):
        """Show side-by-side an input image,
        the ground truth and the prediction.

        Given a tensorflow dataset:
        for image, mask in dataset['train'].take(40):
            sample_image, sample_mask = image, mask
            self.display_sample([sample_image[0], sample_mask[0]])
        """
        plt.figure(figsize=(10, 10))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()
        return

    # --------------------------------------------------------------------------
    # Output Methods
    # --------------------------------------------------------------------------
    def toRasterMask(raster_f, segments, out_tif='segment.tif', ndval=-9999):
        # get geospatial profile, will apply for output file
        with rio.open(raster_f) as src:
            meta = src.profile
            nodatavals = src.read_masks(1).astype('int16')
        print(meta)

        # load numpy array if file is given
        if type(segments) == str:
            segments = np.load(segments)
        segments = segments.astype('int16')
        print(segments.dtype)  # check datatype

        nodatavals[nodatavals == 0] = ndval
        segments[nodatavals == ndval] = nodatavals[nodatavals == ndval]

        out_meta = meta  # modify profile based on numpy array
        out_meta['count'] = 1  # output is single band
        out_meta['dtype'] = 'int16'  # data type is float64

        # write to a raster
        with rio.open(out_tif, 'w', **out_meta) as dst:
            dst.write(segments, 1)

# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("No tests included")