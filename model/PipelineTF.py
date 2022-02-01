import os
import sys
import time
import random
import logging
import cv2
import cupy as cp
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from pathlib import Path
from typing import List
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau
from scipy.ndimage import median_filter, binary_fill_holes

from .ConfigTF import ConfigTF
from .UNetTF import unet_batchnorm
from .Mosaic import from_array

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
AUTOTUNE = tf.data.experimental.AUTOTUNE

class TverskyLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred, beta=0.7):
        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
        r = 1 - (numerator + 1) / (tf.reduce_sum(denominator) + 1)
        return tf.cast(r, tf.float32)
        #y_pred = tf.convert_to_tensor_v2(y_pred)
        #y_true = tf.cast(y_true, y_pred.dtype)
        #return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)


def tversky_loss(y_true, y_pred, beta=0.7):
    """ Tversky index (TI) is a generalization of Dice’s coefficient. TI adds a weight to FP (false positives) and FN (false negatives). """
    def tversky_loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        r = 1 - (numerator + 1) / (tf.reduce_sum(denominator) + 1)
        return tf.cast(r, tf.float32)

    return tf.numpy_function(tversky_loss, [y_true, y_pred], tf.float32)

class PipelineTF(object):

    def __init__(self, conf: DictConfig):

        self.conf = conf
        
        self._dataset_dir = os.path.join(self.conf.data_dir, 'dataset')
        self._images_dir = os.path.join(self.conf.data_dir, 'images')
        self._labels_dir = os.path.join(self.conf.data_dir, 'labels')

        self._model_dir = os.path.join(self.conf.data_dir, 'model')
        self._create_work_dirs([self._images_dir, self._labels_dir, self._model_dir])
        
        self._seed_everything(self.conf.seed)
        self._gpu_strategy = self._set_gpu_strategy(self.conf.gpu_devices)
        
        if self.conf.mixed_precision:
            self._set_mixed_precision()
        
        if self.conf.xla:
            self._set_xla()

    # ------------------------------------------------------------------
    # Main Public Methods
    # ------------------------------------------------------------------
    def preprocess(self):
        
        # Initialize dataframe with data details
        data_df = self._read_dataset_csv(self.conf.dataset_csv)

        # iterate over each file and generate dataset
        for data_filename, label_filename, n_tiles in data_df.values:

            # Read imagery from disk and process both image and mask
            image = rxr.open_rasterio(data_filename, chunks=CHUNKS).load()
            label = rxr.open_rasterio(label_filename, chunks=CHUNKS).values
            print(image.shape, label.shape)

            image = self._modify_bands(
                xraster=image, input_bands=self.conf.input_bands,
                output_bands=self.conf.output_bands)
            print(image.shape, label.shape)

            # Asarray option to force array type
            image = cp.asarray(image.values)
            label = cp.asarray(label)

            label[label == 6] = 5

            # Move from chw to hwc, squeze mask if required
            image = cp.moveaxis(image, 0, -1).astype(np.int16)
            label = cp.squeeze(label) if len(label.shape) != 2 else label
            print(f'Label classes from image: {cp.unique(label)}')

            self._gen_random_tiles(
                image=image, label=label, 
                tile_size=self.conf.tile_size,
                max_patches=n_tiles, 
                include=self.conf.include_classes,
                augment=self.conf.augment,
                output_filename=data_filename
            )
        return
       
    def train(self):

        data_filenames = self._get_dataset_filenames(self._images_dir)
        label_filenames = self._get_dataset_filenames(self._labels_dir)
        print(len(data_filenames), len(label_filenames))

        total_size = len(data_filenames)
        val_size = round(self.conf.test_size * total_size)

        train_x, val_x = train_test_split(
            data_filenames, test_size=val_size, random_state=self.conf.seed)
        train_y, val_y = train_test_split(
            label_filenames, test_size=val_size, random_state=self.conf.seed)

        # Disable AutoShard, data lives in memory, use in memory options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF

        train_dataset = self._tf_train_dataset(train_x, train_y)
        val_dataset = self._tf_val_dataset(val_x, val_y)

        # Disable AutoShard, data lives in memory, use in memory options
        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)

        # Initialize and compile model
        with self._gpu_strategy.scope():

            # initialize UNet model
            model = unet_batchnorm(
                nclass=self.conf.n_classes,
                input_size=(self.conf.tile_size, self.conf.tile_size, len(self.conf.output_bands)),
                maps=[64, 128, 256, 512, 1024]
            )

            # enabling mixed precision to avoid underflow
            # optimizer = mixed_precision.LossScaleOptimizer(optimizer)

            optimizer = tf.keras.optimizers.Adam(self.conf.learning_rate)
            metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), self._iou]
            # model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
            model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)
            # tf.keras.losses.SparseCategoricalCrossentropy()
            #model.compile(loss=TverskyLoss(), optimizer=optimizer, metrics=metrics)
            model.summary()

            callbacks = [
                ModelCheckpoint(
                    filepath=os.path.join(self._model_dir, '{epoch:02d}-{val_loss:.2f}.hdf5'),
                    monitor='val_acc',
                    mode='max',
                    save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
                CSVLogger(os.path.join(self._model_dir, f"{self.conf.experiment_name}.csv")),
                TensorBoard(log_dir=os.path.join(self._model_dir, 'tensorboard_logs')),
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
            ]

        train_steps = len(train_x) // self.conf.batch_size
        val_steps = len(val_x) // self.conf.batch_size

        if len(train_x) % self.conf.batch_size != 0:
            train_steps += 1
        if len(val_x) % self.conf.batch_size != 0:
            val_steps += 1

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.conf.max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks)

        return

    def predict(self):
        
        # Loading the trained model
        #assert model filename

        with self._gpu_strategy.scope():
            model = tf.keras.models.load_model(
                self.conf.model_filename, custom_objects={
                    "_iou": self._iou,
                    "TverskyLoss": TverskyLoss()
                    }
                )
            model.summary()  # print summary of the model

        os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # Get list of files to predict
        inference_filenames = glob(self.conf.inference_regex)
        print(f'Number of files to predict: {len(inference_filenames)}')

        # Tterate over files and predict them
        for filename in inference_filenames:

            save_image = \
                os.path.join(self.conf.inference_save_dir, f'{Path(filename).stem}_clouds.tif')
            #print(save_image)

            if not os.path.isfile(save_image):

                print(f'Starting to predict {filename}')

                # --------------------------------------------------------------------------------
                # Extracting and resizing test and validation data
                # --------------------------------------------------------------------------------
                image = rxr.open_rasterio(filename, chunks=CHUNKS)
                image = image.transpose("y", "x", "band")
                #print(image.shape)

                image = self._modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                #print(image.shape)
                
                # prediction = self._sliding_window(image, model)
                prediction = self._sliding_window(image, model)
                #print(np.unique(prediction))

                #prediction = self._denoise(np.uint8(prediction))
                #prediction = self._binary_fill(prediction)
                #prediction = self._grow(np.uint8(prediction))
                #print(np.unique(prediction))
                
                image = image.drop(dim="band", labels=image.coords["band"].values[1:], drop=True)

                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name='mask',
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs)
                #print(prediction)

                prediction.attrs['long_name'] = ('mask')
                prediction = prediction.transpose("band", "y", "x")

                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(nodata, encoded=True, inplace=True)
                prediction.rio.to_raster(save_image, BIGTIFF="IF_SAFER", compress='LZW')

                del prediction

            # This is the case where the prediction was already saved
            else:
                print(f'{save_image} already predicted.')
        return

    # ------------------------------------------------------------------
    # Main System Private Methods
    # ------------------------------------------------------------------
    def _seed_everything(self, seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        cp.random.seed(seed)

    def _set_gpu_strategy(self, gpu_devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
        devices = tf.config.list_physical_devices('GPU')
        assert len(devices) != 0, "No GPU devices found."
        return tf.distribute.MirroredStrategy()

    def _set_mixed_precision(self):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    def _set_xla(self):
        tf.config.optimizer.set_jit(True)
    
    def _create_work_dirs(self, dirs: list):
        for new_dir in dirs:
            os.makedirs(new_dir, exist_ok=True)

    def _iou(self, y_true, y_pred, smooth=1e-15):
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + smooth) / (union + smooth)
            x = x.astype(np.float32)
            return x
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)

    #Keras
    def _iou_loss(self, y_true, y_pred, smooth=1e-15):
        return 1 - self._iou(y_true, y_pred)

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            print(f'Elapsed time is {time() - start} ms')
            return result
        return wrapper

    def _grow(self, merged_mask, eps=120):
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
        return cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, struct)

    def _denoise(self, merged_mask, eps=30):
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
        return cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, struct)

    def _binary_fill(self, merged_mask):
        return binary_fill_holes(merged_mask)

    def to_tif(raster, filename: str, compress: str = 'LZW', crs: str = None):
        """
        Save TIFF or TIF files, normally used from raster files.
        """
        assert (Path(filename).suffix)[:4] == '.tif', \
            f'to_tif suffix should be one of [.tif, .tiff]'
        #if xp.__name__ == 'cupy':
        #    raster.data = _xarray_to_numpy_(raster.data)
        raster.rio.write_nodata(raster.rio.nodata, encoded=True)
        if crs is not None:
            raster.rio.write_crs(crs, inplace=True)
        raster.rio.to_raster(filename, BIGTIFF="IF_SAFER", compress=compress)

    # ------------------------------------------------------------------
    # Main Preprocessing Private Methods
    # ------------------------------------------------------------------
    def _read_dataset_csv(self, filename: str):
        assert os.path.exists(filename), f'File {filename} not found.'
        data_df = pd.read_csv(filename)
        assert not data_df.isnull().values.any(), f'NaN found: {filename}'
        return data_df

    def _modify_bands(
        self, xraster: xr.core.dataarray.DataArray, input_bands: List[str],
        output_bands: List[str], drop_bands: List[str] = []):
        """
        Drop multiple bands to existing rasterio object
        """
        # Do not modify if image has the same number of output bands
        if xraster.shape[0] == len(output_bands):
            return xraster

        # Drop any bands from input that should not be on output
        for ind_id in list(set(input_bands) - set(output_bands)):
            drop_bands.append(input_bands.index(ind_id)+1)
        return xraster.drop(dim="band", labels=drop_bands, drop=True)

    def _gen_random_tiles(
            self, image: cp.ndarray, label: cp.ndarray,
            tile_size: int = 128, max_patches: int = None,
            include: bool = False, augment: bool = True,
            output_filename: str = 'image'
        ):

        generated_tiles = 0  # counter for generated tiles
        while generated_tiles < max_patches:

            # Generate random integers from image
            x = random.randint(0, image.shape[0] - tile_size)
            y = random.randint(0, image.shape[1] - tile_size)

            ################ fix later #################################

            # Bool values for conditional statement
            #if image[x: (x + tile_size), y: (y + tile_size), :].min() < 0:
            #    continue

            #if label[x: (x + tile_size), y: (y + tile_size)].min() < 0:
            #    continue

            if label[x: (x + tile_size), y: (y + tile_size)].max() > self.conf.n_classes or \
                label[x: (x + tile_size), y: (y + tile_size)].min() < 0:
                continue

            if include and cp.unique(label[x: (x + tile_size), y: (y + tile_size)]).shape[0] < 2:
                continue

            if len(np.unique(label[x: (x + tile_size), y: (y + tile_size)])) > 2:
                print(np.unique(label[x: (x + tile_size), y: (y + tile_size)]))

            #if len(np.unique(image[x: (x + tile_size), y: (y + tile_size), :])) > 2:
            #    print(np.unique(image[x: (x + tile_size), y: (y + tile_size), :]))

            # Add to the tiles counter
            generated_tiles += 1

            # Generate img and mask patches
            image_tile = image[x:(x + tile_size), y:(y + tile_size)]
            label_tile = label[x:(x + tile_size), y:(y + tile_size)]

            # Apply some random transformations
            if augment:

                if cp.random.random_sample() > 0.5:
                    image_tile = cp.fliplr(image_tile)
                    label_tile = cp.fliplr(label_tile)
                if cp.random.random_sample() > 0.5:
                    image_tile = cp.flipud(image_tile)
                    label_tile = cp.flipud(label_tile)
                if cp.random.random_sample() > 0.5:
                    image_tile = cp.rot90(image_tile, 1)
                    label_tile = cp.rot90(label_tile, 1)
                if cp.random.random_sample() > 0.5:
                    image_tile = cp.rot90(image_tile, 2)
                    label_tile = cp.rot90(label_tile, 2)
                if cp.random.random_sample() > 0.5:
                    image_tile = cp.rot90(image_tile, 3)
                    label_tile = cp.rot90(label_tile, 3)

            filename = f'{Path(output_filename).stem}_{generated_tiles}.npy'
            cp.save(os.path.join(self._images_dir, filename), image_tile)
            cp.save(os.path.join(self._labels_dir, filename), label_tile)
        return

    # ------------------------------------------------------------------
    # Main Training Private Methods
    # ------------------------------------------------------------------
    def _get_dataset_filenames(self, data_dir: str, ext: str = '*.npy'):
        data_filenames = glob(os.path.join(data_dir, ext))
        assert len(data_filenames) > 0, f'No files under {data_dir}.'
        return data_filenames

    def _read_data(self, x, y):
        x = (np.load(x) / 10000.0).astype(np.float32)
        #x = np.load(x)
        #for i in range(x.shape[-1]):  # for each channel in images
        #    x[:, :, i] = (x[:, :, i] - np.mean(x[:, :, i])) / (np.std(x[:, :, i]) + 1e-8)
        #y = np.expand_dims(np.load(y), axis=-1).astype(np.float32)
        y = np.load(y)
        y = tf.keras.utils.to_categorical(
            y, num_classes=self.conf.n_classes, dtype='float32'
        )
        return x, y

    def _dataset_preprocessing(self, x, y):
        def _loader(x, y):
            x, y = self._read_data(x.decode(), y.decode())
            return x, y
        
        x, y = tf.numpy_function(_loader, [x, y], [tf.float32, tf.float32])
        x.set_shape(
            [self.conf.tile_size, self.conf.tile_size, len(self.conf.output_bands)])
        y.set_shape(
            [self.conf.tile_size, self.conf.tile_size, self.conf.n_classes])
        return x, y

    def _tf_train_dataset(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(2048)
        dataset = dataset.map(
            self._dataset_preprocessing, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.conf.batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        dataset = dataset.repeat()
        return dataset

    def _tf_val_dataset(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(2048)
        dataset = dataset.map(
            self._dataset_preprocessing, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.conf.batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        dataset = dataset.repeat()
        return dataset

    @tf.function
    def _normalize(self, image: tf.Tensor, label: tf.Tensor) -> tuple:
        image = tf.cast(image, tf.float32) / 10000.0
        return image, label

    @tf.function
    def _augment(self, image, label) -> tuple:
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        return image, label

    @tf.function
    def _load_image_train(self, datapoint: dict) -> tuple:
        image, label = datapoint['image'], datapoint['label']
        if self.conf.augment:
            image, label = self._augment(image, label)
        image, label = self._normalize(image, label)
        return image, label

    @tf.function
    def _load_image_val(self, datapoint: dict) -> tuple:
        image, label = datapoint['image'], datapoint['label']
        return self._normalize(image, label)

    def _sliding_window(self, xraster, model):

        # open rasters and get both data and coordinates
        rast_shape = xraster[:, :, 0].shape  # shape of the wider scene

        # in memory sliding window predictions
        wsy, wsx = self.conf.window_size, self.conf.window_size

        # if the window size is bigger than the image, predict full image
        if wsy > rast_shape[0]:
            wsy = rast_shape[0]
        if wsx > rast_shape[1]:
            wsx = rast_shape[1]

        print(rast_shape, wsy, wsx)
        prediction = np.zeros(rast_shape)  # crop out the window
        print(f'wsize: {wsy}x{wsx}. Prediction shape: {prediction.shape}')

        for sy in tqdm(range(0, rast_shape[0], wsy)):  # iterate over x-axis
            for sx in range(0, rast_shape[1], wsx):  # iterate over y-axis
                y0, y1, x0, x1 = sy, sy + wsy, sx, sx + wsx  # assign window
                if y1 > rast_shape[0]:  # if selected x exceeds boundary
                    y1 = rast_shape[0]  # assign boundary to x-window
                if x1 > rast_shape[1]:  # if selected y exceeds boundary
                    x1 = rast_shape[1]  # assign boundary to y-window
                if y1 - y0 < self.conf.tile_size:  # if x is smaller than tsize
                    y0 = y1 - self.conf.tile_size  # assign boundary to -tsize
                if x1 - x0 < self.conf.tile_size:  # if selected y is small than tsize
                    x0 = x1 - self.conf.tile_size  # assign boundary to -tsize

                window = xraster[y0:y1, x0:x1, :].values  # get window

                #print("First value of window: ", window[0,0,0])

                if np.all(window == window[0,0,0]):
                    print("skipping, everything was nodata")
                    prediction[y0:y1, x0:x1] = window[:, :, 0]
                
                else:
                    window = np.clip(window, 0, 10000)
                    window = from_array(
                        window / 10000.0, (self.conf.tile_size,self.conf.tile_size),
                        overlap_factor=self.conf.inference_overlap, fill_mode='reflect')
                    #window = from_array(
                    #    window, (self.conf.tile_size,self.conf.tile_size),
                    #    overlap_factor=self.conf.inference_overlap, fill_mode='reflect')

                    #print(window.shape, "After from array")

                    window = window.apply(
                        model.predict, progress_bar=True, batch_size=self.conf.batch_size)
                    window = window.get_fusion()

                    #print("After predict: ", window.shape)

                    if self.conf.n_classes > 1:
                        window = np.squeeze(np.argmax(window, axis=-1)).astype(np.int16)
                    else:
                        window = np.squeeze(
                            np.where(window > self.conf.inference_treshold, 1, 0).astype(np.int16))
                    prediction[y0:y1, x0:x1] = window
        return prediction

    def _sliding_window_v2(self, xraster, model):

        # open rasters and get both data and coordinates
        rast_shape = xraster[:, :, 0].shape  # shape of the wider scene
        prediction = np.zeros(rast_shape)  # crop out the window

        window = xraster #.values  # get window
        window = np.clip(window, 0, 10000)
        
        window = from_array(
            window / 10000.0, (self.conf.tile_size,self.conf.tile_size),
            overlap_factor=self.conf.inference_overlap, fill_mode='reflect')

        window = window.apply(
            model.predict, progress_bar=True, batch_size=self.conf.batch_size)
        
        window = window.get_fusion()

        #print("After predict: ", window.shape)

        if self.conf.n_classes > 1:
            window = np.squeeze(np.argmax(window, axis=-1)).astype(np.int16)
        else:
            window = np.squeeze(
                np.where(window > self.conf.inference_treshold, 1, 0).astype(np.int16))
        
        prediction = window
        return prediction

# -----------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------
if __name__ == "__main__":

    schema = OmegaConf.structured(ConfigTF)
    conf = OmegaConf.load("../config/config_clouds/vietnam_clouds.yaml")
    try:
        OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")
