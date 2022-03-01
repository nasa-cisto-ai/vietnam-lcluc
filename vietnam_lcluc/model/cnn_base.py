import os
import logging
import omegaconf
from glob import glob
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau

from . import utils
from .network.unet import unet
from .network.loss import get_loss, TverskyLoss

CHUNKS = {'band': 'auto', 'x': 'auto', 'y': 'auto'}
AUTOTUNE = tf.data.experimental.AUTOTUNE


class CNNPipeline(object):

    def __init__(self, conf: omegaconf.dictconfig.DictConfig):

        # set configuration object
        self.conf = conf

        # set data variables for directory management
        self._dataset_dir = os.path.join(self.conf.data_dir, 'dataset')
        self._images_dir = os.path.join(self.conf.data_dir, 'images')
        self._labels_dir = os.path.join(self.conf.data_dir, 'labels')
        self._model_dir = os.path.join(self.conf.data_dir, 'model')

        # create working directories
        self._create_work_dirs(
            [self._images_dir, self._labels_dir, self._model_dir])

        # set system specifications and hardware strategies
        utils.seed_everything(self.conf.seed)
        self._gpu_strategy = utils.set_gpu_strategy(self.conf.gpu_devices)

        # enable mixed precision
        if self.conf.mixed_precision:
            self._set_mixed_precision()

        # enable linear algebra acceleration
        if self.conf.xla:
            self._set_xla()

    def _create_work_dirs(self, dirs: list) -> None:
        """
        Generate working directories.
        """
        for new_dir in dirs:
            os.makedirs(new_dir, exist_ok=True)
        return

    def _dataset_train_preprocessing(self, x, y):
        def _loader(x, y):
            x, y = self._read_data(x.decode(), y.decode())
            return x, y

        x, y = tf.numpy_function(_loader, [x, y], [tf.float32, tf.float32])
        x.set_shape([
            self.conf.tile_size, self.conf.tile_size,
            len(self.conf.output_bands)])
        y.set_shape(
            [self.conf.tile_size, self.conf.tile_size, self.conf.n_classes])
        return x, y

    def _dataset_val_preprocessing(self, x, y):
        def _loader(x, y):
            x, y = self._read_data(x.decode(), y.decode())
            return x, y

        x, y = tf.numpy_function(_loader, [x, y], [tf.float32, tf.float32])
        x.set_shape([
            self.conf.tile_size, self.conf.tile_size,
            len(self.conf.output_bands)])
        y.set_shape(
            [self.conf.tile_size, self.conf.tile_size, self.conf.n_classes])
        return x, y

    # -------------------------------------------------------------------------
    # _get_dataset_filenames()
    #
    # Get dataset filenames for training
    # -------------------------------------------------------------------------
    def _get_dataset_filenames(
            self, data_dir: str, ext: str = '*.npy') -> list:
        """
        Get dataset filenames for training.
        """
        data_filenames = glob(os.path.join(data_dir, ext))
        assert len(data_filenames) > 0, f'No files under {data_dir}.'
        return data_filenames

    def _read_data(self, x, y):
        """
        Read data from disk and load for training.
        """
        x = np.load(x)
        # x = utils.standardize(x)
        for i in range(x.shape[-1]):  # for each channel in the image
            x[:, :, i] = (x[:, :, i] - self.conf.mean[i]) / \
                (self.conf.std[i] + 1e-8)
        y = np.load(y)
        return x.astype(np.float32), y.astype(np.float32)

    def _read_dataset_csv(self, filename: str) -> pd.core.frame.DataFrame:
        """
        Read dataset CSV from disk and load for preprocessing.
        """
        assert os.path.exists(filename), f'File {filename} not found.'
        data_df = pd.read_csv(filename)
        assert not data_df.isnull().values.any(), f'NaN found: {filename}'
        return data_df

    # -------------------------------------------------------------------------
    # _tf_train_dataset()
    #
    # TF training dataset pipeline
    # -------------------------------------------------------------------------
    def _tf_train_dataset(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(2048)
        dataset = dataset.map(
            self._dataset_train_preprocessing, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.conf.batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        # dataset = dataset.repeat()
        return dataset

    # -------------------------------------------------------------------------
    # _tf_val_dataset()
    #
    # TF validation dataset pipeline
    # -------------------------------------------------------------------------
    def _tf_val_dataset(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(2048)
        dataset = dataset.map(
            self._dataset_val_preprocessing, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.conf.batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        # dataset = dataset.repeat()
        return dataset

    # -------------------------------------------------------------------------
    # preprocess()
    #
    # Preprocessing stage of the pipeline
    # -------------------------------------------------------------------------
    def preprocess(self) -> None:

        logging.info('Starting preprocessing stage')

        # Initialize dataframe with data details
        data_df = self._read_dataset_csv(self.conf.dataset_csv)

        # iterate over each file and generate dataset
        for data_filename, label_filename, n_tiles in data_df.values:

            # Read imagery from disk and process both image and mask
            image = rxr.open_rasterio(data_filename, chunks=CHUNKS).load()
            label = rxr.open_rasterio(label_filename, chunks=CHUNKS).values
            logging.info(f'Image: {image.shape}, Label: {label.shape}')

            # Lower the number of bands if required
            image = utils.modify_bands(
                xraster=image, input_bands=self.conf.input_bands,
                output_bands=self.conf.output_bands)
            logging.info(f'Image: {image.shape}, Label: {label.shape}')

            # Asarray option to force array type
            image = cp.asarray(image.values)
            label = cp.asarray(label)

            # Note: We need to make sure we do not move to int16 since CHM
            # and SAR are not int16 values, do not damage the data
            # image = cp.moveaxis(image, 0, -1).astype(np.int16)

            # Move from chw to hwc, squeze mask if required
            image = cp.moveaxis(image, 0, -1)
            label = cp.squeeze(label) if len(label.shape) != 2 else label
            logging.info(f'Label classes from image: {cp.unique(label)}')

            # ----------------------------------------------------------------
            # preprocessing unique for this project
            # Processing required for this project, we need to convert 6 to 5
            label[label == 6] = 5
            # ----------------------------------------------------------------
            logging.info(f'Label classes from image: {cp.unique(label)}')

            # utils.gen_random_tiles(
            #    image=image,
            #    label=label,
            #    tile_size=self.conf.tile_size,
            #    num_classes=self.conf.n_classes,
            #    max_patches=n_tiles,
            #    include=self.conf.include_classes,
            #    augment=self.conf.augment,
            #    output_filename=data_filename,
            #    out_image_dir=self._images_dir,
            #    out_label_dir=self._labels_dir
            #)
        return

    # -------------------------------------------------------------------------
    # train()
    #
    # Training stage of the pipeline
    # -------------------------------------------------------------------------
    def train(self) -> None:

        logging.info('Starting training stage')

        data_filenames = self._get_dataset_filenames(self._images_dir)
        label_filenames = self._get_dataset_filenames(self._labels_dir)
        logging.info(
            f'Data: {len(data_filenames)}, Label: {len(label_filenames)}')

        total_size = len(data_filenames)
        val_size = round(self.conf.test_size * total_size)
        logging.info(f'Train: {total_size - val_size}, Val: {val_size}')

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

        # Calculate mean and std
        # TODO: make this step optional, output to temporary file
        #       if given, no need to be calculated
        #       make this step with a setter instead of plain self definition
        # self.conf.mean, self.conf.std = utils.get_mean_std_dataset(train_dataset)
        # self.conf.mean, self.conf.std = self.conf.mean.numpy(), self.conf.std.numpy()

        """
        # Here we repeat the dataset
        # TODO: consider doing this step elsewhere to not
        # allow them in the general pipeline
        train_dataset = train_dataset.repeat()
        val_dataset = val_dataset.repeat()

        # Initialize and compile model
        with self._gpu_strategy.scope():

            # initialize UNet model
            # TODO: add unet maps on the configuration file from the model
            #       add get model option?
            model = unet(
                nclass=self.conf.n_classes,
                input_size=(
                    self.conf.tile_size, self.conf.tile_size,
                    len(self.conf.output_bands)
                ),
                maps=[64, 128, 256, 512, 1024]
            )

            # enabling mixed precision to avoid underflow
            # optimizer = mixed_precision.LossScaleOptimizer(optimizer)

            # TODO: add get optimizer function
            #       add get metrics function
            optimizer = tf.keras.optimizers.Adam(self.conf.learning_rate)
            metrics = [
                "acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
            model.compile(
                loss=get_loss(self.conf.loss),
                optimizer=optimizer, metrics=metrics)
            model.summary()

            callbacks = [
                ModelCheckpoint(
                    filepath=os.path.join(
                        self._model_dir, '{epoch:02d}-{val_loss:.2f}.hdf5'),
                    monitor='val_acc',
                    mode='max',
                    save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
                CSVLogger(
                    os.path.join(
                        self._model_dir, f"{self.conf.experiment_name}.csv")),
                TensorBoard(
                    log_dir=os.path.join(self._model_dir, 'tensorboard_logs')),
                EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=False
                )
            ]

        # calculate training and validation steps
        # TODO: consider having its own routine in order to
        # use them in other locations.
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
        """
        return

    # -------------------------------------------------------------------------
    # predict()
    #
    # Predicting stage of the pipeline
    # -------------------------------------------------------------------------
    def predict(self) -> None:

        logging.info('Starting prediction stage')

        # TODO: get last model if no model filename was given

        # TODO: Loading the trained model
        assert os.path.isfile(self.conf.model_filename), \
            f'{self.conf.model_filename} does not exist.'

        with self._gpu_strategy.scope():
            model = tf.keras.models.load_model(
                self.conf.model_filename, custom_objects={
                    # "_iou": self._iou,
                    "TverskyLoss": TverskyLoss()
                    }
                )

        # gather filenames to predict
        data_filenames = sorted(glob(self.conf.inference_regex))
        assert len(data_filenames) > 0, \
            f'No files under {self.conf.inference_regex}.'
        logging.info(f'{len(data_filenames)} files to predict')

        # create inference output directory
        os.makedirs(self.conf.inference_save_dir, exist_ok=True)

        # iterate files, create lock file to avoid predicting the same file
        for filename in data_filenames:

            # output filename to save prediction on
            output_filename = os.path.join(
                self.conf.inference_save_dir,
                f'{Path(filename).stem}.{self.conf.experiment_type}.tif'
            )

            # lock file for multi-node, multi-processing
            lock_filename = f'{output_filename}.lock'

            # predict only if file does not exist and no lock file
            if not os.path.isfile(output_filename) and \
                    not os.path.isfile(lock_filename):

                logging.info(f'Starting to predict {filename}')

                # create lock file
                open(lock_filename, 'w').close()

                # open filename
                image = rxr.open_rasterio(filename, chunks=CHUNKS)
                image = image.transpose("y", "x", "band")
                logging.info(f'Prediction shape: {image.shape}')

                image = utils.modify_bands(
                    xraster=image, input_bands=self.conf.input_bands,
                    output_bands=self.conf.output_bands)
                logging.info(f'Prediction shape after modf: {image.shape}')

                # prediction = self._sliding_window(image, model)
                prediction = utils.sliding_window(
                    xraster=image,
                    model=model,
                    window_size=self.conf.window_size,
                    tile_size=self.conf.tile_size,
                    inference_overlap=self.conf.inference_overlap,
                    inference_treshold=self.conf.inference_treshold,
                    batch_size=self.conf.batch_size,
                    mean=self.conf.mean,
                    std=self.conf.std
                )

                # prediction = self._denoise(np.uint8(prediction))
                # prediction = self._binary_fill(prediction)
                # prediction = self._grow(np.uint8(prediction))

                image = image.drop(
                    dim="band",
                    labels=image.coords["band"].values[1:],
                    drop=True
                )

                prediction = xr.DataArray(
                    np.expand_dims(prediction, axis=-1),
                    name=self.conf.experiment_type,
                    coords=image.coords,
                    dims=image.dims,
                    attrs=image.attrs
                )

                prediction.attrs['long_name'] = (self.conf.experiment_type)
                prediction = prediction.transpose("band", "y", "x")

                nodata = prediction.rio.nodata
                prediction = prediction.where(image != nodata)
                prediction.rio.write_nodata(nodata, encoded=True, inplace=True)
                prediction.rio.to_raster(
                    output_filename, BIGTIFF="IF_SAFER", compress='LZW')

                del prediction

                # delete lock file
                os.remove(lock_filename)

            # This is the case where the prediction was already saved
            else:
                logging.info(f'{output_filename} already predicted.')

        return
