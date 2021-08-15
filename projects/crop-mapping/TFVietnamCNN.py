# -*- coding: utf-8 -*-

import os
import glob
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tifffile import imsave, imread

import segmentation_models as sm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from unet import unet_batchnorm, cloud_net
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow_addons as tfa

from ToolBelt import ConfigYAML, ToolBelt

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# Some global elements for the general pipeline
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
CHUNKS = {'band': 1, 'x': 2048, 'y': 2048}
sm.set_framework('tf.keras')


# -----------------------------------------------------------------------------
# class ConfigYAML
# -----------------------------------------------------------------------------
class TFVietnamCNN(ConfigYAML, ToolBelt):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, configFile):

        ConfigYAML.__init__(self, configFile)  # explicit calls without super
        self.strategy = self._setup()  # setup GPU configurations
        self._createOutputDirs()  # create output directories for local storage

    # --------------------------------------------------------------------------
    # IO Methods
    # --------------------------------------------------------------------------
    def _getDataFrames(self):
        """
        Extract data CSV files to extract tiles from (data and golden tiles)
        """
        # get data CSV file
        assert os.path.exists(self.data_csv), f'File {self.data_csv} not found'
        self.dataDF = pd.read_csv(self.data_csv)

    @ToolBelt.timeit
    def _getDataFromDir(self, dataDir: str, ext: str = 'tif'):
        """
        Read files from given directory.
        """
        data_filenames = glob.glob(f'{dataDir}/*.{ext}')
        imagesList = [imread(imageName) for imageName in data_filenames]
        return np.array(imagesList)

    def _dataAugment(self, image, label):
        """
        Augment data for semantic segmentation.
        Args:
            image (numpy.array): image numpy array
            label (numpy.array): image numpy array
        Return:
            augmented image and label
        ----------
        Example
        ----------
            data_augment(image, label)
        """
        if np.random.random_sample() > 0.5:  # flip left and right
            image = tf.image.random_flip_left_right(image)
            label = tf.image.random_flip_left_right(label)
        if np.random.random_sample() > 0.5:  # reverse second dimension
            image = tf.image.random_flip_up_down(image)
            label = tf.image.random_flip_up_down(label)
        if np.random.random_sample() > 0.5:  # rotate 90 degrees
            image = tf.image.rot90(image, k=1)
            label = tf.image.rot90(label, k=1)
        if np.random.random_sample() > 0.5:  # rotate 180 degrees
            image = tf.image.rot90(image, k=2)
            label = tf.image.rot90(label, k=2)
        if np.random.random_sample() > 0.5:  # rotate 270 degrees
            image = tf.image.rot90(image, k=3)
            label = tf.image.rot90(label, k=3)

        # standardize 0.75, 0.25
        #if np.random.random_sample() > 0.75:
        #    logging.info('Local std using image std')
        #else:
        #    logging.info('Local std using batch std')

        return image, label

    def _dataAugmentTest(self, image, label):
        """
        Augment data for semantic segmentation.
        Args:
            image (numpy.array): image numpy array
            label (numpy.array): image numpy array
        Return:
            augmented image and label
        ----------
        Example
        ----------
            data_augment(image, label)
        """
        return image, label

    # --------------------------------------------------------------------------
    # Preprocessing Methods
    # --------------------------------------------------------------------------
    @ToolBelt.timeit
    def preprocess(self):
        """
        Preprocessing function.
        """
        logging.info('Starting Preprocess Step...')
        self._getDataFrames()  # 2.a.1. Reading data and golden tiles CSV file
        return list(map(self._preprocessRaster, self.dataDF.index))

    @ToolBelt.timeit
    def _modifyBands(self, img, drop_bands=[]):
        """
        Drop multiple bands to existing rasterio object
        """
        # Do not modify if image has the same number of output bands
        if img.shape[0] == len(self.output_bands):
            return img

        # Drop any bands from input that should not be on output
        for ind_id in list(set(self.input_bands) - set(self.output_bands)):
            drop_bands.append(self.input_bands.index(ind_id)+1)
        img = img.drop(dim="band", labels=drop_bands, drop=True)
        return img

    @ToolBelt.timeit
    def _preprocessRaster(self, index: int):
        """
        Generate tiles from each individual raster.
        """
        # 2.a.2. For each file, generate dataset
        logging.info(f'File #{index+1}: ' + self.dataDF['data'][index])

        # Get filename for output purposes
        fileName = self.dataDF['data'][index].split('/')[-1]

        # Read imagery from disk
        img = xr.open_rasterio(
            self.dataDF['data'][index], chunks=CHUNKS
        ).load()
        mask = xr.open_rasterio(
            self.dataDF['label'][index], chunks=CHUNKS
        ).values
        logging.info(
            f'File #{index+1}: {fileName}, img:{img.shape}, label:{mask.shape}'
        )

        # removing bands if necessary
        img = self._modifyBands(img)

        # --------------------------------------------------------------------------
        # Unique for this project end
        # --------------------------------------------------------------------------

        mask[mask == 15] = 5  # merging classes if necessary

        # --------------------------------------------------------------------------
        # Unique for this project start
        # --------------------------------------------------------------------------

        # Moving the first axis to the end for y,x,b format
        img = np.moveaxis(img.values, 0, -1).astype(np.int16)
        img = np.clip(img, 0, 10000)  # necessary due to EVHR mishandling
        mask = np.squeeze(mask)  # squeeze mask to remove single dimension

        # Get region of interest for training
        img = img[
            self.dataDF['ymin'][index]:self.dataDF['ymax'][index],
            self.dataDF['xmin'][index]:self.dataDF['xmax'][index]
        ]
        mask = mask[
            self.dataDF['ymin'][index]:self.dataDF['ymax'][index],
            self.dataDF['xmin'][index]:self.dataDF['xmax'][index]
        ]
        logging.info(f'Post img: {img.shape}, label: {mask.shape}')

        # generate tiles arrays
        imgPatches, maskPatches = self._extractTiles(
            img, mask, tile_size=((self.tile_size, ) * 2),
            random_state=getattr(self, 'seed', 34),
            n_patches=self.dataDF['ntiles'][index]
        )
        logging.info(f'After tiling: {imgPatches.shape}, {maskPatches.shape}')

        # save to disk
        for id in range(imgPatches.shape[0]):

            imsave(
                os.path.join(self.imagesDir, f'{fileName[:-4]}_{id}.tif'),
                imgPatches[id, :, :, :], planarconfig='contig'
            )
            imsave(
                os.path.join(self.labelsDir, f'{fileName[:-4]}_{id}.tif'),
                maskPatches[id, :, :], planarconfig='contig'
            )

        return index

    # --------------------------------------------------------------------------
    # Model Methods
    # --------------------------------------------------------------------------
    def _getCallbacks(self):
        """
        Return data callbacks.
        """
        modelPath = os.path.join(
            self.modelDir, f'{self.experiment_name}'+'-{epoch:02d}.h5'
        )
        checkpoint = ModelCheckpoint(
            modelPath, monitor='val_accuracy', verbose=1,
            save_best_only=True, mode='max'
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        csvPath = os.path.join(self.modelDir, f'{self.experiment_name}.csv')
        log_csv = CSVLogger(csvPath, separator=',', append=False)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(self.modelDir, 'tensorboard')
        )
        return [checkpoint, early_stop, log_csv, tensorboard]

    # Define a function to perform additional preprocessing after datagen.
    # For example, scale images, convert masks to categorical, etc.
    # def _preprocess_data(self, img, mask):
    #    # Scale images
    #    img = self.scaler.fit_transform(
    #        img.reshape(-1, img.shape[-1])
    #    ).reshape(img.shape)
    #    img = self.preprocess_input(img)  # Preprocess based on the backbone
    #    # Convert mask to one-hot
    #    mask = tf.keras.utils.to_categorical(mask, self.n_classes)
    #    return (img, mask)

    # def _trainGenerator(self, train_img_path, train_mask_path):
    #    img_data_gen_args = dict(
    #        horizontal_flip=True,
    #        vertical_flip=True,
    #        fill_mode='reflect'
    #    )
    #
    #    # image_datagen = ImageDataGenerator(**img_data_gen_args)
    #    # mask_datagen = ImageDataGenerator(**img_data_gen_args)
    #    # image_generator = image_datagen.flow_from_directory(
    #    #    train_img_path,
    #    #    class_mode=None,
    #    #    batch_size=self.batch_size,
    #    #    seed=self.seed
    #    # )
    #
    #    # mask_generator = mask_datagen.flow_from_directory(
    #    #    train_mask_path,
    #    #    class_mode=None,
    #    #    color_mode='grayscale',
    #    #    batch_size=self.batch_size,
    #    #    seed=self.seed
    #    # )
    #
    #    # train_generator = zip(image_generator, mask_generator)
    #    # for (img, mask) in train_generator:
    #    #    img, mask = self._preprocess_data(img, mask, self.n_classes)
    #    #    yield (img, mask)

    def _getSTDInfo(self, images, means=list(), stds=list()):
        f = open(f"{self.experiment_name}_norm_data.csv", "w+")
        f.write("i,channel_mean,channel_std\n")
        for i in range(images.shape[-1]):
            mean = np.mean(images[:, :, :, i])
            std = np.std(images[:, :, :, i])
            f.write('{},{},{}\n'.format(i, mean, std))
            means.append(mean)
            stds.append(std)
        f.close()  # close file
        return means, stds

    # --------------------------------------------------------------------------
    # Training Methods
    # --------------------------------------------------------------------------
    @ToolBelt.timeit
    def train(self):
        """
        Train function.
        """
        # lets get the data into memory (since it fits)
        images = self._getDataFromDir(self.imagesDir)
        labels = self._getDataFromDir(self.labelsDir)

        # use this to preprocess input for transfer learning
        # self.BACKBONE = 'resnet34'
        # self.preprocess_input = sm.get_preprocessing(self.BACKBONE)

        # Scale images
        # self.scaler = MinMaxScaler()  # StandardScaler()
        # images = self.scaler.fit_transform(
        #    images.reshape(-1, images.shape[-1])
        # ).reshape(images.shape)
        # images = self.preprocess_input(images)

        # normalize data, prepare for training
        # images = tf.keras.utils.normalize(images, axis=-1, order=2)
        # images = self._contrastStretch(images)
        self.meanList, self.stdList = self._getSTDInfo(images)
        logging.info(f"Means {self.meanList}")
        logging.info(f"Stds {self.stdList}")

        images = self._standardize(images)
        logging.info(f'Images {images.shape}, {images.mean()}, {images.max()}')

        # train_img_path = "data/data_for_keras_aug/train_images/"
        # train_mask_path = "data/data_for_keras_aug/train_masks/"
        # train_img_gen = self._trainGenerator(train_img_path, train_mask_path)

        # val_img_path = "data/data_for_keras_aug/val_images/"
        # val_mask_path = "data/data_for_keras_aug/val_masks/"
        # val_img_gen = self._trainGenerator(val_img_path, val_mask_path)

        self.weights = compute_class_weight(
            'balanced',
            np.unique(np.ravel(labels, order='C')),
            np.ravel(labels, order='C')
        )
        logging.info(f'Calculated weights: {self.weights}')

        # set labels to categorical - using sparse categorical for testing
        labels = tf.keras.utils.to_categorical(
            labels, num_classes=self.n_classes, dtype='float32'
        )
        # labels = np.expand_dims(labels, axis=-1)
        logging.info(f'Training dataset: {images.shape}, {labels.shape}')

        self.seed = getattr(self, 'seed', 34)
        self.batch_size = getattr(self, 'batch_size', 16) * \
            getattr(self, 'strategy.num_replicas_in_sync', 1)
        self.callbacks = self._getCallbacks()

        # divide dataset into train and val
        images, labels = shuffle(images, labels)
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=getattr(self, 'test_size', 0.25),
            random_state=self.seed
        )
        self.trainSize, self.valSize = X_train.shape[0], X_val.shape[0]

        # merge into tensorflow dataset
        dataset = {
            "train": tf.data.Dataset.from_tensor_slices((X_train, y_train)),
            "val": tf.data.Dataset.from_tensor_slices((X_val, y_val))
        }

        # disable AutoShard, data lives in memory, use in memory options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = \
            tf.data.experimental.AutoShardPolicy.OFF

        # prepare pre-fetching for model training
        dataset['train'] = dataset['train'].map(
            self._dataAugment, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset['train'] = dataset['train'].shuffle(
            buffer_size=BUFFER_SIZE, seed=self.seed
        )
        dataset['train'] = dataset['train'].repeat()
        dataset['train'] = dataset['train'].batch(self.batch_size)
        dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
        dataset['train'] = dataset['train'].with_options(options)

        dataset['val'] = dataset['val'].map(self._dataAugmentTest)
        dataset['val'] = dataset['val'].repeat()
        dataset['val'] = dataset['val'].batch(self.batch_size)
        dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
        dataset['val'] = dataset['val'].with_options(options)

        # For testing purposes only
        # for image, mask in dataset['train'].take(40):
        #    sample_image, sample_mask = image, mask
        #    self.display_sample([sample_image[0], sample_mask[0]])

        # initialize the model
        with self.strategy.scope():

            # model = sm.Unet(
            # 'resnet101',
            # input_shape=
            # (self.tile_size, self.tile_size, len(self.output_bands)),
            # encoder_weights=None,
            # classes=1,
            # activation='sigmoid'
            # )

            # works for now
            model = unet_batchnorm(
                nclass=self.n_classes,
                input_size=(
                    self.tile_size, self.tile_size, len(self.output_bands)
                ),
                maps=[64, 128, 256, 512, 1024]
            )

            # model = cloud_net(
            # nclass=self.n_classes,
            # input_size=
            # (self.tile_size, self.tile_size, len(self.output_bands))
            # )

            # model = sm.Unet(
            # self.BACKBONE, encoder_weights=None,
            # input_shape=(
            # self.tile_size, self.tile_size, len(self.output_bands)),
            # classes=self.n_classes, activation='softmax')
            #model = sm.Unet(
            #    backbone_name='resnet34', encoder_weights=None,
            #    input_shape=(None, None, len(self.output_bands)),
            #    classes=self.n_classes, activation='softmax'
            #)

            # enabling mixed precision to avoid underflow
            # optimizer = tf.keras.optimizers.Adam(lr=0.0001)
            optimizer = tfa.optimizers.RectifiedAdam(
                lr=1e-3,
                total_steps=10000,
                warmup_proportion=0.1,
                min_lr=1e-5,
            )

            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

            # loss
            # self.loss = ToolBelt.dice_loss
            # self.loss = sm.losses.DiceLoss(class_weights=self.weights) + \
            #    (1 * sm.losses.CategoricalFocalLoss())
            # self.loss = sm.losses.DiceLoss()  # class_weights=self.weights)
            # self.loss = sm.losses.categorical_focal_jaccard_loss
            self.loss = 'categorical_crossentropy'
            # self.loss = 'sparse_categorical_crossentropy'
            # self.loss = jaccard_distance_loss
            # self.loss = sm.losses.categorical_focal_loss

            model.compile(
                optimizer,
                loss=self.loss,
                # ToolBelt.bcedice_loss,
                # sm.losses.DiceLoss(),
                # #'binary_crossentropy', #sm.losses.DiceLoss(),
                metrics=[sm.metrics.iou_score, 'accuracy'],
            )
        model.summary()

        # fit model
        model_history = model.fit(
            dataset['train'],
            validation_data=dataset['val'],
            shuffle=getattr(self, 'shuffle_train', True),
            batch_size=self.batch_size,
            epochs=getattr(self, 'max_epochs', 50),
            initial_epoch=getattr(self, 'initial_epoch', 0),
            steps_per_epoch=self.trainSize // self.batch_size,
            validation_steps=self.valSize // self.batch_size,
            callbacks=self.callbacks,
        )

        # IOU
        accuracy = model.evaluate(X_val, y_val, batch_size=self.batch_size)
        # y_pred_thresholded = y_pred > 0.5
        # intersection = np.logical_and(y_val, y_pred_thresholded)
        # union = np.logical_or(y_val, y_pred_thresholded)
        # iou_score = np.sum(intersection) / np.sum(union)
        # print("IoU Score: ", iou_score)
        logging.info(f'Model Evaluation: {accuracy}')
        return model_history

    # --------------------------------------------------------------------------
    # Prediction Methods
    # --------------------------------------------------------------------------
    @ToolBelt.timeit
    def _predictRaster(self, fileName):

        logging.info(f'File: {fileName}')

        # Get filename for output purposes
        outRasterName = os.path.join(
            self.inferenceDir, fileName[:-4].split('/')[-1] + '_pred.tif'
        )

        # --------------------------------------------------------------------------------
        # if prediction is not on directory, start predicting
        # (allows for restarting script if it was interrupted at some point)
        # --------------------------------------------------------------------------------
        if not os.path.isfile(outRasterName):

            img = xr.open_rasterio(fileName, chunks=CHUNKS).load()

            # Removing bands if necessary, adding indices if necessary
            img = self._modifyBands(img)

            # Moving the first axis to the end for y, x, b format
            img = np.moveaxis(img.values, 0, -1)
            img = np.clip(img, 0, 10000).astype(np.int16)
            logging.info(f'Tensor shape: {img.shape}')

            predictions_smooth = predict_img_with_smooth_windowing(
                img,
                window_size=self.tile_size,
                subdivisions=2,
                nb_classes=self.n_classes,
                pred_func=(
                    lambda img_batch_subdiv: self.model.predict(
                        self._standardize(
                            self._contrastStretch(
                                img_batch_subdiv
                            )
                        )
                    )
                )
            )

            logging.info(f"After prediction shape: {predictions_smooth.shape}")
            predictions_smooth = np.argmax(predictions_smooth, axis=-1)
            logging.info(f"After prediction shape: {predictions_smooth.shape}")

            # logging.info(f"After prediction min: {predictions_smooth.min()}")
            # logging.info(f"After prediction max: {predictions_smooth.max()}")
            # predictions_smooth = self._pred_mask(
            #    predictions_smooth, threshold=0.75
            # )
            # predictions_smooth = predictions_smooth.astype(np.uint8)
            # predictions_smooth = np.squeeze(predictions_smooth)
            # predictions_smooth = self._grow(predictions_smooth)
            # predictions_smooth = self._denoise(predictions_smooth)
            # predictions_smooth = self._binary_fill(predictions_smooth)

            # output image to disk
            ToolBelt.toRasterMask(
                raster_f=fileName, segments=predictions_smooth,
                out_tif=outRasterName
            )
            del predictions_smooth

        # This is the case where the prediction was already saved
        else:
            logging.info(f'{outRasterName} already predicted.')

    @ToolBelt.timeit
    def predict(self):
        """
        Predict function.
        """
        # get model and setup directories
        try:
            self.model_filename
        except AttributeError:
            modelsList = glob.glob(
                os.path.join(self.modelDir, f'{self.experiment_name}*.h5')
            )
            self.model_filename = max(modelsList, key=os.path.getctime)
        logging.info(f'Loading {self.model_filename}')

        with self.strategy.scope():

            self.model = tf.keras.models.load_model(
                self.model_filename, custom_objects={
                    'dice_loss': ToolBelt.dice_loss,
                    'iou_score': sm.metrics.iou_score
                }
            )

            # get data files to predict
            # self.dataPredict = glob.glob(self.data_predict)
            self.dataPredict = [
                '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN/7-band/Keelin05_20160318_data-7band.tif',
                '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN/7-band/Keelin13_20160318_data-7band.tif',
                '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN/7-band/Keelin23_20110201_data-7band.tif',
                '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN/7-band/Keelin09_20190327__subset_data-7band.tif'
            ]
            logging.info(f'{len(self.dataPredict)} files to predict.')
            # [
            #    '/att/nobackup/jacaraba/DOWNLOAD/Kassassa8bands/WV02_20101020_M1BS_1030010007BBFA00-toa_5000-5000.tif',
            #    '/att/nobackup/jacaraba/DOWNLOAD/Kassassa8bands/WV02_20101020_M1BS_1030010007BBFA00-toa_0-0.tif',
            #    '/att/nobackup/jacaraba/DOWNLOAD/Kassassa8bands/WV03_20150717_M1BS_104001000ED1CC00-toa_5000-0.tif',
            # ]
            # glob.glob('/att/nobackup/mwooten3/Senegal_LCLUC/VHR/priority-tiles/kassassa_M1BS-8band/*.tif')

            # iterate over each file and predict
            for r in self.dataPredict:
                self._predictRaster(r)

    # --------------------------------------------------------------------------
    # Postprocessing Methods
    # --------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    a3D = np.array([[[1, 2, 3], [3, 4, 3]],
                    [[5, 6, 5], [7, 8, 7]]])
    print(len(a3D.shape))
