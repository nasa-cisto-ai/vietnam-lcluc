import os
import random
from tqdm import tqdm
from typing import List
from pathlib import Path

import cupy as cp
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy

from .inference import from_array

AUTOTUNE = tf.data.experimental.AUTOTUNE


def gen_random_tiles(
            image: cp.ndarray, label: cp.ndarray,
            num_classes: int, tile_size: int = 128,
            max_patches: int = None, include: bool = False,
            augment: bool = True, output_filename: str = 'image',
            out_image_dir: str = 'image',  out_label_dir: str = 'label'
        ) -> None:

    generated_tiles = 0  # counter for generated tiles
    while generated_tiles < max_patches:

        # Generate random integers from image
        x = random.randint(0, image.shape[0] - tile_size)
        y = random.randint(0, image.shape[1] - tile_size)

        # first condition, time must have valid classes
        if label[x: (x + tile_size), y: (y + tile_size)].min() < 0 or \
                label[x: (x + tile_size), y: (y + tile_size)].max() \
                >= num_classes:
            continue

        # second condition, if include, number of labels must be at least 2
        if include and cp.unique(
                label[x: (x + tile_size), y: (y + tile_size)]).shape[0] < 2:
            continue

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

        if num_classes > 2:
            label_tile = tf.keras.utils.to_categorical(
                label_tile.get(), num_classes=num_classes)
        else:
            label_tile = np.expand_dims(label_tile, axis=-1)

        filename = f'{Path(output_filename).stem}_{generated_tiles}.npy'
        cp.save(os.path.join(out_image_dir, filename), image_tile)
        cp.save(os.path.join(out_label_dir, filename), label_tile)
    return


def get_mean_std_dataset(tf_dataset):
    """
    Get general mean and std from tensorflow dataset.
    Useful when reading from disk and not from memory.
    """
    #print(type(tf_dataset))  # trying to get the type for the definition
    for data, _ in tf_dataset:
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        channels_sum += tf.reduce_mean(data, [0, 1, 2])
        channels_squared_sum += tf.reduce_mean(data**2, [0, 1, 2])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def modify_bands(
        xraster: xr.core.dataarray.DataArray, input_bands: List[str],
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


def seed_everything(seed: int = 42) -> None:
    """
    Seed libraries.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    cp.random.seed(seed)
    return


def set_gpu_strategy(gpu_devices: str = "0,1,2,3") -> MirroredStrategy:
    """
    Set training strategy.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    devices = tf.config.list_physical_devices('GPU')
    assert len(devices) != 0, "No GPU devices found."
    return MirroredStrategy()


def set_mixed_precision() -> None:
    """
    Enable mixed precision.
    """
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    return


def set_xla() -> None:
    """
    Enable linear acceleration.
    """
    tf.config.optimizer.set_jit(True)
    return


def sliding_window(
            xraster, model, window_size, tile_size,
            inference_overlap, inference_treshold, batch_size,
            mean, std
        ):

    # open rasters and get both data and coordinates
    rast_shape = xraster[:, :, 0].shape  # shape of the wider scene

    # in memory sliding window predictions
    wsy, wsx = window_size, window_size

    # if the window size is bigger than the image, predict full image
    if wsy > rast_shape[0]:
        wsy = rast_shape[0]
    if wsx > rast_shape[1]:
        wsx = rast_shape[1]

    # print(rast_shape, wsy, wsx)
    prediction = np.zeros(rast_shape)  # crop out the window
    print(f'wsize: {wsy}x{wsx}. Prediction shape: {prediction.shape}')

    for sy in tqdm(range(0, rast_shape[0], wsy)):  # iterate over x-axis
        for sx in range(0, rast_shape[1], wsx):  # iterate over y-axis
            y0, y1, x0, x1 = sy, sy + wsy, sx, sx + wsx  # assign window
            if y1 > rast_shape[0]:  # if selected x exceeds boundary
                y1 = rast_shape[0]  # assign boundary to x-window
            if x1 > rast_shape[1]:  # if selected y exceeds boundary
                x1 = rast_shape[1]  # assign boundary to y-window
            if y1 - y0 < tile_size:  # if x is smaller than tsize
                y0 = y1 - tile_size  # assign boundary to -tsize
            if x1 - x0 < tile_size:  # if selected y is small than tsize
                x0 = x1 - tile_size  # assign boundary to -tsize

            window = xraster[y0:y1, x0:x1, :].values  # get window

            if np.all(window == window[0, 0, 0]):
                prediction[y0:y1, x0:x1] = window[:, :, 0]

            else:
                window = np.clip(window, 0, 10000)

                window = from_array(
                    window, (tile_size, tile_size),
                    overlap_factor=inference_overlap, fill_mode='reflect')
                # print("After from_array", window.shape)

                window = window.apply(
                    model.predict, progress_bar=False,
                    batch_size=batch_size, mean=mean, std=std)
                # print("After apply", window.shape)

                window = window.get_fusion()
                # print("After fusion", window.shape)

                if window.shape[-1] > 1:
                    window = np.argmax(window, axis=-1)
                else:
                    window = np.squeeze(
                        np.where(
                            window > inference_treshold, 1, 0).astype(np.int16)
                        )
                prediction[y0:y1, x0:x1] = window
    return prediction
