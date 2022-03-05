import os
import random
from tqdm import tqdm
from typing import List
from pathlib import Path

import cupy as cp
import numpy as np
import xarray as xr
import tensorflow as tf
import scipy.signal
import scipy.signal.windows as w
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
        # MIGHT NEED FIX LATER WITH CROP >=
        if label[x: (x + tile_size), y: (y + tile_size)].min() < 0 or \
                label[x: (x + tile_size), y: (y + tile_size)].max() \
                > num_classes:
            continue

        if image[x: (x + tile_size), y: (y + tile_size)].min() < -100:
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

        if num_classes >= 2:
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
    if xraster['band'].shape[0] == len(output_bands):
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


def spline_window(window_size: int, power: int = 2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    tria = scipy.signal.triang(window_size)
    wind_outer = (abs(2*(tria)) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(tria - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    wind = np.expand_dims(np.expand_dims(wind, 1), 2)
    wind = wind * wind.transpose(1, 0, 2)
    return wind


def sliding_window(
            xraster, model, window_size, tile_size,
            inference_overlap, inference_treshold, batch_size,
            mean, std
        ):

    # open rasters and get both data and coordinates
    rast_shape = xraster[:, :, 0].shape  # shape of the wider scene

    # in memory sliding window predictions
    wsy, wsx = window_size, window_size
    # wsy, wsx = rast_shape[0], rast_shape[1]

    # if the window size is bigger than the image, predict full image
    if wsy > rast_shape[0]:
        wsy = rast_shape[0]
    if wsx > rast_shape[1]:
        wsx = rast_shape[1]

    # smooth window
    # this might be problematic since there might be issues on tiles smaller
    # than actual squares
    # spline = spline_window(wsy)

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
            print(window.shape)
            # window_shape = window.shape

            #window_spline = np.squeeze(
            #    spline[0:window_shape[0], 0:window_shape[1]])

            if np.all(window == window[0, 0, 0]):
                prediction[y0:y1, x0:x1] = window[:, :, 0]

            else:

                # window = np.clip(window, 0, 10000)

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
                    #print('Window shape before spline', window.shape, window_spline.shape)
                    #window = window * window_spline
                else:
                    window = np.squeeze(
                        np.where(
                            window > inference_treshold, 1, 0).astype(np.int16)
                        )
                    #window = window * window_spline
                prediction[y0:y1, x0:x1] = window
    return prediction


def window2d(window_func, window_size, **kwargs):
    window = np.matrix(window_func(M=window_size, sym=False, **kwargs))
    return window.T.dot(window)


def generate_corner_windows(window_func, window_size, **kwargs):
    step = window_size >> 1
    window = window2d(window_func, window_size, **kwargs)
    window_u = np.vstack([np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
    window_b = np.vstack([window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
    window_l = np.hstack([np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
    window_r = np.hstack([window[:, :step], np.tile(window[:, step:step+1], (1, step))])
    window_ul = np.block([
        [np.ones((step, step)), window_u[:step, step:]],
        [window_l[step:, :step], window_l[step:, step:]]])
    window_ur = np.block([
        [window_u[:step, :step], np.ones((step, step))],
        [window_r[step:, :step], window_r[step:, step:]]])
    window_bl = np.block([
        [window_l[:step, :step], window_l[:step, step:]],
        [np.ones((step, step)), window_b[step:, step:]]])
    window_br = np.block([
        [window_r[:step, :step], window_r[:step, step:]],
        [window_b[step:, :step], np.ones((step, step))]])
    return np.array([
        [window_ul, window_u, window_ur],
        [window_l,  window,   window_r],
        [window_bl, window_b, window_br],
    ])


def generate_patch_list(image_width, image_height, window_func, window_size, overlapping=False):
    patch_list = []
    if overlapping:
        step = window_size >> 1
        windows = generate_corner_windows(window_func, window_size)
        max_height = int(image_height/step - 1)*step
        max_width = int(image_width/step - 1)*step
    else:
        step = window_size
        windows = np.ones((window_size, window_size))
        max_height = int(image_height/step)*step
        max_width = int(image_width/step)*step
    for i in range(0, max_height, step):
        for j in range(0, max_width, step):
            if overlapping:
                # Close to border and corner cases
                # Default (1, 1) is regular center window
                border_x, border_y = 1, 1
                if i == 0:
                    border_x = 0
                if j == 0:
                    border_y = 0
                if i == max_height-step:
                    border_x = 2
                if j == max_width-step:
                    border_y = 2
                # Selecting the right window
                current_window = windows[border_x, border_y]
            else:
                current_window = windows
            # The patch is cropped when the patch size is not
            # a multiple of the image size.
            patch_height = window_size
            if i+patch_height > image_height:
                patch_height = image_height - i
            patch_width = window_size
            if j+patch_width > image_width:
                patch_width = image_width - j
            # Adding the patch
            patch_list.append(
                (j, i, patch_width, patch_height,
                    current_window[:patch_height, :patch_width])
            )
    return patch_list

def sliding_window_hann(
            xraster, model, window_size, tile_size,
            inference_overlap, inference_treshold, batch_size,
            mean, std
        ):

    # open rasters and get both data and coordinates
    rast_shape = xraster[:, :, 0].shape  # shape of the wider scene

    # smooth window
    window_func = w.hann
    use_hanning = True

    # print(rast_shape, wsy, wsx)
    prediction = np.zeros((rast_shape[0], rast_shape[1], 2))
    print(f'Prediction shape: {prediction.shape}')

    patch_list = generate_patch_list(
        rast_shape[0], rast_shape[1], window_func, tile_size, use_hanning)

    for patch in patch_list:

        patch_x, patch_y, patch_width, patch_height, window = patch
        input_patch = np.expand_dims(xraster[patch_y:patch_y+patch_height, patch_x:patch_x+patch_width].values, 0)
        window_pred = model.predict(input_patch)
        prediction[patch_y:patch_y+patch_height, patch_x:patch_x+patch_width] += \
            np.squeeze(window_pred) * np.expand_dims(window, -1)

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
    else:
        prediction = np.squeeze(
            np.where(
                prediction > inference_treshold, 1, 0).astype(np.int16))
    return prediction

# second option, same hann window combined with mosaicing
# overalapping big sliding windows, that get hann windows in their corners

def sliding_window_hann_slide(
            xraster, model, window_size, tile_size,
            inference_overlap, inference_treshold, batch_size,
            mean, std
        ):

    # open rasters and get both data and coordinates
    rast_shape = xraster[:, :, 0].shape  # shape of the wider scene

    # in memory sliding window predictions
    wsy, wsx = window_size, window_size
    # wsy, wsx = rast_shape[0], rast_shape[1]

    # if the window size is bigger than the image, predict full image
    if wsy > rast_shape[0]:
        wsy = rast_shape[0]
    if wsx > rast_shape[1]:
        wsx = rast_shape[1]

    # smooth window
    # this might be problematic since there might be issues on tiles smaller
    # than actual squares
    # spline = spline_window(wsy)

    # print(rast_shape, wsy, wsx)
    prediction = np.zeros(rast_shape)  # crop out the window
    print(f'wsize: {wsy}x{wsx}. Prediction shape: {prediction.shape}')

    

    return prediction

"""
def predict_scene(self, scene: Scene, backend: Backend) -> Labels:
    log.info('Making predictions for scene')
    raster_source = scene.raster_source
    label_store = scene.prediction_label_store
    labels = label_store.empty_labels()

    windows = self.get_predict_windows(raster_source.get_extent())

    def predict_batch(chips, windows):
        nonlocal labels
        chips = np.array(chips)
        batch_labels = backend.predict(scene, chips, windows)
        batch_labels = self.post_process_batch(windows, chips,
                                                batch_labels)
        labels += batch_labels

        print('.' * len(chips), end='', flush=True)

    batch_chips, batch_windows = [], []
    for window in windows:
        chip = raster_source.get_chip(window)
        batch_chips.append(chip)
        batch_windows.append(window)

        # Predict on batch
        if len(batch_chips) >= self.config.predict_batch_sz:
            predict_batch(batch_chips, batch_windows)
            batch_chips, batch_windows = [], []
    print()

    # Predict on remaining batch
    if len(batch_chips) > 0:
        predict_batch(batch_chips, batch_windows)

    return self.post_process_predictions(labels, scene)

"""