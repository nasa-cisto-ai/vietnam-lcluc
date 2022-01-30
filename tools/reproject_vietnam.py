import glob
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

#rf_path = '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_rf/images/*'
#fname_list = glob.glob(rf_path)

#cnn_path = '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_cnn/100_unet_viet_cm_4chann_std_Adadelta_256_0.0001_128_1000.h5/images/*'

#cnn_path = '/Users/jacaraba/Desktop/results_mosaic_vietnam_v2/*.tif'
cnn_path = '/Users/jacaraba/Desktop/cloud_training_4band_rgb_fdi_si_ndwi/*.tif'
fname_list = glob.glob(cnn_path)

for filename in fname_list:

    dst_crs = 'EPSG:4326'

    with rasterio.open(filename) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(filename[:-3] + 'epsg4326.tif', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)