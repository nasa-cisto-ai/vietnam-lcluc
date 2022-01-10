import os
from pathlib import Path
from glob import glob
import xarray as xr
import rioxarray as rxr
from typing import List

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

# items to specify
# senegal
#images_list = glob('/Users/jacaraba/Desktop/holidays_work/senegal/data/*.tif')
#output_dir = '/Users/jacaraba/Desktop/holidays_work/senegal/data_fixed'
#input_bands = ['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2']
#output_bands = ['Blue', 'Green', 'Red', 'NIR1']

images_list = glob('/Users/jacaraba/Desktop/holidays_work/vietnam/clouds/data/*.tif')
output_dir = '/Users/jacaraba/Desktop/holidays_work/vietnam/clouds/data_fixed'
input_bands = ['Blue', 'Green', 'Red', 'NIR1', 'HOM1', 'HOM2']
output_bands = ['Blue', 'Green', 'Red', 'NIR1']
os.makedirs(output_dir, exist_ok=True)

for image_filename in images_list:

    image = rxr.open_rasterio(image_filename)
    image = modify_bands(image, input_bands, output_bands)
    image.attrs['long_name'] = tuple(output_bands)
    image.rio.to_raster(os.path.join(output_dir, f'{Path(image_filename).stem}.tif'))

