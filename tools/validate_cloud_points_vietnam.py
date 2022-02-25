"""
    Purpose:
    - Map validation data with binary masks output.

    Input:
    - GEOTIF files with binary masks, together with the CSVs storing the
      validation dataset points. CSV Format: x, y, source, valid, mask

    Output:
    - CSV file with validation data included.

    Notes:
    - This script was intended for a particular use. Modifications may be
      require to apply to other kinds of datasets.

Valid column is the visual validation column.
"""
import os
import xarray as xr
import pandas as pd
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def get_values(val_df, data_path):
    """
    Default Raster initializer
    Args:
        column_id (str): column id
        data_dir (str): directory to read masks from
        val_df (pd df): dataframe with validation dataset
    Return:
        pandas validation values dataframe (single column)
    """
    val_values = list()

    for index in val_df.index:

        raster_regex = val_df['source'][index].capitalize()
        print(raster_regex)

        if raster_regex == 'Keelin17_20091230' or raster_regex == 'Keelin18_20091230':
            mask_filename = os.path.join(
                data_path, f'{raster_regex.capitalize()}_data_a.epsg4326.tif')
            #    data_path, f'{raster_regex.capitalize()}_data_a_clouds.epsg4326.tif')
        else:
            mask_filename = os.path.join(
                data_path, f'{raster_regex.capitalize()}_data.epsg4326.tif')
            #    data_path, f'{raster_regex.capitalize()}_data_clouds.epsg4326.tif')

        raster_data = xr.open_rasterio(mask_filename)
        raster_data = raster_data[0, :, :]

        val = raster_data.sel(
            x=val_df['x'][index], y=val_df['y'][index], method="nearest"
        )
        val_values.append(int(val.values))

    val_df['mask1'] = val_values
    return val_df


def main(validation_csv, data_path):

    # open dataframe
    points_df = pd.read_csv(validation_csv)
    print(points_df)

    # feed data to get values function, return pd column
    points_df = get_values(points_df, data_path)
    
    visual_validation = 'valid'
    mask_validation = 'mask1'

    # compute metrics
    accuracy = accuracy_score(points_df[visual_validation], points_df[mask_validation])
    precision = precision_score(points_df[visual_validation], points_df[mask_validation])
    recall = recall_score(points_df[visual_validation], points_df[mask_validation])
    jaccard = jaccard_score(points_df[visual_validation], points_df[mask_validation])

    #print(points_df.index)
    print('cloud points:', points_df['valid'].value_counts(), points_df['mask'].sum())
    print(f'acc: {accuracy}')
    print(f'prec: {precision}')
    print(f'rec: {recall}')
    print(f'jacc: {jaccard}')

    confs = confusion_matrix(points_df[visual_validation], points_df[mask_validation])
    print(confs)


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Validation CSVs filenames
    validation_csv = '/Users/jacaraba/Desktop/CURRENT_PROJECTS/LCLUC_Vietnam/vietnam_cm_models/validation/keelin_validation.csv'

    # Data filenames
    # data_path = '/Users/jacaraba/Desktop/results_mosaic_vietnam_v2'
    data_path = '/Users/jacaraba/Desktop/cloud_training_4band_rgb_fdi_si_ndwi'

    # Random Forest data paths
    #data_paths = {
    #    'RF': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_rf/images',
    #    'RF POST': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_rf/processed_rf',
    #    'CNN-105': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_cnn/105_unet_viet_cm_4chann_std_Adam_512_0.0001_64_185.h5/images',
    #    'CNN': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_cnn/100_unet_viet_cm_4chann_std_Adadelta_256_0.0001_128_1000.h5/images',
    #    'CNN POST': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_cnn/100_unet_viet_cm_4chann_std_Adadelta_256_0.0001_128_1000.h5/processed'
    #}

    main(
        validation_csv=validation_csv,
        data_path=data_path,
    )
