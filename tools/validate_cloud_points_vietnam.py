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
"""
import xarray as xr
import pandas as pd
from sklearn.metrics import accuracy_score, jaccard_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def get_values(column_id, data_dir, val_df):
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

        # find the file that matches
        raster_regex = val_df['source'][index].capitalize()

        # check if this is for RF or CNN
        if 'RF' in column_id:

            # RF
            if raster_regex == 'Keelin17_20091230' or raster_regex == 'Keelin18_20091230':
                mask_filename = f'{data_dir}/cm_{raster_regex.capitalize()}_data_a.epsg4326.tif'
            else:
                mask_filename = f'{data_dir}/cm_{raster_regex}_data.epsg4326.tif'

        elif 'CNN' in column_id:

            # CNN
            if raster_regex == 'Keelin17_20091230' or raster_regex == 'Keelin18_20091230':
                mask_filename = f'{data_dir}/{raster_regex}_data_a_pred.epsg4326.tif'
            else:
                mask_filename = f'{data_dir}/{raster_regex}_data_pred.epsg4326.tif'

        # open filename, TODO: avoid opening new filenames, cache previous
        dask_size = {'band': 1, 'x': 2048, 'y': 2048}
        raster_data = xr.open_rasterio(mask_filename, chunks=dask_size)
        raster_data = raster_data[0, :, :]

        # from latitud and longitude, get pixel value
        # use the .sel() method to retrieve the value of the nearest cell
        val = raster_data.sel(
            x=val_df['x'][index], y=val_df['y'][index], method="nearest"
        )

        val_values.append(int(val.values))

    val_df[column_id] = val_values
    return val_df


def main(validation_csv, data_paths):







    # csv_filenames, data_paths


    # file to add metrics
    f = open("rf_cnn_metrics.csv", "a")

    # iterate over each CSV validation dataset
    for csv_filename in csv_filenames:

        # Write information on file
        f.write(f'{csv_filename}\n')
        f.write('model,accuracy,precision,recall,jaccard\n')

        # open dataframe
        df = pd.read_csv(csv_filename)
        print(df)

        # iterate over each set of data values, RF vs CNN
        for column_id in data_paths:

            # feed data to get values function, return pd column
            df = get_values(column_id, data_paths[column_id], df)

            # compute metrics
            accuracy = accuracy_score(df['valid'], df[column_id])
            precision = precision_score(df['valid'], df[column_id])
            recall = recall_score(df['valid'], df[column_id])
            jaccard = jaccard_score(df['valid'], df[column_id])

            # write some metrics to file
            f.write(f'{column_id},{accuracy},{precision},{recall},{jaccard}\n')

        # save csv
        df.to_csv(f'{csv_filename[:-4]}_val.csv', index=False)
    f.close()

# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Validation CSVs filenames
    validation_csv = '/Users/jacaraba/Desktop/CURRENT_PROJECTS/LCLUC_Vietnam/vietnam_cm_models/validation/keelin_validation.csv'

    # Random Forest data paths
    data_paths = {
        'RF': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_rf/images',
        'RF POST': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_rf/processed_rf',
        'CNN-105': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_cnn/105_unet_viet_cm_4chann_std_Adam_512_0.0001_64_185.h5/images',
        'CNN': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_cnn/100_unet_viet_cm_4chann_std_Adadelta_256_0.0001_128_1000.h5/images',
        'CNN POST': '/Users/jacaraba/Desktop/vietnam_cm_models/cloudmask_keelin_4bands_cnn/100_unet_viet_cm_4chann_std_Adadelta_256_0.0001_128_1000.h5/processed'
    }

    main(
        csv_filenames=csv_filenames,
        data_paths=data_paths,
    )
