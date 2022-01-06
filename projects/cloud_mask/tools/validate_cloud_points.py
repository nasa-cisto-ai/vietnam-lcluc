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
import os
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
        rregex = val_df['source'][index].capitalize()
        print(rregex)

        # check if this is for RF or CNN
        if 'RF' in column_id:

            # RF
            if rregex == 'Keelin17_20091230' or rregex == 'Keelin18_20091230':
                mask_filename = os.path.join(
                    data_dir, f'{rregex}_data_a.epsg.4326.tif')
            else:
                mask_filename = os.path.join(
                    data_dir, f'{rregex}_data.epsg.4326.tif')

        elif 'CNN' in column_id:

            # CNN
            if rregex == 'Keelin17_20091230' or rregex == 'Keelin18_20091230':
                mask_filename = os.path.join(
                    data_dir, f'{rregex}_data_a_pred.epsg4326.tif')
            else:
                mask_filename = os.path.join(
                    data_dir, f'{rregex}_data_pred.epsg4326.tif')

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


def main(validation_csv, data_dict):

    # file to add metrics to
    f = open("rf_cnn_metrics.csv", "a")

    # Write information on file
    f.write(f'{validation_csv}\n')
    f.write('model,accuracy,precision,recall,jaccard\n')

    # open dataframe
    df = pd.read_csv(validation_csv)
    df = df.drop(['OBJECTID'], axis=1)
    df.drop(
        df.columns[
            df.columns.str.contains(
                'unnamed',case = False)
            ],axis = 1, inplace = True)

    print(df)

    # iterate over each set of data values, RF vs CNN
    for column_id in data_dict:

        # feed data to get values function, return pd column
        print(column_id, data_dict[column_id], df)
        df = get_values(column_id, data_dict[column_id], df)

        # compute metrics
        accuracy = accuracy_score(df['valid'], df[column_id])
        precision = precision_score(df['valid'], df[column_id])
        recall = recall_score(df['valid'], df[column_id])
        jaccard = jaccard_score(df['valid'], df[column_id])

        # write some metrics to file
        f.write(f'{column_id},{accuracy},{precision},{recall},{jaccard}\n')

    # save csv
    df.to_csv(f'{validation_csv[:-4]}_val_20211028.csv', index=False)

    # confusion matrix
    true_values = df['valid'].to_numpy()
    rf_values = df['RF'].to_numpy()
    cnn_values = df['CNN'].to_numpy()

    print('RF')
    print(confusion_matrix(true_values, rf_values))
    print('CNN')
    print(confusion_matrix(true_values, cnn_values))

    print(true_values.sum())

    f.close()


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # validation csv filenames
    validation_csv = '/Users/jacaraba/Desktop/CURRENT_PROJECTS/LCLUC_Vietnam/vietnam_cm_models/validation/keelin_validation.csv'

    data_dict = {
        'RF': '/Users/jacaraba/Desktop/CURRENT_PROJECTS/LCLUC_Vietnam/vietnam_cm_models/cloudmask_keelin_4bands_rf_article',
        'CNN': '/Users/jacaraba/Desktop/CURRENT_PROJECTS/LCLUC_Vietnam/vietnam_cm_models/cloudmask_keelin_4bands_cnn/105_unet_viet_cm_4chann_std_Adam_512_0.0001_64_185.h5/processed_cnn_2021-10-28'
    }
    main(validation_csv=validation_csv, data_dict=data_dict)
