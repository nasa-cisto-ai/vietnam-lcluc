import os
import sys
import glob
import argparse
import logging

from tqdm import tqdm
import xarray as xr
import numpy as np
import rasterio.features as riofeat
import rasterio as rio
from scipy.ndimage import median_filter, binary_fill_holes

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"


def npy_to_tif(raster_f='image.tif', segments='segment.npy',
               outtif='segment.tif', ndval=-9999
               ):
    """
    Args:
        raster_f:
        segments:
        outtif:
    Returns:
    """
    # get geospatial profile, will apply for output file
    with rio.open(raster_f) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')
    print(meta)

    # load numpy array if file is given
    if type(segments) == str:
        segments = np.load(segments)
    segments = segments.astype('int16')
    print(segments.dtype)  # check datatype

    nodatavals[nodatavals == 0] = ndval
    segments[nodatavals == ndval] = nodatavals[nodatavals == ndval]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(outtif, 'w', **out_meta) as dst:
        dst.write(segments, 1)


def main():

    # Process command-line args.
    desc = 'Use this application to postprocess Cloud mask TIF files.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-i', '--input-rasters', type=str, required=True,
        dest='input_rasters', help='Regex with raster dir (e.g. /home/*.tif')

    parser.add_argument(
        '-o', '--output-dir', type=str, required=True,
        dest='output_dir', help='Output directory to save rasters')

    parser.add_argument(
        '-s', '--sieve-size', type=int, required=False, default=800,
        dest='sieve_size', help='Sieve size')

    parser.add_argument(
        '-m', '--median-size', type=int, required=False, default=20,
        dest='median_size', help='Median size')

    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # set logging
    # --------------------------------------------------------------------------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)  # set stdout handler
    ch.setLevel(logging.INFO)

    # set formatter and handlers
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --------------------------------------------------------------------------------
    # postprocessing step
    # --------------------------------------------------------------------------------
    filenames = glob.glob(args.input_rasters)
    logging.info(f'{len(filenames)} files to posprocess...')
    os.system('mkdir -p {}'.format(args.output_dir))

    for mask_f in tqdm(filenames):

        # open mask
        raster_mask = xr.open_rasterio(
            mask_f, chunks={'band': 1, 'x': 2048, 'y': 2048}).values

        # out filename
        out_filename = os.path.join(args.output_dir, mask_f.split('/')[-1])

        # riofeat.sieve(prediction, size, out, mask, connectivity)
        riofeat.sieve(raster_mask, args.sieve_size, raster_mask, None, 8)

        # median filter and binary filter
        raster_mask = median_filter(raster_mask, size=args.median_size)
        raster_mask = binary_fill_holes(raster_mask).astype(int)

        npy_to_tif(raster_f=mask_f, segments=raster_mask, outtif=out_filename)

    return


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
