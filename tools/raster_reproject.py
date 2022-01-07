import os
import sys
import glob
import logging
import argparse
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def main():

    # Process command-line args.
    desc = 'Use this application to reproject TIF files.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '-i', '--input-rasters', type=str, required=True,
        dest='input_rasters', help='Regex with raster dir (e.g. /home/*.tif')

    parser.add_argument(
        '-o', '--output-dir', type=str, required=True,
        dest='output_dir', help='Output directory to save rasters')

    parser.add_argument(
        '-c', '--crs', type=str, required=False, default='EPSG:4326',
        dest='crs', help='Output crs to save')

    args = parser.parse_args()

    # --------------------------------------------------------------------------------
    # set logging
    # --------------------------------------------------------------------------------
    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)
    #ch = logging.StreamHandler(sys.stdout)  # set stdout handler
    #ch.setLevel(logging.INFO)

    # set formatter and handlers
    #formatter = logging.Formatter(
    #    "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S")
    #ch.setFormatter(formatter)
    #logger.addHandler(ch)

    # --------------------------------------------------------------------------------
    # reproject step
    # --------------------------------------------------------------------------------
    filenames = glob.glob(args.input_rasters)

    for filename in filenames:

        output_filename = os.path.join(
            args.output_dir,
            filename.split('/')[-1][:-3] +
            args.crs.replace(':', '.').lower() + '.tif')

        with rasterio.open(filename) as src:
            transform, width, height = calculate_default_transform(
                src.crs, args.crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': args.crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(output_filename, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=args.crs,
                        resampling=Resampling.nearest
                    )


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
