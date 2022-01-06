# code to convert labeled polygons into rasters

# Process:
#  Rasterize feature class, setting NoData value and using data.tif for pixel size, projection, etc.
#   Actually, projection should carry over and just hardcode pixel size 

from Raster import Raster
import os

# layer - change
layerName = 'Keelin23_20110201'

ddir = '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN'
inFC = os.path.join(ddir, 'relabelPolygons.gdb')
data7Dir = '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN/7-band'
dataTif = os.path.join(data7Dir, '{}_data-7band.tif'.format(layerName))
outDir = os.path.join(ddir, 'labeled_training')
outTif = os.path.join(outDir, '{}__training.tif'.format(layerName))

ndVal = 15
res = 0.5
outType = 'Int16'
fieldName = 'new_gridcode'


# Arguments: field/attribute name, layer name, pixel size,
#            no data val, lzw compression, 
#            target aligned pixels (?), output type

# Get extent from raster
dataRast = Raster(dataTif)
listExt = [str(i) for i in dataRast.extent()]

command = 'gdal_rasterize -l {} -a {} -tr {} {} -te {} -a_nodata {} -ot {} -co COMPRESS=LZW {} {}'.format(layerName, fieldName, res, res, ' '.join(listExt), ndVal, outType, inFC, outTif)
print command

os.system(command)




