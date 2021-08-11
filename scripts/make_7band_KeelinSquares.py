# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:31:55 2021
@author: mwooten3

PURPOSE: To bypass the need for PCI* and create the 7-band (not including 
texture metrics) images for Keelin squares so they can be used in the CNN
    *will revisit automating PCI if it turns out we need the isoclusters
    
PROCESS:
For each Keelin square:
    Get the 6 band .tif
    Extract the bands we need: 1-4
    Create the NDVI, NDWI, BAI
    Create the 7-band image
"""

import os, sys
import glob

import time

from Raster import Raster


inDir = '/att/pubrepo/ILAB/projects/Vietnam/Sarah/data/'
outDir = '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/ISO/KeelinData-7bands'
tempDir = os.path.join(outDir, 'temp') # to store individual bands created
logDir = os.path.join(outDir, 'logs')

for d in [tempDir, logDir]:
    os.system('mkdir -p {}'.format(d)) # -p will create outDir as well if need be
    
bandsDict = {'Blue': 1, 'Green': 2, 'Red': 3, 'NIR': 4, 't1': 5, 't2': 6} # do not care about t1 or t2 

outNoDataVal = 11000 # To be used for output no data values. The range for no layer should be outside of this       
    
def createIndexTif(inTif, bandTifDict, calculation, indexName, inputs):
    # thought about generalizing but nah, too messy bc inputs (would have to 
    # either explicitly supply inputs or replace names in formulas with 
    # letters) and just don't feel like it
    return outTif, bandTifDict

def createBai(inTif, bandTifDict):
    
    # Formula (the one we got to work in PCI after trial and error):
    # BAI --> (1/((0.1-(Red/10000))^2 + (0.06-(NIR/10000))^2))
    
    outTif = os.path.join(tempDir, '{}-BAI.tif'.format(inTif.baseName))
    
    calc = '(1/((0.1-(A/10000))**2 + (0.06-(B/10000))**2))'
    cmd = "gdal_calc.py --calc='{}' --outfile={} -A {} -B {} --co COMPRESS=LZW \
    --co BIGTIFF=YES --NoDataValue={} --type=Int16".format(calc, outTif,    \
                        bandTifDict['Red'], bandTifDict['NIR'], outNoDataVal)
            
    print "\nCreating BAI .tif:"
    print cmd
    os.system(cmd)
    
    # Add output to dictionary
    if not os.path.isfile(outTif):
        raise RuntimeError("The index file {} was not created\n".format(outTif))
    else: 
        bandTifDict['BAI'] = outTif
    
    return bandTifDict

def createNdvi(inTif, bandTifDict):
    
    # Formula
    # NDVI --> ((NIR - Red)/ (NIR + Red))*10000

    outTif = os.path.join(tempDir, '{}-NDVI.tif'.format(inTif.baseName))
    
    calc = '((1.0*(B-A))/(B+A))*10000'
    cmd = "gdal_calc.py --calc='{}' --outfile={} -A {} -B {} --co COMPRESS=LZW \
    --co BIGTIFF=YES --NoDataValue={} --type=Int16".format(calc, outTif,    \
                        bandTifDict['Red'], bandTifDict['NIR'], outNoDataVal)
            
    print "\nCreating NDVI .tif:"
    print cmd
    os.system(cmd)
    
    # Add output to dictionary
    if not os.path.isfile(outTif):
        raise RuntimeError("The index file {} was not created\n".format(outTif))
    else: 
        bandTifDict['NDVI'] = outTif
    
    return bandTifDict

def createNdwi(inTif, bandTifDict):
    
    # Formula
    # NDWI --> ((Green -NIR)/ (Green + NIR))*10000

    outTif = os.path.join(tempDir, '{}-NDWI.tif'.format(inTif.baseName))
    
    calc = '((1.0*(B-A))/(B+A))*10000'
    cmd = "gdal_calc.py --calc='{}' --outfile={} -A {} -B {} --co COMPRESS=LZW \
    --co BIGTIFF=YES --NoDataValue={} --type=Int16".format(calc, outTif,      \
                        bandTifDict['NIR'], bandTifDict['Green'], outNoDataVal)
            
    print "\nCreating NDWI .tif:"
    print cmd
    os.system(cmd)

    # Add output to dictionary
    if not os.path.isfile(outTif):
        raise RuntimeError("The index file {} was not created\n".format(outTif))
    else: 
        bandTifDict['NDWI'] = outTif 
        
    return bandTifDict

def createFinalTif(bandTifDict, bandOrder, outTif):
    # --> vrt then translate to .tif
    
    # Output temp .vrt
    outVrt = outTif.replace(outDir, tempDir).replace('.tif', '.vrt')
    
    # Get the vrt input band .tifs and the corresponding NoData values
    srcNoData = []
    inputTifs = []
    for band in bandOrder:
        srcNoData.append(str(Raster(bandTifDict[band]).noDataValue))
        inputTifs.append(bandTifDict[band])
    
    # for whatever reason, -vrtnodata is being overwritten with the nd value
    # of the first input (9999), so leave out argument and try to change the 
    # value in translate step. Not super important as there doesn't appear to 
    # be any NoData pixels anyways (as is probably true for the other KSs)
    cmd = 'gdalbuildvrt -separate -srcnodata "{}" -vrtnodata None {} {}'.format(\
                                " ".join(srcNoData), outVrt, " ".join(inputTifs))

    print "\n\nCreating temporary .vrt:"
    print cmd
    os.system(cmd)
    
    
    cmd2 = 'gdal_translate -ot Int16 -a_nodata 11000 -co COMPRESS=LZW {} {}'\
                                                      .format(outVrt, outTif)
    print "\n\nCreating final .tif:"
    print cmd2
    os.system(cmd2)
    print ''
    
    return None


def extractBandTifs(inTif, bands):
    
    # Start to build the dictionary that will contain path to all 7 bands in the output
    bandTifDict = {}
    
    print "\nExtracting select bands ({}) from input .tif {}:".format(bands, inTif.filePath)
    
    for band in bands:
        outBandTif = os.path.join(tempDir, '{}-{}.tif'.format(inTif.baseName, band))
        
        bandTifDict[band] = outBandTif
        
        if os.path.isfile(outBandTif):
            continue # skip it
            
        bandN = bandsDict[band]
        
        inTif.extractBand(bandN, outBandTif)
        
    return bandTifDict

def logOutput(bname):
    
    # Log output:
    logFile = os.path.join(logDir, 'create_data-7bands__{}__Log.txt'.format(bname))
    #print "See {} for log\n".format(logFile)
    so = se = open(logFile, 'a', 0) # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) # re-open stdout without buffering
    os.dup2(so.fileno(), sys.stdout.fileno()) # redirect stdout and stderr to the log file opened above
    os.dup2(se.fileno(), sys.stderr.fileno())
    
    return logFile

def processKeelinSquare(inTifPath):
    
    # Get tif obj, set up output vars and begin logging
    inTif = Raster(inTifPath)
    bname = inTif.baseName
    logOutput(bname)
    
    #if bname != 'Keelin13_20160318_data': # test w/ sample square
      #  return None # TEMP BLOCK
        
    outTifPath = os.path.join(outDir, '{}-7band.tif'.format(bname))
    
    print "BEGIN: {}".format(time.strftime("%m-%d-%y %I:%M:%S"))
    print " Input .tif: {}".format(inTifPath)
    print " Output .tif: {}\n".format(outTifPath)
    
    # First, start building the band tif dict for the input KS
    ## The bandTifDict will have all of the individual bands that will be merged into the final output
    bands = ['Blue', 'Green', 'Red', 'NIR'] # The bands we are interested in from input .tif
    bandTifDict = extractBandTifs(inTif, bands = bands)
    
    # Then, create the indices (same order as PCI - BAI, NDVI, NDWI):
    # These .tifs will also be added to dict as output
    bandTifDict = createBai(inTif, bandTifDict)
    bandTifDict = createNdvi(inTif, bandTifDict)
    bandTifDict = createNdwi(inTif, bandTifDict)
    
    # Lastly, create the 7-band .tif, using xplicit order of bands (same as input + index bands)
    createFinalTif(bandTifDict, bands + ['BAI', 'NDVI', 'NDWI'], outTifPath)

    print "END: {}\n\n".format(time.strftime("%m-%d-%y %I:%M:%S"))
    
    """ doesnt work - some day figure this out 
    sys.stdout.close()
    del logFile """
    
    return outTifPath


# Iterate through Keelin squares and process
inTifs = glob.glob(os.path.join(inDir, 'Keelin*tif'))
nTifs  = len(inTifs)
print "Creating 7-band outputs for {} geoTIFFs...".format(nTifs)
print " See logs in {}\n".format(logDir)

c = 0

for tif in inTifs:
    c+=1
    
    # was not working with logging (not sure if because in function or not)
    #print "{}/{}: Processing {}".format(c, nTifs, tif)
      
    outTif = processKeelinSquare(tif)






