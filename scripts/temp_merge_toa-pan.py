# need temp code to try gdal_merge step on crane nodes outside of process
# not trying to make this a thing outside of this


"""
ddir = '/att/gpfsfs/briskfs01/ppl/mwooten3/AIST/EVHR_tests/requests/vietnamPan-gEyFT2Vs0UriU82TBgY_h_eZfWpduppRCBsn7FDP/'
In files:
5-toas/WV02_20091205_P1BS_1030010002315A00_BAND_P.r100-ortho-toa.tif
Ortho xml:
4-orthos/WV02_20091205_P1BS_1030010002315A00_BAND_P.r100-ortho.xml
bandFiles = glob.glob(os.path.join(ddir, '5-toas', 'W*BAND_P*tif'))
for each BANDB file:
'gdal_merge.py -co COMPRESS=LZW -co BIGTIFF=YES -ot Int16 -separate -n -9999 
-a_nodata -9999 -o {} {}'. \
                format(outFileName, \
                       ' '.join(bandFiles))
"""
import os, glob

ddir = '/att/gpfsfs/briskfs01/ppl/mwooten3/AIST/EVHR_tests/requests/vietnamPan-gEyFT2Vs0UriU82TBgY_h_eZfWpduppRCBsn7FDP/'
#inf = ddir + '5-toas/WV02_20091205_P1BS_1030010002315A00_BAND_P.r100-ortho-toa.tif'
c=0
for inf in glob.glob(os.path.join(ddir, '5-toas', 'W*BAND_P*tif')):
#    if os.path.basename(inf) != 'WV02_20091205_P1BS_1030010002315A00_BAND_P.r100-ortho-toa.tif':
#        continue
    c+=1
    out = inf.replace('_BAND_P.r100-ortho-toa.tif', '-toa.tif')
    strip = os.path.basename(out).strip('-toa.tif')
    fromXml = os.path.join(ddir, '4-orthos', '{}_BAND_P.r100-ortho.xml'.format(strip))
    toXml = os.path.join(ddir, '5-toas', '{}-toa.xml'.format(strip))
    
    print "Processing {}: {}".format(c, strip)
    # out was created but not written
    os.system('rm {}'.format(out))
    # covert BAND_P to final toa with merge
    cmd = 'gdal_merge.py -co COMPRESS=LZW -co BIGTIFF=YES -ot Int16 -separate -n -9999 -a_nodata -9999 -o {} {}'.format(out, inf)
    print cmd
    os.system(cmd)
    
    # copy ortho xml
    print 'cp {} {}'.format(fromXml, toXml)
    os.system('cp {} {}'.format(fromXml, toXml))

#    print inf
#    print out
#    print fromXml
#    print toXml
    print ''
print c