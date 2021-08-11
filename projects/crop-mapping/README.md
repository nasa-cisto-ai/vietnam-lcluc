# Vietnam Crop Mapping

Mapping of land cover in Vietnam using deep learning.

Author: Jordan A. Caraballo-Vega, <jordan.a.caraballo-vega@nasa.gov>
Main POC: Margaret Wooten, <margaret.wooten@nasa.gov>

## Data Sources

Data directory and bands: /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN/7-band/*.tif, 93 Keelin squares.

Band Order:
1 = Blue
2 = Green
3 = Red
4 = NIR
5 = BAI (Burned Area Index)
6 = NDVI
7 = NDWI (Normalized Difference Water Index)

Training directory and classes: /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/CNN/labeled_training/*.tif

Classes:
0 = Growing
1 = Sparse
2 = Fallow
3 = Burned
4 = Water
5 = Other
6 = IDK (what is this?)

Ouput directory: /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_LCLUC

## Methodology

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Margaret Wooten, margaret.wooten@nasa.gov

# References

