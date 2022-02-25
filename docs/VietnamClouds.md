# Vietnam Clouds

Training data 8-bands:
Training data 4-bands:
Training data 4-bands + 3 indices: /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_rgb_fdi_si_ndwi.csv

## Random Forest

Training

```bash
python rf_pipeline.py --train-csv /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_rgb_fdi_si_ndwi.csv --step train --output-model /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_rgb_fdi_si_ndwi.pkl
```

Prediction

```
python rf_pipeline.py --step predict --output-model /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_rgb_fdi_si_ndwi.pkl --rasters '/att/pubrepo/ILAB/projects/Vietnam/Sarah/data/*.tif' --output-dir /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/predictions/cloud_training_4band_rgb_fdi_si_ndwi
```

Explainable AI

```bash
```