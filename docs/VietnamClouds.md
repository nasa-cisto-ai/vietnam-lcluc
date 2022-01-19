# Vietnam Clouds

Training data 8-bands:
Training data 4-bands:
Training data 4-bands + 3 indices: /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_rgb_fdi_si_ndwi.csv

## Random Forest

Training

```bash
python rf_pipeline.py --train-csv /Users/jacaraba/Desktop/development/ilab/vietnam-lcluc/data/cloud_training_4band_rgb_fdi_si_ndwi.csv --step train --output-model /Users/jacaraba/Desktop/development/ilab/vietnam-lcluc/data/cloud_training_4band_rgb_fdi_si_ndwi.pkl
```

Prediction

Explainable AI

```bash
```