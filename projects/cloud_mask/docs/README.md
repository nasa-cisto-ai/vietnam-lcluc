# Vietnam Cloud Masking

Random Forest and Convolutional Neural Network for Cloud Masking in
World View Imagery

## Data

- Project Directory: /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD
- Data Directory:

## Random Forest

Random Forest Classification

### Random Forest Data

- Original Training CSV: /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training.csv
- Training Labels Directory: /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels
- Combinations:
  - R G B NIR1
  - B G R NIR1
  - B G R NIR1 FDI SI NDWI
  - CB B G Y R RE NIR1 NIR2
  - CB B G Y R RE NIR1 NIR2 FDI SI NDWI

### Random Forest Preparation

Shell into the container:

```bash
singularity shell --nv -B $NOBACKUP:$NOBACKUP,/att/gpfsfs/atrepo01/ILAB:/att/gpfsfs/atrepo01/ILAB /lscratch/jacaraba/vietnam-lcluc/container/nccs-lcluc
```

B G R NIR1 Dataset:

```bash
python modify_csv.py --input-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training.csv --output-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_BGRNIR1.csv --input-columns CB B G Y R RE NIR1 NIR2 FDI SI NDWI L --output-columns B G R NIR1 L
```

R G B NIR1 Dataset:

```bash
python modify_csv.py --input-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training.csv --output-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_RGBNIR1.csv --input-columns CB B G Y R RE NIR1 NIR2 FDI SI NDWI L --output-columns R G B NIR1 L
```

CB B G Y R RE NIR1 NIR2:

```bash
python modify_csv.py --input-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training.csv --output-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_8band.csv --input-columns CB B G Y R RE NIR1 NIR2 FDI SI NDWI L --output-columns CB B G Y R RE NIR1 NIR2 L
```

### Random Forest Training

| Syntax                                | Train  | Predict | Validate |
| ------------------------------------- | ------ | ------- | -------- |
| R G B NIR1                            |        |         |          |
| B G R NIR1                            |        |         |          |
| B G R NIR1 FDI SI NDWI                |        |         |          |
| CB B G Y R RE NIR1 NIR2               |        |         |          |
| CB B G Y R RE NIR1 NIR2 FDI SI NDWI   |        |         |          |

R G B NIR1: cloud_training_4band_RGBNIR1.csv

```bash
python rf_driver.py --step train --train-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_RGBNIR1.csv --seed 42 --test-size 0.20 --n-trees 20 --max-features log2 --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_RGBNIR1/cloud_training_4band_RGBNIR1.pkl
```

B G R NIR1: cloud_training_4band_BGRNIR1.csv

```bash
```

B G R NIR1 FDI SI NDWI: cloud_training_4band_fdi_si_ndwi.csv

```bash
```

CB B G Y R RE NIR1 NIR2: cloud_training_8band.csv

```bash
```

CB B G Y R RE NIR1 NIR2 FDI SI NDWI: cloud_training_8band_fdi_si_ndwi.csv

```bash
```

### Random Forest Prediction

```bash
```

## Convolutional Neural Network

- Data Directory:
- Labels Directory:

## Article Notes

### Data Location

- Validation Dataset:
- Final CNN Results:
- Final CNN Results Postprocessed:

- Final RF Results:
- Final RF Results Postprocessed:

### Feature Importance and Decision Tree Visualization

- Visualize feature importance per band
- Visualiza decission trees

### Images in Result Section

- Buildings Illustration: KeelinXXX
- Thin Clouds Illustration: KeelinXXX
- Flooded Field Illustration: KeelinXXX
- Flooded Field Boats: KeeelinXXX
