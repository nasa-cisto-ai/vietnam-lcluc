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
| R G B NIR1                            |  x      |         |          |
| B G R NIR1                            |  x      |         |          |
| B G R NIR1 FDI SI NDWI                |        |         |          |
| CB B G Y R RE NIR1 NIR2               |        |         |          |
| CB B G Y R RE NIR1 NIR2 FDI SI NDWI   |        |         |          |

R G B NIR1: cloud_training_4band_RGBNIR1.csv

```bash
(rapids) Singularity> python rf_driver.py --step train --train-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_RGBNIR1.csv --seed 42 --test-size 0.20 --n-trees 20 --max-features log2 --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_RGBNIR1/cloud_training_4band_RGBNIR1.pkl
2021-10-18 18:31:22; INFO; Open /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_RGBNIR1.csv dataset for training.
2021-10-18 18:31:22; INFO; X size: (799999, 4)
2021-10-18 18:31:22; INFO; Y size:  (799999,)
2021-10-18 18:31:22; INFO; ntrees=20, maxfeat=log2
2021-10-18 18:31:22; INFO; Training model via RAPIDS.
2021-10-18 18:31:24; INFO; init
2021-10-18 18:31:24; INFO; Test Accuracy:  0.99993
2021-10-18 18:31:24; INFO; Test Precision: 0.9998801617815949
2021-10-18 18:31:24; INFO; Test Recall:    0.999980024968789
2021-10-18 18:31:24; INFO; Test F-Score:   0.9999300908818536
2021-10-18 18:31:25; INFO; Model has been saved as /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_RGBNIR1/cloud_training_4band_RGBNIR1.pkl
```

B G R NIR1: cloud_training_4band_BGRNIR1.csv

```bash
(rapids) Singularity> python rf_driver.py --step train --train-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_BGRNIR1.csv --seed 42 --test-size 0.20 --n-trees 20 --max-features log2 --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_BGRNIR1/cloud_training_4band_BGRNIR1.pkl
2021-10-18 18:33:00; INFO; Open /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_BGRNIR1.csv dataset for training.
2021-10-18 18:33:00; INFO; X size: (799999, 4)
2021-10-18 18:33:00; INFO; Y size:  (799999,)
2021-10-18 18:33:00; INFO; ntrees=20, maxfeat=log2
2021-10-18 18:33:00; INFO; Training model via RAPIDS.
2021-10-18 18:33:01; INFO; init
2021-10-18 18:33:02; INFO; Test Accuracy:  0.99994
2021-10-18 18:33:02; INFO; Test Precision: 0.9998900846348312
2021-10-18 18:33:02; INFO; Test Recall:    0.999990006695514
2021-10-18 18:33:02; INFO; Test F-Score:   0.9999400431689184
2021-10-18 18:33:02; INFO; Model has been saved as /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_BGRNIR1/cloud_training_4band_BGRNIR1.pkl
```

B G R NIR1 FDI SI NDWI: cloud_training_4band_fdi_si_ndwi.csv

```bash
(rapids) Singularity> python rf_driver.py --step train --train-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_fdi_si_ndwi.csv --seed 42 --test-size 0.20 --n-trees 20 --max-features log2 --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_fdi_si_ndwi/cloud_training_4band_fdi_si_ndwi.pkl
2021-10-18 18:34:19; INFO; Open /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_fdi_si_ndwi.csv dataset for training.
2021-10-18 18:34:19; INFO; X size: (799999, 7)
2021-10-18 18:34:19; INFO; Y size:  (799999,)
2021-10-18 18:34:19; INFO; ntrees=20, maxfeat=log2
2021-10-18 18:34:19; INFO; Training model via RAPIDS.
2021-10-18 18:34:21; INFO; init
2021-10-18 18:34:22; INFO; Test Accuracy:  0.99995
2021-10-18 18:34:22; INFO; Test Precision: 0.9999100233939175
2021-10-18 18:34:22; INFO; Test Recall:    0.999990001799676
2021-10-18 18:34:22; INFO; Test F-Score:   0.9999500109975805
2021-10-18 18:34:22; INFO; Model has been saved as /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_fdi_si_ndwi/cloud_training_4band_fdi_si_ndwi.pkl
```

CB B G Y R RE NIR1 NIR2: cloud_training_8band.csv

```bash
(rapids) Singularity> python rf_driver.py --step train --train-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_8band.csv --seed 42 --test-size 0.20 --n-trees 20 --max-features log2 --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_8band/cloud_training_8band.pkl
2021-10-18 18:35:35; INFO; Open /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_8band.csv dataset for training.
2021-10-18 18:35:35; INFO; X size: (799999, 8)
2021-10-18 18:35:35; INFO; Y size:  (799999,)
2021-10-18 18:35:35; INFO; ntrees=20, maxfeat=log2
2021-10-18 18:35:35; INFO; Training model via RAPIDS.
2021-10-18 18:35:37; INFO; init
2021-10-18 18:35:38; INFO; Test Accuracy:  0.999935
2021-10-18 18:35:38; INFO; Test Precision: 0.9998697159808382
2021-10-18 18:35:38; INFO; Test Recall:    1.0
2021-10-18 18:35:38; INFO; Test F-Score:   0.9999348537466612
2021-10-18 18:35:38; INFO; Model has been saved as /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_8band/cloud_training_8band.pkl
```

CB B G Y R RE NIR1 NIR2 FDI SI NDWI: cloud_training_8band_fdi_si_ndwi.csv

```bash
(rapids) Singularity> python rf_driver.py --step train --train-csv /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_8band_fdi_si_ndwi.csv --seed 42 --test-size 0.20 --n-trees 20 --max-features log2 --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_8band_fdi_si_ndwi/cloud_training_8band_fdi_si_ndwi.pkl
2021-10-18 18:36:31; INFO; Open /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_8band_fdi_si_ndwi.csv dataset for training.
2021-10-18 18:36:32; INFO; X size: (799999, 11)
2021-10-18 18:36:32; INFO; Y size:  (799999,)
2021-10-18 18:36:32; INFO; ntrees=20, maxfeat=log2
2021-10-18 18:36:32; INFO; Training model via RAPIDS.
2021-10-18 18:36:33; INFO; init
2021-10-18 18:36:34; INFO; Test Accuracy:  0.999945
2021-10-18 18:36:34; INFO; Test Precision: 0.9999000549697666
2021-10-18 18:36:34; INFO; Test Recall:    0.999990004597885
2021-10-18 18:36:34; INFO; Test F-Score:   0.9999450277609807
2021-10-18 18:36:34; INFO; Model has been saved as /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_8band_fdi_si_ndwi/cloud_training_8band_fdi_si_ndwi.pkl
```

### Random Forest Visualization

```bash
python rf_driver.py --step vis --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_RGBNIR1/cloud_training_4band_RGBNIR1.pkl --bands R G B NIR1
```

### Random Forest Prediction

```bash
python rf_driver.py --step predict --output-pkl /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_BGRNIR1/cloud_training_4band_BGRNIR1.pkl --output-dir /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_RGBNIR1/predictions --rasters --window-size 8192
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
