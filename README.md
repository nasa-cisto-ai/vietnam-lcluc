# Vietnam LCLUC

Vietnam LCLUC Deep Learning Repository

## Business Case

The following repository stores several experiments for the task of instance and semantic
segmentation of LCLUC in very high-resolution satellite imagery. Many of the instructions
listed below are guided towards utilizing GSFC NASA Center for Climate Simulation (NCCS)
computing resources, particularly the PRISM GPU cluster.

A system with NVIDIA GPUs is required to run the scripts located in this repository.

- projects/vietnam_clouds: semantic segmentation of clouds
- projects/vietnam_crop: multi-class segmentation of land cover classes
- projects/vietnam_trees: semantic segmentation of trees

## Quickstart

```bash
module load singularity
singularity build --sandbox /lscratch/jacaraba/container/tf-container docker://gitlab.nccs.nasa.gov:5050/nccs-ci/nccs-containers/rapids-tensorflow/nccs-ubuntu20-rapids-tensorflow
singularity shell --nv -B /att,/lscratch,/adapt/nobackup/projects/ilab /lscratch/jacaraba/container/tf-container
```

## Final LCLUC Product

Data products:

- cloud mask
- tree/shadow
- crop/field

Individual land use classes:

- 0: Growing Rice/Row Crop 
- 1: Sparse/Barren 
- 2: Intercrop 
- 3: Burned 
- 4: Water 
- 5: Roads/Urban/Other (Built up) 
- 6: Other veg (include reeds/marshy areas)

Trees and buildings:

- 0: trees
- 1: water
- 2: build
- 3: shadow
- 4: other 
- 5: tree

At the end, we care about these classes:

- From crop: 0-growing active, 4-water
- From trees: 0-tree, 3-shadow, 1-water
- From clouds: 1-cloud

## References

[1] Chollet, François; et all, Keras, (2015), GitHub repository, <https://github.com/keras-team/keras>.
[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, <https://github.com/pytorch/pytorch>.
[3] Google Brain Team; et all, TensorFlow, (2015), GitHub repository, <https://github.com/tensorflow/tensorflow>.
