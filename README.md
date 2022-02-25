# Vietnam LCLUC

Vietnam LCLUC AI/ML Repository

## Business Case

The following repository stores several experiments for the task of instance and semantic
segmentation of LCLUC in very high-resolution satellite imagery. Many of the instructions
listed below are guided towards utilizing GSFC NASA Center for Climate Simulation (NCCS)
computing resources, particularly the PRISM GPU cluster.

A system with NVIDIA GPUs is required to run the scripts located in this repository.

- projects/cloud_mask: semantic segmentation of clouds
- projects/crop_mapping: multi-class segmentation of land cover classes
- projects/tree_cover: semantic segmentation of trees

## Final LCLUC Product

cloud mask
tree/shadow
crop/field

0: Growing Rice/Row Crop 
1: Sparse/Barren 
2: Intercrop 
3: Burned 
4: Water 
5: Roads/Urban/Other (Built up) 
6: Other veg (include reeds/marshy areas)

Trees and buildings

0: trees
1: water
2: build
3: shadow
4: other 
5: tree

We care about

From crop: 0-growing active, 4-water
From trees: 0-tree, 3-shadow, 1-water
From clouds: 1-cloud

## References

[1] Chollet, François; et all, Keras, (2015), GitHub repository, <https://github.com/keras-team/keras>.
[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, <https://github.com/pytorch/pytorch>.
[3] Google Brain Team; et all, TensorFlow, (2015), GitHub repository, <https://github.com/tensorflow/tensorflow>.
