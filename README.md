## Deep learning T2 mapping using phase-cycled bSSFP signals/images

This project aims to achieve faster T2 mapping with minimal error and inputs via two machine learning methods as alternatives to [PLANET](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26717) algorithm.  
Through [existing bSSFP simulation](https://github.com/yv17/FYP-Python/blob/master/bssfp_data_generator/bssfp.py), two types of phase-cycled bSSFP data can be generated: 

(i)  Sequence of complex signals, in which the magnitude of each complex signal is the individual pixel of a magnitude image

(ii) Series of magnitude images, with each image constituted from a matrix of signal magnitudes. 

Either type of phase-cycled bSSFP data can be used to train a corresponding type of supervised regression neural network to predict T2 values:

i)  [Voxelwise regression](https://github.com/yv17/FYP-Python/blob/master/voxel_reg.ipynb)

ii) [Image-to-image regression](https://github.com/yv17/FYP-Python/blob/master/voxel_reg.ipynb)
