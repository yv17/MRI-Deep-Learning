## Deep learning T2 mapping using phase-cycled bSSFP signals/images

This project aims to achieve faster and noise-robust T2 mapping using phase-cycled bSSFP signals via deep learning methods, as alternative to parametric approach [PLANET](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26717).  

Through [existing bSSFP simulation](https://github.com/yv17/FYP-Python/blob/master/bssfp_data_generator/bssfp.py), phase-cycled bSSFP signals can be presented in form of arrays or images. Either form can be used to train a corresponding type of supervised regression neural network to predict T2 values:

i)  [Voxelwise Regression](https://github.com/yv17/FYP-Python/blob/master/voxel_reg.ipynb)

ii) [Image-to-image Regression](https://github.com/yv17/FYP-Python/blob/master/voxel_reg.ipynb)
