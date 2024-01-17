# Perform Non-Rigid Registration Using Neural Tracking + DynamicFusion

# Run
```
python3 fusion.py --datadir <path-to-folder>
``` 

Each datadir folder contains
1. color: folder containing rgb images as 00%d.png
2. depth: folder containing depth image as 00%d.png 
3. intrinsics.txt: 4x4 camera intrinsic matrix 
4. graph_config.json (optional): parameters to generate graph 

# Dependencies 
```
pip install pynput pycuda cupy pykdtree
```  

# Full installation with all dependencies

## 1.a Conda/Pip install - importing my environment
```shell
conda env create -f environment.yml
# conda env export | grep -v "^prefix: " > environment.yml
```

or

## 1.b Conda/Pip install - commands

```shell
# Recreating yourself
conda create -n occlusionfuf python=3.10 && conda activate occlusionfuf
# chose that version or pytorch and python to match with conda built version of pytorch3d
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# install pytorch3d (painfull)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py310_cu118_pyt210.tar.bz2 #it fails: conda install pytorch3d::pytorch3d

# install other packages
conda install pyg::pytorch-scatter pyg::pytorch-sparse pyg::pytorch-cluster pyg::pytorch-spline-conv pyg::pyg
conda install conda-forge::kornia conda-forge::cupy conda-forge::pycuda conda-forge::pykdtree

pip install scikit-image numba tensorboardx nibabel easydict open3d opencv-python pybind11 plyfile pynput
```

## 2. Building dependencies

```shell

# NeuralNRT
cd csrc
python setup.py install

#MVRegC
cd NonRigidICP/cxx
python setup.py install

#Lepard
cd lepard/cpp_wrappers
./compile_wrappers.sh

# Lietorch (conda install -c lietorch lietorch leads to symbol not found)
#cd NonRigidICP/lietorch
#git clone https://github.com/princeton-vl/lietorch.git
export CUDA_HOME=/usr/local/cuda #otherwise cuda.h missing error
#export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
#export PATH=${CUDA_HOME}/bin:${PATH}
python setup.py install
# todo try instead: pip install git+https://github.com/princeton-vl/lietorch.git
```


# Files: 
- [ ] fusion.py 					// Main file for fusion based on dynamic fusion 

- [ ] frameLoader.py 				// Loads RGBD frames 
- [ ] tsdf.py 						// Class to store fused model and its volumetric represenation
- [ ] embedded_deformation_graph.py // Class to strore + update graph data 
- [ ] warpfield.py 					// Stores skinning weights,deformation and transformation parameters of each timestep 
- [ ] log/visualizer.py 			// Base class to visualize details using open3d, plotly, matplotlib  
- [ ] run_model.py 					// Run Neural Tracking to estimate transformation parameters

- [ ] run_tests.py 					// Run tests for checking if everything in module is working correctly on indepedant tests. 
- [ ] evaluate.py 					// Run different evaluations such as deformation error and geometric error defined in DeepDeform

# Todo:- 
2. Update max depth in fusion.py, currently sending maximum depth of first frame. Can do better. Need to calculate the bound of the segmented object in complete video
3. Refactor TSDF code, one bigger kernel better that smaller kernel but we can create a folder dedicated to TSDF for better understanding
7. Deleting optical flow data for now. Might need to saved later. 
14. Visvualize num_iterations hyperparamater for erosotion while adding new nodes 
16. Replace direct get such as tsdf.canonical_model, warpfiled.deformed_model with get_canonical_model, get_deformed_graph_nodes 
18. Update logger to show colored log, defined in self.log 
19. Add to warpfield clear, is_deformed defined in tsdf, needs to be updated 
21. Write save/write functions after all modules are working.
22. Clean ARAP function in run_model, replace graph,warpfield, self.graph
23. Remove canonical_node_cuda, sending canonical volume data doesn't work.
24. No checks for valid invalid nodes during optical flow. Basically in neural tracking using pixel anchors and weights we know valid source and target points. Current implementation doesn't use that. 
25. No interpolation while getting optical flow values currently, add interpolation for pixels  
26. Check values deformed graph node poisitions using optical flow flow
27. Add tests to measure deformation of nodes using OcclusionFusion Model 
28. Add tests to see write values are getting passed to occlusion fusion and optimization
29. Refactor motion_complete_model_test.py & optimization_tests.py, each test has only slight changes described in commnents. Create a class instead of functions, write each test seperately.   
30. Merge vis_utils.py and vis_open3d.py
# Tests: 
```
python3 ./tests.sh // Run Tests to see if working perfectly
```
The file runs the following tests
1. Check if deformation happening correctly
2. Check with and without gpu 
3. Check if ARAP is working
4. Check optimization

Test TODO:- 
1. Check normals are getting deformed perfectly. 
4. Evaluate error on DeformingThings4D 

# Evaluate 
```
python3 evaluate.py 
```
Calculate evaluation scores on:
1. DeepDeform
2. DeformingThings4D 
3. DonkeyDoll Dataset
4. Human Dataset 

# Licesne 


# Acknowlegement
1. Neural Tracking
2. OcclusionFusion 