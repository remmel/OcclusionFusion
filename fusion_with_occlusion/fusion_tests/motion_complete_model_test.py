import os 
import numpy as np

# Creating data from deformingThings4D 
# 1. Create movement of graph nodes from deformingThings4D 

# Import Fusion Modules 
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf  
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  
from vis import get_visualizer # Visualizer 
from run_model import Deformnet_runner # Neural Tracking + ARAP Moudle 

# Test imports 
from .test_utils import Dict2Class




datapath = "/media/srialien/Elements/AT-Datasets/DeformingThings4D/DeformingThings4D/animals"

def test1(random=True,use_gpu=True):

	# Load random anime file 

	# create graph from mesh 

	# Track its position 

	# Predict position of visible nodes from motion completea net

	return 