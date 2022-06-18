import os
import json 
from .visualizer import Visualizer

import ipyvolume as ipv

class VisualizerIpyVolume(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)	

	def plot_tsdf_volume(self):
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		tsdf,color,weight = self.tsdf.get_volume()	
		abc = ipv.quickvolshow(tsdf,opacity=0.03, level_width=0.1)	

		return abc
		