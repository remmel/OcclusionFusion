# Creating data from deformingThings4D 
# Given raw data from lepard create data in the format for OcclusionFusion/NeuralTracking

import os 
import sys
import numpy as np
from skimage import io

savedir = "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test"

def create_sample(raw_depth_path):
	if raw_depth_path[-1] == '/': raw_depth_path = raw_depth_path[:-1] 
	assert os.path.isdir(os.path.join(raw_depth_path,"depth"))


	sample_name = os.path.basename(raw_depth_path)



	for cams in os.listdir(raw_depth_path):
		if "intr" in cams: # Found intrensic matrix 
			cam_name = cams.replace("intr.txt","")
			savepath = os.path.join(savedir,f"{sample_name}_{cam_name}")
			os.makedirs(savepath,exist_ok=True)
			
			print(f"Saving under name:{savepath}")
			
			os.system(f"cp {os.path.join(raw_depth_path,cams)} {os.path.join(savepath,'intrinsics.txt')}")
			os.system(f"cp {os.path.join(raw_depth_path,cams.replace('intr','extr'))} {os.path.join(savepath,'extrinsics.txt')}")
			
			os.makedirs(os.path.join(savepath,"depth"),exist_ok=True)
			for depth_file in os.listdir(os.path.join(raw_depth_path,'depth')):
				if cam_name in depth_file:
					os.system(f"cp {os.path.join(raw_depth_path,'depth',depth_file)}\
						{os.path.join(savepath,'depth',depth_file.split('_')[-1])}")

			os.makedirs(os.path.join(savepath,"mask"),exist_ok=True) 
			os.makedirs(os.path.join(savepath,"color"),exist_ok=True)
			# Here we are making the assumption that all objects are already segmented and so mask = depth > 0

			for depth_file in os.listdir(os.path.join(savepath,"depth")):
				print(os.path.join(savepath,"depth",depth_file))
				depth = io.imread(os.path.join(savepath,"depth",depth_file))
				mask = (255*(depth>0)).astype(np.uint8)

				io.imsave(os.path.join(savepath,"mask",depth_file),mask)

				color = np.zeros((depth.shape[0],depth.shape[1],3),dtype=np.uint8)
				color[depth>0,0] = 224
				color[depth>0,1] = 0
				color[depth>0,2] = 125

				io.imsave(os.path.join(savepath,"color",depth_file),color)



if __name__ == "__main__": 
	raw_depth_path = sys.argv[1]
	create_sample(raw_depth_path) 
