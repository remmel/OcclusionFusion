# Given a series of point clouds, use lepards for sceneflow and NICP between 2 point clouds 

import os 
import numpy as np
import torch
import json

from tsdf import TSDFVolume # Create main TSDF module where the 3D volume is stored
from vis import get_visualizer # Visualizer 
from frame_loader import RGBDVideoLoader
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf 
from run_lepard import Lepard_runner # SceneFlow module 
from NonRigidICP.model.registration_fusion import Registration as PytorchRegistration
from lepard.inference import Lepard	
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  
from .test_utils import Dict2Class


def dict_to_numpy(model_data):	
	# Convert every torch tensor to np.array to save results 
	for k in model_data:
		if type(model_data[k]) == torch.Tensor:
			model_data[k] = model_data[k].cpu().data.numpy()
			# print(k,model_data[k].shape)
		elif type(model_data[k]) == list:
			for i,r in enumerate(model_data[k]): # Numpy does not handle variable length, this will produce error 
				if type(r) == torch.Tensor:
					model_data[k][i] = model_data[k][i].cpu().data.numpy()
					# print(k,i,model_data[k][i].shape)
		elif type(model_data[k]) == dict:
			for r in model_data[k]:
				if type(model_data[k][r]) == torch.Tensor:
					model_data[k][r] = model_data[k][r].cpu().data.numpy()
					# print(k,r,model_data[k][r].shape)

	return model_data


def test1(use_gpu=True):
	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/dinoET_act2_cam2",\
		# "datadir": "/media/srialien/2e32b92f-33a0-4f62-a89f-445621302f09/DeepDeform_Tests/dinoET_act2_cam2",\
		# "datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/mannequin_Running_cam1",\
		"skip_rate":1,
		"voxel_size": 0.02})
	vis = get_visualizer(fopt)
	
	frame_loader = RGBDVideoLoader(fopt.datadir)

	source_frame_data = frame_loader.get_source_data(fopt.source_frame)


	lepard = Lepard(os.path.join(os.getcwd(),"../lepard/configs/test/4dmatch.yaml"))

	source_mask = source_frame_data["im"][-1] > 0	
	source_pcd = source_frame_data["im"][3:,source_mask].T
	mask_indices = np.where(source_mask)
	bbox = [np.min(mask_indices[1]),np.min(mask_indices[0]),np.max(mask_indices[1]),np.max(mask_indices[0])] # x_min, y_min, x_max,y_max\n
	
	# intrinsics
	cam_intr = np.eye(3)
	cam_intr[0, 0] = source_frame_data["intrinsics"][0]
	cam_intr[1, 1] = source_frame_data["intrinsics"][1]
	cam_intr[0, 2] = source_frame_data["intrinsics"][2]
	cam_intr[1, 2] = source_frame_data["intrinsics"][3]


	max_depth = source_frame_data["im"][-1].max()

	# Create a new tsdf volume
	tsdf = TSDFVolume(bbox,max_depth+1, source_frame_data["intrinsics"], fopt,vis)
	vis.tsdf = tsdf
	tsdf.integrate(source_frame_data)

	graph = EDGraph(tsdf,vis,source_frame_data)

	tsdf.graph = graph 		# Add graph to tsdf		
	vis.graph  = graph 		# Add graph to visualizer 

	warpfield = WarpField(graph,tsdf,vis)

	tsdf.warpfield = warpfield  # Add warpfield to tsdf
	vis.warpfield = warpfield   # Add warpfield to visualizer

	gradient_descent_optimizer = PytorchRegistration(source_pcd,graph,warpfield,cam_intr,vis)		
	warpfield.optimizer = gradient_descent_optimizer
	# gradient_descent_optimizer.config.iters = 150


	for i in range(fopt.skip_rate,len(frame_loader),fopt.skip_rate):
		print(f"Registering frame:{i}")
		target_frame_data = frame_loader.get_target_data(fopt.source_frame+i,source_frame_data["cropper"])



		target_mask = target_frame_data["im"][-1] > 0	
		target_pcd = target_frame_data["im"][3:,target_mask].T


		scene_flow,corresp,valid_verts = lepard(source_pcd,target_pcd)
		target_matches = source_pcd.copy()
		target_matches[valid_verts] += scene_flow[valid_verts]


		print(source_pcd.shape,target_matches.shape)

		scene_flow_data = {'source':source_pcd,'scene_flow': scene_flow,"valid_verts":valid_verts,"target_matches":target_matches,'landmarks':corresp}	
		optical_flow_data = {'source_id':fopt.source_frame,'target_id':fopt.source_frame+i}

		estimated_transformations = gradient_descent_optimizer.optimize(optical_flow_data,
											scene_flow_data,
											target_frame_data)

		estimated_transformations = dict_to_numpy(estimated_transformations)


		save_convergance_info(fopt,estimated_transformations["convergence_info"],optical_flow_data["source_id"],optical_flow_data["target_id"])

		# Update warpfield parameters, warpfield maps to target frame  
		warpfield.update_transformations(estimated_transformations)

		# Register TSDF, tsdf maps to target frame  
		# tsdf.update(target_frame_data["im"],target_frame_data["id"])
		target_pcd = vis.get_target_RGBD()

		tsdf.integrate(target_frame_data)

		# self.vis.plot_skinned_model()

		# Add new nodes to warpfield and graph if any
		# update = self.warpfield.update_graph() 
		update = False
		
		# if source_frame > 0:		
		# vis.show_zoomed_in(scene_flow_data,fopt.source_frame,debug=False) # plot registration details 
		vis.show(scene_flow_data,fopt.source_frame,debug=False) # plot registration details 

		tsdf.clear()



def test2(use_gpu=True): # Passing canoconical model instead of source point cloud 
	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/foxWDFS_Actions2_cam2",\
		# "datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/mannequin_Running_cam1",\
		"skip_rate":15,
		"voxel_size": 0.02})
	vis = get_visualizer(fopt)
	
	frame_loader = RGBDVideoLoader(fopt.datadir)

	source_frame_data = frame_loader.get_source_data(fopt.source_frame)

	lepard = Lepard(os.path.join(os.getcwd(),"../lepard/configs/test/4dmatch.yaml"))

	source_mask = source_frame_data["im"][-1] > 0	
	source_pcd = source_frame_data["im"][3:,source_mask].T

	mask_indices = np.where(source_mask)
	bbox = [np.min(mask_indices[1]),np.min(mask_indices[0]),np.max(mask_indices[1]),np.max(mask_indices[0])] # x_min, y_min, x_max,y_max\n

	# intrinsics
	cam_intr = np.eye(3)
	cam_intr[0, 0] = source_frame_data["intrinsics"][0]
	cam_intr[1, 1] = source_frame_data["intrinsics"][1]
	cam_intr[0, 2] = source_frame_data["intrinsics"][2]
	cam_intr[1, 2] = source_frame_data["intrinsics"][3]


	max_depth = source_frame_data["im"][-1].max()

	# Create a new tsdf volume
	tsdf = TSDFVolume(bbox, max_depth+1, source_frame_data["intrinsics"], fopt,vis)
	vis.tsdf = tsdf
	tsdf.integrate(source_frame_data)

	graph = EDGraph(tsdf,vis,source_frame_data)
	vis.graph  = graph 		# Add graph to visualizer 

	warpfield = WarpField(graph,tsdf,vis)

	tsdf.warpfield = warpfield  # Add warpfield to tsdf
	vis.warpfield = warpfield   # Add warpfield to visualizer

	source_verts = tsdf.get_canonical_model()[0]

	gradient_descent_optimizer = PytorchRegistration(source_verts,graph,warpfield,cam_intr,vis)		
	# gradient_descent_optimizer.config.iters = 150


	for i in range(fopt.skip_rate,len(frame_loader),fopt.skip_rate):
		print(f"Registering frame:{i}")
		target_frame_data = frame_loader.get_target_data(fopt.source_frame+i,source_frame_data["cropper"])




		target_mask = target_frame_data["im"][-1] > 0	
		target_pcd = target_frame_data["im"][3:,target_mask].T

		source_verts = tsdf.get_canonical_model()[0]


		scene_flow,corresp,valid_verts = lepard(source_verts,target_pcd)
		target_matches = source_verts.copy()
		target_matches[valid_verts] += scene_flow[valid_verts]


		print(source_verts.shape,target_matches.shape)

		scene_flow_data = {'source':source_verts,'scene_flow': scene_flow,"valid_verts":valid_verts,"target_matches":target_matches,'landmarks':corresp}	
		optical_flow_data = {'source_id':fopt.source_frame,'target_id':fopt.source_frame+i}

		estimated_transformations = gradient_descent_optimizer.optimize(optical_flow_data,
											scene_flow_data,
											target_frame_data)

		estimated_transformations = dict_to_numpy(estimated_transformations)


		save_convergance_info(fopt,estimated_transformations["convergence_info"],optical_flow_data["source_id"],optical_flow_data["target_id"])

		# Update warpfield parameters, warpfield maps to target frame  
		warpfield.update_transformations(estimated_transformations)

		# Register TSDF, tsdf maps to target frame  
		tsdf.integrate(target_frame_data)

		# self.vis.plot_skinned_model()

		# Add new nodes to warpfield and graph if any
		# update = self.warpfield.update_graph() 
		update = False
		print("canoconical model before:",tsdf.get_canonical_model()[0].shape)
		gradient_descent_optimizer.update(tsdf.get_canonical_model()[0])
		print("canoconical model after:",tsdf.get_canonical_model()[0].shape)


		# if source_frame > 0:		
		vis.show(scene_flow_data,fopt.source_frame,debug=False) # plot registration details 

		tsdf.clear()



def save_convergance_info(fopt,convergence_info_dict,source_id,target_id):
	savepath = os.path.join(fopt.datadir,"results",f"optimization_convergence_info_{source_id}_{target_id}.json")
	with open(savepath, "w") as f:
		json.dump(convergence_info_dict, f)