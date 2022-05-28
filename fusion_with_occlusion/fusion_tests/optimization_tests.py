# This folder contains optimization tests for OcclusionFusion/NeuralTracking

import os

# Python Imports
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Import Fusion Modules 
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf  
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  
from vis import get_visualizer # Visualizer 
from run_model import Deformnet_runner # Neural Tracking + ARAP Moudle 

# Test imports 
from .test_utils import Dict2Class

from .ssdr import SSDR


# Base TSDF class (instead of using a volumetric representation using Mesh)
class TSDFMesh:
	def __init__(self,fopt,mesh):
		"""
			Given a mesh TSDF mesh contains the gr data. 
			@params:
				fopt: 


		"""
		self.fopt = fopt
		# Estimate normals for future use
		self.frame_id = fopt.source_frame

		self.mesh = mesh

	def set_data(self,trajectory,trajectory_normals):
		self.trajectory = trajectory
		self.trajectory_normals = trajectory_normals

	def get_mesh(self):
		faces = np.asarray(self.mesh.triangles)         

		return self.trajectory[0],faces,self.trajectory_normals[0],None

	def get_canonical_model(self):
		return self.get_mesh()	

	def check_visibility(self,points):
		return np.ones(points.shape[0],dtype=np.bool),np.zeros(points.shape[0],dtype=np.bool)

	def get_source_data(self):
		return self.trajectory[self.frame_id], self.trajectory_normals[self.frame_id]

	def get_target_data(self):
		return self.trajectory[self.frame_id+self.fopt.skip_rate], self.trajectory_normals[self.frame_id+self.fopt.skip_rate]


class TestModel(Deformnet_runner):
	def __init__(self,vis,fopt):
		super().__init__(vis,fopt)

	def get_predicted_location(self,optical_flow_data, vertices,valid_verts,intrinsics):

		N = vertices.shape[0] 
		valid_mask = valid_verts & np.ones(N,dtype=np.bool)

		source_points,source_points_normals = self.tsdf.get_source_data()
		target_points,target_points_normals = self.tsdf.get_target_data()
		
		fx = intrinsics[0]
		fy = intrinsics[1]
		cx = intrinsics[2]
		cy = intrinsics[3]


		source_points_px = source_points[:,0]*fx/source_points[:,2] + cx
		source_points_py = source_points[:,1]*fy/source_points[:,2] + cy

		target_points_px = target_points[:,0]*fx/target_points[:,2] + cx
		target_points_py = target_points[:,1]*fy/target_points[:,2] + cy
		# self.vis.plot_opticalflow(source_points_px,source_points_py,target_points_px,target_points_py,debug=True)

		return valid_mask,target_points,target_points_normals,target_points_px,target_points_py
		

def test1(use_gpu=True):
	"""
		Rotating and translating sphere.
	"""  	

	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir":"/media/srialien/Elements/DeepDeform_Tests/sphere",\
		"skip_rate":1})
	vis = get_visualizer(fopt)




	# Create sphere
	mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
	# center = np.array([-0.3,0,-2])
	# mesh_sphere.translate(center)
	mesh_sphere.paint_uniform_color([1.0,0,0]) # Red color sphere


	# Generate data
	trajectory = [np.asarray(mesh_sphere.vertices)]
	
	mesh_sphere.compute_vertex_normals(normalized=True)
	trajectory_normals = [np.asarray(mesh_sphere.vertex_normals)]

	rotations = [np.eye(3)]
	translations = [np.zeros(3)]

	rotmat = R.from_rotvec(np.random.random(3)*np.pi).as_matrix().astype(np.float32)
	trans = [0,0,0]
	for _ in range(0,100):
		
		# Create random rotations to deform object
		mesh_sphere.rotate(rotmat)
		mesh_sphere.translate(trans)

		center = mesh_sphere.get_center()
		print("Center:",center)

		rotations.append(rotmat@rotations[-1])
		translations.append(rotmat@translations[-1] - rotmat@center + center + trans)

		trajectory.append(np.array(mesh_sphere.vertices))

		mesh_sphere.compute_vertex_normals(normalized=True)
		trajectory_normals.append(np.array(mesh_sphere.vertex_normals))


	rotations = np.array(rotations)
	translations = np.array(translations)	
	trajectory = np.array(trajectory,dtype=np.float32)
	trajectory_normals = np.array(trajectory_normals,dtype=np.float32)

	mesh_sphere.vertices = o3d.utility.Vector3dVector(trajectory[0])


	# Create fusion modules 
	tsdf = TSDFMesh(fopt,mesh_sphere)
	tsdf.set_data(trajectory,trajectory_normals)

	graph = EDGraph(tsdf,vis)
	warpfield = WarpField(graph,tsdf,vis)
	model = TestModel(vis,fopt)	

	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield
	
	# Add modules to model
	model.graph = graph
	model.warpfield = warpfield
	model.tsdf = tsdf
	model.vis = vis

	# Update tsdf 
	tsdf.reduced_graph_dict = {"valid_nodes_mask":np.ones(graph.nodes.shape[0],dtype=np.bool)}

	# Create sphere
	# vis.plot_skinned_model(debug=True)	




	T = tsdf.trajectory.shape[0]

	intrinsics = np.array([1,1,0,0],dtype=np.float32)

	for target_frame_id in range(fopt.source_frame,T-1,fopt.skip_rate):
		
		print(f"Frame id:{target_frame_id}")
		# Optical flow data not required, returned by tsdf for the test
		optical_flow_data = {"source_id":target_frame_id - fopt.skip_rate,"target_id":target_frame_id}

		# Get deformed graph positions  
		source_vertices = tsdf.trajectory[target_frame_id - fopt.skip_rate]
		source_graph_nodes = source_vertices[graph.node_indices]
		
		target_vertices = tsdf.trajectory[target_frame_id]
		target_graph_nodes = target_vertices[graph.node_indices]

		node_motion_data = target_graph_nodes - source_graph_nodes, np.ones(graph.nodes.shape[0],dtype=np.float32)

		estimated_transformations = model.optimize(optical_flow_data,node_motion_data,intrinsics,None)


		# Update warpfield parameters, warpfield maps to target frame  
		warpfield.update_transformations(estimated_transformations)

		# TSDF gets updated last
		tsdf.frame_id = target_frame_id		


		# Evaluate results 
		rec_err_per_sample = np.linalg.norm(warpfield.deformed_nodes - target_vertices[graph.node_indices],axis=1)
		print("Reconstructon Error:",np.mean(rec_err_per_sample))

		# high_error_visible_nodes = np.where(rec_err_per_sample > 1e-3)[0]
		# print("Problembatic Nodes:",high_error_visible_nodes)

		# print("Reconstruction Differnce per sample:",rec_err_per_sample[high_error_visible_nodes])


		# gr_translation = -tsdf.rotations[target_frame_id]@center + center + tsdf.translations[target_frame_id]

		# print("Gr Translation:",gr_translation)
		# print("High Error Node Translations:",warpfield.translations[high_error_visible_nodes])
		# print("Difference in translations:", warpfield.translations[high_error_visible_nodes] - gr_translation[None,:])

		# print("Gr rotations:",tsdf.rotations)
		# print("High Error Node rotations:",warpfield.rotations[high_error_visible_nodes])




		# Plot deformed graph with different color 
		init_graph = vis.get_rendered_graph(graph.nodes,graph.edges) # Initial graph 
		if not hasattr(vis,'bbox'):
			vis.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

		bbox = vis.bbox

		deformed_graph = vis.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox)

		target_graph = vis.get_rendered_graph(target_graph_nodes,graph.edges,trans=np.array([2,0,0])*bbox) # Actualy position 

		deformed_model = vis.get_deformed_model_from_tsdf(trans=np.array([1,0,0])*bbox)

		vis.plot(init_graph + deformed_graph + target_graph,"Deformed Graph",False)		


def transform_matrix2rotation_translation(relative_transform_matrix): 
	T,N = relative_transform_matrix.shape
	T //=4
	N //= 4

	rotations = np.zeros((T,N,3,3),dtype=np.float32)
	translations = np.zeros((T,N,3),dtype=np.float32)

	for t in range(T):
		for n in range(N):
			for i in range(3):
				for j in range(3):
					rotations[t,n,i,j] = relative_transform_matrix[4*t+i,4*n+j]
				translations[t,n,i] = relative_transform_matrix[4*t+i,4*n+3]

	return rotations,translations

def test2(use_gpu=True):
	"""
		Tests on Anime Files from deforming things4D
	"""  
	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/mannequin_Running",\
		"skip_rate":1})
	vis = get_visualizer(fopt)

	ssdr = SSDR()
	filname = fopt.datadir.split("/")[-1] + '.anime'
	trajectory,faces = ssdr.load_anime_file(os.path.join(fopt.datadir,filname))


	mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(trajectory[0]),
			o3d.utility.Vector3iVector(faces))

	# Compute normals 
	trajectory_normals = []
	for traj in trajectory:
		mesh.vertices = o3d.utility.Vector3dVector(traj)
		mesh.compute_vertex_normals(normalized=True)
		normals = np.array(mesh.vertex_normals)
		trajectory_normals.append(normals)

	trajectory_normals = np.array(trajectory_normals)	


	# Create fusion modules 
	tsdf = TSDFMesh(fopt,mesh)
	tsdf.set_data(trajectory,trajectory_normals)


	graph = EDGraph(tsdf,vis)
	warpfield = WarpField(graph,tsdf,vis)
	model = TestModel(vis,fopt)	
	# model.model.gn_num_iter = 1
	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield
	
	# Add modules to model
	model.graph = graph
	model.warpfield = warpfield
	model.tsdf = tsdf
	model.vis = vis

	tsdf.reduced_graph_dict = {"valid_nodes_mask":np.ones(graph.nodes.shape[0],dtype=np.bool)}

	deformed_vertices,deformed_normals,vert_anchors,vert_weights,valid_verts = warpfield.deform_mesh(trajectory[0],trajectory_normals[0])
	gr_deformation,relative_transform_matrix,rmse_error = ssdr.get_transforms(trajectory,faces,vert_anchors,vert_weights,graph.nodes)
	gr_rotations, gr_translations = transform_matrix2rotation_translation(relative_transform_matrix)
	# print("Relative Transforms:",relative_transform_matrix)
	print("Rsme Error:",rmse_error)
	# print("Gr Rotations:",gr_rotations[:,0])
	print("Gr Translations:",gr_translations[:,0])



	T = tsdf.trajectory.shape[0]

	intrinsics = np.array([500,500,250,250],dtype=np.float32)

	for target_frame_id in range(fopt.source_frame+fopt.skip_rate,T-1,fopt.skip_rate):
		
		print(f"Frame id:{target_frame_id}")
		# Optical flow data not required, returned by tsdf for the test
		optical_flow_data = {"source_id":target_frame_id - fopt.skip_rate,"target_id":target_frame_id}

		# # Get deformed graph positions  
		source_vertices = tsdf.trajectory[target_frame_id - fopt.skip_rate]
		source_graph_nodes = source_vertices[graph.node_indices]
		
		target_vertices = tsdf.trajectory[target_frame_id]
		target_graph_nodes = target_vertices[graph.node_indices]

		node_motion_data = target_graph_nodes - source_graph_nodes, np.ones(graph.nodes.shape[0],dtype=np.float32)

		# Setting gr as input to optimization to create baseline 
		# warpfield.rotations = gr_rotations[target_frame_id]
		# warpfield.translations = gr_translations[target_frame_id]
		# warpfield.deformed_nodes = np.array([gr_rotations[target_frame_id,i]@graph.nodes[i] for i in range(graph.num_nodes)]) + gr_translations[target_frame_id]
		# warpfield.frame_id = optical_flow_data["source_id"]

		scene_flow_data = {'source':source_vertices,'scene_flow': target_vertices - source_vertices ,"valid_verts":np.ones(target_vertices.shape[0],dtype=np.bool),"target_matches":target_vertices}	


		estimated_transformations = model.optimize(optical_flow_data,node_motion_data,scene_flow_data,intrinsics,None,gr_deformation=gr_deformation[target_frame_id])

		# Update warpfield parameters, warpfield maps to target frame  
		warpfield.update_transformations(estimated_transformations)
		# TSDF gets updated last
		tsdf.frame_id = target_frame_id		


		# Evaluate results 
		rec_err_per_sample = np.linalg.norm(warpfield.deformed_nodes - target_vertices[graph.node_indices],axis=1)
		print("Reconstructon Error:",np.mean(rec_err_per_sample))

		print("Original Error:",np.mean(np.linalg.norm(gr_deformation[target_frame_id,graph.node_indices] - target_vertices[graph.node_indices],axis=1)))
		# high_error_visible_nodes = np.where(rec_err_per_sample > 1e-3)[0]
		# print("Problembatic Nodes:",high_error_visible_nodes)

		# print("Reconstruction Differnce per sample:",rec_err_per_sample[high_error_visible_nodes])


		# gr_translation = -tsdf.rotations[target_frame_id]@center + center + tsdf.translations[target_frame_id]

		# print("Gr Translation:",gr_translation)
		# print("High Error Node Translations:",warpfield.translations[high_error_visible_nodes])
		# print("Difference in translations:", warpfield.translations[high_error_visible_nodes] - gr_translation[None,:])

		# print("Gr rotations:",tsdf.rotations)
		# print("High Error Node rotations:",warpfield.rotations[high_error_visible_nodes])




		# Plot deformed graph with different color 
		init_graph = vis.get_rendered_graph(graph.nodes,graph.edges) # Initial graph 
		if not hasattr(vis,'bbox'):
			vis.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

		bbox = vis.bbox

		deformed_graph = vis.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox)

		target_graph = vis.get_rendered_graph(target_graph_nodes,graph.edges,trans=np.array([2,0,0])*bbox) # Actualy position 
		image_name = f"optimization_test2_ssdr_{target_frame_id:02d}.png"
		vis.plot(init_graph + deformed_graph + target_graph,"Deformed Graph",False,savename=image_name)		
