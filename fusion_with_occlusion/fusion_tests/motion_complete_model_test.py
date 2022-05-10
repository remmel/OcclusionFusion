import os 
import numpy as np


# Python Imports
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Import Fusion Modules 
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf  
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  
from vis import get_visualizer # Visualizer 
from run_model import Deformnet_runner # Neural Tracking + ARAP Moudle 
from run_motion_model import MotionCompleteNet_Runner # OcclusionFusion GNN

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

		self.savepath = os.path.join(self.fopt.datadir,"results")
		os.makedirs(self.savepath,exist_ok=True)
		os.makedirs(os.path.join(self.savepath,"visible_nodes"),exist_ok=True)


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
		Tests on Anime Files from deforming things4D
		In this test gr is passed at source
		Random nodes are sampled and are made not visible for Occlusion Fusion
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
	motion_model = MotionCompleteNet_Runner(fopt)	

	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield
	
	# Add modules to model
	motion_model.graph = graph



	deformed_vertices,deformed_normals,vert_anchors,vert_weights,valid_verts = warpfield.deform_mesh(trajectory[0],trajectory_normals[0])
	relative_transform_matrix,rmse_error = ssdr.get_transforms(trajectory,faces,vert_anchors,vert_weights,graph.nodes)
	
	print("Relative Transforms:",relative_transform_matrix)
	print("Rsme Error:",rmse_error)



	T = tsdf.trajectory.shape[0]

	intrinsics = np.array([1,1,0,0],dtype=np.float32)

	for i,invisible_node_percentage in enumerate(range(25,100,25)):
		
		visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
		num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
		invisible_nodes = np.random.choice(graph.num_nodes,size=num_invisible_nodes,replace=False)
		visible_nodes[invisible_nodes] = False
		tsdf.reduced_graph_dict = {"valid_nodes_mask":visible_nodes}

		for target_frame_id in range(fopt.skip_rate,T-1,fopt.skip_rate):
			
			print(f"Frame id:{target_frame_id}")
			# Optical flow data not required, returned by tsdf for the test
			optical_flow_data = {"source_id":target_frame_id - fopt.skip_rate,"target_id":target_frame_id}

			# Save node details for future use by Occlusion fusion
			np.save(os.path.join(tsdf.savepath,"visible_nodes",f"{optical_flow_data['source_id']}.npy"),visible_nodes)


			# Get deformed graph positions  
			source_vertices = tsdf.trajectory[target_frame_id - fopt.skip_rate]
			source_graph_nodes = source_vertices[graph.node_indices]
			
			target_vertices = tsdf.trajectory[target_frame_id]
			target_graph_nodes = target_vertices[graph.node_indices]

			predicted_motion,predicted_motion_confidence = motion_model(optical_flow_data["source_id"],
				source_graph_nodes,
				target_graph_nodes,
				visible_nodes)

			# TSDF gets updated last
			warpfield.deformed_nodes = source_graph_nodes + predicted_motion
			warpfield.frame_id = target_frame_id
			tsdf.frame_id = target_frame_id		


			# Save data for occlusion fusion 
			np.save(os.path.join(warpfield.savepath,"deformed_nodes",f"{target_frame_id}.npy"),warpfield.deformed_nodes)




			# Evaluate results 
			rec_err_per_sample = np.linalg.norm(warpfield.get_deformed_nodes() - target_vertices[graph.node_indices],axis=1)
			print("Percentage visible:",invisible_node_percentage)
			print("Reconstructon Error:",np.mean(rec_err_per_sample))

			# Plot deformed graph with different color 
			init_graph = vis.get_rendered_graph(graph.nodes,graph.edges,color=visible_nodes) # Initial graph 
			if not hasattr(vis,'bbox'):
				vis.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

			bbox = vis.bbox


			deformed_graph_color = predicted_motion_confidence[:,None]*np.tile(vis.colors[1:2],(graph.num_nodes,1)) \
									+ (1-predicted_motion_confidence[:,None])*np.tile(vis.colors[0:1],(graph.num_nodes,1))
			deformed_graph = vis.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox,color=deformed_graph_color)

			target_graph = vis.get_rendered_graph(target_graph_nodes,graph.edges,trans=np.array([2,0,0])*bbox) # Actualy position 
			vis.plot(init_graph + deformed_graph + target_graph,"Deformed Graph",False)

def test2(use_gpu=True):
	"""
		Tests on Anime Files from deforming things4D
		In this test gr is passed at source
		Based on Normals, nodes facing the camera are found and some percentage of them are removed 
	"""  
	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/mannequin_Running",\
		"skip_rate":1})
	vis = get_visualizer(fopt)

	ssdr = SSDR()
	filname = fopt.datadir.split("/")[-1] + '.anime'
	trajectory,faces = ssdr.load_anime_file(os.path.join(fopt.datadir,filname))



	# Compute normals 
	trajectory_normals = []
	for traj in trajectory:
		mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(traj),
			o3d.utility.Vector3iVector(faces))
		mesh.compute_vertex_normals(normalized=True)
		normals = np.array(mesh.vertex_normals)
		trajectory_normals.append(normals)

	trajectory_normals = np.array(trajectory_normals)	


	mesh = o3d.geometry.TriangleMesh(
		o3d.utility.Vector3dVector(trajectory[0]),
		o3d.utility.Vector3iVector(faces))

	# Create fusion modules 
	tsdf = TSDFMesh(fopt,mesh)
	tsdf.set_data(trajectory,trajectory_normals)

	graph = EDGraph(tsdf,vis)
	warpfield = WarpField(graph,tsdf,vis)
	motion_model = MotionCompleteNet_Runner(fopt)	

	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield
	
	# Add modules to model
	motion_model.graph = graph



	deformed_vertices,deformed_normals,vert_anchors,vert_weights,valid_verts = warpfield.deform_mesh(trajectory[0],trajectory_normals[0])
	relative_transform_matrix,rmse_error = ssdr.get_transforms(trajectory,faces,vert_anchors,vert_weights,graph.nodes)
	
	print("Relative Transforms:",relative_transform_matrix)
	print("Rsme Error:",rmse_error)




	T = tsdf.trajectory.shape[0]

	intrinsics = np.array([1,1,0,0],dtype=np.float32)

	for i,invisible_node_percentage in enumerate(range(95,100,25)):

		# Find Invisible nodes
		source_graph_normals = tsdf.trajectory_normals[0][graph.node_indices]
		direction_difference = (source_graph_normals@np.array([[0],[-1],[0]])).reshape(-1)
		difference_sorted = np.argsort(direction_difference)

		visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
		num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
		invisible_nodes = difference_sorted[:num_invisible_nodes]
		visible_nodes[invisible_nodes] = False
		tsdf.reduced_graph_dict = {"valid_nodes_mask":visible_nodes}
		# Save node details for future use by Occlusion fusion
		np.save(os.path.join(tsdf.savepath,"visible_nodes",f"0.npy"),visible_nodes)


		for target_frame_id in range(fopt.skip_rate,T-1,fopt.skip_rate):
			
			print(f"Frame id:{target_frame_id}")
			# Optical flow data not required, returned by tsdf for the test
			optical_flow_data = {"source_id":target_frame_id - fopt.skip_rate,"target_id":target_frame_id}

			# Find Invisible nodes
			source_graph_normals = tsdf.trajectory_normals[optical_flow_data['source_id']][graph.node_indices]
			direction_difference = (source_graph_normals@np.array([[0],[-1],[0]])).reshape(-1)
			difference_sorted = np.argsort(direction_difference)

			visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
			num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
			invisible_nodes = difference_sorted[:num_invisible_nodes]

			print(invisible_nodes)
			print(difference_sorted[invisible_nodes])

			visible_nodes[invisible_nodes] = False
			tsdf.reduced_graph_dict = {"valid_nodes_mask":visible_nodes}
			# Save node details for future use by Occlusion fusion
			np.save(os.path.join(tsdf.savepath,"visible_nodes",f"{optical_flow_data['source_id']}.npy"),visible_nodes)

			# Get deformed graph positions  
			source_vertices = tsdf.trajectory[target_frame_id - fopt.skip_rate]
			source_graph_nodes = source_vertices[graph.node_indices]
			
			target_vertices = tsdf.trajectory[target_frame_id]
			target_graph_nodes = target_vertices[graph.node_indices]

			input_source_graph_nodes = source_graph_nodes.copy()
			input_source_graph_nodes[invisible_nodes,:] = 0

			input_target_graph_nodes = target_graph_nodes.copy()
			input_target_graph_nodes[invisible_nodes,:] = 0

			predicted_motion,predicted_motion_confidence = motion_model(optical_flow_data["source_id"],
				input_source_graph_nodes,
				input_target_graph_nodes,
				visible_nodes)

			# TSDF gets updated last
			warpfield.deformed_nodes = source_graph_nodes + predicted_motion
			warpfield.frame_id = target_frame_id
			tsdf.frame_id = target_frame_id		


			# Save data for occlusion fusion 
			np.save(os.path.join(warpfield.savepath,"deformed_nodes",f"{target_frame_id}.npy"),warpfield.deformed_nodes)




			# Evaluate results 
			rec_err_per_sample = np.linalg.norm(warpfield.get_deformed_nodes() - target_vertices[graph.node_indices],axis=1)
			print("Percentage visible:",invisible_node_percentage,"Reconstructon Error:",np.mean(rec_err_per_sample))

			# Plot deformed graph with different color 
			init_graph = vis.get_rendered_graph(graph.nodes,graph.edges,color=visible_nodes) # Initial graph 
			if not hasattr(vis,'bbox'):
				vis.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

			bbox = vis.bbox


			deformed_graph_color = predicted_motion_confidence[:,None]*np.tile(vis.colors[1:2],(graph.num_nodes,1)) \
									+ (1-predicted_motion_confidence[:,None])*np.tile(vis.colors[0:1],(graph.num_nodes,1))
			deformed_graph = vis.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox,color=deformed_graph_color)

			target_graph = vis.get_rendered_graph(target_graph_nodes,graph.edges,trans=np.array([2,0,0])*bbox) # Actualy position 
			vis.plot(init_graph + deformed_graph + target_graph,"Deformed Graph",False)


def test3(use_gpu=True):
	"""
		Tests on Anime Files from deforming things4D
		Unlike test2, in this test previously predicted node positions are used as source
		Based on Normals, nodes facing the camera are found and some percentage of them are removed. 

		Note:- This is failing even when 1% of the nodes are made invisible.
	"""  
	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/mannequin_Running",\
		"skip_rate":1})
	vis = get_visualizer(fopt)

	ssdr = SSDR()
	filname = fopt.datadir.split("/")[-1] + '.anime'
	trajectory,faces = ssdr.load_anime_file(os.path.join(fopt.datadir,filname))



	# Compute normals 
	trajectory_normals = []
	for traj in trajectory:
		mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(traj),
			o3d.utility.Vector3iVector(faces))
		mesh.compute_vertex_normals(normalized=True)
		normals = np.array(mesh.vertex_normals)
		trajectory_normals.append(normals)

	trajectory_normals = np.array(trajectory_normals)	


	mesh = o3d.geometry.TriangleMesh(
		o3d.utility.Vector3dVector(trajectory[0]),
		o3d.utility.Vector3iVector(faces))

	# Create fusion modules 
	tsdf = TSDFMesh(fopt,mesh)
	tsdf.set_data(trajectory,trajectory_normals)

	graph = EDGraph(tsdf,vis)
	warpfield = WarpField(graph,tsdf,vis)
	motion_model = MotionCompleteNet_Runner(fopt)	

	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield
	
	# Add modules to model
	motion_model.graph = graph



	deformed_vertices,deformed_normals,vert_anchors,vert_weights,valid_verts = warpfield.deform_mesh(trajectory[0],trajectory_normals[0])
	relative_transform_matrix,rmse_error = ssdr.get_transforms(trajectory,faces,vert_anchors,vert_weights,graph.nodes)
	
	print("Relative Transforms:",relative_transform_matrix)
	print("Rsme Error:",rmse_error)




	T = tsdf.trajectory.shape[0]

	intrinsics = np.array([1,1,0,0],dtype=np.float32)

	for i,invisible_node_percentage in enumerate(range(1,100,25)):

		# Find Invisible nodes
		source_graph_normals = tsdf.trajectory_normals[0][graph.node_indices]
		direction_difference = (source_graph_normals@np.array([[0],[-1],[0]])).reshape(-1)
		difference_sorted = np.argsort(direction_difference)

		visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
		num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
		invisible_nodes = difference_sorted[:num_invisible_nodes]
		visible_nodes[invisible_nodes] = False
		tsdf.reduced_graph_dict = {"valid_nodes_mask":visible_nodes}
		# Save node details for future use by Occlusion fusion
		np.save(os.path.join(tsdf.savepath,"visible_nodes",f"0.npy"),visible_nodes)


		for target_frame_id in range(fopt.skip_rate,T-1,fopt.skip_rate):
			
			print(f"Frame id:{target_frame_id}")
			# Optical flow data not required, returned by tsdf for the test
			optical_flow_data = {"source_id":target_frame_id - fopt.skip_rate,"target_id":target_frame_id}

			# Find Invisible nodes
			source_graph_normals = tsdf.trajectory_normals[optical_flow_data['source_id']][graph.node_indices]
			direction_difference = (source_graph_normals@np.array([[0],[-1],[0]])).reshape(-1)
			difference_sorted = np.argsort(direction_difference)

			visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
			num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
			invisible_nodes = difference_sorted[:num_invisible_nodes]

			print(invisible_nodes)
			print(difference_sorted[invisible_nodes])

			visible_nodes[invisible_nodes] = False
			tsdf.reduced_graph_dict = {"valid_nodes_mask":visible_nodes}
			# Save node details for future use by Occlusion fusion
			np.save(os.path.join(tsdf.savepath,"visible_nodes",f"{optical_flow_data['source_id']}.npy"),visible_nodes)

			# Get deformed graph positions  
			source_graph_nodes = warpfield.deformed_nodes
			
			target_vertices = tsdf.trajectory[target_frame_id]
			target_graph_nodes = target_vertices[graph.node_indices]

			input_source_graph_nodes = source_graph_nodes.copy()
			input_source_graph_nodes[invisible_nodes,:] = 0

			input_target_graph_nodes = target_graph_nodes.copy()
			input_target_graph_nodes[invisible_nodes,:] = 0

			predicted_motion,predicted_motion_confidence = motion_model(optical_flow_data["source_id"],
				input_source_graph_nodes,
				input_target_graph_nodes,
				visible_nodes)

			# TSDF gets updated last
			warpfield.deformed_nodes = source_graph_nodes + predicted_motion
			warpfield.frame_id = target_frame_id
			tsdf.frame_id = target_frame_id		


			# Save data for occlusion fusion 
			np.save(os.path.join(warpfield.savepath,"deformed_nodes",f"{target_frame_id}.npy"),warpfield.deformed_nodes)




			# Evaluate results 
			rec_err_per_sample = np.linalg.norm(warpfield.get_deformed_nodes() - target_vertices[graph.node_indices],axis=1)
			print("Percentage visible:",invisible_node_percentage,"Reconstructon Error:",np.mean(rec_err_per_sample))

			# Plot deformed graph with different color 
			init_graph = vis.get_rendered_graph(graph.nodes,graph.edges,color=visible_nodes) # Initial graph 
			if not hasattr(vis,'bbox'):
				vis.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

			bbox = vis.bbox


			deformed_graph_color = predicted_motion_confidence[:,None]*np.tile(vis.colors[1:2],(graph.num_nodes,1)) \
									+ (1-predicted_motion_confidence[:,None])*np.tile(vis.colors[0:1],(graph.num_nodes,1))
			deformed_graph = vis.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox,color=deformed_graph_color)

			target_graph = vis.get_rendered_graph(target_graph_nodes,graph.edges,trans=np.array([2,0,0])*bbox) # Actualy position 

			image_name = f"test3_{target_frame_id:02d}.png"
			vis.plot(init_graph + deformed_graph + target_graph,"Deformed Graph",False,savename=image_name)


def test4(use_gpu=True):
	"""
		Tests on Anime Files from deforming things4D
		Unlike test2, in this test previously predicted node positions are used as source
		Unlike test3, main model's optimization is used with only motion loss and arap loss to solve problems of test 3  
		Based on Normals, nodes facing the camera are found and some percentage of them are removed. 

		Note:- This is failing even when 1% of the nodes are made invisible.
	"""  
	fopt = Dict2Class({"source_frame":0,\
		"gpu":use_gpu,"visualizer":"open3d",\
		"datadir": "/media/srialien/Elements/AT-Datasets/DeepDeform/new_test/mannequin_Running",\
		"skip_rate":1})
	vis = get_visualizer(fopt)

	ssdr = SSDR()
	filname = fopt.datadir.split("/")[-1] + '.anime'
	trajectory,faces = ssdr.load_anime_file(os.path.join(fopt.datadir,filname))

	# Compute normals 
	trajectory_normals = []
	for traj in trajectory:
		mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(traj),
			o3d.utility.Vector3iVector(faces))
		mesh.compute_vertex_normals(normalized=True)
		normals = np.array(mesh.vertex_normals)
		trajectory_normals.append(normals)

	trajectory_normals = np.array(trajectory_normals)	


	mesh = o3d.geometry.TriangleMesh(
		o3d.utility.Vector3dVector(trajectory[0]),
		o3d.utility.Vector3iVector(faces))

	# Create fusion modules 
	tsdf = TSDFMesh(fopt,mesh)
	tsdf.set_data(trajectory,trajectory_normals)

	graph = EDGraph(tsdf,vis)
	warpfield = WarpField(graph,tsdf,vis)
	model = TestModel(vis,fopt)	
	motion_model = MotionCompleteNet_Runner(fopt)	

	# Add modules to vis
	vis.tsdf = tsdf
	vis.graph = graph
	vis.warpfield = warpfield
	
	# Add modules to model
	motion_model.graph = graph
	
	# Add modules to model
	model.graph = graph
	model.warpfield = warpfield
	model.tsdf = tsdf
	model.vis = vis



	deformed_vertices,deformed_normals,vert_anchors,vert_weights,valid_verts = warpfield.deform_mesh(trajectory[0],trajectory_normals[0])
	relative_transform_matrix,rmse_error = ssdr.get_transforms(trajectory,faces,vert_anchors,vert_weights,graph.nodes)
	
	print("Relative Transforms:",relative_transform_matrix)
	print("Rsme Error:",rmse_error)




	T = tsdf.trajectory.shape[0]

	intrinsics = np.array([1,1,0,0],dtype=np.float32)

	for i,invisible_node_percentage in enumerate(range(1,100,25)):

		# Find Invisible nodes
		source_graph_normals = tsdf.trajectory_normals[0][graph.node_indices]
		direction_difference = (source_graph_normals@np.array([[0],[-1],[0]])).reshape(-1)
		difference_sorted = np.argsort(direction_difference)

		visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
		num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
		invisible_nodes = difference_sorted[:num_invisible_nodes]
		visible_nodes[invisible_nodes] = False
		tsdf.reduced_graph_dict = {"valid_nodes_mask":visible_nodes}
		# Save node details for future use by Occlusion fusion
		np.save(os.path.join(tsdf.savepath,"visible_nodes",f"0.npy"),visible_nodes)


		for target_frame_id in range(fopt.skip_rate,T-1,fopt.skip_rate):
			
			print(f"Frame id:{target_frame_id}")
			# Optical flow data not required, returned by tsdf for the test
			optical_flow_data = {"source_id":target_frame_id - fopt.skip_rate,"target_id":target_frame_id}

			# Find Invisible nodes
			source_graph_normals = tsdf.trajectory_normals[optical_flow_data['source_id']][graph.node_indices]
			direction_difference = (source_graph_normals@np.array([[0],[-1],[0]])).reshape(-1)
			difference_sorted = np.argsort(direction_difference)

			visible_nodes = np.ones(graph.num_nodes,dtype=np.bool)	
			num_invisible_nodes = graph.num_nodes*invisible_node_percentage//100
			invisible_nodes = difference_sorted[:num_invisible_nodes]

			print(invisible_nodes)
			print(difference_sorted[invisible_nodes])

			visible_nodes[invisible_nodes] = False
			tsdf.reduced_graph_dict = {"valid_nodes_mask":visible_nodes}
			# Save node details for future use by Occlusion fusion
			np.save(os.path.join(tsdf.savepath,"visible_nodes",f"{optical_flow_data['source_id']}.npy"),visible_nodes)

			# Get deformed graph positions  
			source_graph_nodes = warpfield.deformed_nodes
			
			target_vertices = tsdf.trajectory[target_frame_id]
			target_graph_nodes = target_vertices[graph.node_indices]

			input_source_graph_nodes = source_graph_nodes.copy()
			input_source_graph_nodes[invisible_nodes,:] = 0

			input_target_graph_nodes = target_graph_nodes.copy()
			input_target_graph_nodes[invisible_nodes,:] = 0

			node_motion_data = motion_model(optical_flow_data["source_id"],
				input_source_graph_nodes,
				input_target_graph_nodes,
				visible_nodes)

			estimated_transformations = model.optimize(optical_flow_data,node_motion_data,intrinsics,None)


			# Update warpfield parameters, warpfield maps to target frame  
			warpfield.update_transformations(estimated_transformations)


			# TSDF gets updated last
			tsdf.frame_id = target_frame_id		


			# Save data for occlusion fusion 
			np.save(os.path.join(warpfield.savepath,"deformed_nodes",f"{target_frame_id}.npy"),warpfield.deformed_nodes)




			# Evaluate results 
			rec_err_per_sample = np.linalg.norm(warpfield.get_deformed_nodes() - target_vertices[graph.node_indices],axis=1)
			print("Percentage visible:",invisible_node_percentage,"Reconstructon Error:",np.mean(rec_err_per_sample))

			# Plot deformed graph with different color 
			init_graph = vis.get_rendered_graph(graph.nodes,graph.edges,color=visible_nodes) # Initial graph 
			if not hasattr(vis,'bbox'):
				vis.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

			bbox = vis.bbox


			deformed_graph_color = predicted_motion_confidence[:,None]*np.tile(vis.colors[1:2],(graph.num_nodes,1)) \
									+ (1-predicted_motion_confidence[:,None])*np.tile(vis.colors[0:1],(graph.num_nodes,1))
			deformed_graph = vis.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox,color=deformed_graph_color)

			target_graph = vis.get_rendered_graph(target_graph_nodes,graph.edges,trans=np.array([2,0,0])*bbox) # Actualy position 

			image_name = f"test3_{target_frame_id:02d}.png"
			vis.plot(init_graph + deformed_graph + target_graph,"Deformed Graph",False,savename=image_name)

