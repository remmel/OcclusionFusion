	# Python Imports
import os
import numpy as np
import open3d as o3d
import json


# Nueral Tracking Modules
from utils import image_proc
import utils.viz_utils as viz_utils
import utils.line_mesh as line_mesh_utils

# Fusion Modules
from .visualizer import Visualizer

class VisualizeOpen3D(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)

		# Camera parameters, load camera parameters from savepath
		# To save camera parameters in Open3D press D. Then copy DepthCapture*.json to savepath/camera_params.json 
		if os.path.isfile(os.path.join(self.savepath,"camera_params.json")):
			self.camera_params = o3d.io.read_pinhole_camera_parameters(\
				os.path.join(self.savepath,"camera_params.json"))

		if os.path.isfile(os.path.join(self.savepath,"camera_params_zoomed.json")):
			self.camera_params_zoomed = o3d.io.read_pinhole_camera_parameters(\
				os.path.join(self.savepath,"camera_params_zoomed.json"))

	######################################
	# Helper modules 					 #	
	######################################	
	def plot(self,object_list,title,debug,savename=None,zoomed_in=True):
		"""
			Main Function which takes all open3d objects ans plots them
			
			@params: 
				object_list: List of open3D objects need to be plotted
				title: Title of the plot
				debug: Whether to stop program when visualizing results
		"""	



		print(f"::{title} Debug:{debug}")	
		if debug: 
			self.vis_debug = o3d.visualization.Visualizer()
			self.vis_debug.create_window(width=1280, height=1280,window_name="Fusion Pipeline")
			
			for o in object_list:
				self.vis_debug.add_geometry(o)

			
			ctr = self.vis_debug.get_view_control()

			# ctr.set_up([0,1,0])
			# ctr.set_lookat([0,0,0])	

			if hasattr(self,"camera_params") and not zoomed_in :	
				ctr.convert_from_pinhole_camera_parameters(self.camera_params)	

			if hasattr(self,"camera_params_zoomed") and zoomed_in :	
				ctr.convert_from_pinhole_camera_parameters(self.camera_params_zoomed)	
			# ctr.set_zoom(0.3)
			self.vis_debug.run() # Plot and halt the program
			self.vis_debug.destroy_window()
			self.vis_debug.close()

		else:
			if hasattr(self,'vis'):
				self.vis.clear_geometries() # Clear previous dataq
			else:	
				# Create visualization object
				self.vis = o3d.visualization.Visualizer()
				self.vis.create_window(width=1280, height=1280,window_name="Fusion Pipeline")

			for o in object_list:
				self.vis.add_geometry(o)

			ctr = self.vis.get_view_control()
			# ctr.set_up([0,1,0])
			# ctr.set_lookat([0,0,0])
			# ctr.set_zoom(0.3)


			if hasattr(self,"camera_params") and not zoomed_in :	
				ctr.convert_from_pinhole_camera_parameters(self.camera_params)	

			if hasattr(self,"camera_params_zoomed") and zoomed_in :	
				ctr.convert_from_pinhole_camera_parameters(self.camera_params_zoomed)


			self.vis.poll_events()
			self.vis.update_renderer()

			if savename is not None: # Save images only when not debugging
				self.vis.capture_screen_image(os.path.join(self.savepath,"images",savename)) # TODO: Returns segfault

	
	def get_mesh(self,verts,faces,trans=np.zeros((3,1)),color=None,normals=None):
		"""
			Create Open3D Mesh  
		"""		
		canonical_mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(verts)),
			o3d.utility.Vector3iVector(faces))
		if color is not None:
			color = self.get_color(color)
			canonical_mesh.vertex_colors = o3d.utility.Vector3dVector(color)
		if normals is not None: 
			canonical_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

		canonical_mesh.translate(trans)
			
		return canonical_mesh

	def get_model_from_tsdf(self,trans=np.zeros((3,1))): 
		"""
			Create open3D object to visualize tsdf 
		"""
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		verts, faces, normals, color = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
		
		# print("Canonical Model vertices:",verts.shape)

		return self.get_mesh(verts,faces,trans=trans,color=color,normals=normals)	

	def get_deformed_model_from_tsdf(self,trans=np.zeros((3,1))):
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 

		verts,faces,normals,color = self.tsdf.get_deformed_model()
		return self.get_mesh(verts,faces,trans=trans,color=color,normals=normals)

	def get_rendered_graph(self,nodes,edges,color=None,trans=np.zeros((1,3))):
		"""
			Get graph in a graph structure that can be plotted using Open3D
			@params:
				color: Color of nodes (could be a label, rgb color)
				trans: np.ndarray(1,3): Global Translation of the graph for plotting 
		"""
		color = self.get_color(color) # Get color

		# Motion Graph
		rendered_graph = viz_utils.create_open3d_graph(
			viz_utils.transform_pointcloud_to_opengl_coords(nodes) + trans,
			edges,
			color=color)
		
		return rendered_graph

	def get_rendered_reduced_graph(self,trans=np.zeros((1,3))):
		"""
			Get graph in a graph structure that can be plotted using Open3D
			@params:
				color: Color of nodes (could be a label, rgb color)
				trans: np.ndarray(1,3): Global Translation of the graph for plotting 
		"""
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		assert hasattr(self.tsdf,'reduced_graph_dict'),  "Visible nodes not calculated. Can't show reduded graph" 

		nodes = self.tsdf.reduced_graph_dict["valid_nodes_at_source"]
		edges = self.tsdf.reduced_graph_dict["graph_edges"]
		
		rendered_graph = self.get_rendered_graph(nodes,edges,trans=trans)
		return rendered_graph

	def get_rendered_deformed_graph(self,color=None,trans=np.zeros((1,3))):
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		assert hasattr(self,'warpfield'),  "Warpfield not defined. Add warpfield as attribute to visualizer first." 
		
		nodes = self.warpfield.get_deformed_nodes()
		edges = self.warpfield.graph.edges

		if color is None:
			color = self.tsdf.reduced_graph_dict["valid_nodes_mask"]

		return self.get_rendered_graph(nodes,edges,color=color,trans=trans)


	def get_source_RGBD(self,trans=np.zeros((3,1))):
		assert hasattr(self,'warpfield'),  "Warpfield not defined. Add warpfield as attribute to visualizer first." 

		if hasattr(self.warpfield,'source_im'):
			source_pcd = viz_utils.get_pcd(self.warpfield.source_im) # Get point cloud with max 10000 points
		elif hasattr(self.tsdf,'im'):
			source_pcd = viz_utils.get_pcd(self.tsdf.im) # Get point cloud with max 10000 points

		source_pcd.translate(trans)

		return source_pcd

	def get_target_RGBD(self,trans=np.zeros((3,1))):
		"""
			Get Target image from TSDF
			Plot as point cloud  
		"""	
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		assert hasattr(self.tsdf,'im'),  "Target image not defined. Update/integrate target image to tsdf first." 

		target_pcd = viz_utils.get_pcd(self.tsdf.im) # Get point cloud with max 10000 points

		# Update boundary mask color
		# boundary_points = np.where(target_data["target_boundary_mask"].reshape(-1) > 0)[0]
		# points_color = np.asarray(target_pcd.colors)
		# points_color[boundary_points, 0] = 1.0
		# target_pcd.colors = o3d.utility.Vector3dVector(points_color)  # Mark boundary points in read



		# Save it somewhere 
		target_depth_savepath = os.path.join(self.savepath,"target_pcd",f"{self.tsdf.frame_id}.xyz")
		o3d.io.write_point_cloud(target_depth_savepath, target_pcd)
		target_pcd.translate(trans)
		




		return target_pcd


	# Make functions defined in sub-classes based on method used 
	def plot_skinned_model(self,debug=True):
		"""
			Plot the skinning of the mesh 
		"""	

		color_list = self.get_color(np.arange(self.graph.nodes.shape[0]+1)) # Common color for graph and mesh 
		color_list[-1] = 0. # Last/Background color is black
		rendered_graph = self.get_rendered_graph(self.graph.nodes,self.graph.edges,color=color_list,trans=np.array([0,0,0.01]))
		
		verts, faces, normals, _ = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
		vert_anchors,vert_weights,valid_verts = self.warpfield.skin(verts)
		mesh_color = np.array([vert_weights[i,:]@color_list[vert_anchors[i,:],:] for i in range(verts.shape[0])])
		
		reshape_gpu_vol = [verts.shape[0],1,1]        
		deformed_vertices = self.warpfield.deform(verts,vert_anchors,vert_weights,reshape_gpu_vol,valid_verts)    

		mesh = self.get_mesh(deformed_vertices,faces,color=mesh_color,normals=normals)	

		self.plot([mesh] + rendered_graph,"Skinned Object",debug=debug)


	def plot_graph(self,color,title="Embedded Graph",debug=False):
		"""
			@parama:
				color: Color of nodes (could be a None,label, rgb color)
				debug: bool: Stop program to show the plot
		"""
		
		assert hasattr(self,'graph'),  "Graph not defined. Add graph as attribute to visualizer first." 
		rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges,color)
		self.plot([rendered_graph_nodes,rendered_graph_edges],title,debug)

		# source_pcd = self.get_source_RGBD()
		# rendered_reduced_graph_nodes,rendered_reduced_graph_edges = self.get_rendered_reduced_graph()

		# self.plot([source_pcd,rendered_reduced_graph_nodes,rendered_reduced_graph_edges],"Showing reduced graph",debug)


	def plot_deformed_graph(self,debug=False):
		
		init_graph = self.get_rendered_graph(self.graph.nodes,self.graph.edges) # Initial graph 
		if not hasattr(self,'bbox'):
			self.bbox = (init_graph[0].get_max_bound() - init_graph[0].get_min_bound()) # Compute bounding box using nodes of init graph

		bbox = self.bbox

		deformed_graph = self.get_rendered_deformed_graph(trans=np.array([1,0,0])*bbox)

		self.plot(init_graph + deformed_graph,"Deformed Graph",debug)	

	def init_plot(self,debug=False):	
		"""
			Plot the initial TSDF and graph used for registration
		"""
		title = "Initization"
		canonical_mesh = self.get_model_from_tsdf()
		rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges)
		self.plot([canonical_mesh,rendered_graph_nodes,rendered_graph_edges],title,debug)
	
	def plot_correspondence(self,target_matches,valid_correspondences,target_image_data,debug=True,savename=None):

		# print(target_matches)
		deformed_model = self.get_model_from_tsdf()
		target_matches = viz_utils.transform_pointcloud_to_opengl_coords(target_matches)
		
		
		source_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(deformed_model.vertices)))

		corresp_weights = np.ones(target_matches.shape[0])


		good_matches_set, good_weighted_matches_set = viz_utils.create_matches_lines(valid_correspondences , np.array([0.0, 0.8, 0]),
																					 np.array([0.8, 0, 0.0]),
																					 source_points, target_matches,
																					 corresp_weights, 0.3,
																					 1.)	


		if savename is not None:
			savename = os.path.join(self.savepath,"images",savename)

		if target_image_data is not None:
			target_pcd = viz_utils.get_pcd(target_image_data["im"]) # Get point cloud with max 10000 points
		
			self.plot([deformed_model,target_pcd,good_matches_set],title="Deformed Model Correspondences",debug=debug,savename=savename)	
		else:
			self.plot([deformed_model,good_matches_set],title="Deformed Model Correspondences",debug=debug,savename=savename)	

	
	def plot_optimization_sceneflow(self,frame_id,n_iter,warped_points,target_points,target_matches,deformed_nodes,landmarks,motion_complete_graph_location,debug=True):
		"""
			Plot registration during optimization 
		"""


		deformed_nodes = viz_utils.transform_pointcloud_to_opengl_coords(deformed_nodes)
		# deformed_nodes_rendered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(deformed_nodes)))
		# deformed_nodes_rendered.colors = o3d.utility.Vector3dVector(np.array([[0/255, 0/255, 255/255] for i in range(len(deformed_nodes))]))

		deformed_graph_rendered = self.get_rendered_graph(viz_utils.transform_pointcloud_to_opengl_coords(deformed_nodes),self.graph.edges)

		if motion_complete_graph_location is not None:
			motion_complete_graph_color = np.ones(motion_complete_graph_location.shape[0])
			motion_complete_graph_rendered = self.get_rendered_graph(motion_complete_graph_location,self.graph.edges,color=motion_complete_graph_color)
		else: 	
			motion_complete_graph_rendered = []

		warped_points = viz_utils.transform_pointcloud_to_opengl_coords(warped_points)
		warped_points_rendered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(warped_points)))
		warped_points_rendered.colors = o3d.utility.Vector3dVector(np.array([[0/255, 255/255, 0/255] for i in range(len(warped_points))]))

		target_points = viz_utils.transform_pointcloud_to_opengl_coords(target_points)
		target_points_rendered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(target_points)))
		target_points_rendered.colors = o3d.utility.Vector3dVector(np.array([[0/255, 0/255, 255/255] for i in range(len(target_points))]))

		n_match_matches = len(landmarks)
		if n_match_matches > 1000: 
			n_match_matches = 1000
			inds = np.random.choice(len(landmarks),1000,replace=False)
			landmarks = landmarks[:,ind]


		warped_points = warped_points[landmarks[0]]
		target_matches = target_matches[landmarks[1]]
		match_matches_points = np.concatenate([warped_points, viz_utils.transform_pointcloud_to_opengl_coords(target_matches)], axis=0)
		
		match_matches_lines = [[i, i + n_match_matches] for i in range(0, n_match_matches, 1)]


		# --> Create match (unweighted) lines 
		match_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(match_matches_lines))]
		match_matches_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(match_matches_points),
			lines=o3d.utility.Vector2iVector(match_matches_lines),
		)
		match_matches_set.colors = o3d.utility.Vector3dVector(match_matches_colors)

		savename = f"NICP_Lepard_{n_iter}_{frame_id}.png"
		self.plot(deformed_graph_rendered + motion_complete_graph_rendered + [warped_points_rendered,target_points_rendered,match_matches_set],f"Iteration:{n_iter}",debug=debug,savename=savename)

	def plot_optimization(self,frame_id,n_iter,deformed_points,valid_verts,target_points,debug=True):
		"""
			Plot registration during optimization 
		"""
		deformed_points = deformed_points.reshape(-1,3)

		target_matches = np.zeros((valid_verts.shape[0],3))
		target_matches[valid_verts] = target_points.reshape(-1,3)


		verts, faces, normals, color = self.tsdf.get_canonical_model()  # Extract the new canonical pose using marching cubes
		verts[valid_verts] = deformed_points
		deformed_model = self.get_mesh(verts,faces,color=color,normals=normals)

		target_matches = viz_utils.transform_pointcloud_to_opengl_coords(target_matches)

		source_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(deformed_model.vertices)))

		corresp_weights = np.ones(target_matches.shape[0])

		good_matches_set, good_weighted_matches_set = viz_utils.create_matches_lines(valid_verts , np.array([0.0, 0.8, 0]),
																					 np.array([0.8, 0, 0.0]),
																					 source_points, target_matches,
																					 corresp_weights, 0.3,
																					 1.)


		target_pcd = o3d.geometry.PointCloud()
		target_pcd.points = o3d.utility.Vector3dVector(target_matches)

		self.plot([deformed_model,good_matches_set,target_pcd],f"Iteration:{n_iter}",debug=debug)

	def plot_opticalflow(self,source_px,source_py, target_px,target_py,trans=np.zeros((3,1)),debug=False):
	
		assert len(source_px) == len(source_py) == len(target_px) == len(target_py),\
			f"Source and target must be of same length, source_px:{len(source_px)} source_py:{len(source_py)} target_px:{len(target_px)} target_px:{len(target_py)}"	
		
		N = len(source_px)	
		# Create selfource pcloud
		source = np.zeros((N,3),dtype=np.float32)
		source[:,0] = source_px	
		source[:,1] = source_py	


		target = np.zeros((N,3),dtype=np.float32)
		target[:,0] = target_px	
		target[:,1] = target_py	

		# Create optical flow color 
		optical_flow_magnitude = np.linalg.norm(target - source,axis=1) # Find magnitude
		optical_flow_magnitude = (optical_flow_magnitude - optical_flow_magnitude.min(keepdims=True))/(optical_flow_magnitude.max(keepdims=True) - optical_flow_magnitude.min(keepdims=True)) # Normalized

		colors = np.zeros((N,3),dtype=np.float32)
		colors[:,0] = optical_flow_magnitude # Red
		colors[:,2] = 1 - optical_flow_magnitude # Blue 

		source_pcd = o3d.geometry.PointCloud()
		source_pcd.points = o3d.utility.Vector3dVector(source)
		source_pcd.colors = o3d.utility.Vector3dVector(colors)

		target_pcd = o3d.geometry.PointCloud()
		target_pcd.points = o3d.utility.Vector3dVector(target)
		target_pcd.colors = o3d.utility.Vector3dVector(colors)

		self.plot([source_pcd,target_pcd],f"Optical flow",debug=debug)



	def plot_alignment(self,source_frame_data,\
			target_frame_data,graph_data,skin_data,\
			model_data):
		"""
			Plot Alignment similiar to neural tracking

		"""

		# Params for visualization correspondence info
		weight_thr = 0.3
		weight_scale = 1

		# Source
		source_pcd = self.get_source_RGBD()

		# keep only object using the mask
		valid_source_mask = np.moveaxis(model_data["valid_source_points"], 0, -1).reshape(-1).astype(bool)
		source_object_pcd = source_pcd.select_by_index(np.where(valid_source_mask)[0])

		# Source warped
		warped_deform_pred_3d_np = image_proc.warp_deform_3d(
			source_frame_data["im"], skin_data["pixel_anchors"], skin_data["pixel_weights"], graph_data["valid_nodes_at_source"],
			model_data["node_rotations"], model_data["node_translations"]
		)

		source_warped = np.copy(source_frame_data["im"])
		source_warped[3:, :, :] = warped_deform_pred_3d_np
		warped_pcd = viz_utils.get_pcd(source_warped).select_by_index(np.where(valid_source_mask)[0])
		warped_pcd.paint_uniform_color([1, 0.706, 0])

		# TARGET
		target_pcd = self.get_target_RGBD()


		####################################
		# GRAPH #
		####################################
		rendered_graph = viz_utils.create_open3d_graph(
			viz_utils.transform_pointcloud_to_opengl_coords(graph_data["valid_nodes_at_source"] + model_data["node_translations"]), graph_data["graph_edges"])

		# Correspondences
		# Mask
		mask_pred_flat = model_data["mask_pred"].reshape(-1)
		valid_correspondences = model_data["valid_correspondences"].reshape(-1).astype(bool)
		# target matches
		target_matches = np.moveaxis(model_data["target_matches"], 0, -1).reshape(-1, 3)
		target_matches = viz_utils.transform_pointcloud_to_opengl_coords(target_matches)

		# "Good" matches
		good_mask = valid_correspondences & (mask_pred_flat >= weight_thr)
		good_matches_set, good_weighted_matches_set = viz_utils.create_matches_lines(good_mask, np.array([0.0, 0.8, 0]),
																					 np.array([0.8, 0, 0.0]),
																					 source_pcd, target_matches,
																					 mask_pred_flat, weight_thr,
																					 weight_scale)

		bad_mask = valid_correspondences & (mask_pred_flat < weight_thr)
		bad_matches_set, bad_weighted_matches_set = viz_utils.create_matches_lines(bad_mask, np.array([0.0, 0.8, 0]),
																				   np.array([0.8, 0, 0.0]), source_pcd,
																				   target_matches, mask_pred_flat,
																				   weight_thr, weight_scale)


		####################################
		# Generate info for aligning source to target (by interpolating between source and warped source)
		####################################
		warped_points = np.asarray(warped_pcd.points)
		valid_source_points = np.asarray(source_object_pcd.points)
		assert warped_points.shape[0] == np.asarray(source_object_pcd.points).shape[
			0], f"Warp points:{warped_points.shape} Valid Source Points:{valid_source_points.shape}"
		line_segments = warped_points - valid_source_points
		line_segments_unit, line_lengths = line_mesh_utils.normalized(line_segments)
		line_lengths = line_lengths[:, np.newaxis]
		line_lengths = np.repeat(line_lengths, 3, axis=1)

		####################################
		# Draw
		####################################

		geometry_dict = {
			"source_pcd": source_pcd,
			"source_obj": source_object_pcd,
			"target_pcd": target_pcd,
			"graph": rendered_graph,
			# "deformed_graph":    rendered_deformed_graph
		}

		alignment_dict = {
			"valid_source_points": valid_source_points,
			"line_segments_unit": line_segments_unit,
			"line_lengths": line_lengths
		}

		matches_dict = {
			"good_matches_set": good_matches_set,
			"good_weighted_matches_set": good_weighted_matches_set,
			"bad_matches_set": bad_matches_set,
			"bad_weighted_matches_set": bad_weighted_matches_set
		}

		#####################################################################################################
		# Open viewer
		#####################################################################################################
		manager = viz_utils.CustomDrawGeometryWithKeyCallback(
			geometry_dict, alignment_dict, matches_dict
		)
		manager.custom_draw_geometry_with_key_callback()		

	def show(self,scene_flow_data,source_frame,matches=None,debug=True):
		"""
			For visualizing the tsdf integration: 
			1. Source RGBD + Graph(visible nodes)   2. Target RGBD as Point Cloud 
			3. Canonical Model + Graph   			3. Deformed Model   
		"""

		# Top left
		# source_pcd = self.get_source_RGBD()
		source_pcd = o3d.geometry.PointCloud()
		source_pcd.points = o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['source']))
		source_pcd.colors = o3d.utility.Vector3dVector(np.array([[0/255, 255/255, 0/255] for i in range(len(scene_flow_data['source']))]))

	
		# Create bounding box for later use 
		if not hasattr(self,'bbox'):
			self.bbox = (source_pcd.get_max_bound() - source_pcd.get_min_bound())

		bbox = self.bbox
		# rendered_reduced_graph_nodes,rendered_reduced_graph_edges = self.get_rendered_reduced_graph(trans=np.array([0.0, 0, 0.03]) * bbox)

		# Top right 
		target_pcd = self.get_target_RGBD(trans=np.array([1.0, 0, 0]) * bbox)

		scene_flow = scene_flow_data['scene_flow'] 
		scene_flow[:,0] += bbox[0]

		# n_match_matches = scene_flow_data['source'][scene_flow_data['valid_verts']].shape[0]
		# match_matches_points = np.concatenate([viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['source'][scene_flow_data['valid_verts']]), viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['source'][scene_flow_data['valid_verts']]+scene_flow[scene_flow_data['valid_verts']])], axis=0)
		# match_matches_lines = [[i, i + n_match_matches] for i in range(0, n_match_matches, 1)]

		# # --> Create match (unweighted) lines 
		# match_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(match_matches_lines))]
		# match_matches_set = o3d.geometry.LineSet(
		# 	points=o3d.utility.Vector3dVector(match_matches_points),
		# 	lines=o3d.utility.Vector2iVector(match_matches_lines),
		# )
		# match_matches_set.colors = o3d.utility.Vector3dVector(match_matches_colors)



		# color = np.load(os.path.join(self.savepath,"visible_nodes",f"{source_frame}.npy"))
		color = None
		rendered_deformed_nodes,rendered_deformed_edges = self.get_rendered_graph(self.warpfield.get_deformed_nodes(),self.graph.edges,color=color,trans=np.array([0.0, -1.0, 0.00]) * bbox)


		# Bottom left
		canonical_mesh = self.get_model_from_tsdf(trans=np.array([0, -1.0, 0]) * bbox)
		# rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges,color=color,trans=np.array([0, -1.0, 0.01]) * bbox)

		# Bottom right
		deformed_mesh = self.get_deformed_model_from_tsdf(trans=np.array([1.0, -1.0, 0]) * bbox)
		# rendered_reduced_graph_nodes2,rendered_reduced_graph_edges2 = self.get_rendered_reduced_graph(trans=np.array([1.5, -1.5, 0.01]) * bbox)
		# rendered_deformed_nodes,rendered_deformed_edges = self.get_rendered_graph(self.warpfield.get_deformed_nodes(),self.graph.edges,color=color,trans=np.array([1.0, -1.0, 0.01]) * bbox)

		# Add matches
		# print(matches)
		# trans = np.array([0, 0, 0]) * bbox
		# matches[0].translate(trans)
		# matches[1].translate(trans)
		# if matches is not None:
			# vis.add_geometry(matches[0])
			# vis.add_geometry(matches[1])



		image_path = os.path.join(self.savepath,"images",f"subplot_{self.tsdf.frame_id}.png")
		print("Saving results to:",image_path)
		self.plot([source_pcd,
			target_pcd,
			# match_matches_set,
			rendered_deformed_nodes,rendered_deformed_edges,
			# canonical_mesh,rendered_graph_nodes,rendered_graph_edges,
			deformed_mesh],"Showing frame",debug,savename=image_path)



		# if debug:
		# 	self.vis_debug.capture_screen_image(image_path) # TODO: Returns segfault
		# else:
		# 	self.vis.capture_screen_image(image_path) # TODO: Returns segfault

		# Save results 
		canonical_mesh.translate(np.array([0.0, 1.0, 0]) * bbox)
		o3d.io.write_triangle_mesh(os.path.join(self.savepath,f"canonical_model/{self.tsdf.frame_id}.ply"),canonical_mesh) 

		deformed_mesh.translate(np.array([-1.0, 1.0, 0]) * bbox)
		o3d.io.write_triangle_mesh(os.path.join(self.savepath,f"deformed_model/{self.tsdf.frame_id}.ply"),deformed_mesh) 


	def show_zoomed_in(self,scene_flow_data,source_frame,matches=None,debug=True):
		"""
			For visualizing the tsdf integration: 
			1. Source RGBD + Graph(visible nodes)   2. Target RGBD as Point Cloud 
			3. Canonical Model + Graph   			3. Deformed Model   
		"""

		# Top left
		# source_pcd = self.get_source_RGBD()
		# source_pcd = o3d.geometry.PointCloud()
		# source_pcd.points = o3d.utility.Vector3dVector(viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['source']))
		# source_pcd.colors = o3d.utility.Vector3dVector(np.array([[0/255, 255/255, 0/255] for i in range(len(scene_flow_data['source']))]))

	
		# # Create bounding box for later use 
		# if not hasattr(self,'bbox'):
		# 	self.bbox = (source_pcd.get_max_bound() - source_pcd.get_min_bound())

		# bbox = self.bbox
		# rendered_reduced_graph_nodes,rendered_reduced_graph_edges = self.get_rendered_reduced_graph(trans=np.array([0.0, 0, 0.03]) * bbox)

		# Top right 
		target_pcd = self.get_target_RGBD()
		target_pcd.colors = o3d.utility.Vector3dVector(np.array([[0/255, 0/255, 255/255] for i in range(len(target_pcd.points))]))

		scene_flow = scene_flow_data['scene_flow'] 
		# scene_flow[:,0] += bbox[0]
		n_match_matches = scene_flow_data['source'][scene_flow_data['valid_verts']].shape[0]
		match_matches_points = np.concatenate([viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['source'][scene_flow_data['valid_verts']])[::100], viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['source'][scene_flow_data['valid_verts']]+scene_flow[scene_flow_data['valid_verts']])[::100]], axis=0)
		match_matches_lines = [[i, i + n_match_matches] for i in range(0, n_match_matches, 1)]

		# --> Create match (unweighted) lines 
		match_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(match_matches_lines))]
		match_matches_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(match_matches_points),
			lines=o3d.utility.Vector2iVector(match_matches_lines),
		)
		match_matches_set.colors = o3d.utility.Vector3dVector(match_matches_colors)



		# color = np.load(os.path.join(self.savepath,"visible_nodes",f"{source_frame}.npy"))
		color = None
		rendered_deformed_nodes,rendered_deformed_edges = self.get_rendered_graph(self.warpfield.get_deformed_nodes(),self.graph.edges,color=color)


		# Bottom left
		canonical_mesh = self.get_model_from_tsdf()
		# rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges,color=color,trans=np.array([0, -1.0, 0.01]) * bbox)

		# Bottom right
		deformed_mesh = self.get_deformed_model_from_tsdf()
		# rendered_reduced_graph_nodes2,rendered_reduced_graph_edges2 = self.get_rendered_reduced_graph(trans=np.array([1.5, -1.5, 0.01]) * bbox)
		# rendered_deformed_nodes,rendered_deformed_edges = self.get_rendered_graph(self.warpfield.get_deformed_nodes(),self.graph.edges,color=color,trans=np.array([1.0, -1.0, 0.01]) * bbox)

		# Add matches
		# print(matches)
		# trans = np.array([0, 0, 0]) * bbox
		# matches[0].translate(trans)
		# matches[1].translate(trans)
		# if matches is not None:
			# vis.add_geometry(matches[0])
			# vis.add_geometry(matches[1])



		image_path = os.path.join(self.savepath,"images",f"{self.tsdf.frame_id}.png")
		print("Saving results to:",image_path)
		self.plot([
			# source_pcd,
			target_pcd,
			# match_matches_set,
			rendered_deformed_nodes,rendered_deformed_edges,
			# canonical_mesh,rendered_graph_nodes,rendered_graph_edges,
			deformed_mesh],"Showing frame",debug,savename=image_path,zoomed_in=True)



		# if debug:
		# 	self.vis_debug.capture_screen_image(image_path) # TODO: Returns segfault
		# else:
		# 	self.vis.capture_screen_image(image_path) # TODO: Returns segfault

		# Save results 
		# canonical_mesh.translate(np.array([0.0, 1.0, 0]) * bbox)
		o3d.io.write_triangle_mesh(os.path.join(self.savepath,f"canonical_model/{self.tsdf.frame_id}.ply"),canonical_mesh) 

		# deformed_mesh.translate(np.array([-1.0, 1.0, 0]) * bbox)
		o3d.io.write_triangle_mesh(os.path.join(self.savepath,f"deformed_model/{self.tsdf.frame_id}.ply"),deformed_mesh) 
