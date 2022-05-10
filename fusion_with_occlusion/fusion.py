# fusion.py is the main file to connect everything. 
# Run python fusion.py --datadir <path-to-RGBD-frames>

import sys
import argparse # To parse arguments 
import logging # To log info 


sys.path.append("../")  # Making it easier to load modules

# Fusion Modules 
from frame_loader import RGBDVideoLoader
from tsdf import TSDFVolume # Create main TSDF module where the 3D volume is stored
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf 
from vis import get_visualizer # Visualizer 
from run_model import Deformnet_runner # Neural Tracking Moudle 
from run_motion_model import MotionCompleteNet_Runner
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  




class DynamicFusion:
	def __init__(self,opt):
		self.frameloader = RGBDVideoLoader(opt.datadir)
		self.opt = opt 
		
		
		# For logging results
		self.log = logging.getLogger(__name__)

		# Define visualizer
		self.vis = get_visualizer(opt)

		self.model = Deformnet_runner(self.vis,opt)
		self.motion_complete_model = MotionCompleteNet_Runner(opt)	

		self.model.vis = self.vis

	def create_tsdf(self):
		# Need to load initial frame 
		self.target_frame = self.opt.source_frame
		source_data = self.frameloader.get_source_data(self.opt.source_frame)  # Get source data

		# TODO: Needs to be updated (Iterate over all frames and find max depth only of the semgmented object)
		max_depth = source_data["im"][-1].max()

		# Create a new tsdf volume
		self.tsdf = TSDFVolume(max_depth+1, source_data["intrinsics"], self.opt,self.vis)

		# Add TSDF to visualizer
		self.vis.tsdf = self.tsdf
		self.model.tsdf = self.tsdf

		# Integrate source frame 
		assert "mask" in source_data, "Source frame must contain segmented object for graph generation"
		mask = source_data["mask"]

		source_data["im"][:, mask == 0] = 0
		
		self.tsdf.integrate(source_data)


	def create_graph(self):
		# Assert TSDF already initialized  
		assert hasattr(self,'tsdf'), "TSDF not defined. Run create_tsdf first." 
	
		# Initialize graph 
		self.graph = EDGraph(self.tsdf,self.vis)

		self.tsdf.graph = self.graph 		# Add graph to tsdf		
		self.model.graph = self.graph 		# Add graph to Model 
		self.motion_complete_model.graph = self.graph # Add graph to motion compelte net model 
		self.vis.graph  = self.graph 		# Add graph to visualizer 

		assert hasattr(self,'graph'),  "Graph not defined. Run create_graph first." 

		self.warpfield = WarpField(self.graph,self.tsdf,self.vis)


		self.tsdf.warpfield = self.warpfield  # Add warpfield to tsdf
		self.model.warpfield = self.warpfield # Add warpfield to Model
		self.vis.warpfield = self.warpfield   # Add warpfield to visualizer


		self.warpfield.model = self.model

	def register_new_frame(self): 

		# Check next frame can be registered
		if self.target_frame + self.opt.skip_rate >= len(self.frameloader):
			return False,"Registration Completed"

		self.log.info(f"Registering {self.target_frame}th frame to {self.target_frame + self.opt.skip_rate}th frame")

		success,msg = self.register_frame(self.target_frame,self.target_frame + self.opt.skip_rate)
		
		# Update frame number 
		self.target_frame = self.target_frame + self.opt.skip_rate

		return success,msg

	def register_frame(self,source_frame,target_frame):
		"""
			Main step of the algorithm. Register new target frame 
			Args: 
				source_frame(int): which source frame id to integrate  
				target_frame(int): which target frame id to integrate 

			Returns:
				success(bool): Return whether sucess or failed in registering   
		"""

		source_frame_data = self.frameloader.get_source_data(source_frame)	
		target_frame_data = self.frameloader.get_target_data(target_frame,source_frame_data["cropper"])	
		# find target location of visible nodes
		optical_flow_data = self.model.estimate_optical_flow(source_frame_data,target_frame_data)

		# Get visible nodes 
		deformed_nodes_at_source = self.warpfield.get_deformed_nodes()
		visible_nodes_mask = self.tsdf.get_visible_nodes() # Assuming previous frame was used as source 

		updated_visible_nodes_mask,deformed_visible_nodes_predicted_location = self.model.get_predicted_location(optical_flow_data,deformed_nodes_at_source,visible_nodes_mask,source_frame_data["intrinsics"])[:2] 
		
		# Use occlusion fusion to find motion of complete graph
		estimated_complete_nodes_motion_data = self.motion_complete_model(source_frame_data["id"],
			deformed_nodes_at_source,
			deformed_visible_nodes_predicted_location,
			updated_visible_nodes_mask)

		# Compute optical flow and estimate transformation for graph using neural tracking
		estimated_transformations = self.model.optimize(optical_flow_data,estimated_complete_nodes_motion_data,source_frame_data["intrinsics"],target_frame_data)

		# Update warpfield parameters, warpfield maps to target frame  
		self.warpfield.update_transformations(estimated_transformations)

		# Register TSDF, tsdf maps to target frame  
		self.tsdf.integrate(target_frame_data)

		# self.vis.plot_skinned_model()

		# Add new nodes to warpfield and graph if any
		# update = self.warpfield.update_graph() 
		update = False
		
		self.vis.show(source_frame,debug=False) # plot registration details 


		# Return whether sucess or failed in registering 
		return True, f"Registered {source_frame}th frame to {target_frame}th frame. Added graph nodes:{update}"

	def clear_frame_data(self):
		if hasattr(self,'tsdf'):  self.tsdf.clear() # Clear image information, remove deformed model 
		# if hasattr(self,'warpfield'):  self.warpfield.clear() # Clear image information, remove deformed model 
		# if hasattr(self,'graph'): self.graph.clear()   # Remove estimated deformation from previous timesteps 	

			
	def __call__(self):

		# Initialize
		self.create_tsdf()
		self.create_graph()

		# self.vis.init_plot()

		# Run fusion 
		while True: 

			success, msg = self.register_new_frame()
			self.log.info(msg) # Print data

			if not success: 
				break
			self.clear_frame_data() # Reset information 

		self.vis.create_video("./results.mp4")



if __name__ == "__main__":
	args = argparse.ArgumentParser() 
	# Path locations
	args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')

	# Arguments for tsdf
	args.add_argument('--voxel_size', default=0.005, type=float, help='length of each voxel cube in TSDF')

	# For GPU
	args.add_argument('--gpu', 	  dest='gpu', action="store_true",help='Try to use GPU for faster optimization')
	args.add_argument('--no-gpu', dest='gpu', action="store_false",help='Uses CPU')
	args.set_defaults(gpu=True)

	# Arguments for loading frames 
	args.add_argument('--source_frame', default=0, type=int, help='frame index to create the deformable model')
	args.add_argument('--skip_rate', default=1, type=int, help='frame rate while running code')


	# Arguments for debugging  
	args.add_argument('--debug', default=True, type=bool, help='Whether debbugging or not. True: Logging + Visualization, False: Only Critical information and plots')
	args.add_argument('--visualizer', default='open3d', type=str, help='if debugging_level == 2, also plot registration')

	opt = args.parse_args()

	# Logging details 
	LOGFORMAT = "[%(filename)s:%(lineno)s - %(funcName)3s() ] %(message)s"
	logging.basicConfig(stream=sys.stdout, format=LOGFORMAT,level=logging.INFO if opt.debug is False else logging.DEBUG) 
	logging.getLogger('numba').setLevel(logging.WARNING)
	logging.getLogger('PIL').setLevel(logging.WARNING)



	dfusion = DynamicFusion(opt)
	dfusion()