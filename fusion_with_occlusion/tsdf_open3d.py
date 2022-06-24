# Perform Non-rigid fusion using open3d instead of pycuda 

import os 
import sys
import numpy as np
import open3d as o3d 



def integrate_random_image(datadir): 
volume = o3d.pipelines.integration.ScalableTSDFVolume(
	voxel_length=4.0 / 512.0,
	sdf_trunc=0.04,
	color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


color = o3d.io.read_image(os.path.join(datadir,'color',"0000.png"))
depth = o3d.io.read_image(os.path.join(datadir,'depth',"0000.png"))

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
	color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)

camera = np.loadtxt(os.path.join(datadir,"intrinsics.txt"))
camera = o3d.camera.PinholeCameraIntrinsic(512,500,*camera[ [0,1,0,1],[0,1,2,2]])

volume.integrate(
	rgbd,
	camera,
	np.eye(4))

	mesh = volume.extract_triangle_mesh()
	mesh.compute_vertex_normals()
	o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__": 
	datadir = sys.argv[1]
	integrate_random_image(datadir)