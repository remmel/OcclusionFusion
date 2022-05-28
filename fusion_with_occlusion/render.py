import sys
import cv2
import numpy as np
import torch
import torch.utils.dlpack as torch_dlpack
import os
import typing
from skimage import io

import open3d as o3d
# import open3d.core as o3c

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh import MeshRasterizer, RasterizationSettings, MeshRenderer, SoftPhongShader, TexturesVertex
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.structures.meshes import Meshes

sys.path.append("../")
from model.dataset import StaticCenterCrop


def make_ndc_intrinsic_matrix(image_size, intrinsic_matrix,torch_device):
	"""
	Makes an intrinsic matrix in NDC (normalized device coordinates) coordinate system
	:param image_size: size of the output image, (height, width)
	:param intrinsic_matrix: 3x3 or 4x4 projection matrix of the camera
	:param torch_device: device on which to initialize the intrinsic matrix
	:return:
	"""

	image_height, image_width = image_size
	half_image_width = image_width / 2
	half_image_height = image_height / 2

	fx_screen = intrinsic_matrix[0]
	fy_screen = intrinsic_matrix[1]
	px_screen = intrinsic_matrix[2]
	py_screen = intrinsic_matrix[3]

	# fx = fx_screen
	# fy = fy_screen
	# px = px_screen
	# py = py_screen

	fx = (640/480)*fx_screen / half_image_width
	fy = fy_screen / half_image_height
	px = -(px_screen - half_image_width) / half_image_width
	py = -(py_screen - half_image_height) / half_image_height
	# TODO due to what looks like a PyTorch3D bug, we have to use the 1.0 values here, not the below commented code
	#  values, and then use the non-identity rotation matrix...
	ndc_intrinsic_matrix = torch.tensor([[[fx, 0.0, px, 0.0],
										  [0.0, fy, py, 0.0],
										  [0.0, 0.0, 0.0, 1.0],
										  [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32, device=torch_device)
	# ndc_intrinsic_matrix = torch.tensor([[[fx, 0.0, px, 0.0],
	#                                       [0.0, fy, py, 0.0],
	#                                       [0.0, 0.0, 0.0, -1.0],
	#                                       [0.0, 0.0, -1.0, 0.0]]], dtype=torch.float32, device=torch_device)
	return ndc_intrinsic_matrix


class PyTorch3DRenderer:
	def __init__(self, image_size, intrinsic_list):
		"""
		Construct a renderer to render to the specified size using the specified device and projective camera intrinsics.
		:param image_size: tuple (height, width) for the rendered image size
		:param device: the device to use for rendering
		:param intrinsic_matrix: a 3x3 or 4x4 intrinsics tensor
		"""

		self.torch_device = torch.device("cuda:0")
		self.lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
								  specular_color=((0.0, 0.0, 0.0),), device=self.torch_device, location=[[0.0, 0.0, -3.0]])

		self.K = make_ndc_intrinsic_matrix(image_size, intrinsic_list, self.torch_device)
		# FIXME (see comments in pipeline/subprocedure_examples/pytorch3d_rendering_test.py)
		# camera_rotation = (torch.eye(3, dtype=torch.float32, device=self.torch_device)).unsqueeze(0)
		camera_rotation = np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, -1]]], dtype=np.float32)
		self.cameras: PerspectiveCameras \
			= PerspectiveCameras(device=self.torch_device,
								 R=camera_rotation,
								 T=torch.zeros((1, 3), dtype=torch.float32, device=self.torch_device),
								 K=self.K)

		self.rasterization_settings = RasterizationSettings(image_size=image_size,
															perspective_correct=True,
															cull_backfaces=False,
															cull_to_frustum=False,
															z_clip_value=0.5,
															faces_per_pixel=1)

		self.rasterizer = MeshRasterizer(self.cameras, raster_settings=self.rasterization_settings)
		self.image_size = image_size
		self.renderer = MeshRenderer(
			rasterizer=self.rasterizer,
			shader=SoftPhongShader(
				device=self.torch_device,
				cameras=self.cameras,
				lights=self.lights
			)
		)

	def render_mesh(self, mesh: o3d.geometry.TriangleMesh,
					extrinsics = None,
					depth_scale=1000.0) -> typing.Tuple[np.ndarray, np.ndarray]:
		"""
		Render mesh to depth & color images compatible with typical RGB-D input depth & rgb images
		If the extrinsics matrix is provided, camera extrinsics are also updated for all subsequent renderings.
		Otherwise, the previous extrinsics are used. If no extrinsics were ever specified, uses an identity transform.
		:param mesh: the mesh to render
		:param extrinsics: an optional 4x4 camera transformation matrix.
		:param depth_scale: factor to scale depth (meters) by, commonly 1,000 in off-the-shelf RGB-D sensors
		:return:
		"""


		vertices_numpy = np.array(mesh.vertices, dtype=np.float32)

		vertex_colors_numpy = np.fliplr(np.array(mesh.vertex_colors, dtype=np.float32)).copy()
		faces_numpy = np.array(mesh.triangles, dtype=np.int64)
		print(np.max(vertices_numpy,axis=0))

		vertices_torch = torch.from_numpy(vertices_numpy).cuda().unsqueeze(0)
		vertices_rgb = torch.from_numpy(vertex_colors_numpy).cuda().unsqueeze(0)
		textures = TexturesVertex(verts_features=vertices_rgb)
		faces_torch = torch.from_numpy(faces_numpy).cuda().unsqueeze(0)

		meshes_torch3d = Meshes(vertices_torch, faces_torch, textures)

		fragments = self.rasterizer.forward(meshes_torch3d)
		rendered_depth = fragments.zbuf.cpu().numpy().reshape(self.image_size[0], self.image_size[1])

		mask = rendered_depth != -1.0
		rendered_depth[~mask] = 0.0
		rendered_depth *= depth_scale
		rendered_depth = rendered_depth.astype(np.int16)

		images = self.renderer(meshes_torch3d)
		rendered_color = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
		rendered_color[~mask] = 0
		rendered_color = rendered_color[:,:,[2,1,0]]

		rendered_mask = mask.astype(bool)

		return rendered_depth, rendered_color,rendered_mask


if __name__ == "__main__":
	import matplotlib.pyplot as plt 
	mesh = sys.argv[1] 
	intrinsic_matrix = np.loadtxt(os.path.join(os.path.dirname(mesh),"../../","intrinsics.txt"))

	# intrinsic_matrix[1,2] += (480 - 480) / 2
	# intrinsic_matrix[0,2] += (640 - 640) / 2



	depth_path = os.path.join(os.path.dirname(mesh),"../../depth", f"00{int(mesh.split('/')[-1].split('.')[0])}.png".zfill(4))
	depth_image = io.imread(depth_path).astype(np.int16)


	color_path = os.path.join(os.path.dirname(mesh),"../../color", f"00{int(mesh.split('/')[-1].split('.')[0])}.png".zfill(4))
	color_image = io.imread(color_path)

	mask_path = os.path.join(os.path.dirname(mesh),"../../mask", f"001.png")
	mask_image = (io.imread(mask_path,as_gray=True) > 0).astype(np.float32)


	mesh = o3d.io.read_triangle_mesh(mesh)
	renderer = PyTorch3DRenderer((480,640),[intrinsic_matrix[0,0],intrinsic_matrix[1,1],intrinsic_matrix[0,2],intrinsic_matrix[1,2]])
	depth,color,mask = renderer.render_mesh(mesh)

	cropper = StaticCenterCrop(color_image.shape[:2], (480, 640))
	depth_image = cropper(depth_image)
	color_image = cropper(color_image)
	mask_image = cropper(mask_image)


	fig = plt.figure()
	ax = fig.add_subplot(331)
	ax.imshow(depth)


	ax = fig.add_subplot(334)
	ax.imshow(depth_image)

	ax = fig.add_subplot(337)
	ax.imshow(depth_image - depth)

	print(depth_image.max())
	print(depth.max())
	print("Max Differnce:", np.max(depth_image - depth))

	depth_diff = depth_image - depth
	print(depth_diff[depth_diff > 0])


	ax = fig.add_subplot(332)
	ax.imshow(color)

	ax = fig.add_subplot(335)
	ax.imshow(color_image)

	ax = fig.add_subplot(338)
	ax.imshow(color_image - color)

	# print(np.mean(color_image - color,axis=(0,1))/255)


	ax = fig.add_subplot(333)
	ax.imshow(mask)

	ax = fig.add_subplot(336)
	ax.imshow(mask_image)

	ax = fig.add_subplot(339)
	ax.imshow(mask_image - mask)


	plt.show()