import os
import cv2
import numpy as np
import logging
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage import io
from PIL import Image
from timeit import default_timer as timer
from easydict import EasyDict as edict
import datetime
import argparse
from .geometry import *

from scipy.spatial.transform import Rotation as R
import yaml
import matplotlib.pyplot as plt
import torch
from lietorch import SO3, SE3, LieGroupParameter
import torch.optim as optim
from .loss import *
from .point_render import PCDRender

def load_config(config_path):
    with open(config_path,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = edict(config)  

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')

    return config


class Registration():


    def __init__(self, canonical_vertices, graph,warpfield, K,vis):


        self.config = load_config(os.path.join(os.path.dirname(__file__),"../config.yaml"))
        self.device = self.config.device

        self.deformation_model = self.config.deformation_model
        self.intrinsics = K

        self.graph = graph
        self.warpfield = warpfield 

        self.update(canonical_vertices)

        """define differentiable pcd renderer"""
        # self.renderer = PCDRender(K, img_size=image_size)
        self.renderer = PCDRender(K)

        self.vis = vis

        self.prev_R = None
        self.prev_rot = None
        self.prev_trans = None

        self.log = logging.getLogger(__name__)

    def update(self,canonical_vertices):

        """initialize deformation graph"""
        self.graph_nodes = torch.from_numpy(self.graph.nodes).to(self.device)
        self.graph_edges = torch.from_numpy(self.graph.edges).long().to(self.device)
        self.graph_edges_weights = torch.from_numpy(self.graph.edges_weights).to(self.device)
        self.graph_clusters = torch.from_numpy(self.graph.clusters).long() #.to(self.device)

        vert_anchors,vert_weights,valid_verts = self.warpfield.skin(canonical_vertices)

        # Find visible vertices and update

        """initialize point clouds"""
        # valid_pixels = torch.sum(data['pixel_anchors'], dim=-1) > -4
        self.valid_source_verts = valid_verts
        self.source_pcd = torch.from_numpy(canonical_vertices[valid_verts]).to(self.device)
        self.point_anchors = torch.from_numpy(vert_anchors[valid_verts]).long().to(self.device)
        self.anchor_weight = torch.from_numpy(vert_weights[valid_verts]).to(self.device)
        self.anchor_loc = self.graph_nodes[self.point_anchors].to(self.device)
        self.frame_point_len = [ len(self.source_pcd)]


        # print("New verts:",deformed_vertices.shape)
        # print("Valid verts:",valid_verts.shape)
        # print("Valid verts:",valid_verts)
        # print("Old:",self.source_pcd.shape)


        """pixel to pcd map"""
        # self.pix_2_pcd_map = [ self.map_pixel_to_pcd(valid_pixels).to(self.config.device) ] # TODO


    def optimize(self, optical_flow_data,scene_flow_data,complete_node_motion_data,target_frame_data, landmarks=None):
        """
        :param tgt_depth_path:
        :return:
        """

        """load target frame"""
        tgt_depth = target_frame_data["im"][-1]
        depth_mask = torch.from_numpy(tgt_depth > 0)
        tgt_pcd = depth_2_pc(tgt_depth, self.intrinsics).transpose(1,2,0)
        self.tgt_pcd = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float().to(self.device)
        pix_2_pcd = self.map_pixel_to_pcd( depth_mask ).to(self.device)

        target_matches = scene_flow_data["target_matches"]

        target_matches = target_matches[self.valid_source_verts] # Only those matches for which source has valid skinning

        target_matches = torch.from_numpy(target_matches).float().to(self.device)
        valid_verts = scene_flow_data["valid_verts"]

        landmarks = np.array([(i,i) for i in range(self.source_pcd.shape[0]) if valid_verts[i]]).T
        # if landmarks is not None: # Marks the correspondence between source and the target pixel in depth images
        #     s_uv , t_uv = landmarks
        #     s_id = self.pix_2_pcd_map[-1][ s_uv[:,1], s_uv[:,0] ]
        #     t_id = pix_2_pcd [ t_uv[:,1], t_uv[:,0]]
        #     valid_id = (s_id>-1) * (t_id>-1)
        #     s_ldmk = s_id[valid_id]
        #     t_ldmk = t_id[valid_id]

        #     landmarks = (s_ldmk, t_ldmk)

        # self.visualize_results(self.tgt_pcd)

        target_graph_node_location, target_graph_node_confidence = complete_node_motion_data
        target_graph_node_location = torch.from_numpy(target_graph_node_location).to(self.device) if target_graph_node_location is not None else None
        target_graph_node_confidence = torch.from_numpy(target_graph_node_confidence).to(self.device) if target_graph_node_confidence is not None else None


        estimated_transforms = self.solve(target_matches=target_matches,landmarks=landmarks,\
            target_graph_node_location=target_graph_node_location,
            target_graph_node_confidence=target_graph_node_confidence)
        # self.visualize_results( self.tgt_pcd, estimated_transforms["warped_verts"])

        estimated_transforms["source_frame_id"] = optical_flow_data["source_id"]
        estimated_transforms["target_frame_id"] = optical_flow_data["target_id"]

        # print(estimated_transforms)
        return estimated_transforms



    def solve(self, **kwargs ):


        if self.deformation_model == "ED":
            # Embeded_deformation, c.f. https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
            return self.optimize_ED(**kwargs)

    
    def deform_ED(self,points,anchors,weights,valid_pts,batch_size=128**3):

        # convert to pytorch
        with torch.no_grad():
            return_points = points.copy()

            # Run in batches since above certain points leads to out of memory issue
            chunks = [ (i,min(points.shape[0],i + batch_size)) for i in range(0, points.shape[0], batch_size)]
            
            for chunk in chunks:  
                l,r = chunk
                points_batch = torch.from_numpy(points[l:r]).to(self.device)
                weights_batch = torch.from_numpy(weights[l:r]).to(self.device)
                anchors_batch = anchors[l:r]

                anchor_trn_batch = self.t[anchors_batch]
                anchor_rot_batch = self.R[anchors_batch]
                anchor_loc_batch = self.graph_nodes[anchors_batch]

                warped_points_batch = ED_warp(points_batch, anchor_loc_batch, anchor_rot_batch, anchor_trn_batch, weights_batch)        

                valid_pts_batch = valid_pts[l:r]

                return_points[l:r][valid_pts_batch] = warped_points_batch[valid_pts_batch].detach().cpu().numpy() # Returns points padded with invalid points



            return return_points


    def optimize_ED(self, target_matches=None, landmarks=None,
            target_graph_node_location=None,
            target_graph_node_confidence=None):
        '''
        :param landmarks:
        :return:
        '''

        rotations,translations = self.warpfield.get_transformation_wrt_origin(self.warpfield.rotations,self.warpfield.translations)

        """translations"""
        if self.prev_trans is None:
            node_translations = torch.zeros_like(self.graph_nodes)
        else: 
            node_translations = self.prev_trans.detach().clone()
        # node_translations = torch.from_numpy(translations).float().to(self.device)
        # print(node_translations)
        self.t = torch.nn.Parameter(node_translations)
        self.t.requires_grad = True

        """rotations"""
        if self.prev_rot is None:
            phi = torch.zeros_like(self.graph_nodes)
            node_rotations = SO3.exp(phi)
        else: 
            node_rotations = SO3(self.prev_rot.detach().clone())
        # node_rotations = R.from_matrix(rotations).as_quat()
        # node_rotations = torch.from_numpy(node_rotations).float().to(self.device)
        # node_rotations = SO3(node_rotations)

        # print(node_rotations)

        self.R = LieGroupParameter(node_rotations)


        """optimizer setup"""
        optimizer = optim.Adam([self.R, self.t], lr= self.config.lr )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config.gamma)


        """render reference pcd"""
        sil_tgt, d_tgt, _ = self.render_pcd( self.tgt_pcd )

        convergence_info = {                
                "lmdk":[],
                "arap":[],
                "total":[],
                "depth":[],
                "chamfer":[],
                "silh":[],
                "motion": [],
                "smooth_rot": [],
                "smooth_trans": [],
                "grad_trans":[],
                "grad_rot":[]}

        # Transform points
        for i in range(self.config.iters):

            anchor_trn = self.t [self.point_anchors]
            anchor_rot = self.R [ self.point_anchors]
            # print(self.source_pcd.shape, self.anchor_loc.shape, anchor_rot.shape, anchor_trn.shape, self.anchor_weight.shape)
            warped_pcd = ED_warp(self.source_pcd, self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)

            err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)
            if target_matches is None:
                err_ldmk = landmark_cost(warped_pcd, self.tgt_pcd, landmarks) if landmarks is not None else 0
            else:    
                err_ldmk = landmark_cost(warped_pcd, target_matches, landmarks) if landmarks is not None else 0

            if self.config.w_silh > 0 or self.config.w_depth > 0: 
                sil_src, d_src, _ = self.render_pcd(warped_pcd) 
            err_silh = silhouette_cost(sil_src, sil_tgt) if self.config.w_silh > 0 else 0
            err_depth,depth_error_image = projective_depth_cost(d_src, d_tgt) if self.config.w_depth > 0 else 0,0

            # print(self.graph_nodes  + self.t)
            # print(target_graph_node_location)

            err_motion = occlusion_fusion_graph_motion_cost(self.graph_nodes,self.t,
                    target_graph_node_location,
                    target_graph_node_confidence) if self.config.w_motion > 0 and target_graph_node_location is not None and target_graph_node_confidence is not None else 0


            # print("Motion Error:",err_motion)
            cd = chamfer_dist(warped_pcd, self.tgt_pcd) if self.config.w_chamfer > 0 else 0

        
            err_smooth_trans = ((self.t - self.prev_trans)**2).mean() if self.prev_trans is not None and self.config.w_smooth_trans > 0 else 0     
            err_smooth_rot = ((self.R - self.prev_R)**2).mean() if self.prev_R is not None and self.config.w_smooth_rot > 0 else 0

            loss = \
                err_arap * self.config.w_arap + \
                err_ldmk * self.config.w_ldmk + \
                err_silh * self.config.w_silh + \
                err_depth * self.config.w_depth + \
                cd * self.config.w_chamfer + \
                err_motion * self.config.w_motion + \
                err_smooth_rot * self.config.w_smooth_rot + \
                err_smooth_trans * self.config.w_smooth_trans
            

            if loss.item() < 1e-7:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()    

            lr = optimizer.param_groups[0]["lr"]
            # print((warped_pcd[landmarks[0]]-target_matches[landmarks[1]]).max())
            print(f"Frame:{self.warpfield.frame_id}" + "\t-->Iteration: {0}. Lr:{1:.5f} Loss: arap = {2:.3f}, ldmk = {3:.6f}, chamher:{4:.6f} silh = {5:.3f} depth = {6:.7f} motion = {7:.7f} smooth rot:{8:.7f} smooth trans:{9:.7f} total = {10:.3f}".format(i, lr,err_arap, err_ldmk, cd, err_silh,err_depth,err_motion,err_smooth_rot,err_smooth_trans,loss.item()))


            convergence_info["lmdk"].append(np.sqrt(err_ldmk.item()/landmarks.shape[0]) if landmarks is not None else 0)
            convergence_info["arap"].append(err_arap.item())
            convergence_info["chamfer"].append(cd.item() if self.config.w_chamfer > 0 else 0 )
            convergence_info["silh"].append(err_silh.item() if self.config.w_silh > 0 else 0 )
            convergence_info["depth"].append(err_depth.item() if self.config.w_depth > 0 else 0 )
            convergence_info["motion"].append(err_motion.item() if self.config.w_motion > 0 and target_graph_node_location is not None else 0 )
            convergence_info["smooth_trans"].append(err_smooth_trans.item() if self.prev_trans is not None and self.config.w_smooth_trans > 0 else 0 )
            convergence_info["smooth_rot"].append(err_smooth_rot.item() if self.prev_R is not None and self.config.w_smooth_rot > 0 else 0 )
            convergence_info["total"].append(loss.item())

        # # Smooth graph using just arap
        # for i in range(100):

        #     err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)

        #     if err_arap.item() < 1e-7:
        #         break

        #     optimizer.zero_grad()
        #     err_arap.backward()
        #     optimizer.step()
        #     scheduler.step()    

        #     lr = optimizer.param_groups[0]["lr"]
        #     print("\t-->Smoothing Iteration: {0}. Lr:{1:.5f} Loss: arap = {2:.3f}".format(i, lr,err_arap))

        # only chamfer and arap loss
        # for i in range(100):

        #     anchor_trn = self.t [self.point_anchors]
        #     anchor_rot = self.R [ self.point_anchors]
        #     # print(self.source_pcd.shape, self.anchor_loc.shape, anchor_rot.shape, anchor_trn.shape, self.anchor_weight.shape)
        #     warped_pcd = ED_warp(self.source_pcd, self.anchor_loc, anchor_rot, anchor_trn, self.anchor_weight)

        #     err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)

        #     # print("Motion Error:",err_motion)
        #     cd = chamfer_dist(warped_pcd, self.tgt_pcd) if self.config.w_chamfer > 0 else 0

        #     loss = \
        #         err_arap * self.config.w_arap + \
        #         cd * self.config.w_chamfer/10

        #     if err_arap.item() < 1e-7:
        #         break

        #     optimizer.zero_grad()
        #     err_arap.backward()
        #     optimizer.step()
        #     scheduler.step()    

        #     lr = optimizer.param_groups[0]["lr"]
        #     print(f"Frame:{self.warpfield.frame_id}" + "\t-->Final Smoothing Iteration: {0}. Lr:{1:.5f} Loss: chamfer:{2:.7f} arap = {3:.7f} total: {4:.7f}".format(i, lr,err_arap,cd, loss.item()))


        #     # Plot results:
        #     if self.warpfield.frame_id > 45:
        #         deformed_nodes = (self.graph_nodes+self.t).detach().cpu().numpy()
        #         targtarget_graph_node_location = target_graph_node_location.detach().cpu().numpy() if target_graph_node_location is not None else None 
        #         self.vis.plot_optimization_sceneflow(self.warpfield.frame_id,i,warped_pcd.detach().cpu().numpy(),self.tgt_pcd.detach().cpu().numpy(),target_matches.detach().cpu().numpy(),deformed_nodes,landmarks,target_graph_node_location, debug=False)
        #          # self.vis.plot_depth_images([sil_src,d_src,sil_tgt,d_tgt,depth_error_image],savename=f"RenderedImages_{i}.png")

        quat_data = self.R.retr().data

        self.prev_R = self.R.detach()
        self.prev_rot = quat_data.clone()
        self.prev_trans = self.t.detach().clone()

        quat_data_np = quat_data.detach().cpu().numpy()
        rotmats = R.from_quat(quat_data_np).as_matrix()
        t = self.t.cpu().data.numpy()
        estimated_transforms = {'warped_verts':warped_pcd,\
        "node_rotations":rotmats, "node_translations":self.t.cpu(),\
        "deformed_nodes_to_target": self.graph_nodes + self.t,
        "convergence_info":convergence_info}

        return estimated_transforms

    def render_pcd (self, x):
        INF = 0
        px, dx = self.renderer(x)
        px, dx  = map(lambda feat: feat.squeeze(), [px, dx ])
        dx[dx < 0] = INF
        mask = px[..., 0] > 0
        return px, dx, mask


    def map_pixel_to_pcd(self, valid_pix_mask):
        ''' establish pixel to point cloud mapping, with -1 filling for invalid pixels
        :param valid_pix_mask:
        :return:
        '''
        image_size = valid_pix_mask.shape
        pix_2_pcd_map = torch.cumsum(valid_pix_mask.view(-1), dim=0).view(image_size).long() - 1
        pix_2_pcd_map [~valid_pix_mask] = -1
        return pix_2_pcd_map


    def visualize_results(self, tgt_pcd, warped_pcd=None):

        # import mayavi.mlab as mlab
        import open3d as o3d
        c_red = (224. / 255., 0 / 255., 125 / 255.)
        c_pink = (224. / 255., 75. / 255., 232. / 255.)
        c_blue = (0. / 255., 0. / 255., 255. / 255.)
        scale_factor = 0.007
        source_pcd = self.source_pcd.cpu().numpy()
        tgt_pcd = tgt_pcd.cpu().numpy()

        # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
        if warped_pcd is None:
            # mlab.points3d(source_pcd[ :, 0], source_pcd[ :, 1], source_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_red)
            warped_o3d = o3d.geometry.PointCloud()
            warped_o3d.points = o3d.utility.Vector3dVector(source_pcd)
            warped_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array(c_red).reshape((1,3)),(source_pcd.shape[0],1)))
        else:
            warped_pcd = warped_pcd.detach().cpu().numpy()
            warped_o3d = o3d.geometry.PointCloud()
            warped_o3d.points = o3d.utility.Vector3dVector(warped_pcd)
            warped_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array(c_pink).reshape((1,3)),(source_pcd.shape[0],1)))
            # mlab.points3d(warped_pcd[ :, 0], warped_pcd[ :, 1], warped_pcd[:,  2], resolution=4, scale_factor=scale_factor , color=c_pink)

        target_o3d = o3d.geometry.PointCloud()
        target_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
        target_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array(c_blue).reshape((1,3)),(source_pcd.shape[0],1)))
        # mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_blue)
        # mlab.show()
        o3d.visualization.draw_geometries([warped_o3d,target_o3d])