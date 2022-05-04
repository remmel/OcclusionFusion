
import numpy as np

class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

# Base TSDF class (instead of using a volumetric representation using Mesh)
class TSDFMesh:
    def __init__(self,fopt,mesh):
        self.fopt = fopt
        self.mesh = mesh

        # Estimate normals for future use
        self.mesh.compute_vertex_normals(normalized=True)
        self.frame_id = fopt.source_frame
    def get_mesh(self):
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)         

        return vertices,faces,None,None