from slam import topology as stop
import scipy.sparse as sp
import numpy as np

# Computes a curvature with the umbrella operator (see Desbrun et al)
def curvatureTopologic(mesh):
    A=stop.edges_to_adjacency_matrix(mesh)
    L=sp.diags([A.sum(axis=0)],[0],A.shape)-A
    L=L.toarray()
    curvature=np.sum(np.dot(L, mesh.vertices)*mesh.vertex_normals,axis=1)
    return curvature