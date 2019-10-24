from slam import topology as stop
import scipy.sparse as sp
import numpy as np


# Computes a curvature with the umbrella operator (see Desbrun et al)
def curvatureTopologic(mesh):
    curvature = np.sum(np.dot(graph_laplacian(mesh), mesh.vertices) * mesh.vertex_normals, axis=1)
    return curvature


# Computes the graph laplacian (purely topological)
def graph_laplacian(mesh):
    A = stop.edges_to_adjacency_matrix(mesh)
    Lsparse = sp.diags([A.sum(axis=0)], [0], A.shape) - A
    L = Lsparse.toarray()
    return L,Lsparse
