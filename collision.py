from mathfunc import closestPointTriangle, cross_dim_2, norm_dim_3
import numpy as np
import math
from numba import jit, njit, prange
from sklearn.neighbors import KDTree
import sys

# Generates point-triangle proximity lists (NNLt) using the linked cell algorithm
@jit
def createNNLtriangle(NNLt, coordinates, faces, nodal_idx, n_surface_nodes, n_faces, prox_skin, bw, mw):
  mx = max(1, int(bw/mw))  # = 40 cells, bw=3.2, mw=0.08
  head = [-1]*mx*mx*mx  # mx*mx*mx cells number, size mx*mx*mx list with all values are -1, 40*40*40 = 64000
  lists = [0]*n_faces
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  for i in range(n_faces):  # Divide triangle faces into cells, i index of face
    cog = (coordinates[faces[i,0]] + coordinates[faces[i,1]] + coordinates[faces[i,2]])/3.0
    xa = int((cog[0]+0.5*bw)/bw*mx)
    ya = int((cog[1]+0.5*bw)/bw*mx)
    za = int((cog[2]+0.5*bw)/bw*mx)
    tmp = mx*mx*za + mx*ya + xa
    # print ('cog.x is ' + str(cog[0]) + ' cog y is ' + str(cog[1]) + ' cog.z is ' + str(cog[2]) + ' xa is ' + str(xa) + ' ya is ' + str(ya) + ' za is ' + str(za) + ' tmp is ' + str(tmp))
    lists[i] = head[mx*mx*za + mx*ya + xa]
    head[mx*mx*za + mx*ya + xa] = i

  for i in range(n_surface_nodes):   # Search cells around each surface point and build proximity list
    pt = nodal_idx[i]
    NNLt[i][:] = []
    xa = int((coordinates[pt,0]+0.5*bw)/bw*mx)
    ya = int((coordinates[pt,1]+0.5*bw)/bw*mx)
    za = int((coordinates[pt,2]+0.5*bw)/bw*mx)

    for xi, yi, zi in zip(range(max(0,xa-1), min(mx-1, xa+1)+1), range(max(0,ya-1), min(mx-1, ya+1)+1), range(max(0,za-1), min(mx-1, za+1)+1)): # Browse head list
      tri = head[mx*mx*zi + mx*yi + xi]
      while tri != -1:
        if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:
          pc, ubt, vbt, wbt = closestPointTriangle(coordinates[pt], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]], ub, vb, wb)
          if np.linalg.norm(pc - coordinates[pt]) < prox_skin:
            NNLt[i].append(tri)
        tri = lists[tri]

  return NNLt


# Calculate contact forces
#@jit
def contactProcess(coordinates, Ft, nodal_idx, Utold, n_surface_nodes, NNLt, faces, n_faces, bounding_box, cell_width, prox_skin, repuls_skin, contact_stiffness, mesh_spacing, gr):
  maxDist = 0.0
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  maxDist = max(norm_dim_3(coordinates[nodal_idx[:]] - Utold[:]))
  if maxDist > 0.5*(prox_skin-repuls_skin):
    #NNLt = createNNLtriangle(NNLt, Ut, faces, SN, n_surface_nodes, nf, prox_skin, bw, mw) # Generates point-triangle proximity lists (NNLt[n_surface_nodes]) using the linked cell algorithm
    Utold[:] = coordinates[nodal_idx[:]]
    if np.any(np.isinf(coordinates[nodal_idx[:]])) == True or np.any(np.isnan(coordinates[nodal_idx[:]])) == True:
      print('Computational divergence')
      coordinates[nodal_idx[:]] = np.nan_to_num(coordinates[nodal_idx[:]])
    tree = KDTree(coordinates[nodal_idx[:]])
    ind = tree.query_radius(coordinates[nodal_idx[:]], r=0.5*mesh_spacing)  # Generates point-points proximity index arrays (ind) using the Kd-Tree algorithm (looks up the nearest neighbors of any point)
    ind = [[indice for indice in ind[i] if indice != i] for i in range(len(ind))]  # Remove the index of the point itself
    for i in range(n_surface_nodes):
      #ind[i] = [SN[ind[i][j]] for j in range(len(ind[i]))] # Find corresponding surface node index for "ind"
      NNLt[i] = [np.where(nodal_idx[ind[i][tp]] == faces[:,:])[0] for tp in range(len(ind[i]))]  # Find corresponding proximity triangle indexes by the nearest neighbouring points indexes of a point
      NNLt[i] = [item for sublist in NNLt[i] for item in sublist] # Merge all proximity triangle indexes for a point
      NNLt[i] = list(set(NNLt[i]))  # Remove the same proximity triangle indexes for a point
      NNLt[i] = [item for item in NNLt[i] if nodal_idx[i] != faces[item,0] and nodal_idx[i] != faces[item,1] and nodal_idx[i] != faces[item,2]] # Determine if the point is inside the proximity triangle or not
  for i in range(n_surface_nodes): # Loop through surface points
    for tp in range(len(NNLt[i])): # Loop through corresponding proximity triangles
      pt = nodal_idx[i]
      tri = NNLt[i][tp] # A proximity triangle index
      #if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:   # Determine if the point is inside the proximity triangle or not
      pc, ubt, vbt, wbt = closestPointTriangle(coordinates[pt], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]], ub, vb, wb)   # Find the closest point in the triangle to the point and barycentric coordinates of the triangle
      cc = pc - coordinates[pt]   # The closest point in the triangle subtracts to the point
      rc = np.linalg.norm(cc)   # Distance between the closest point in the triangle to the point, sqrt(x*x+y*y+z*z)
      if rc < repuls_skin and gr[pt] + gr[faces[tri,0]] > 0.0:  # Calculate contact force if within the contact range
        cc *= 1.0/rc
        Ntri = cross_dim_2(coordinates[faces[tri,1]] - coordinates[faces[tri,0]], coordinates[faces[tri,2]] - coordinates[faces[tri,0]]) # Triangle normal
        Ntri *= 1.0/np.linalg.norm(Ntri)
        fn = cc*(rc-repuls_skin)/repuls_skin*contact_stiffness*mesh_spacing*mesh_spacing # kc = 10.0*K Contact stiffness
        if np.dot(fn,Ntri) < 0.0:
          fn -= Ntri*np.dot(fn,Ntri)*2.0
        Ft[faces[tri,0]] -= fn*ubt
        Ft[faces[tri,1]] -= fn*vbt
        Ft[faces[tri,2]] -= fn*wbt
        Ft[pt] += fn

  return Ft, NNLt
