from mathfunc import closestPointTriangle, cross_dim_2, norm_dim_3
import numpy as np
import math
from numba import jit, njit, prange

# Generates point-triangle proximity lists (NNLt) using the linked cell algorithm
@jit(forceobj=True)
def createNNLtriangle(NNLt, coordinates, faces, nodal_idx, n_surface_nodes, n_faces, prox_skin, bounding_box, cell_width):
  mx = max(1, int(bounding_box/cell_width))  # = 40 cells, bounding_box=3.2, cell_width=0.08
  head = np.array([-1]*mx*mx*mx, dtype=np.int64)  # mx*mx*mx cells nomber, size mx*mx*mx list with all values are -1, 40*40*40 = 64000
  lists = np.zeros(n_faces, dtype=np.int64)
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  cog = np.zeros((n_faces,3), dtype=np.float64)
  xa = np.zeros(n_faces, dtype=np.int64)
  ya = np.zeros(n_faces, dtype=np.int64)
  za = np.zeros(n_faces, dtype=np.int64)
  tmp = np.zeros(n_faces, dtype=np.int64)
  xxa = np.zeros(n_surface_nodes, dtype=np.int64)
  yya = np.zeros(n_surface_nodes, dtype=np.int64)
  zza = np.zeros(n_surface_nodes, dtype=np.int64)
  cog[:] = (coordinates[faces[:,0]] + coordinates[faces[:,1]] + coordinates[faces[:,2]])/3.0
  xa[:] = (cog[:,0]+0.5*bounding_box)/bounding_box*mx
  ya[:] = (cog[:,1]+0.5*bounding_box)/bounding_box*mx
  za[:] = (cog[:,2]+0.5*bounding_box)/bounding_box*mx
  tmp[:] = mx*mx*za[:] + mx*ya[:] + xa[:]
  # print ('cog.x is ' + str(cog[0]) + ' cog y is ' + str(cog[1]) + ' cog.z is ' + str(cog[2]) + ' xa is ' + str(xa) + ' ya is ' + str(ya) + ' za is ' + str(za) + ' tmp is ' + str(tmp))
  #Divide triangle faces into cells
  lists[:] = head[tmp[:]]
  head[tmp[:]] = np.arange(0,n_faces)

  # Search cells around each surface point and build proximity list
  NNLt[:][:] = []
  xxa[:] = (coordinates[nodal_idx[:],0] + 0.5*bounding_box)/bounding_box*mx
  yya[:] = (coordinates[nodal_idx[:],1] + 0.5*bounding_box)/bounding_box*mx
  zza[:] = (coordinates[nodal_idx[:],2] + 0.5*bounding_box)/bounding_box*mx
  for i in range(n_surface_nodes):
    for xi, yi, zi in zip(range(max(0,xxa[i]-1), min(mx-1, xxa[i]+1)+1), range(max(0,yya[i]-1), min(mx-1, yya[i]+1)+1), range(max(0,zza[i]-1), min(mx-1, zza[i]+1)+1)): # Browse head list
      tri = head[mx*mx*zi + mx*yi + xi]
      while tri != -1:
        if nodal_idx[i] != faces[tri,0] and nodal_idx[i] != faces[tri,1] and nodal_idx[i] != faces[tri,2]:
          pc, ubt, vbt, wbt = closestPointTriangle(coordinates[nodal_idx[i]], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]], ub, vb, wb)
          if np.linalg.norm(pc - coordinates[nodal_idx[i]]) < prox_skin:
            NNLt[i].append(tri)
        tri = lists[tri]
		#NNLt[i] = NNLt[i,1:]
	#NNLt = make_2D_array(NNLt)

  return NNLt

# Calculate contact forces
#@jit
def contactProcess(coordinates, Ft, nodal_idx, Utold, n_surface_nodes, NNLt, faces, n_faces, bounding_box, cell_width, prox_skin, repuls_skin, contact_stiffness, mesh_spacing, gr):
  maxDist = 0.0
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  maxDist = max(norm_dim_3(coordinates[nodal_idx[:]] - Utold[:]))
  if maxDist > 0.5*(prox_skin-repuls_skin):
    NNLt = createNNLtriangle(NNLt, coordinates, faces, nodal_idx, n_surface_nodes, n_faces, prox_skin, bounding_box, cell_width) # Generates point-triangle proximity lists (NNLt[n_surface_nodes]) using the linked cell algorithm
    Utold[:] = coordinates[nodal_idx[:]]
  for i in range(n_surface_nodes): # Loop through surface points
    for tp in range(len(NNLt[i])):
      pt = nodal_idx[i] # A surface point index
      tri = NNLt[i][tp] # A proximity triangle index
      pc, ubt, vbt, wbt = closestPointTriangle(coordinates[pt], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]], ub, vb, wb)  # Find the nearest point to Barycentric
      cc = pc - coordinates[pt] # moinus to all nodes
      # closestPointTriangle returns the closest point of triangle abc to point p (returns a or b or c, if not, pt projection through the barycenter inside the triangle)
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
