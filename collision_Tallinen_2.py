from mathfunc import closestPointTriangle, cross_dim_2, norm_dim_3
import numpy as np
import math
from numba import jit, njit, prange

# Generates point-triangle proximity lists (NNLt) using the linked cell algorithm
@jit(forceobj=True)
def createNNLtriangle(NNLt, coordinates, faces, SN, nsn, nf, hs, bw, mw):
  mx = max(1, int(bw/mw))  # = 40 cells, bw=3.2, mw=0.08
  head = np.array([-1]*mx*mx*mx, dtype=np.int64)  # mx*mx*mx cells nomber, size mx*mx*mx list with all values are -1, 40*40*40 = 64000
  lists = np.zeros(nf, dtype=np.int64)
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  cog = np.zeros((nf,3), dtype=np.float64)
  xa = np.zeros(nf, dtype=np.int64)
  ya = np.zeros(nf, dtype=np.int64)
  za = np.zeros(nf, dtype=np.int64)
  tmp = np.zeros(nf, dtype=np.int64)
  xxa = np.zeros(nsn, dtype=np.int64)
  yya = np.zeros(nsn, dtype=np.int64)
  zza = np.zeros(nsn, dtype=np.int64)
  cog[:] = (coordinates[faces[:,0]] + coordinates[faces[:,1]] + coordinates[faces[:,2]])/3.0
  xa[:] = (cog[:,0]+0.5*bw)/bw*mx
  ya[:] = (cog[:,1]+0.5*bw)/bw*mx
  za[:] = (cog[:,2]+0.5*bw)/bw*mx
  tmp[:] = mx*mx*za[:] + mx*ya[:] + xa[:]
  # print ('cog.x is ' + str(cog[0]) + ' cog y is ' + str(cog[1]) + ' cog.z is ' + str(cog[2]) + ' xa is ' + str(xa) + ' ya is ' + str(ya) + ' za is ' + str(za) + ' tmp is ' + str(tmp))
  #Divide triangle faces into cells
  lists[:] = head[tmp[:]]
  head[tmp[:]] = np.arange(0,nf)

  # Search cells around each surface point and build proximity list
  NNLt[:][:] = []
  xxa[:] = (coordinates[SN[:],0] + 0.5*bw)/bw*mx
  yya[:] = (cooridnates[SN[:],1] + 0.5*bw)/bw*mx
  zza[:] = (coordinates[SN[:],2] + 0.5*bw)/bw*mx
  for i in range(nsn):
    for xi, yi, zi in zip(range(max(0,xxa[i]-1), min(mx-1, xxa[i]+1)+1), range(max(0,yya[i]-1), min(mx-1, yya[i]+1)+1), range(max(0,zza[i]-1), min(mx-1, zza[i]+1)+1)): # Browse head list
      tri = head[mx*mx*zi + mx*yi + xi]
      while tri != -1:
        if SN[i] != faces[tri,0] and SN[i] != faces[tri,1] and SN[i] != faces[tri,2]:
          pc, ubt, vbt, wbt = closestPointTriangle(coordinates[SN[i]], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]], ub, vb, wb)
          if np.linalg.norm(pc - coordinates[SN[i]]) < hs:
            NNLt[i].append(tri)
        tri = lists[tri]
		#NNLt[i] = NNLt[i,1:]
	#NNLt = make_2D_array(NNLt)

  return NNLt

# Calculate contact forces
#@jit
def contactProcess(coordinates, Ft, SN, Utold, nsn, NNLt, faces, nf, bw, mw, hs, hc, kc, a, gr):
  maxDist = 0.0
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  maxDist = max(norm_dim_3(coordinates[SN[:]] - Utold[:]))
  if maxDist > 0.5*(hs-hc):
    NNLt = createNNLtriangle(NNLt, coordinates, faces, SN, nsn, nf, hs, bw, mw) # Generates point-triangle proximity lists (NNLt[nsn]) using the linked cell algorithm
    Utold[:] = coordinates[SN[:]]
  for i in range(nsn): # Loop through surface points
    for tp in range(len(NNLt[i])):
      pt = SN[i] # A surface point index
      tri = NNLt[i][tp] # A proximity triangle index
      pc, ubt, vbt, wbt = closestPointTriangle(coordinates[pt], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]], ub, vb, wb)  # Find the nearest point to Barycentric
      cc = pc - coordinates[pt] # moinus to all nodes
      # closestPointTriangle returns the closest point of triangle abc to point p (returns a or b or c, if not, pt projection through the barycenter inside the triangle)
      rc = np.linalg.norm(cc)   # Distance between the closest point in the triangle to the point, sqrt(x*x+y*y+z*z)
      if rc < hc and gr[pt] + gr[faces[tri,0]] > 0.0:  # Calculate contact force if within the contact range
        cc *= 1.0/rc
        Ntri = cross_dim_2(coordinates[faces[tri,1]] - coordinates[faces[tri,0]], coordinates[faces[tri,2]] - coordinates[faces[tri,0]]) # Triangle normal
        Ntri *= 1.0/np.linalg.norm(Ntri)
        fn = cc*(rc-hc)/hc*kc*a*a # kc = 10.0*K Contact stiffness
        if np.dot(fn,Ntri) < 0.0:
          fn -= Ntri*np.dot(fn,Ntri)*2.0
        Ft[faces[tri,0]] -= fn*ubt
        Ft[faces[tri,1]] -= fn*vbt
        Ft[faces[tri,2]] -= fn*wbt
        Ft[pt] += fn

  return Ft, NNLt
