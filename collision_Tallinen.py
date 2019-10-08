from mathfunc import closestPointTriangle, cross_dim_2, norm_dim_3
import numpy as np
import math
from numba import jit, njit, prange

# Generates point-triangle proximity lists (NNLt) using the linked cell algorithm
@jit(forceobj=True)
def createNNLtriangle(NNLt, Ut, faces, SN, nsn, nf, hs, bw, mw):
  mx = max(1, int(bw/mw))  # = 40 cells, bw=3.2, mw=0.08
  head = [-1]*mx*mx*mx # mx*mx*mx cells nomber, size mx*mx*mx list with all values are -1, 40*40*40 = 64000
  lists = [0]*nf
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  for i in range(nf):  # Divide triangle faces into cells, i index of face
    cog = (Ut[faces[i,0]] + Ut[faces[i,1]] + Ut[faces[i,2]])/3.0
    xa = int((cog[0]+0.5*bw)/bw*mx)
    ya = int((cog[1]+0.5*bw)/bw*mx)
    za = int((cog[2]+0.5*bw)/bw*mx)
    tmp = mx*mx*za + mx*ya + xa
    # print ('cog.x is ' + str(cog[0]) + ' cog y is ' + str(cog[1]) + ' cog.z is ' + str(cog[2]) + ' xa is ' + str(xa) + ' ya is ' + str(ya) + ' za is ' + str(za) + ' tmp is ' + str(tmp))
    lists[i] = head[mx*mx*za + mx*ya + xa]
    head[mx*mx*za + mx*ya + xa] = i

  for i in range(nsn):   # Search cells around each surface point and build proximity list
    pt = SN[i]
    NNLt[i][:] = []
    xa = int((Ut[pt,0]+0.5*bw)/bw*mx)
    ya = int((Ut[pt,1]+0.5*bw)/bw*mx)
    za = int((Ut[pt,2]+0.5*bw)/bw*mx)

    for xi, yi, zi in zip(range(max(0,xa-1), min(mx-1, xa+1)+1), range(max(0,ya-1), min(mx-1, ya+1)+1), range(max(0,za-1), min(mx-1, za+1)+1)): # Browse head list
      tri = head[mx*mx*zi + mx*yi + xi]
      while tri != -1:
        if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:
          pc, ubt, vbt, wbt = closestPointTriangle(Ut[pt], Ut[faces[tri,0]], Ut[faces[tri,1]], Ut[faces[tri,2]], ub, vb, wb)
          if np.linalg.norm(pc - Ut[pt]) < hs:
            NNLt[i].append(tri)
        tri = lists[tri]

  return NNLt

# Calculate contact forces
@jit(forceobj=True)
def contactProcess(Ut, Ft, SN, Utold, nsn, NNLt, faces, nf, bw, mw, hs, hc, kc, a, gr):
  maxDist = 0.0
  ub = vb = wb = 0.0  # Barycentric coordinates of triangles
  maxDist = max(norm_dim_3(Ut[SN[:]] - Utold[:]))
  if maxDist > 0.5*(hs-hc):
    NNLt = createNNLtriangle(NNLt, Ut, faces, SN, nsn, nf, hs, bw, mw) # Generates point-triangle proximity lists (NNLt[nsn]) using the linked cell algorithm
    for i in range(nsn):
      Utold[i] = Ut[SN[i]]
  for i in range(nsn): # Loop through surface points
    for tp in range(len(NNLt[i])):
      pt = SN[i] # A surface point index
      tri = NNLt[i][tp] # A proximity triangle index
      pc, ubt, vbt, wbt = closestPointTriangle(Ut[pt], Ut[faces[tri,0]], Ut[faces[tri,1]], Ut[faces[tri,2]], ub, vb, wb)  # Find the nearest point to Barycentric
      cc = pc - Ut[pt] # moinus to all nodes
      # closestPointTriangle returns the closest point of triangle abc to point p (returns a or b or c, if not, pt projection through the barycenter inside the triangle)
      rc = np.linalg.norm(cc)   # Distance between the closest point in the triangle to the point, sqrt(x*x+y*y+z*z)
      if rc < hc and gr[pt] + gr[faces[tri][0]] > 0.0:  # Calculate contact force if within the contact range
        cc *= 1.0/rc
        Ntri = cross_dim_2(Ut[faces[tri,1]] - Ut[faces[tri,0]], Ut[faces[tri,2]] - Ut[faces[tri,0]]) # Triangle normal
        Ntri *= 1.0/np.linalg.norm(Ntri)
        fn = cc*(rc-hc)/hc*kc*a*a # kc = 10.0*K Contact stiffness
        if np.dot(fn,Ntri) < 0.0:
          fn -= Ntri*np.dot(fn,Ntri)*2.0
        Ft[faces[tri,0]] -= fn*ubt
        Ft[faces[tri,1]] -= fn*vbt
        Ft[faces[tri,2]] -= fn*wbt
        Ft[pt] += fn

  return Ft, NNLt
