import numpy as np
import math
from numba import jit, njit, prange
from mathfunc import det, cross

# Import mesh, each line as a list
def importMesh(path):
  mesh = []
  with open(path) as inputfile:
    for line in inputfile:
      mesh.append(line.strip().split(' '))
    for i in range(len(mesh)):
      mesh[i] = list(filter(None, mesh[i]))
      mesh[i] = np.array([float(a) for a in mesh[i]])

  return mesh

# Read nodes, get undeformed coordinates x y z and save them in Ut0, initialize deformed coordinates Ut
@njit(parallel=True)
def vertex(mesh):
  nn = np.int64(mesh[0][0])
  Ut0 = np.zeros((nn,3), dtype=np.float64) # Undeformed coordinates of nodes
  #Ut = np.zeros((nn,3), dtype = float) # Deformed coordinates of nodes
  for i in prange(nn):
    Ut0[i] = np.array([float(mesh[i+1][1]),float(mesh[i+1][0]),float(mesh[i+1][2])]) # Change x, y (Netgen?)
    
  Ut = Ut0 # Initialize deformed coordinates of nodes
  
  return Ut0, Ut, nn

# Read element indices (tets: index of four vertices of tetrahedra) and get number of elements (ne)
@njit(parallel=True)
def tetraVerticesIndices(mesh, nn):
  ne = np.int64(mesh[nn+1][0])
  tets = np.zeros((ne,4), dtype=np.int64) # Index of four vertices of tetrahedra
  for i in prange(ne):
    tets[i] = np.array([int(mesh[i+nn+2][1])-1,int(mesh[i+nn+2][2])-1,int(mesh[i+nn+2][4])-1,int(mesh[i+nn+2][3])-1])  # Note the switch of handedness (1,2,3,4 -> 1,2,4,3) - the code uses right handed tets
  
  return tets, ne

# Read surface triangle indices (faces: index of three vertices of triangles) and get number of surface triangles (nf)
@njit(parallel=True)
def triangleIndices(mesh, nn, ne):
  nf = np.int64(mesh[nn+ne+2][0])
  faces = np.zeros((nf,3), dtype=np.int64) # Index of three vertices of triangles
  for i in prange(nf):
    faces[i] = np.array([int(mesh[i+nn+ne+3][1])-1,int(mesh[i+nn+ne+3][2])-1,int(mesh[i+nn+ne+3][3])-1])

  return faces, nf

# Determine surface nodes and index maps
@jit
def numberSurfaceNodes(faces, nn, nf):
  nsn = 0 # Number of nodes at the surface
  SNb = np.zeros(nn, dtype = int) # SNb: Nodal index map from full mesh to surface. Initialization SNb with all 0
  '''for i in range(nn):
  SNb[i] = 0'''
  for i in range(nf):
    SNb[faces[i][0]] = 1
    SNb[faces[i][1]] = 1
    SNb[faces[i][2]] = 1
  for i in range(nn):
    if SNb[i] == 1:
      nsn += 1 # Determine surface nodes
  SN = np.zeros(nsn, dtype = int) # SN: Nodal index map from surface to full mesh
  p = 0 # Iterator
  for i in range(nn):
    if SNb[i] == 1:
      SN[p] = i
      SNb[i] = p
      p += 1

  return nsn, SN, SNb

# Return the total volume of a tetrahedral mesh
@njit(parallel=True)
def volume_mesh(Vn_init, nn, ne, tets, Ut):
  #Vn_init = np.zeros(nn, dtype = float)
  #for i in prange(nn):
   #Vn_init[i] = 0.0
  A_init = np.zeros((3,3), dtype=np.float64)
  for i in range(ne):
    n1 = tets[i][0]
    n2 = tets[i][1]
    n3 = tets[i][2]
    n4 = tets[i][3]

    x1_init = Ut[n2] - Ut[n1]
    x2_init = Ut[n3] - Ut[n1]
    x3_init = Ut[n4] - Ut[n1]
    #A_init = np.array([x1_init, x2_init, x3_init])
    A_init[0] = x1_init
    A_init[1] = x2_init
    A_init[2] = x3_init
    vol_init = det(A_init.transpose())/6.0
    Vn_init[n1] += vol_init/4.0
    Vn_init[n2] += vol_init/4.0
    Vn_init[n3] += vol_init/4.0
    Vn_init[n4] += vol_init/4.0

  Vm_init = 0.0
  for i in prange(nn):
    Vm_init += Vn_init[i]

  return Vm_init

# Mark non-growing areas
@njit(parallel=True)
def markgrowth(Ut0, nn):
  gr = np.zeros(nn, dtype = np.float64)
  for i in prange(nn):
    qp = Ut0[i]
    rqp = np.linalg.norm(np.array([(qp[0]+0.1)*0.714, qp[1], qp[2]-0.05]))
    if rqp < 0.6:
      gr[i] = max(1.0 - 10.0*(0.6-rqp), 0.0)
    else:
      gr[i] = 1.0

  return gr

# Configuration of tetrahedra at reference state (A0)
@jit(nopython=True, parallel=True)
def configRefer(Ut0, tets, ne):
  A0 = np.zeros((ne,3,3), dtype=np.float64)
  for i in range(ne):
    xr1 = Ut0[tets[i][1]] - Ut0[tets[i][0]]
    xr2 = Ut0[tets[i][2]] - Ut0[tets[i][0]]
    xr3 = Ut0[tets[i][3]] - Ut0[tets[i][0]]
    A0[i][0] = xr1 # Reference state
    A0[i][1] = xr2
    A0[i][2] = xr3
    #A0[i] = np.matrix([xr1, xr2, xr3])
    A0[i] = A0[i].transpose()

  return A0

# Configuration of a deformed tetrahedron (At)
@jit
def configDeform(Ut, tets, i):
  At = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
  x1 = Ut[tets[i][1]] - Ut[tets[i][0]]
  x2 = Ut[tets[i][2]] - Ut[tets[i][0]]
  x3 = Ut[tets[i][3]] - Ut[tets[i][0]]
  At[0] = x1
  At[1] = x2
  At[2] = x3
  #At = np.matrix([x1, x2, x3])
  At = At.transpose()

  return At

# Calculate normals of each surface triangle and apply these normals to surface nodes
@njit(parallel=True)
def normalSurfaces(Ut0, faces, SNb, nf, nsn, N0):
  for i in prange(nf):
    Ntmp = cross(Ut0[faces[i][1]] - Ut0[faces[i][0]], Ut0[faces[i][2]] - Ut0[faces[i][0]])
    N0[SNb[faces[i][0]]] += Ntmp
    N0[SNb[faces[i][1]]] += Ntmp
    N0[SNb[faces[i][2]]] += Ntmp
  #N0 = preprocessing.normalize(N0)
  for i in prange(nsn):
    N0_norm = np.linalg.norm(N0[i])
    N0[i] *= 1.0/N0_norm

  return N0

# Calculate normals of each deformed tetrahedron
@jit
def tetraNormals(N0, csn, tets, i):
  Nt = N0[csn[tets[i][0]]] + N0[csn[tets[i][1]]] + N0[csn[tets[i][2]]] + N0[csn[tets[i][3]]]
  Nt_norm = np.linalg.norm(Nt)
  Nt *= 1.0/Nt_norm

  return Nt

# Calculate undeformed (Vn0) and deformed (Vn) nodal volume
# Computes the volume measured at each point of a tetrahedral mesh as the sum of 1/4 of the volume of each of the tetrahedra to which it belongs
@jit(nopython=True, parallel=True)     #(nopython=True, parallel=True)
def volumeNodal(G, A0, tets, Ut, ne, nn):
  #for i in prange(nn):
    #Vn0[i] = 0.0
    #Vn[i] = 0.0
  Vn0 = np.zeros(nn, dtype=np.float64) #Initialize nodal volumes in reference state
  Vn = np.zeros(nn, dtype=np.float64)  #Initialize deformed nodal volumes
  At = np.zeros((3,3), dtype=np.float64)
  for i in range(ne):
    vol0 = det(np.dot(G[i], A0[i]))/6.0
    #vol0 = np.linalg.det(G[i]*np.array(A0[i]))/6.0
    Vn0[tets[i][0]] += vol0/4.0
    Vn0[tets[i][1]] += vol0/4.0
    Vn0[tets[i][2]] += vol0/4.0
    Vn0[tets[i][3]] += vol0/4.0

    #At = configDeform(Ut, tets, i)
    x1 = Ut[tets[i][1]] - Ut[tets[i][0]]
    x2 = Ut[tets[i][2]] - Ut[tets[i][0]]
    x3 = Ut[tets[i][3]] - Ut[tets[i][0]]
    At[0] = x1
    At[1] = x2
    At[2] = x3
    #At = np.array([x1, x2, x3])
    vol = det(At.transpose())/6.0
    Vn[tets[i][0]] += vol/4.0
    Vn[tets[i][1]] += vol/4.0
    Vn[tets[i][2]] += vol/4.0
    Vn[tets[i][3]] += vol/4.0

  return Vn0, Vn

# Midplane
@njit(parallel=True)
def midPlane(Ut, Ut0, Ft, SN, nsn, mpy, a, hc, K):
  for i in prange(nsn):
    pt = SN[i]
    if Ut0[pt][1] < mpy - 0.5*a and Ut[pt][1] > mpy:
      Ft[pt][1] -= (mpy - Ut[pt][1])/hc*a*a*K
    if Ut0[pt][1] > mpy + 0.5*a and Ut[pt][1] < mpy:
      Ft[pt][1] -= (mpy - Ut[pt][1])/hc*a*a*K

  return Ft

# Calculate the longitudinal length of the real brain
@jit
def longitLength(t):
  L = -0.98153*t**2+3.4214*t+1.9936
  #L = -41.6607*t**2+101.7986*t+58.843 #for the case without normalisation

  return L

# Obtain zoom parameter by checking the longitudinal length of the brain model
@jit
def paraZoom(Ut, SN, L, nsn):
  ymin = 1.0
  ymax = -1.0
  xmin = 1.0
  xmax = -1.0
  for i in range(nsn):
    xmin = min(xmin, Ut[SN[i]][0])
    xmax = max(xmax, Ut[SN[i]][0])
    ymin = min(ymin, Ut[SN[i]][1])
    ymax = max(ymax, Ut[SN[i]][1])

  # Zoom parameter
  zoom_pos = L/(xmax-xmin)

  return zoom_pos
