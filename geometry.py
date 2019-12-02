import numpy as np
import math
import numba
from numba import jit, njit, prange
from mathfunc import det_dim_3, det_dim_2, cross_dim_3, dot_mat_dim_3, transpose_dim_3, normalize_dim_3
import slam.io as sio
from scipy import spatial
from scipy.optimize import curve_fit
from slam import topology as stop
from curvatureCoarse import graph_laplacian
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import scipy.special as spe

# Import mesh, each line as a list
def importMesh(path):
  mesh = []
  with open(path) as inputfile:
    for line in inputfile:
      mesh.append(line.strip().split(' '))
    for i in range(len(mesh)):
      mesh[i] = list(filter(None, mesh[i]))
      mesh[i] = np.array([float(a) for a in mesh[i]])
  #mesh = np.asarray(mesh, dtype=np.float64)
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
  SNb = np.zeros(nn, dtype=int) # SNb: Nodal index map from full mesh to surface. Initialization SNb with all 0
  SNb[faces[:,0]] = SNb[faces[:,1]] = SNb[faces[:,2]] = 1
  for i in range(nn):
    if SNb[i] == 1:
      nsn += 1 # Determine surface nodes
  SN = np.zeros(nsn, dtype=int) # SN: Nodal index map from surface to full mesh
  p = 0 # Iterator
  for i in range(nn):
    if SNb[i] == 1:
      SN[p] = i
      SNb[i] = p
      p += 1

  return nsn, SN, SNb

# Check minimum, maximum and average edge lengths (average mesh spacing) at the surface
@jit(nopython=True, parallel=True)
def edge_length(Ut, faces, nf):
  mine = 1e9
  maxe = ave = 0.0
  for i in range(nf):
    mine = min(np.linalg.norm(Ut[faces[i,1]] - Ut[faces[i,0]]), mine)
    mine = min(np.linalg.norm(Ut[faces[i,2]] - Ut[faces[i,0]]), mine)
    mine = min(np.linalg.norm(Ut[faces[i,2]] - Ut[faces[i,1]]), mine)
    maxe = max(np.linalg.norm(Ut[faces[i,1]] - Ut[faces[i,0]]), maxe)
    maxe = max(np.linalg.norm(Ut[faces[i,2]] - Ut[faces[i,0]]), maxe)
    maxe = max(np.linalg.norm(Ut[faces[i,2]] - Ut[faces[i,1]]), maxe)
    ave += np.linalg.norm(Ut[faces[i,2]] - Ut[faces[i,1]]) + np.linalg.norm(Ut[faces[i,2]] - Ut[faces[i,0]]) + np.linalg.norm(Ut[faces[i,1]] - Ut[faces[i,0]])
  ave /= 3.0*nf

  return mine, maxe, ave

# Return the total volume of a tetrahedral mesh
@jit(nopython=True, parallel=True)
def volume_mesh(Vn_init, nn, ne, tets, Ut):
  A_init = np.zeros((ne,3,3), dtype=np.float64)
  vol_init = np.zeros(ne, dtype=np.float64)

  A_init[:,0] = Ut[tets[:,1]] - Ut[tets[:,0]]
  A_init[:,1] = Ut[tets[:,2]] - Ut[tets[:,0]]
  A_init[:,2] = Ut[tets[:,3]] - Ut[tets[:,0]]
  vol_init[:] = det_dim_3(transpose_dim_3(A_init[:]))/6.0

  for i in range(ne):
    #vol_init[i] = np.linalg.det(np.transpose(A_init[i]))/6.0
    Vn_init[tets[i,:]] += vol_init[i]/4.0

  Vm_init = np.sum(Vn_init)

  return Vm_init

# Define the label for each surface node
@jit
def tetra_labels_surface(mesh_file, method, n_clusters, Ut0, SN, tets):
  mesh = sio.load_mesh(mesh_file)
  if method.__eq__("Kmeans"):
  # 1) Simple K-means to start simply
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh.vertices)
  else:
  # 2) Another method: spectral clustering
    Lsparse = graph_laplacian(mesh)
    evals, evecs = eigs(Lsparse, k=n_clusters - 1, which='SM')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(evecs))
  #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh.vertices)
  labels = kmeans.labels_ 
  mesh.vertices[:,[0,1]]=mesh.vertices[:,[1,0]]
  # Find the nearest reference surface nodes to our surface nodes (csn) and distribute the labels to our surface nodes (labels_surface)
  tree = spatial.KDTree(mesh.vertices)
  csn = tree.query(Ut0[SN[:]])
  #labels_surface = np.zeros(nsn, dtype = np.int64)
  #labels_surface_2 = np.zeros(nsn, dtype = np.int64)
  labels_surface = kmeans.labels_[csn[1]]

  return labels_surface, labels

# Define the label for each tetrahedron
@jit
def tetra_labels_volume(Ut0, SN, tets, labels_surface):
  # Find the nearest surface nodes to barycenters of tetahedra (csn_t) and distribute the label to each tetahedra (labels_volume)
  Ut_barycenter = (Ut0[tets[:,0]] + Ut0[tets[:,1]] + Ut0[tets[:,2]] + Ut0[tets[:,3]])/4.0
  tree = spatial.KDTree(Ut0[SN[:]])
  csn_t = tree.query(Ut_barycenter[:,:])
  #labels_volume = np.zeros(ne, dtype = np.int64)
  labels_volume = labels_surface[csn_t[1]]

  return labels_volume

# Define Gaussian function for temporal growth rate
@jit
def func(x,a,c,sigma):

  return a*np.exp(-(70*x-c)**2/sigma)

# Define asymmetric normal function for temporal growth rate
@jit
def skew(X, a, e, w, p):
  X = (p*X-e)/w
  Y = 2*np.exp(-X**2/2)/np.sqrt(2*np.pi)
  Y *= 1/w*spe.ndtr(a*X)  #ndtr: gaussian cumulative distribution function

  return Y

# Curve-fit of temporal growth for each label
#@jit
def Curve_fitting(texture_file, labels, n_clusters):
  ages=[29, 29, 28, 28.5, 31.5, 32, 31, 32, 30.5, 32, 32, 31, 35.5, 35, 34.5, 35, 34.5, 35, 36, 34.5, 37.5, 35, 34.5, 36, 34.5, 33, 33]
  xdata=np.array(ages)
  tp_model = 6.926*10**(-5)*xdata**3-0.00665*xdata**2+0.250*xdata-3.0189  #time of numerical model

  texture = sio.load_texture(texture_file)

  peak=np.zeros((n_clusters,))
  amplitude=np.zeros((n_clusters,))
  latency=np.zeros((n_clusters,))
  multiple=np.zeros((n_clusters,))
  for k in range(n_clusters):
    ydata=np.mean(texture.darray[:,np.where(labels == k)[0]], axis=1)
    popt, pcov=curve_fit(func, tp_model, ydata, [0.16, 32, 25.])  #p0=[2, 32., 20., 85]) 
    peak[k]=popt[1]
    amplitude[k]=popt[0]
    latency[k]=popt[2]

  return peak, amplitude, latency

# Mark non-growing areas
@njit(parallel=True)
def markgrowth(Ut0, nn):
  gr = np.zeros(nn, dtype = np.float64)
  for i in prange(nn):
    rqp = np.linalg.norm(np.array([(Ut0[i,0]+0.1)*0.714, Ut0[i,1], Ut0[i,2]-0.05]))
    if rqp < 0.6:
      gr[i] = max(1.0 - 10.0*(0.6-rqp), 0.0)
    else:
      gr[i] = 1.0

  return gr

# Configuration of tetrahedra at reference state (A0)
@jit
def configRefer(Ut0, tets, ne):
  A0 = np.zeros((ne,3,3), dtype=np.float64)
  A0[:,0] = Ut0[tets[:,1]] - Ut0[tets[:,0]] # Reference state
  A0[:,1] = Ut0[tets[:,2]] - Ut0[tets[:,0]]
  A0[:,2] = Ut0[tets[:,3]] - Ut0[tets[:,0]]
  A0[:] = transpose_dim_3(A0[:])

  return A0

# Configuration of a deformed tetrahedron (At)
@jit
def configDeform(Ut, tets, ne):
  At = np.zeros((ne,3,3), dtype=np.float64)
  At[:,0] = Ut[tets[:,1]] - Ut[tets[:,0]]
  At[:,1] = Ut[tets[:,2]] - Ut[tets[:,0]]
  At[:,2] = Ut[tets[:,3]] - Ut[tets[:,0]]
  #At = np.matrix([x1, x2, x3])
  At[:] = transpose_dim_3(At[:])

  return At

# Calculate normals of each surface triangle and apply these normals to surface nodes
@jit(nopython=True, parallel=True) 
def normalSurfaces(Ut0, faces, SNb, nf, nsn, N0):
  Ntmp = np.zeros((nf,3), dtype=np.float64)
  Ntmp = cross_dim_3(Ut0[faces[:,1]] - Ut0[faces[:,0]], Ut0[faces[:,2]] - Ut0[faces[:,0]])
  for i in range(nf):
    N0[SNb[faces[i,:]]] += Ntmp[i]
  for i in range(nsn):
    N0[i] *= 1.0/np.linalg.norm(N0[i])
  #N0 = normalize_dim_3(N0)

  return N0

# Calculate normals of each deformed tetrahedron
@jit
def tetraNormals(N0, csn, tets, ne):
  Nt = np.zeros((ne,3), dtype=np.float64)
  Nt[:] = N0[csn[tets[:,0]]] + N0[csn[tets[:,1]]] + N0[csn[tets[:,2]]] + N0[csn[tets[:,3]]]
  Nt = normalize_dim_3(Nt)
  """for i in prange(ne):
    Nt[i] *= 1.0/np.linalg.norm(Nt[i])"""

  return Nt

# Calculate undeformed (Vn0) and deformed (Vn) nodal volume
# Computes the volume measured at each point of a tetrahedral mesh as the sum of 1/4 of the volume of each of the tetrahedra to which it belongs
@jit(nopython=True, parallel=True)   #(nopython=True, parallel=True)
def volumeNodal(G, A0, tets, Ut, ne, nn):
  Vn0 = np.zeros(nn, dtype=np.float64) #Initialize nodal volumes in reference state
  Vn = np.zeros(nn, dtype=np.float64)  #Initialize deformed nodal volumes
  At = np.zeros((ne,3,3), dtype=np.float64)
  vol0 = np.zeros(ne, dtype=np.float64)
  vol = np.zeros(ne, dtype=np.float64)
  At[:,0] = Ut[tets[:,1]] - Ut[tets[:,0]]
  At[:,1] = Ut[tets[:,2]] - Ut[tets[:,0]]
  At[:,2] = Ut[tets[:,3]] - Ut[tets[:,0]]
  vol0[:] = det_dim_3(dot_mat_dim_3(G[:], A0[:]))/6.0
  #vol0[:] = det_dim_3(dot_const_mat_dim_3(G, A0[:]))/6.0
  vol[:] = det_dim_3(transpose_dim_3(At[:]))/6.0
  for i in range(ne):
    #vol0[i] = np.linalg.det(np.dot(G[i], A0[i]))/6.0
    #vol[i] = np.linalg.det(np.transpose(At[i]))/6.0
    Vn0[tets[i,:]] += vol0[i]/4.0
    Vn[tets[i,:]] += vol[i]/4.0

  return Vn0, Vn

# Midplane
@njit(parallel=True)
def midPlane(Ut, Ut0, Ft, SN, nsn, mpy, a, hc, K):
  for i in prange(nsn):
    pt = SN[i]
    if Ut0[pt,1] < mpy - 0.5*a and Ut[pt,1] > mpy:
      Ft[pt,1] -= (mpy - Ut[pt,1])/hc*a*a*K
    if Ut0[pt,1] > mpy + 0.5*a and Ut[pt,1] < mpy:
      Ft[pt,1] -= (mpy - Ut[pt,1])/hc*a*a*K

  return Ft

# Calculate the longitudinal length of the real brain
@jit
def longitLength(t):
  #L = -0.81643*t**2+2.1246*t+1.3475
  L = -0.98153*t**2+3.4214*t+1.9936
  #L = -41.6607*t**2+101.7986*t+58.843 #for the case without normalisation

  return L

# Obtain zoom parameter by checking the longitudinal length of the brain model
@jit
def paraZoom(Ut, SN, L, nsn):
  #xmin = ymin = 1.0
  #xmax = ymax = -1.0

  xmin = min(Ut[SN[:],0])
  xmax = max(Ut[SN[:],0])
  ymin = min(Ut[SN[:],1])
  ymax = max(Ut[SN[:],1])

  # Zoom parameter
  zoom_pos = L/(xmax-xmin)

  return zoom_pos
