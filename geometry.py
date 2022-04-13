import numpy as np
from numba import jit, njit, prange
from mathfunc import det_dim_3, cross_dim_3, dot_mat_dim_3, transpose_dim_3, normalize_dim_3, normalize
import slam.io as sio
from scipy import spatial
from scipy.optimize import curve_fit
from curvatureCoarse import graph_laplacian
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import scipy.special as spe

'''
Structure of netgen mesh data:
  mesh [0] = number of nodes
  mesh[1:6761]: nodes, defined as 3 point numpy arrays

  mesh [6762] = number of tets
  mesh [6763:42259]: tests, defines as 1 indicator for handeness + 4 summits.

  mesh [42260] = number of faces
  mesh [42261:46424] = faces, defines as 1 indicator for handeness + 3 indexes.

'''

def netgen_to_array(path):
  '''
  Converts a netgen mesh to a list of np.arrays

  Args:
  path (string): path of mesh file

  Returns:
  mesh (list): list of np.arrays
  
  '''
  mesh = []
  with open(path) as inputfile:
    for line in inputfile:
      mesh.append(line.strip().split(' '))
    for i in range(len(mesh)):
      mesh[i] = list(filter(None, mesh[i]))
      mesh[i] = np.array([float(a) for a in mesh[i]])
  return mesh

@njit(parallel=True)
def get_nodes(mesh):
  '''
  Extract coordinates and number of nodes from mesh

  Args:
  mesh (list): mesh file as a list of np.arrays

  Returns:
  coordinates (np.array): list of 3 cartesian points
  n_nodes (int): number of nodes
  '''
  n_nodes = np.int64(mesh[0][0])
  coordinates = np.zeros((n_nodes,3), dtype=np.float64) # Undeformed coordinates of nodes
  for i in prange(n_nodes):
    coordinates[i] = np.array([float(mesh[i+1][1]),float(mesh[i+1][0]),float(mesh[i+1][2])]) # Change x, y (Netgen)
  
  return coordinates, n_nodes

@njit(parallel=True)
def get_tetra_indices(mesh, n_nodes):
  '''
  Takes a list of arrays as an input and returns tets and number of tets. Tets are defined as 4 indexes of vertices from the coordinate list

  Args:
  mesh (list): mesh file as a list of np.arrays
  n_nodes (int): number of nodes

  Returns:
  tets (np.array): list of 4 vertices indexes
  n_tets(int): number of tets in the mesh
  '''
  n_tets = np.int64(mesh[n_nodes+1][0])
  tets = np.zeros((n_tets,4), dtype=np.int64) # Index of four vertices of tetrahedra
  for i in prange(n_tets):
    tets[i] = np.array([int(mesh[i+n_nodes+2][1])-1,int(mesh[i+n_nodes+2][2])-1,int(mesh[i+n_nodes+2][4])-1,int(mesh[i+n_nodes+2][3])-1])  # Note the switch of handedness (1,2,3,4 -> 1,2,4,3) - the code uses right handed tets
  
  return tets, n_tets

@njit(parallel=True)
def get_face_indices(mesh, n_nodes, n_tets):
  '''
  Takes a list of arrays as an input and returns faces and number of faces. Faces are defined as 3 indexes of vertices from the coordinate list, only on the surface

  Args:
  mesh (list): mesh file as a list of np.arrays
  n_nodes (int): number of nodes

  Returns:
  faces (np.array): list of 3 vertices indexes
  n_faces(int): number of faces in the mesh
  '''
  n_faces = np.int64(mesh[n_nodes+n_tets+2][0])
  faces = np.zeros((n_faces,3), dtype=np.int64) # Index of three vertices of triangles
  for i in prange(n_faces):
    faces[i] = np.array([int(mesh[i+n_nodes+n_tets+3][1])-1,int(mesh[i+n_nodes+n_tets+3][2])-1,int(mesh[i+n_nodes+n_tets+3][3])-1])

  return faces, n_faces

@njit
def get_nb_surface_nodes(faces, n_nodes):
  """
  Define number of surface nodes and nodal indexes used for contact processing and NNLt triangle
  Args:
  faces (numpy array): faces index
  n_nodes (int): number of nodes
  n_faces (int): number of faces
  Returns:
  n_surface_nodes (int): Number of surface nodes
  nodal_idx (list): nodal index map from surface to full mesh, stops at last surface node
  nodal_idxb (list): nodal index map from full mesh to surface, non surface nodes are labelled with 0
  """
  n_surface_nodes = 0 

  nodal_idx_b = np.zeros(n_nodes, dtype=np.int64) # Successive nodes indices which are surface nodes
  nodal_idx_b[faces[:,0]] = nodal_idx_b[faces[:,1]] = nodal_idx_b[faces[:,2]] = 1

  for i in range(n_nodes):
    if nodal_idx_b[i] == 1:
      n_surface_nodes += 1 # Determine surface nodes

  nodal_idx = np.zeros(n_surface_nodes, dtype=np.int64) # Successive count of surface nodes, in the total-nodes-size array

  p = 0 
  for i in range(n_nodes):
    if nodal_idx_b[i] == 1:
      nodal_idx[p] = i
      nodal_idx_b[i] = p
      p += 1

  return n_surface_nodes, nodal_idx, nodal_idx_b

@jit(nopython=True, parallel=True)
def edge_length(coordinates, faces, n_faces):
  """
  Calculate minimum, maximum and average edge length at the surface of mesh
  Args:
  coordinates (numpy array): cartesian cooridnates of vertices
  faces (numpy array): faces index
  n_faces (int): number of faces
  Returns:
  mine (float): minimum edge length
  maxe (float): maximum edge length
  ave: average edge length
  """
  mine = 1e9
  maxe = ave = 0.0
  for i in range(n_faces):
    mine = min(np.linalg.norm(coordinates[faces[i,1]] - coordinates[faces[i,0]]), mine)
    mine = min(np.linalg.norm(coordinates[faces[i,2]] - coordinates[faces[i,0]]), mine)
    mine = min(np.linalg.norm(coordinates[faces[i,2]] - coordinates[faces[i,1]]), mine)
    maxe = max(np.linalg.norm(coordinates[faces[i,1]] - coordinates[faces[i,0]]), maxe)
    maxe = max(np.linalg.norm(coordinates[faces[i,2]] - coordinates[faces[i,0]]), maxe)
    maxe = max(np.linalg.norm(coordinates[faces[i,2]] - coordinates[faces[i,1]]), maxe)
    ave += np.linalg.norm(coordinates[faces[i,2]] - coordinates[faces[i,1]]) + np.linalg.norm(coordinates[faces[i,2]] - coordinates[faces[i,0]]) + np.linalg.norm(coordinates[faces[i,1]] - coordinates[faces[i,0]])
  ave /= 3.0*n_faces

  return mine, maxe, ave

@jit(nopython=True, parallel=True)
def volume_mesh(n_nodes, n_tets, tets, coordinates):
  '''
  Calculate total volume of the mesh, used for information only
  Args:
  n_nodes (int): number of nodes
  n_tets (int): number of tets
  tets (numpy array): tetras index
  coordinates (numpy array): cartesian cooridnates of vertices
  Returns:
  Vm_init (float): total volume of mesh (sign for inversion)
  '''
  Vn_init = np.zeros(n_nodes, dtype = np.float64)
  A_init = np.zeros((n_tets,3,3), dtype=np.float64)
  vol_init = np.zeros(n_tets, dtype=np.float64)

  A_init[:,0] = coordinates[tets[:,1]] - coordinates[tets[:,0]]
  A_init[:,1] = coordinates[tets[:,2]] - coordinates[tets[:,0]]
  A_init[:,2] = coordinates[tets[:,3]] - coordinates[tets[:,0]]
  vol_init[:] = det_dim_3(transpose_dim_3(A_init[:]))/6.0

  for i in range(n_tets):
    Vn_init[tets[i,:]] += vol_init[i]/4.0

  Vm_init = np.sum(Vn_init)

  return -Vm_init

@jit
def tetra_labels_surface_half(mesh_file, method, n_clusters, coordinates0, nodal_idx, tets, lobes):
  '''
  Define the label for each surface node for half brain
  Args:
  mesh_file (str): path of surface mesh file
  method (str): method for clustering
  n_clusters (int): number of clusters
  coordinates0 (numpy array): initial cartesian cooridnates of vertices
  nodal_idx (list): nodal index map from surface to full mesh, stops at last surface node
  tets (numpy array): tetras index
  lobes (numpy array): lobar labels of all nodes of surface mesh
  
  Returns:
  labels_surface (numpy array): lobar labels of all surface nodes of mesh for simulation
  labels (numpy array): labels of all nodes of surface mesh after clustering
  '''
  mesh = sio.load_mesh(mesh_file)
  if method.__eq__("Kmeans"):
  # 1) Simple K-means to start simply
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh.vertices)
    labels = kmeans.labels_
  elif method.__eq__("Spectral"):
  # 2) Another method: spectral clustering
    Lsparse = graph_laplacian(mesh)
    evals, evecs = eigs(Lsparse, k=n_clusters - 1, which='SM')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(evecs))
    labels = kmeans.labels_
  #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh.vertices)
  labels = lobes #kmeans.labels_ 
  # Find the nearest reference surface nodes to our surface nodes (nearest_surf_node) and distribute the labels to our surface nodes (labels_surface)
  mesh.vertices[:,[0,1]]=mesh.vertices[:,[1,0]]
  tree = spatial.cKDTree(mesh.vertices)
  nearest_surf_node = tree.query(coordinates0[nodal_idx[:]])
  #labels_surface = np.zeros(n_surface_nodes, dtype = np.int64)
  #labels_surface_2 = np.zeros(n_surface_nodes, dtype = np.int64)
  labels_surface = labels[nearest_surf_node[1]]

  return labels_surface, labels

@jit
def tetra_labels_volume_half(coordinates0, nodal_idx, tets, labels_surface):
  '''
  Define the label for each tetrahedron for half brain
  Args:
  coordinates0 (numpy array): initial cartesian cooridnates of vertices
  nodal_idx (list): nodal index map from surface to full mesh, stops at last surface node
  tets (numpy array): tetras index
  labels_surface (numpy array): lobar labels of all surface nodes of mesh for simulation
  
  Returns:
  labels_volume (numpy array): lobar labels of all tetrahedrons of mesh for simulation
  '''
  # Find the nearest surface nodes to barycenters of tetahedra (nearest_surf_node_t) and distribute the label to each tetahedra (labels_volume)
  Ut_barycenter = (coordinates0[tets[:,0]] + coordinates0[tets[:,1]] + coordinates0[tets[:,2]] + coordinates0[tets[:,3]])/4.0
  tree = spatial.cKDTree(coordinates0[nodal_idx[:]])
  nearest_surf_node_t = tree.query(Ut_barycenter[:,:])
  #labels_volume = np.zeros(ne, dtype = np.int64)
  labels_volume = labels_surface[nearest_surf_node_t[1]]

  return labels_volume

@jit
def tetra_labels_surface_whole(mesh_file, mesh_file_2, method, n_clusters, coordinates0, nodal_idx, tets, indices_a, indices_b, lobes, lobes_2):
  '''
  Define the label for each surface node for whole brain
  Args:
  mesh_file (str): path of right hemisphere surface mesh file
  mesh_file_2 (str): path of left hemisphere surface mesh file
  method (str): method for clustering
  n_clusters (int): number of clusters
  coordinates0 (numpy array): initial cartesian cooridnates of vertices
  nodal_idx (list): nodal index map from surface to full mesh, stops at last surface node
  tets (numpy array): tetras index
  indices_a (numpy array): right hemisphere surface node indices
  indices_b (numpy array): left hemisphere surface node indices
  lobes (numpy array): lobar labels of all nodes of right hemisphere surface mesh
  lobes_2 (numpy array): lobar labels of all nodes of left hemisphere surface mesh
  
  Returns:
  labels_surface (numpy array): lobar labels of right hemisphere surface nodes of mesh for simulation
  labels_surface_2 (numpy array): lobar labels of left hemisphere surface nodes of mesh for simulation
  labels (numpy array): labels of all nodes of right hemisphere surface mesh after clustering
  labels_2 (numpy array): labels of all nodes of left hemisphere surface mesh after clustering
  '''
  mesh = sio.load_mesh(mesh_file)
  mesh_2 = sio.load_mesh(mesh_file_2)
  if method.__eq__("Kmeans"):
  # 1) Simple K-means to start simply
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh.vertices)
    kmeans_2 = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh_2.vertices)
    labels = kmeans.labels_
    labels_2 = kmeans_2.labels_
  elif method.__eq__("Spectral"):
  # 2) Another method: spectral clustering
    Lsparse = graph_laplacian(mesh)
    Lsparse_2 = graph_laplacian(mesh_2)
    evals, evecs = eigs(Lsparse, k=n_clusters - 1, which='SM')
    evals_2, evecs_2 = eigs(Lsparse_2, k=n_clusters - 1, which='SM')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(evecs))
    kmeans_2 = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(evecs_2))
    labels = kmeans.labels_
    labels_2 = kmeans_2.labels_
  #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh.vertices)
  labels = lobes #kmeans.labels_
  labels_2 = lobes_2 #kmeans_2.labels_ 
  # Find the nearest reference surface nodes to our surface nodes (nearest_surf_node) and distribute the labels to our surface nodes (labels_surface)
  mesh.vertices[:,[0,1]]=mesh.vertices[:,[1,0]]
  mesh_2.vertices[:,[0,1]]=mesh_2.vertices[:,[1,0]]
  tree_1 = spatial.cKDTree(mesh.vertices)
  tree_2 = spatial.cKDTree(mesh_2.vertices)
  nearest_surf_node = tree_1.query(coordinates0[nodal_idx[indices_a]])
  nearest_surf_node_2 = tree_2.query(coordinates0[nodal_idx[indices_b]])
  #labels_surface = np.zeros(n_surface_nodes, dtype = np.int64)
  #labels_surface_2 = np.zeros(n_surface_nodes, dtype = np.int64)
  labels_surface = labels[nearest_surf_node[1]]
  labels_surface_2 = labels_2[nearest_surf_node_2[1]]

  return labels_surface, labels_surface_2, labels, labels_2

@jit
def tetra_labels_volume_whole(coordinates0, nodal_idx, tets, indices_a, indices_b, indices_c, indices_d, labels_surface, labels_surface_2):
  '''
  Define the label for each tetrahedron for whole brain
  Args:
  coordinates0 (numpy array): initial cartesian cooridnates of vertices
  nodal_idx (list): nodal index map from surface to full mesh, stops at last surface node
  tets (numpy array): tetras indices
  indices_a (numpy array): right hemisphere surface node indices
  indices_b (numpy array): left hemisphere surface node indices
  indices_c (numpy array): right hemisphere tetras indices
  indices_d (numpy array): left hemisphere tetras indices
  labels_surface (numpy array): lobar labels of right hemisphere surface nodes of mesh for simulation
  labels_surface_2 (numpy array): lobar labels of left hemisphere surface nodes of mesh for simulation
  
  Returns:
  labels_volume (numpy array): lobar labels of right hemisphere tetras of mesh for simulation
  labels_volume_2 (numpy array): lobar labels of left hemisphere tetras of mesh for simulation
  '''
  # Find the nearest surface nodes to barycenters of tetahedra (nearest_surf_node_t) and distribute the label to each tetahedra (labels_volume)
  #indices_c = np.where((Ut0[tets[:,0],1]+Ut0[tets[:,1],1]+Ut0[tets[:,2],1]+Ut0[tets[:,3],1])/4 >= 0.0)[0]  #lower part
  #indices_d = np.where((Ut0[tets[:,0],1]+Ut0[tets[:,1],1]+Ut0[tets[:,2],1]+Ut0[tets[:,3],1])/4 < 0.0)[0]  #upper part
  Ut_barycenter_c = (coordinates0[tets[indices_c,0]] + coordinates0[tets[indices_c,1]] + coordinates0[tets[indices_c,2]] + coordinates0[tets[indices_c,3]])/4.0
  Ut_barycenter_d = (coordinates0[tets[indices_d,0]] + coordinates0[tets[indices_d,1]] + coordinates0[tets[indices_d,2]] + coordinates0[tets[indices_d,3]])/4.0
  tree_3 = spatial.cKDTree(coordinates0[nodal_idx[indices_a]])
  nearest_surf_node_t = tree_3.query(Ut_barycenter_c[:,:])
  tree_4 = spatial.cKDTree(coordinates0[nodal_idx[indices_b]])
  nearest_surf_node_t_2 = tree_4.query(Ut_barycenter_d[:,:])
  #labels_volume = np.zeros(ne, dtype = np.int64)
  labels_volume = labels_surface[nearest_surf_node_t[1]]
  labels_volume_2 = labels_surface_2[nearest_surf_node_t_2[1]]

  return labels_volume, labels_volume_2

# Define Gaussian function for temporal growth rate
@jit
def func(x, a, c, sigma):

  return a*np.exp(-(x-c)**2/sigma)

# Define asymmetric normal function for temporal growth rate
@jit
def skew(X, a, e, w):
  X = (X-e)/w
  Y = 2*np.exp(-X**2/2)/np.sqrt(2*np.pi)
  Y *= 1/w*spe.ndtr(a*X)  #ndtr: gaussian cumulative distribution function
  
  return Y

# Define polynomial for temporal growth
@jit
def poly(x, a, b, c):
  
  return a*x**2+b*x+c

# Define gompertz model for temporal growth
@jit
def gompertz(x, a, b, c):

  return a*np.exp(-np.exp(-b*(x-c)))

#@jit
def Curve_fitting_half(texture_file, labels, n_clusters, lobes):
  '''
  Curve-fit of temporal growth for each label for half brain
  Args:
  texture_file (str): path of .gii texture file
  labels (numpy array): labels of all nodes of hemisphere surface mesh after clustering
  n_clusters (int): number of clusters
  lobes (numpy array): lobar labels of all nodes of hemisphere surface mesh
  
  Returns:
  amplitude, peak, latency (numpy array): parameters of Gompertz function for each lobe of hemisphere
  '''
  ages=[29, 29, 28, 28.5, 31.5, 32, 31, 32, 30.5, 32, 32, 31, 35.5, 35, 34.5, 35, 34.5, 35, 36, 34.5, 37.5, 35, 34.5, 36, 34.5, 33, 33]
  xdata=np.array(ages)
  
  croissance_globale_R = np.array([1.30, 1.56, 1.22, 1.46, 1.35,1.24,1.47,1.58,1.45,1.31,1.66,1.78,1.67,1.42,1.44,1.22,1.75,1.56,1.40,1.33,1.79,1.27,1.52,1.34,1.68,1.81,1.78])

  # Calculate the local (true) cortical growth
  texture = sio.load_texture(texture_file)
  texture_sujet2_R = np.array([texture.darray[1], texture.darray[5], texture.darray[13]])
  
  ages_sujet2 = np.array([31, 33, 37, 44])
  tp_model = 6.926*10**(-5)*ages_sujet2**3-0.00665*ages_sujet2**2+0.250*ages_sujet2-3.0189  #time of numerical model
  croissance_globale_sujet2_R = np.array([croissance_globale_R[1], croissance_globale_R[5], croissance_globale_R[13]])
  croissance_true_relative_R = np.zeros(texture_sujet2_R.shape)
  for i in range(texture_sujet2_R.shape[0]):
    croissance_true_relative_R[i, :] = texture_sujet2_R[i,:]*croissance_globale_sujet2_R[i]

  latency=np.zeros(len(np.unique(lobes)), dtype=np.float64)
  amplitude=np.zeros(len(np.unique(lobes)), dtype=np.float64)
  peak=np.zeros(len(np.unique(lobes)), dtype=np.float64)
  ydata=np.zeros((3, len(np.unique(lobes))), dtype = np.float64)
  ydata_new=np.zeros((4, len(np.unique(lobes))), dtype = np.float64)
  
  p = 0
  for k in np.unique(lobes):      #= int(np.unique(lobes)[5])
    for j in range(texture_sujet2_R.shape[0]):
      ydata[j, p]=np.mean(croissance_true_relative_R[j, np.where(lobes == k)[0]])
    p += 1
  ydata_new[0,:] = ydata[0,:]-1
  ydata_new[1,:] = ydata[0, :]*ydata[1, :]-1
  ydata_new[2,:] = ydata[0, :]*ydata[1, :]*ydata[2, :]-1
  ydata_new[3,:] = np.full(len(np.unique(lobes)), 1.829)

  m = 0
  for j in range(len(np.unique(lobes))):    
    popt, pcov = curve_fit(gompertz, tp_model, ydata_new[:,j])   
    peak[m]=popt[1]
    amplitude[m]=popt[0]
    latency[m]=popt[2]
    #multiple[m]=popt[3]
    m += 1

  return peak, amplitude, latency

#@jit
def curve_fitting_whole(texture_file, texture_file_2, labels, labels_2, n_clusters, lobes, lobes_2):
  '''
  Curve-fit of temporal growth for each label for whole brain
  Args:
  texture_file (str): path of .gii texture file of right hemisphere
  texture_file_2 (str): path of .gii texture file of left hemisphere
  labels (numpy array): labels of all nodes of right hemisphere surface mesh after clustering
  labels_2 (numpy array): labels of all nodes of left hemisphere surface mesh after clustering
  n_clusters (int): number of clusters
  lobes (numpy array): lobar labels of all nodes of right hemisphere surface mesh
  lobes (numpy array): lobar labels of all nodes of left hemisphere surface mesh
  
  Returns:
  amplitude, peak, latency (numpy array): parameters of Gompertz function for each lobe of right hemisphere
  amplitude_2, peak_2, latency_2 (numpy array): parameters of Gompertz function for each lobe of left hemisphere
  '''
  ages=[29, 29, 28, 28.5, 31.5, 32, 31, 32, 30.5, 32, 32, 31, 35.5, 35, 34.5, 35, 34.5, 35, 36, 34.5, 37.5, 35, 34.5, 36, 34.5, 33, 33]
  xdata=np.array(ages)
  
  croissance_globale_L = np.array([1.26, 1.51, 1.21, 1.38, 1.37, 1.22, 1.44, 1.57, 1.45, 1.34, 1.54, 1.67, 1.70, 1.48, 1.50, 1.19, 1.78, 1.59, 1.49, 1.34, 1.77, 1.29 ,1.57, 1.45, 1.71, 1.76, 1.81])
  croissance_globale_R = np.array([1.30, 1.56, 1.22, 1.46, 1.35,1.24,1.47,1.58,1.45,1.31,1.66,1.78,1.67,1.42,1.44,1.22,1.75,1.56,1.40,1.33,1.79,1.27,1.52,1.34,1.68,1.81,1.78])
  """xdata_new = np.zeros(5)
  xdata_new[0]=22
  xdata_new[1]=27
  xdata_new[2]=31
  xdata_new[3]=33
  xdata_new[4]=37"""
  #tp_model = 6.926*10**(-5)*xdata**3-0.00665*xdata**2+0.250*xdata-3.0189  #time of numerical model

  # Calculate the local (true) cortical growth
  """popt, pcov = curve_fit(poly, np.array([27, 31, 33, 37]), np.array([1, 1.26, 1.26*1.37, 1.26*1.37*1.70]))
  croissance_globale = np.array(poly(xdata,*popt))"""

  texture = sio.load_texture(texture_file)
  texture_2 = sio.load_texture(texture_file_2)
  texture_sujet2_R = np.array([texture.darray[1], texture.darray[5], texture.darray[13]])
  texture_sujet2_L = np.array([texture_2.darray[1], texture_2.darray[5], texture_2.darray[13]])
  
  ages_sujet2 = np.array([31, 33, 37, 44])
  tp_model = 6.926*10**(-5)*ages_sujet2**3-0.00665*ages_sujet2**2+0.250*ages_sujet2-3.0189  #time of numerical model
  croissance_globale_sujet2_R = np.array([croissance_globale_R[1], croissance_globale_R[5], croissance_globale_R[13]])
  croissance_globale_sujet2_L = np.array([croissance_globale_L[1], croissance_globale_L[5], croissance_globale_L[13]])
  croissance_true_relative_R = np.zeros(texture_sujet2_R.shape)
  croissance_true_relative_L = np.zeros(texture_sujet2_L.shape)
  for i in range(texture_sujet2_R.shape[0]):
    croissance_true_relative_R[i, :] = texture_sujet2_R[i,:]*croissance_globale_sujet2_R[i]
    croissance_true_relative_L[i, :] = texture_sujet2_L[i,:]*croissance_globale_sujet2_L[i]
  """croissance_true = np.zeros(texture.darray.shape, dtype=np.float64)
  for i in range(texture.darray.shape[0]):
    croissance_true[i,:] = texture.darray[i,:]*croissance_globale[i]
  croissance_length = np.sqrt(croissance_true)

  texture_2 = sio.load_texture(texture_file_2)
  croissance_true_2 = np.zeros(texture_2.darray.shape, dtype=np.float64)
  for j in range(texture_2.darray.shape[0]):
    croissance_true_2[j,:] = texture_2.darray[j,:]*croissance_globale[j]
  croissance_length_2 = np.sqrt(croissance_true_2)"""

  """peak=np.zeros((n_clusters,))
  amplitude=np.zeros((n_clusters,))
  latency=np.zeros((n_clusters,))
  peak_2=np.zeros((n_clusters,))
  amplitude_2=np.zeros((n_clusters,))
  latency_2=np.zeros((n_clusters,))
  for k in range(n_clusters):
    ydata=np.mean(croissance_length[:,np.where(labels == k)[0]], axis=1)
    ydata_2=np.mean(croissance_length_2[:,np.where(labels_2 == k)[0]], axis=1)
    popt, pcov=curve_fit(func, tp_model, ydata, p0=[1.5, 0.9, 0.09])
    popt_2, pcov_2=curve_fit(func, tp_model, ydata_2, p0=[1.5, 0.9, 0.09])
    peak[k]=popt[1]   
    amplitude[k]=popt[0]
    latency[k]=popt[2]
    peak_2[k]=popt_2[1]   
    amplitude_2[k]=popt_2[0]
    latency_2[k]=popt_2[2]"""

  latency=np.zeros(len(np.unique(lobes)), dtype=np.float64)
  amplitude=np.zeros(len(np.unique(lobes)), dtype=np.float64)
  peak=np.zeros(len(np.unique(lobes)), dtype=np.float64)
  latency_2=np.zeros(len(np.unique(lobes_2)), dtype=np.float64)
  amplitude_2=np.zeros(len(np.unique(lobes_2)), dtype=np.float64)
  peak_2=np.zeros(len(np.unique(lobes_2)), dtype=np.float64)
  ydata=np.zeros((3, len(np.unique(lobes))), dtype = np.float64)
  ydata_new=np.zeros((4, len(np.unique(lobes))), dtype = np.float64)
  ydata_2=np.zeros((3, len(np.unique(lobes_2))), dtype = np.float64)
  ydata_new_2=np.zeros((4, len(np.unique(lobes_2))), dtype = np.float64)

  #for k in range(n_clusters):  
  """m = 0
  for k in np.unique(lobes):
    #ydata=np.mean(texture_new[:,np.where(labels == k)[0]], axis=1)
    #ydata_2=np.mean(texture_new_2[:,np.where(labels_2 == k)[0]], axis=1)
    ydata=np.mean(croissance_length[:, np.where(lobes == k)[0]], axis=1)
    popt, pcov=curve_fit(gompertz, tp_model, ydata, p0=[0.94, 2.16, 3.51, 1.0]) #p0=[1.5, 0.9, 0.09] =[0.94, 2.16, 3.51, 0.65])
    peak[m]=popt[1]
    amplitude[m]=popt[0]
    latency[m]=popt[2]
    multiple[m]=popt[3]
    m += 1
  m_2 = 0
  for k in np.unique(lobes_2):
    ydata_2=np.mean(croissance_length_2[:, np.where(lobes_2 == k)[0]], axis=1)
    popt_2, pcov_2=curve_fit(gompertz, tp_model, ydata_2, p0=[0.94, 2.16, 3.51, 1.0])
    peak_2[m_2]=popt_2[1]   
    amplitude_2[m_2]=popt_2[0]
    latency_2[m_2]=popt_2[2]
    multiple_2[m_2]=popt_2[3]
    m_2 += 1"""
  
  p = 0
  for k in np.unique(lobes):      #= int(np.unique(lobes)[5])
    for j in range(texture_sujet2_R.shape[0]):
      ydata[j, p]=np.mean(croissance_true_relative_R[j, np.where(lobes == k)[0]])
    p += 1
  ydata_new[0,:] = ydata[0,:]-1
  ydata_new[1,:] = ydata[0, :]*ydata[1, :]-1
  ydata_new[2,:] = ydata[0, :]*ydata[1, :]*ydata[2, :]-1
  ydata_new[3,:] = np.full(len(np.unique(lobes)), 1.829)

  m = 0
  for j in range(len(np.unique(lobes))):    
    popt, pcov = curve_fit(gompertz, tp_model, ydata_new[:,j])   
    peak[m]=popt[1]
    amplitude[m]=popt[0]
    latency[m]=popt[2]
    #multiple[m]=popt[3]
    m += 1

  p_2 = 0
  for k in np.unique(lobes_2):
    for j in range(texture_sujet2_L.shape[0]):
      ydata_2[j, p_2]=np.mean(croissance_true_relative_L[j, np.where(lobes_2 == k)[0]])
    p_2 += 1
  ydata_new_2[0,:] = ydata_2[0,:]-1
  ydata_new_2[1,:] = ydata_2[0, :]*ydata_2[1, :]-1
  ydata_new_2[2,:] = ydata_2[0, :]*ydata_2[1, :]*ydata_2[2, :]-1
  ydata_new_2[3,:] = np.full(len(np.unique(lobes_2)), 1.829)

  m_2 = 0
  for j in range(len(np.unique(lobes_2))):    
    popt_2, pcov_2 = curve_fit(gompertz, tp_model, ydata_new_2[:,j])   
    peak_2[m_2]=popt_2[1]
    amplitude_2[m_2]=popt_2[0]
    latency_2[m_2]=popt_2[2]
    #multiple_2[m_2]=popt[3]
    m_2 += 1

  return peak, amplitude, latency, peak_2, amplitude_2, latency_2

@njit(parallel=True)
def mark_nogrowth(coordinates0, n_nodes):
  '''
  Mark non-growing areas
  Args:
  coordinates0 (numpy array): initial cartesian cooridnates of vertices
  n_nodes (int): number of nodes
  Returns:
  gr (numpy array): growth factors that control the magnitude of growth of each region
  '''
  gr = np.zeros(n_nodes, dtype = np.float64)
  for i in prange(n_nodes):
    rqp = np.linalg.norm(np.array([(coordinates0[i,0]+0.1)*0.714, coordinates0[i,1], coordinates0[i,2]-0.05]))
    if rqp < 0.6:
      gr[i] = max(1.0 - 10.0*(0.6-rqp), 0.0)
    else:
      gr[i] = 1.0

  return gr

@jit
def config_refer(coordinates0, tets, n_tets):
  '''
  Calculate the reference configuration of tetrahendrons (Ar), used for elasticity/deformation calculation
  Args:
  coordinates0 (numpy array): initial cartesian cooridnates of vertices
  tets (numpy array): indices of the tetrahedrons
  n_tets (int): number of tetrahedrons
  Returns:
  ref_state_tets (numpy array): Reference configuration of tetrahedrons
  '''
  ref_state_tets = np.zeros((n_tets,3,3), dtype=np.float64)
  ref_state_tets[:,0] = coordinates0[tets[:,1]] - coordinates0[tets[:,0]]
  ref_state_tets[:,1] = coordinates0[tets[:,2]] - coordinates0[tets[:,0]]
  ref_state_tets[:,2] = coordinates0[tets[:,3]] - coordinates0[tets[:,0]]
  ref_state_tets[:] = transpose_dim_3(ref_state_tets[:]) 

  return ref_state_tets

@jit(nopython=True)
def config_deform(coordinates, tets, n_tets):
  '''
  Calculate the deformed configuration of tetrahendrons (At), used for elasticity/deformation calculation
  Args:
  coordinates (numpy array): deformed cartesian cooridnates of vertices
  tets (numpy array): indices of the tetrahedrons
  n_tets (int): number of tetrahedrons
  Returns:
  material_tets(numpy array): Deformed configuration of tetrahedrons
  '''
  material_tets = np.zeros((n_tets,3,3), dtype=np.float64)
  material_tets[:,0] = coordinates[tets[:,1]] - coordinates[tets[:,0]]
  material_tets[:,1] = coordinates[tets[:,2]] - coordinates[tets[:,0]]
  material_tets[:,2] = coordinates[tets[:,3]] - coordinates[tets[:,0]]
  material_tets[:] = transpose_dim_3(material_tets[:])

  return material_tets

@jit(forceobj=True, parallel=True) 
def normals_surfaces(coordinates0, faces, nodal_idx_b, n_faces, n_surface_nodes, surf_node_norms):
  '''
  TODO: Why take surf_node_norms as parameter ?
  Calculate normal of each face and average result for each surface node with normalised length. 
  Args:
  coordinates0 (numpy array): deformed cartesian cooridnates of vertices
  faces (numpy array): indices of faces
  nodal_idx_b (numpy array): list of surface node indices
  n_faces (int): number of faces
  n_surface_nodes (int): number of surface nodes
  surf_node_norms (np array): normals of surface nodes

  Returns:
  surf_node_norms (np array): normals of surface nodes
  '''
  Ntmp = np.zeros((n_faces,3), dtype=np.float64)
  Ntmp = cross_dim_3(coordinates0[faces[:,1]] - coordinates0[faces[:,0]], coordinates0[faces[:,2]] - coordinates0[faces[:,0]])
  for i in prange(n_faces):
    surf_node_norms[nodal_idx_b[faces[i,:]]] += Ntmp[i]
  for i in prange(n_surface_nodes): #because several norms are added to each face
    surf_node_norms[i] *= 1.0/np.linalg.norm(surf_node_norms[i])

  return surf_node_norms

# Calculate normals of each deformed tetrahedron
@jit
def tetra_normals_leg(surf_node_norms, nearest_surf_node, tets, n_tets):
  Nt = np.zeros((n_tets,3), dtype=np.float64)
  Nt[:] = surf_node_norms[nearest_surf_node[tets[:,0]]] + surf_node_norms[nearest_surf_node[tets[:,1]]] + surf_node_norms[nearest_surf_node[tets[:,2]]] + surf_node_norms[nearest_surf_node[tets[:,3]]]
  Nt = normalize_dim_3(Nt) #prob line for nopython mode

  return Nt
# Calculate normals of each deformed tetrahedron  
@jit(nopython=True)
def tetra_normals(surf_node_norms, nearest_surf_node, tets, n_tets):
  """
  Propagate the normal of nearest surface nodes to all tetrahedrons
  Args:
  surf_node_norms (np array): normals of surface nodes
  nearest_surf_node (np array): nearest surface node for each node
  test (np array): node indices of tetrahedrons
  n_tets (int): number of tets
  Returns:
  tet_norms (np array): normal for each tetrahedron
  """
  tet_norms = np.zeros((n_tets,3), dtype=np.float64)
  tet_norms[:] = surf_node_norms[nearest_surf_node[tets[:,0]]] + surf_node_norms[nearest_surf_node[tets[:,1]]] + surf_node_norms[nearest_surf_node[tets[:,2]]] + surf_node_norms[nearest_surf_node[tets[:,3]]]
  tet_norms = normalize(tet_norms)

  return tet_norms

# Computes the volume measured at each point of a tetrahedral mesh as the sum of 1/4 of the volume of each of the tetrahedra to which it belongs
@jit(nopython=True, parallel=True) 
def calc_vol_nodal(tan_growth_tensor, ref_state_tets, material_tets, tets, n_tets, n_nodes):
  """
  Calculates the undeformed and deformed nodal volume for each node. Volume per tetra is calculated and then distributed equally on each node
  
  Args:
  tan_growth_tensor (np array): growth tensor for each tetrahedron
  ref_state_tets (np array): Reference state of tetrahedrons
  tets (np array): indices of tetrahedrons
  material_tets (np array): Current configuration of the tetrahedrons
  n_tets (int) number of tetrahedrons

  Returns:
  Vn0 (np array): Inital volume of each node
  Vn (np array): Deformed volume of each node
  """
  Vn0 = np.zeros(n_nodes, dtype=np.float64) #Initialize nodal volumes in reference state
  Vn = np.zeros(n_nodes, dtype=np.float64)  #Initialize deformed nodal volumes
  vol0 = np.zeros(n_tets, dtype=np.float64)
  vol = np.zeros(n_tets, dtype=np.float64)

  vol0[:] = det_dim_3(dot_mat_dim_3(tan_growth_tensor[:], ref_state_tets[:]))/6.0
  vol[:] = det_dim_3(material_tets[:])/6.0
  
  for i in prange(n_tets):
    for tet in tets[i]:
      Vn0[tet] += vol0[i]/4.0
      Vn[tet] += vol[i]/4.0

  return Vn0, Vn

# Midplane, what is midplane_pos exactly?
@njit
def calc_mid_plane(coordinates, coordinates0, Ft, nodal_idx, n_surface_nodes, midplane_pos, mesh_spacing, repuls_skin, bulk_modulus):
  '''
  Check a box condition and restrict growth for the outer layer for each surface node
  '''
  for i in prange(n_surface_nodes):
    pt = nodal_idx[i]
    if coordinates0[pt,1] < midplane_pos - 0.5*mesh_spacing and coordinates[pt,1] > midplane_pos:
      Ft[pt,1] -= (midplane_pos - coordinates[pt,1])/repuls_skin*mesh_spacing*mesh_spacing*bulk_modulus
    if coordinates0[pt,1] > midplane_pos + 0.5*mesh_spacing and coordinates[pt,1] < midplane_pos:
      Ft[pt,1] -= (midplane_pos - coordinates[pt,1])/repuls_skin*mesh_spacing*mesh_spacing*bulk_modulus

  return Ft

@jit
def calc_longi_length(t):
  '''
  TODO: add reference for calculation
  Calculate the expected longitudinal length of the brain depending on simulation time. Used for info and visualisation only
  args:
  t (float): time of simulation
  returns:
  longi_length (float): calculated longitudinal length of the brain
  '''
  #L = -0.81643*t**2+2.1246*t+1.3475
  longi_length = -0.98153*t**2+3.4214*t+1.9936
  #L = -41.6607*t**2+101.7986*t+58.843 #for the case without normalisation

  return longi_length

# Obtain zoom parameter by checking the longitudinal length of the brain model
@jit
def paraZoom(coordinates, nodal_idx, longi_length):
  '''
  Obtain zoom parameter by checking the longitudinal length of the brain model, used for outputs
  args:
  coordinates (np array): cartesian coordinates of nodes
  nodal_idx (np array): index list of surface nodes
  longi_length (float): calculated length of the foetal brain
  returns:
  zoom_pos (float): zomm position used for paraview and denormalisation
  '''
  #xmin = ymin = 1.0
  #xmax = ymax = -1.0

  xmin = min(coordinates[nodal_idx[:],0])
  xmax = max(coordinates[nodal_idx[:],0])
  ymin = min(coordinates[nodal_idx[:],1])
  ymax = max(coordinates[nodal_idx[:],1])

  # Zoom parameter
  zoom_pos = longi_length/(xmax-xmin)

  return zoom_pos
