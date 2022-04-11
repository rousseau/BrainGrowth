import numpy as np
import math
from numba import jit, njit, prange

@jit(nopython=True)
def growthRate(GROWTH_RELATIVE, t, n_tets, filter = 1.0):
  """
  Calculates global relative growth rate for half or whole brain
  Args:
  GROWTH_RELATIVE (float): Constante set at simulation level
  t (float): current time of the simulation
  n_tets(int): number of tetrahedrons in the model
  Returns:
  at (array): growth rate per tetrahedre
  filter (array): Optional factor for smoothing
  """
  at = np.zeros(n_tets, dtype=np.float64)
  at[:] = GROWTH_RELATIVE*t*filter

  return at

"""
##############################################
###LEGACY code for reference, safely ignore###
##############################################
#Smoothing using slam
    if step == 0:
        #create filtering for each surface node
        stl_path = "./res/sphere5/pov_H0.042000AT1.829000/B0.stl"
        stl_mesh = trimesh.load (stl_path)
        filtering = np.ones (len(stl_mesh.vertices))
        filtering = laplacian_texture_smoothing (stl_mesh, filtering, 10, dt)
        #Expand filtering via nearest surface node (vectorisable)
        gauss = np.ones (n_nodes, dtype = np.float64)
        for i in range (len(gauss)):
          gauss[i] = filtering[nearest_surf_node[i]]
        
        #conversion from node to tet length, take the average from nodes
        gauss_tets = np.ones (n_tets, dtype = np.float64)
        for i in range (len(tets)):
          x = 0
          for j in tets[i]:
            x += gauss [j]
          gauss_tets [i] = x/4
    
    #apply filter (why not before ? Because stl not available at this point)
    at *= gauss_tets

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

#optional gaussian filtering on at
  # centroids = calc_tets_center(coordinates, tets)
  # gauss = gaussian_filter (centroids)
    """

@jit(nopython=True, parallel=True)
def calc_growth_filter(growth_filter, dist_2_surf, n_tets, tets, cortex_thickness):
  for i in prange(n_tets):
    if float(dist_2_surf[tets[i][0]]) < cortex_thickness:
      growth_filter[i] = 1.0 #is the first somit of the tet deeper in the brain than cortical tickness?
    else:
      growth_filter[i] = 1.2
    return growth_filter

@jit
def growthRate_2_half(t, n_tets, n_surface_nodes, labels_surface, labels_volume, peak, amplitude, latency, lobes):
  """
  TODO: test and finish function
  Calculates the regional relative growth rate for half brain
  Args:
  t (float): current time of simulation
  n_tets (int): number of tetrahedrons in model
  n_surface_nodes (int): number of surface nodes
  labels_surface: lobar labels of surafces nodes
  labels_volume: lobar labels of tetrahedrons
  peak: parameter of Gompertz function
  aplitude: parameter of Gompertz function
  latency: parameter of Gompertz function
  lobes: lobar labels of all nodes of surface mesh
  """
  at = np.zeros(n_tets, dtype=np.float64)
  bt = np.zeros(n_surface_nodes, dtype=np.float64)
  m = 0
  for i in np.unique(lobes):
    at[np.where(labels_volume == i)[0]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    bt[np.where(labels_surface == i)[0]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    m += 1
  at = np.where(at > 0.0, at, 0.0)
  bt = np.where(bt > 0.0, bt, 0.0)

  return at, bt

# Calculate the regional relative growth rate for whole brain
@jit
def growthRate_2_whole(t, n_tets, n_surface_nodes, labels_surface, labels_surface_2, labels_volume, labels_volume_2, peak, amplitude, latency, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d):
  at = np.zeros(n_tets, dtype=np.float64)
  bt = np.zeros(n_surface_nodes, dtype=np.float64)
  #for i in range(n_clusters):
  m = 0
  for i in np.unique(lobes):
    at[indices_c[np.where(labels_volume == i)[0]]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    bt[indices_a[np.where(labels_surface == i)[0]]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    m += 1
  m_2 = 0
  for i in np.unique(lobes_2):
    at[indices_d[np.where(labels_volume_2 == i)[0]]] = amplitude[m_2]*np.exp(-np.exp(-peak[m_2]*(t-latency[m_2])))
    bt[indices_b[np.where(labels_surface_2 == i)[0]]] = amplitude[m_2]*np.exp(-np.exp(-peak[m_2]*(t-latency[m_2])))
    m_2 += 1
  at = np.where(at > 0.0, at, 0.0)
  bt = np.where(bt > 0.0, bt, 0.0)

  return at, bt

@njit(parallel=True)
def shear_modulus(dist_2_surf, cortex_thickness, tets, n_tets, muw, mug, gr):
  """
  Calculates global shear modulus for white and gray matter for each tetrahedron

  Args:
  dist_2_surf (array): distance to surface for each node
  cortex_thickness (float): thickness of growing layer
  tets (array): tetrahedrons index
  n_tets (int): number of tetrahedron
  muw (float): Shear modulus of white matter
  mug (float): Shear modulus of gray matter
  gr (array): yes/no growth mask for each node.

  Returns:
  gm (array): differential growth term depending on tetrahedron initial position
  mu (array): global shear modulus for each tetrahedron
  """
  gm = np.zeros(n_tets, dtype=np.float64)
  mu = np.zeros(n_tets, dtype=np.float64)

  for i in prange(n_tets):
    gm[i] = 1.0/(1.0 + math.exp(10.0*(0.25*(dist_2_surf[tets[i,0]] + dist_2_surf[tets[i,1]] + dist_2_surf[tets[i,2]] + dist_2_surf[tets[i,3]])/cortex_thickness - 1.0)))*0.25*(gr[tets[i,0]] + gr[tets[i,1]] + gr[tets[i,2]] + gr[tets[i,3]])
    mu[i] = muw*(1.0 - gm[i]) + mug*gm[i]  # Global modulus of white matter and gray matter

  return gm, mu

@jit(nopython=True)
def at_tets_to_nodes(n_nodes, tets, at):
  """ For each node, mean the 'at' biological growth term of all proximal tets. """

  at_nodes = np.zeros(n_nodes, dtype=np.float64)
  proximal_tets = np.zeros(n_nodes, dtype=np.float64) 

  for i in prange(len(tets)):
    at_nodes[tets[i,0]] += at[i] #biological growth is a time-dependant but no spatial-dependant term, so only need to mean the term over all proximal tetrahedrons around the node.
    at_nodes[tets[i,1]] += at[i]
    at_nodes[tets[i,2]] += at[i]
    at_nodes[tets[i,3]] += at[i]
    proximal_tets[tets[i,0]] += 1 #proximal_tets[nodeindice0 for teti]
    proximal_tets[tets[i,1]] += 1
    proximal_tets[tets[i,2]] += 1
    proximal_tets[tets[i,3]] += 1

  for j in prange(n_nodes): 
    at_nodes[j] /= proximal_tets[j]
    
  return at_nodes

@njit(parallel=True)
def tangential_cortical_expansion_ratio(dist_2_surf, cortex_thickness, tets, n_nodes, gr, at): 
  """ 
  Calculate the nodal (differential) tangential growth term g(y,t) for all nodes. Whereas gm(y,t) was tetrahedric. 
  Enable to visualize the nodal growth. 
  """
  
  gm_nodal = np.zeros(n_nodes, dtype=np.float64)
  g = np.zeros(n_nodes, dtype=np.float64)

  at_nodes = at_tets_to_nodes(n_nodes, tets, at)

  for i in prange(n_nodes):
    gm_nodal[i] = gr[i]/(1.0 + math.exp(10.0*(dist_2_surf[i]/cortex_thickness - 1.0))) # grey/white matter ponderation term
    g[i] = 1 + at_nodes[i] * gm_nodal[i] # tangential expansion gradient term
  
  return gm_nodal, g

@jit(nopython=True)
def growth_tensor_tangen(tet_norms, gm, at, tan_growth_tensor, n_tets):
    '''
    Calculate relative (relates to dist_2_surf) tangential growth factor G
    '''
    A = np.zeros((n_tets,3,3), dtype=np.float64)
    A[:,0,0] = tet_norms[:,0]*tet_norms[:,0]
    A[:,0,1] = tet_norms[:,0]*tet_norms[:,1]
    A[:,0,2] = tet_norms[:,0]*tet_norms[:,2]
    A[:,1,0] = tet_norms[:,0]*tet_norms[:,1]
    A[:,1,1] = tet_norms[:,1]*tet_norms[:,1]
    A[:,1,2] = tet_norms[:,1]*tet_norms[:,2]
    A[:,2,0] = tet_norms[:,0]*tet_norms[:,2]
    A[:,2,1] = tet_norms[:,1]*tet_norms[:,2]
    A[:,2,2] = tet_norms[:,2]*tet_norms[:,2]

    gm = np.reshape(np.repeat(gm, 9), (n_tets, 3, 3))
    at = np.reshape(np.repeat(at, 9), (n_tets, 3, 3))
    #identity = np.resize(identity, (n_tets, 3, 3)) // not compatible numba, but apparently broacasting similar
    tan_growth_tensor = np.identity(3) + (np.identity(3) - A) * gm * at

    return tan_growth_tensor

""" @jit
def growthTensor_homo(G, n_tets, GROWTH_RELATIVE, t):
 
  #Calculates homogenous growth factor G
  #Args:
  #G (array): normals for each tetrahedron
  #n_tets (int): number of tetrahedrons
  #GROWTH_RELATIVE (float): constante set at simulation level
  #t (float): current time of simulation
  #Returns:
  #G (array): homogenous growth factor G

  for i in prange(n_tets):
    G[i] = 1.0 + GROWTH_RELATIVE*t

  return G

# Calculate homogeneous growth factor G (2nd version)
@jit
def growthTensor_homo_2(G, n_tets, GROWTH_RELATIVE):
  for i in prange(n_tets):
    G[i] = GROWTH_RELATIVE

  return G

@jit
def growthTensor_relahomo(gm, G, n_tets, GROWTH_RELATIVE, t):

  #Calculates cortical layer (related to dist_2_surf) homogenous growth factor G

  for i in prange(n_tets):
  #G[i] = np.full((3, 3), gm*GROWTH_RELATIVE)
    G[i] = 1.0 + GROWTH_RELATIVE*t*gm

  return G """