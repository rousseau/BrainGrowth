import numpy as np
import math
from math import sqrt
from numba import jit, njit, prange
from mathfunc import dot_vec_dim_3
from scipy import spatial
from scipy import pi,sqrt,exp
import scipy.special as sp

@jit
def dist2surf(coordinates0, nodal_idx):
  """
  Find the nearest surface node to each node and associated distance. Function needed to set up growth of gray matter
  Args:
  coordinates0 (array): vertice coordinates
  nodal_idx (array): index of surface nodes
  Returns:
  csn (array): nearest surface node for each node
  dist_2_surf: distance to surface for each node
  """
  tree = spatial.KDTree(coordinates0[nodal_idx[:]])
  pp = tree.query(coordinates0)
  csn = pp[1]  
  dist_2_surf = pp[0] 

  return csn, dist_2_surf

@jit
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

@jit
def growthRate_2_half(t, n_tets, n_surface_nodes, labels_surface, labels_volume, peak, amplitude, latency, lobes):
  """
  TODO: test and finish function
  Calculates the regional relative growth rate for half brain
  Args:
  t (float): current time of simulation
  n_tets (int): number of tetrahedrons in model
  n_surface_nodes (int): number of surface nodes
  labels_surface: ???
  labels_volume: ???
  peak: ???
  aplitude ???
  latency: ???
  lobes: ???
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

@jit
def cortexThickness(THICKNESS_CORTEX, t):
  """
  Calculates thickness of growing layer
  Args:
  THICKNESS_CORTEX (float): constante set at simulation level
  t (float): current time of simulation
  Returns:
  H (float): current growing layer thickness
  """
  H = THICKNESS_CORTEX + 0.01*t
  return H

# Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
@njit(parallel=True)
def shearModulus(dist_2_surf, cortex_thickness, tets, n_tets, muw, mug, gr):
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
  gm (array): gray matter shear modulus for each tetrahedron
  mu (array): white matter shear modulus for each tetrahedron
  """
  gm = np.zeros(n_tets, dtype=np.float64)
  mu = np.zeros(n_tets, dtype=np.float64)
  for i in prange(n_tets):
    gm[i] = 1.0/(1.0 + math.exp(10.0*(0.25*(dist_2_surf[tets[i,0]] + dist_2_surf[tets[i,1]] + dist_2_surf[tets[i,2]] + dist_2_surf[tets[i,3]])/cortex_thickness - 1.0)))*0.25*(gr[tets[i,0]] + gr[tets[i,1]] + gr[tets[i,2]] + gr[tets[i,3]])
    mu[i] = muw*(1.0 - gm[i]) + mug*gm[i]  # Global modulus of white matter and gray matter

  return gm, mu

# Calculate relative (relates to dist_2_surf) tangential growth factor G
@jit(nopython=True)
def growthTensor_tangen_leg(tet_norms, gm, at, tan_growth_tensor, n_tets):
  '''
  TODO: vectorizable
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
  for i in prange(n_tets):
    tan_growth_tensor[i] = np.identity(3) + (np.identity(3) - A[i])*gm[i]*at[i]
  #G[i] = np.identity(3) + (np.identity(3) - np.matrix([[Nt[0]*Nt[0], Nt[0]*Nt[1], Nt[0]*Nt[2]], [Nt[0]*Nt[1], Nt[1]*Nt[1], Nt[1]*Nt[2]], [Nt[0]*Nt[2], Nt[1]*Nt[2], Nt[2]*Nt[2]]]))*gm*at

  return tan_growth_tensor

# Calculate relative (relates to dist_2_surf) tangential growth factor G
@jit (nopython=True)
def growthTensor_tangen(tet_norms, gm, at, tan_growth_tensor, n_tets):
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

@jit
def growthTensor_homo(G, n_tets, GROWTH_RELATIVE, t):
  """
  Calculates homogenous growth factor G
  Args:
  G (array): normals for each tetrahedron
  n_tets (int): number of tetrahedrons
  GROWTH_RELATIVE (float): constante set at simulation level
  t (float): current time of simulation
  Returns:
  G (array): homogenous growth factor G
  """
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
  """
  Calculates cortical layer (related to dist_2_surf) homogenous growth factor G
  """
  for i in prange(n_tets):
  #G[i] = np.full((3, 3), gm*GROWTH_RELATIVE)
    G[i] = 1.0 + GROWTH_RELATIVE*t*gm

  return G
