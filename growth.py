import numpy as np
import math
from math import sqrt
from numba import jit, njit, prange
from mathfunc import dot_vec_dim_3
from scipy import spatial
from scipy import pi,sqrt,exp
import scipy.special as sp

# Find the nearest surface nodes to nodes (csn) and distances to them (dist_2_surf) - these are needed to set up the growth of the gray matter
@jit
def dist2surf(coordinates0, nodal_idx):
  #csn = np.zeros(nn)  # Nearest surface nodes
  #dist_2_surf = np.zeros(nn)  # Distances to nearest surface node
  """for i in prange(nn):
    d2 = dot_vec_dim_3(Ut0[SN[:]] - Ut0[i], Ut0[SN[:]] - Ut0[i])
    csn[i] = np.argmin(d2)
    dist_2_surf[i] = sqrt(np.min(d2))"""
  tree = spatial.KDTree(coordinates0[nodal_idx[:]])
  pp = tree.query(coordinates0)
  csn = pp[1]  # Nearest surface nodes
  dist_2_surf = pp[0]  # Distances to nearest surface node

  return csn, dist_2_surf

# Calculate the global relative growth rate for half or whole brain
@jit
def growthRate(GROWTH_RELATIVE, t, n_tets):
  at = np.zeros(n_tets, dtype=np.float64)
  #if t < 0.0:
  #at[indices_b[:]] = 1.5*GROWTH_RELATIVE*t   #3.658
  #at[indices_a[:]] = GROWTH_RELATIVE*t     #1.829
  #for i in prange(ne):
    #at[i] = 2.0*GROWTH_RELATIVE*t/(1+math.exp((Ut0[tets[i,0],2]+Ut0[tets[i,1],2]+Ut0[tets[i,2],2]+Ut0[tets[i,3],2])/4))  #~1.829*2max, ~1.829min
  at[:] = GROWTH_RELATIVE*t
    #at[i] = GROWTH_RELATIVE + 7.4*t
  #if t >= 0.0: 
    #at = GROWTH_RELATIVE - GROWTH_RELATIVE*t

  return at

# Calculate the regional relative growth rate for half brain
@jit
def growthRate_2_half(t, n_tets, n_surface_nodes, n_clusters, labels_surface, labels_volume, peak, amplitude, latency, lobes):
  at = np.zeros(n_tets, dtype=np.float64)
  bt = np.zeros(n_surface_nodes, dtype=np.float64)
  #for i in range(n_clusters):
  m = 0
  for i in np.unique(lobes):
    #at[np.where(labels == i)[0]] = 2*np.exp(-((t-peak[i])/latency[i])**2/2)/np.sqrt(2*np.pi) * 1/latency[i] * sp.ndtr(amplitude[i]*(t-peak[i])/latency[i])
    at[np.where(labels_volume == i)[0]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    bt[np.where(labels_surface == i)[0]] = amplitude[m]*np.exp(-np.exp(-peak[m]*(t-latency[m])))
    m += 1
  at = np.where(at > 0.0, at, 0.0)
  bt = np.where(bt > 0.0, bt, 0.0)

  return at, bt

# Calculate the regional relative growth rate for whole brain
@jit
def growthRate_2_whole(t, n_tets, n_surface_nodes, n_clusters, labels_surface, labels_surface_2, labels_volume, labels_volume_2, peak, amplitude, latency, peak_2, amplitude_2, latency_2, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d):
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

# Calculate the thickness of growing layer
@jit
def cortexThickness(THICKNESS_CORTEX, t):
  #if t < 0.0:
  H = THICKNESS_CORTEX + 0.01*t
  #if t >= 0.0:
    #H = THICKNESS_CORTEX + THICKNESS_CORTEX*t

  return H

# Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
@njit(parallel=True)
def shearModulus(dist_2_surf, cortex_thickness, tets, n_tets, muw, mug, gr):
  gm = np.zeros(n_tets, dtype=np.float64)
  mu = np.zeros(n_tets, dtype=np.float64)
  for i in prange(n_tets):
    gm[i] = 1.0/(1.0 + math.exp(10.0*(0.25*(dist_2_surf[tets[i,0]] + dist_2_surf[tets[i,1]] + dist_2_surf[tets[i,2]] + dist_2_surf[tets[i,3]])/cortex_thickness - 1.0)))*0.25*(gr[tets[i,0]] + gr[tets[i,1]] + gr[tets[i,2]] + gr[tets[i,3]])
    #wm[i] = 1.0 - gm[i]
    mu[i] = muw*(1.0 - gm[i]) + mug*gm[i]  # Global modulus of white matter and gray matter

  return gm, mu

# Calculate relative (relates to dist_2_surf) tangential growth factor G
@jit
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
  for i in prange(n_tets):
    tan_growth_tensor[i] = np.identity(3) + (np.identity(3) - A[i])*gm[i]*at[i]
  #G[i] = np.identity(3) + (np.identity(3) - np.matrix([[Nt[0]*Nt[0], Nt[0]*Nt[1], Nt[0]*Nt[2]], [Nt[0]*Nt[1], Nt[1]*Nt[1], Nt[1]*Nt[2]], [Nt[0]*Nt[2], Nt[1]*Nt[2], Nt[2]*Nt[2]]]))*gm*at

  return tan_growth_tensor

# Calculate homogeneous growth factor G
@jit
def growthTensor_homo(G, n_tets, GROWTH_RELATIVE, t):
  for i in prange(n_tets):
    G[i] = 1.0 + GROWTH_RELATIVE*t

  return G

# Calculate homogeneous growth factor G (2nd version)
@jit
def growthTensor_homo_2(G, n_tets, GROWTH_RELATIVE):
  for i in prange(n_tets):
    G[i] = GROWTH_RELATIVE

  return G

# Calculate cortical layer (relates to dist_2_surf) homogeneous growth factor G
@jit
def growthTensor_relahomo(gm, G, n_tets, GROWTH_RELATIVE, t):
  for i in prange(n_tets):
  #G[i] = np.full((3, 3), gm*GROWTH_RELATIVE)
    G[i] = 1.0 + GROWTH_RELATIVE*t*gm

  return G
