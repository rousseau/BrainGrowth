from __future__ import division

from numpy.lib.function_base import vectorize
from mathfunc import det_dim_2, det_dim_3, inv, inv_dim_3, cross_dim_2, transpose_dim_3, dot_mat_dim_3, EV, Eigensystem, dot_tetra
import numpy as np
from math import sqrt
import time
import cython
cimport cython
from cython.parallel import prange, parallel
 
@cython.boundscheck(False)
@cython.wraparound(False)
def c_tetra_elasticity(material_tets: np.float64, ref_state_tets: np.float64, Ft: np.float64, tan_growth_tensor: np.float64, bulk_modulus: float, k_param: float, mu: np.float64, tets: np.int64, Vn: np.float64, Vn0: np.float64, n_tets: int, eps: float) -> np.float64:
  """
  Calculates elastic forces
  """
  #tetraelasticty variables
  # cdef double [:,:,::1] ref_state_growth
  # cdef double [:,:,::1] deformation_grad
  # cdef double [:,:,::1] left_cauchy_grad
  # cdef double [:,:,::1] rel_vol_chg
  # cdef double [:,:,::1] rel_vol_chg1
  # cdef double [:,:,::1] rel_vol_chg2
  # cdef double [:,:,::1] rel_vol_chg3
  # cdef double [:,:,::1] rel_vol_chg4
  # cdef double [:,:,::1] rel_vol_chg_av

  ref_state_growth = np.zeros ((n_tets, 3, 3), dtype=np.float64) #Ar
  deformation_grad = np.zeros((n_tets,3,3), dtype=np.float64)
  left_cauchy_grad = np.zeros((n_tets,3,3), dtype=np.float64)
  rel_vol_chg = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg1 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg2 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg3 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg4 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg_av = np.zeros(n_tets, dtype=np.float64)
  
  #Apply growth to reference state
  ref_state_growth = dot_mat_dim_3(tan_growth_tensor, ref_state_tets)
  cdef double [:,:,::1] c_ref_state_growth = ref_state_growth.copy()
  #Calculate deformation gradient F //combine relative volume change ?
  deformation_grad = dot_mat_dim_3(material_tets, inv_dim_3(ref_state_growth))   
  cdef double [:,:,::1] c_deformation_grad = deformation_grad.copy()
  #Calculate Left-Cauchy-Green gradient B
  left_cauchy_grad = dot_mat_dim_3(deformation_grad, np.transpose(deformation_grad, (0, 2, 1)))
  cdef double [:,:,::1] c_left_cauchy_grad = left_cauchy_grad.copy()
  #relative volume change J
  rel_vol_chg = det_dim_3(deformation_grad)
  cdef double [:] c_rel_vol_chg = rel_vol_chg
  #averaged volume change
  rel_vol_chg1 = Vn[tets[:,0]]/Vn0[tets[:,0]]
  cdef double [:] c_rel_vol_chg1 = rel_vol_chg1
  rel_vol_chg2 = Vn[tets[:,1]]/Vn0[tets[:,1]]
  cdef double [:] c_rel_vol_chg2 = rel_vol_chg2
  rel_vol_chg3 = Vn[tets[:,2]]/Vn0[tets[:,2]]
  cdef double [:] c_rel_vol_chg3 = rel_vol_chg3
  rel_vol_chg4 = Vn[tets[:,3]]/Vn0[tets[:,3]]
  cdef double [:] c_rel_vol_chg4 = rel_vol_chg4
  rel_vol_chg_av = (rel_vol_chg1 + rel_vol_chg2 + rel_vol_chg3 + rel_vol_chg4)/4.0   
  cdef double [:] c_rel_vol_chg_av = rel_vol_chg_av
    
  #decide if need SVD or not
  # with nogil,parallel(): #prange only usable without gil, but for that you need to get rid of np objects
  cdef double [:,::1] c_identity = np.identity(3)
  cdef double c_bulk_modulus = bulk_modulus
  cdef double c_powJ23
  cdef double c_S 
  cdef double [:,:] c_P = np.zeros((3, 3))
  cdef double c_cauchy_trace = 0.0
  cdef double [:,:,:] c_inv_trans_deformation_grad = np.linalg.inv(np.transpose(deformation_grad, (0, 2, 1)))  #TODO: verify is equivalent
  cdef double [:] c_mu = mu
  # cdef double [:] xr1, xr2, xr3, N1, N2, N3, N4


  cdef cython.int i
  cdef cython.int c_n_tets = n_tets
  cdef cython.int j #matrix loop variable
  cdef cython.int k #matrix loop variable

  for i in range(c_n_tets): #nogil + //
        
    # ll1, ll2, ll3 = EV(left_cauchy_grad[i])
    # ll1, ll3, ll2 = np.linalg.eig(left_cauchy_grad[i])
        
    # if ll3 >= eps*eps and rel_vol_chg[i] > 0.0:  # No need for SVD
    c_powJ23 = c_rel_vol_chg[i] ** (2.0/3.0) #those python objects ? Need to cdef ? Also, better to do function or do it by hand ? like (1/)
    cauchy_trace = np.trace(left_cauchy_grad[i])/3.0

    for j in range(3):
      for k in range(3):
        c_S = (c_left_cauchy_grad[i][j][k] - c_identity[j][k] * c_cauchy_trace)*c_mu[i]/(c_rel_vol_chg[i]*c_powJ23) + c_identity[j][k]*c_bulk_modulus*(c_rel_vol_chg_av[i]-1.0)
        c_P[j][k] = (c_S * c_inv_trans_deformation_grad[i][j][k]) * c_rel_vol_chg[i] 
      # W = 0.5*mu[i]*(np.trace(left_cauchy_grad[i])/powJ23 - 3.0) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))*0.25
        
        
    # else:   #need SVD, does not proc on sphere5 data
    #   C = np.dot(deformation_grad[i].transpose(), deformation_grad[i])
    #   w2, v2 = np.linalg.eigh(C)
    #   v2 = -v2
    
    #   l1 = sqrt(w2[0])
    #   l2 = sqrt(w2[1])
    #   l3 = sqrt(w2[2])
    
    #   if np.linalg.det(v2) < 0.0:
    #     v2[0,0] = -v2[0,0]
    #     v2[1,0] = -v2[1,0]
    #     v2[2,0] = -v2[2,0]
    
    #   Fdi = identity
    #   if l1 >= 1e-25:
    #     Fdi[0,0] = 1.0/l1
    #     Fdi[1,1] = 1.0/l2
    #     Fdi[2,2] = 1.0/l3
    
    #   U = np.dot(deformation_grad[i], np.dot(v2, Fdi))
    
    #   if l1 < 1e-25:
    #     U[0,0] = U[1,1]*U[2,2] - U[2,1]*U[1,2]
    #     U[1,0] = U[2,1]*U[0,2] - U[0,1]*U[2,2]
    #     U[2,0] = U[0,1]*U[1,2] - U[1,1]*U[0,2]
    
    #   if np.linalg.det(deformation_grad[i]) < 0.0:
    #     l1 = -l1
    #     U[0,0] = -U[0,0]
    #     U[1,0] = -U[1,0]
    #     U[2,0] = -U[2,0]
    
      # Pd = identity
      # pow23 = np.power(eps*l2*l3, 2.0/3.0)
      # Pd[0,0] = mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23 + k_param*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l2*l3
      # Pd[1,1] = mu[i]/3.0*(-eps*eps/l2 + 2.0*l2 - l3*l3/l2)/pow23 + mu[i]/9.0*(-4.0*eps/l2 - 4.0/eps*l2 + 2.0/eps/l2*l3*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l3
      # Pd[2,2] = mu[i]/3.0*(-eps*eps/l3 - l2*l2/l3 + 2.0*l3)/pow23 + mu[i]/9.0*(-4.0*eps/l3 + 2.0/eps*l2*l2/l3 - 4.0/eps*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l2
      # P = np.dot(U, np.dot(Pd, v2.transpose()))
      ##W = 0.5*mu[i]*((eps*eps + l2*l2 + l3*l3)/pow23 - 3.0) + mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23*(l1-eps) + 0.5*k_param*(l1-eps)*(l1-eps) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))/4.0
     
    # Calculate tetra face negative normals (because traction Ft=-P*n)
    # xr1[0:2] = c_ref_state_growth[i,0,0], c_ref_state_growth[i,1,0], c_ref_state_growth[i,2,0]
    # xr2[0:2] = c_ref_state_growth[i,0,1], c_ref_state_growth[i,1,1], c_ref_state_growth[i,2,1]
    # xr3[0:2] = c_ref_state_growth[i,0,2], c_ref_state_growth[i,1,2], c_ref_state_growth[i,2,2]
    xr1 = np.array([ref_state_growth[i,0,0], ref_state_growth[i,1,0], ref_state_growth[i,2,0]])
    xr2 = np.array([ref_state_growth[i,0,1], ref_state_growth[i,1,1], ref_state_growth[i,2,1]])
    xr3 = np.array([ref_state_growth[i,0,2], ref_state_growth[i,1,2], ref_state_growth[i,2,2]])
    N1 = np.cross(xr3, xr1)
    N2 = np.cross(xr2, xr3)
    N3 = np.cross(xr1, xr2)
    N4 = np.cross(xr2-xr3, xr1-xr3)
        
      # Distribute forces among tetra vertices, probably not vectorizable. Surprising that its //
    Ft[tets[i,0]] += np.dot(c_P, (N1 + N2 + N3).T)/6.0
    Ft[tets[i,1]] += np.dot(c_P, (N1 + N3 + N4).T)/6.0
    Ft[tets[i,2]] += np.dot(c_P, (N2 + N3 + N4).T)/6.0
    Ft[tets[i,3]] += np.dot(c_P, (N1 + N2 + N4).T)/6.0
        
  return Ft


  
def tetra_elasticity(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps):
  """
  Calculates elastic forces
  """
    
  #tetraelasticty variables
  ref_state_growth = np.zeros ((n_tets, 3, 3), dtype=np.float64) #Ar
  deformation_grad = np.zeros((n_tets,3,3), dtype=np.float64)
  left_cauchy_grad = np.zeros((n_tets,3,3), dtype=np.float64)
  rel_vol_chg = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg1 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg2 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg3 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg4 = np.zeros(n_tets, dtype=np.float64)
  rel_vol_chg_av = np.zeros(n_tets, dtype=np.float64)
    
  #Apply growth to reference state
  ref_state_growth = dot_mat_dim_3(tan_growth_tensor, ref_state_tets)
  #Calculate deformation gradient F //combine relative volume change ?
  deformation_grad = dot_mat_dim_3(material_tets, inv_dim_3(ref_state_growth))   
  #Calculate Left-Cauchy-Green gradient B
  left_cauchy_grad = dot_mat_dim_3(deformation_grad, np.transpose(deformation_grad, (0, 2, 1)))
  #relative volume change J
  rel_vol_chg = det_dim_3(deformation_grad)
  #averaged volume change
  rel_vol_chg1 = Vn[tets[:,0]]/Vn0[tets[:,0]]
  rel_vol_chg2 = Vn[tets[:,1]]/Vn0[tets[:,1]]
  rel_vol_chg3 = Vn[tets[:,2]]/Vn0[tets[:,2]]
  rel_vol_chg4 = Vn[tets[:,3]]/Vn0[tets[:,3]]
  rel_vol_chg_av = (rel_vol_chg1 + rel_vol_chg2 + rel_vol_chg3 + rel_vol_chg4)/4.0 

  identity = np.identity(3)  
    
  #decide if need SVD or not
  for i in range(n_tets):
        
    ll1, ll2, ll3 = EV(left_cauchy_grad[i])
    # ll1, ll3, ll2 = np.linalg.eig(left_cauchy_grad[i])
        
    if ll3 >= eps*eps and rel_vol_chg[i] > 0.0:  # No need for SVD
      powJ23 = np.power(rel_vol_chg[i], 2.0/3.0)
      S = (left_cauchy_grad[i] - identity*np.trace(left_cauchy_grad[i])/3.0)*mu[i]/(rel_vol_chg[i]*powJ23) + identity*bulk_modulus*(rel_vol_chg_av[i]-1.0)
      P = np.dot(S, np.linalg.inv(deformation_grad[i].transpose()))*rel_vol_chg[i] #P = cauchy stress
      W = 0.5*mu[i]*(np.trace(left_cauchy_grad[i])/powJ23 - 3.0) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))*0.25
        
        
    else:   #need SVD, does not proc on sphere5 data
      C = np.dot(deformation_grad[i].transpose(), deformation_grad[i])
      w2, v2 = np.linalg.eigh(C)
      v2 = -v2
    
      l1 = sqrt(w2[0])
      l2 = sqrt(w2[1])
      l3 = sqrt(w2[2])
    
      if np.linalg.det(v2) < 0.0:
        v2[0,0] = -v2[0,0]
        v2[1,0] = -v2[1,0]
        v2[2,0] = -v2[2,0]
    
      Fdi = identity
      if l1 >= 1e-25:
        Fdi[0,0] = 1.0/l1
        Fdi[1,1] = 1.0/l2
        Fdi[2,2] = 1.0/l3
    
      U = np.dot(deformation_grad[i], np.dot(v2, Fdi))
    
      if l1 < 1e-25:
        U[0,0] = U[1,1]*U[2,2] - U[2,1]*U[1,2]
        U[1,0] = U[2,1]*U[0,2] - U[0,1]*U[2,2]
        U[2,0] = U[0,1]*U[1,2] - U[1,1]*U[0,2]
    
      if np.linalg.det(deformation_grad[i]) < 0.0:
        l1 = -l1
        U[0,0] = -U[0,0]
        U[1,0] = -U[1,0]
        U[2,0] = -U[2,0]
    
      Pd = identity
      pow23 = np.power(eps*l2*l3, 2.0/3.0)
      Pd[0,0] = mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23 + k_param*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l2*l3
      Pd[1,1] = mu[i]/3.0*(-eps*eps/l2 + 2.0*l2 - l3*l3/l2)/pow23 + mu[i]/9.0*(-4.0*eps/l2 - 4.0/eps*l2 + 2.0/eps/l2*l3*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l3
      Pd[2,2] = mu[i]/3.0*(-eps*eps/l3 - l2*l2/l3 + 2.0*l3)/pow23 + mu[i]/9.0*(-4.0*eps/l3 + 2.0/eps*l2*l2/l3 - 4.0/eps*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l2
      P = np.dot(U, np.dot(Pd, v2.transpose()))
      #W = 0.5*mu[i]*((eps*eps + l2*l2 + l3*l3)/pow23 - 3.0) + mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23*(l1-eps) + 0.5*k_param*(l1-eps)*(l1-eps) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))/4.0
     
    # Calculate tetra face negative normals (because traction Ft=-P*n)
    xr1 = np.array([ref_state_growth[i,0,0], ref_state_growth[i,1,0], ref_state_growth[i,2,0]])
    xr2 = np.array([ref_state_growth[i,0,1], ref_state_growth[i,1,1], ref_state_growth[i,2,1]])
    xr3 = np.array([ref_state_growth[i,0,2], ref_state_growth[i,1,2], ref_state_growth[i,2,2]])
    N1 = np.cross(xr3, xr1)
    N2 = np.cross(xr2, xr3)
    N3 = np.cross(xr1, xr2)
    N4 = np.cross(xr2-xr3, xr1-xr3)
        
      # Distribute forces among tetra vertices, probably not vectorizable. Surprising that its //
    Ft[tets[i,0]] += np.dot(P, (N1 + N2 + N3).T)/6.0
    Ft[tets[i,1]] += np.dot(P, (N1 + N3 + N4).T)/6.0
    Ft[tets[i,2]] += np.dot(P, (N2 + N3 + N4).T)/6.0
    Ft[tets[i,3]] += np.dot(P, (N1 + N2 + N4).T)/6.0
        
  return Ft


