from __future__ import division

from numpy.lib.function_base import vectorize
from mathfunc import det_dim_2, det_dim_3, inv, inv_dim_3, cross_dim_2, transpose_dim_3, dot_mat_dim_3, EV, Eigensystem
import numpy as np
from math import sqrt
from numba import jit, njit, prange
import time

@jit(nopython=True)
def tetra1(tets, tan_growth_tensor, ref_state_tets, ref_state_growth, material_tets, Vn, Vn0):
  """
  Calculates elastic forces
  """
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

  return left_cauchy_grad, rel_vol_chg, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, rel_vol_chg_av, deformation_grad, ref_state_growth

#pure np bit slower than numba version
def tetra1_np(tets, tan_growth_tensor, ref_state_tets, ref_state_growth, material_tets, Vn, Vn0):
  """
  Calculates elastic forces
  """
  #Apply growth to reference state
  ref_state_growth = tan_growth_tensor @ ref_state_tets
  #Calculate deformation gradient F //combine relative volume change ?
  deformation_grad = material_tets @ np.linalg.inv(ref_state_growth) 
  #Calculate Left-Cauchy-Green gradient B
  left_cauchy_grad = deformation_grad @ np.transpose(deformation_grad, (0, 2, 1))
  #relative volume change J
  rel_vol_chg = np.linalg.det(deformation_grad)
  #averaged volume change
  rel_vol_chg1 = Vn[tets[:,0]]/Vn0[tets[:,0]]
  rel_vol_chg2 = Vn[tets[:,1]]/Vn0[tets[:,1]]
  rel_vol_chg3 = Vn[tets[:,2]]/Vn0[tets[:,2]]
  rel_vol_chg4 = Vn[tets[:,3]]/Vn0[tets[:,3]]
  rel_vol_chg_av = (rel_vol_chg1 + rel_vol_chg2 + rel_vol_chg3 + rel_vol_chg4)/4.0  

  return left_cauchy_grad, rel_vol_chg, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, rel_vol_chg_av, deformation_grad, ref_state_growth

@jit(nopython=True, parallel=True, nogil=True)
def tetra2(n_tets, tets, Ft, left_cauchy_grad, mu, eps, rel_vol_chg, bulk_modulus,rel_vol_chg_av, deformation_grad, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, k_param, ref_state_growth):
  #decide if need SVD or not
  for i in prange (n_tets):
        
    ll1, ll2, ll3 = EV(left_cauchy_grad[i])
        
    if ll3 >= eps*eps and rel_vol_chg[i] > 0.0:  # No need for SVD
      powJ23 = np.power(rel_vol_chg[i], 2.0/3.0)
      S = (left_cauchy_grad[i] - np.identity(3)*np.trace(left_cauchy_grad[i])/3.0)*mu[i]/(rel_vol_chg[i]*powJ23) + np.identity(3)*bulk_modulus*(rel_vol_chg_av[i]-1.0)
      P = np.dot(S, np.linalg.inv(deformation_grad[i].transpose()))*rel_vol_chg[i]
      #W = 0.5*mu[i]*(np.trace(left_cauchy_grad[i])/powJ23 - 3.0) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))*0.25
        
        
    else:   #need SVD
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
    
      Fdi = np.identity(3)
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
    
      Pd = np.identity(3)
      pow23 = np.power(eps*l2*l3, 2.0/3.0)
      Pd[0,0] = mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23 + k_param*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l2*l3
      Pd[1,1] = mu[i]/3.0*(-eps*eps/l2 + 2.0*l2 - l3*l3/l2)/pow23 + mu[i]/9.0*(-4.0*eps/l2 - 4.0/eps*l2 + 2.0/eps/l2*l3*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l3
      Pd[2,2] = mu[i]/3.0*(-eps*eps/l3 - l2*l2/l3 + 2.0*l3)/pow23 + mu[i]/9.0*(-4.0*eps/l3 + 2.0/eps*l2*l2/l3 - 4.0/eps*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l2
      P = np.dot(U, np.dot(Pd, v2.transpose()))
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

@jit(nopython=True, parallel=False)   
def tetra_elasticity_test(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps):
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
    
  #decide if need SVD or not
  for i in prange (n_tets):
        
    ll1, ll2, ll3 = EV(left_cauchy_grad[i])
        
    if ll3 >= eps*eps and rel_vol_chg[i] > 0.0:  # No need for SVD
      powJ23 = np.power(rel_vol_chg[i], 2.0/3.0)
      S = (left_cauchy_grad[i] - np.identity(3)*np.trace(left_cauchy_grad[i])/3.0)*mu[i]/(rel_vol_chg[i]*powJ23) + np.identity(3)*bulk_modulus*(rel_vol_chg_av[i]-1.0)
      P = np.dot(S, np.linalg.inv(deformation_grad[i].transpose()))*rel_vol_chg[i]
      W = 0.5*mu[i]*(np.trace(left_cauchy_grad[i])/powJ23 - 3.0) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))*0.25
        
        
    else:   #need SVD
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
    
      Fdi = np.identity(3)
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
    
      Pd = np.identity(3)
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

@jit(nopython=True, parallel=True)   
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
    
  #decide if need SVD or not
  for i in prange (n_tets):
        
    ll1, ll2, ll3 = EV(left_cauchy_grad[i])
        
    if ll3 >= eps*eps and rel_vol_chg[i] > 0.0:  # No need for SVD
      powJ23 = np.power(rel_vol_chg[i], 2.0/3.0)
      S = (left_cauchy_grad[i] - np.identity(3)*np.trace(left_cauchy_grad[i])/3.0)*mu[i]/(rel_vol_chg[i]*powJ23) + np.identity(3)*bulk_modulus*(rel_vol_chg_av[i]-1.0)
      P = np.dot(S, np.linalg.inv(deformation_grad[i].transpose()))*rel_vol_chg[i] #P = cauchy stress
      W = 0.5*mu[i]*(np.trace(left_cauchy_grad[i])/powJ23 - 3.0) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))*0.25
        
        
    else:   #need SVD
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
    
      Fdi = np.identity(3)
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
    
      Pd = np.identity(3)
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

#pure np/python version
@jit(nopython=True)
def tetraElasticity_np(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps):

  # Apply growth to reference state
  Ar = np.zeros((n_tets,3,3), dtype=np.float64)
  Ar[:] = tan_growth_tensor[:] @ ref_state_tets[:]

  # Calculate deformation gradient
  F = np.zeros((n_tets,3,3), dtype=np.float64)
  F[:] = material_tets[:] @ np.linalg.inv(Ar[:])   # Ar: rest tetra, At: material tetra

  # Calculate left Cauchy-Green strain tensor
  B = np.zeros((n_tets,3,3), dtype=np.float64)
  B[:] = F[:] @ np.transpose(F, (0, 2, 1))

  # Calculate relative volume change and averaged nodal volume change
  J = np.zeros(n_tets, dtype=np.float64)
  J1 = np.zeros(n_tets, dtype=np.float64)
  J2 = np.zeros(n_tets, dtype=np.float64)
  J3 = np.zeros(n_tets, dtype=np.float64)
  J4 = np.zeros(n_tets, dtype=np.float64)
  Ja = np.zeros(n_tets, dtype=np.float64)
  J[:] = np.linalg.det(F[:]) # Relative volume change
  J1[:] = Vn[tets[:,0]]/Vn0[tets[:,0]]
  J2[:] = Vn[tets[:,1]]/Vn0[tets[:,1]]
  J3[:] = Vn[tets[:,2]]/Vn0[tets[:,2]]
  J4[:] = Vn[tets[:,3]]/Vn0[tets[:,3]]
  Ja[:] = (J1[:] + J2[:] + J3[:] + J4[:])/4.0   # Averaged nodal volume change

  # Decide if need for SVD or not
  for i in prange(n_tets):

    ll1, ll3, ll2 = np.linalg.eigvals(B[i])   #retest, some error between ll3 and ll2, longer than numa EV, sortable de la boucle

    if ll3 >= eps*eps and J[i] > 0.0: # No need for SVD

     # Calculate the total stress (shear stress + bulk stress)
      powJ23 = np.power(J[i], 2.0/3.0)
      S = (B[i] - np.identity(3)*np.trace(B[i])/3.0)*mu[i]/(J[i]*powJ23) + np.identity(3)*bulk_modulus*(Ja[i]-1.0)
      P = np.dot(S, np.linalg.inv(F[i].transpose()))*J[i]
      W = 0.5*mu[i]*(np.trace(B[i])/powJ23 - 3.0) + 0.5*bulk_modulus*((J1[i]-1.0)*(J1[i]-1.0) + (J2[i]-1.0)*(J2[i]-1.0) + (J3[i]-1.0)*(J3[i]-1.0) + (J4[i]-1.0)*(J4[i]-1.0))*0.25

    else:  # Needs SVD

      C = np.dot(F[i].transpose(), F[i])

      w2, v2 = np.linalg.eigh(C)
      v2 = -v2

      l1 = sqrt(w2[0])
      l2 = sqrt(w2[1])
      l3 = sqrt(w2[2])

      if np.linalg.det(v2) < 0.0:
        v2[0,0] = -v2[0,0]
        v2[1,0] = -v2[1,0]
        v2[2,0] = -v2[2,0]

      Fdi = np.identity(3)
      if l1 >= 1e-25:
        Fdi[0,0] = 1.0/l1
        Fdi[1,1] = 1.0/l2
        Fdi[2,2] = 1.0/l3

      U = np.dot(F[i], np.dot(v2, Fdi))

      if l1 < 1e-25:
        U[0,0] = U[1,1]*U[2,2] - U[2,1]*U[1,2]
        U[1,0] = U[2,1]*U[0,2] - U[0,1]*U[2,2]
        U[2,0] = U[0,1]*U[1,2] - U[1,1]*U[0,2]

      if np.linalg.det(F[i]) < 0.0:
        l1 = -l1
        U[0,0] = -U[0,0]
        U[1,0] = -U[1,0]
        U[2,0] = -U[2,0]

      Pd = np.identity(3)
      pow23 = np.power(eps*l2*l3, 2.0/3.0)
      Pd[0,0] = mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23 + k_param*(l1-eps) + bulk_modulus*(Ja[i]-1.0)*l2*l3
      Pd[1,1] = mu[i]/3.0*(-eps*eps/l2 + 2.0*l2 - l3*l3/l2)/pow23 + mu[i]/9.0*(-4.0*eps/l2 - 4.0/eps*l2 + 2.0/eps/l2*l3*l3)/pow23*(l1-eps) + bulk_modulus*(Ja[i]-1.0)*l1*l3
      Pd[2,2] = mu[i]/3.0*(-eps*eps/l3 - l2*l2/l3 + 2.0*l3)/pow23 + mu[i]/9.0*(-4.0*eps/l3 + 2.0/eps*l2*l2/l3 - 4.0/eps*l3)/pow23*(l1-eps) + bulk_modulus*(Ja[i]-1.0)*l1*l2
      P = np.dot(U, np.dot(Pd, v2.transpose()))
      W = 0.5*mu[i]*((eps*eps + l2*l2 + l3*l3)/pow23 - 3.0) + mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23*(l1-eps) + 0.5*k_param*(l1-eps)*(l1-eps) + 0.5*bulk_modulus*((J1[i]-1.0)*(J1[i]-1.0) + (J2[i]-1.0)*(J2[i]-1.0) + (J3[i]-1.0)*(J3[i]-1.0) + (J4[i]-1.0)*(J4[i]-1.0))/4.0

  # Increment total elastic energy
  #if J*J > 1e-50:
    #Ue += W*vol/J

    # Calculate tetra face negative normals (because traction Ft=-P*n)
    xr1 = np.array([Ar[i,0,0], Ar[i,1,0], Ar[i,2,0]])
    xr2 = np.array([Ar[i,0,1], Ar[i,1,1], Ar[i,2,1]])
    xr3 = np.array([Ar[i,0,2], Ar[i,1,2], Ar[i,2,2]])
    N1 = np.cross(xr3, xr1)
    N2 = np.cross(xr2, xr3)
    N3 = np.cross(xr1, xr2)
    N4 = np.cross(xr2-xr3, xr1-xr3)

    # Distribute forces among tetra vertices
    Ft[tets[i,0]] += np.dot(P, (N1 + N2 + N3).T)/6.0
    Ft[tets[i,1]] += np.dot(P, (N1 + N3 + N4).T)/6.0
    Ft[tets[i,2]] += np.dot(P, (N2 + N3 + N4).T)/6.0
    Ft[tets[i,3]] += np.dot(P, (N1 + N2 + N4).T)/6.0

  return Ft

@jit(nopython=True) #vectorized version
def move(n_nodes, Ft, Vt, coordinates, damping_coef, Vn0, mass_density, dt):
  """
  Integrate forces and velocities to displacement, reinitialize Ft
  Args:
  n_nodes (int): number of nodes
  Ft (array): forces applied to nodes
  Vt (array): Velocity of nodes
  coordinates (array): coordinates of vertices
  damping coef (float): 
  Vn0 (array): volume for each node
  mass_density (float): 
  dt (float): 
  """ 
  vol = np.reshape(np.repeat(Vn0, 3), (n_nodes, 3))
  Ft -= Vt * damping_coef * vol
  Vt += Ft/(vol*mass_density)*dt
  coordinates += Vt*dt
  Ft = np.zeros((n_nodes, 3), dtype=np.float64)
    
  return Ft, coordinates, Vt
