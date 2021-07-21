import numpy as np   
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray

ctypedef np.float64_t DTYPE_t


cdef double[:,:,:] matmul_dim_3(double[:,:,:,]a, double[:,:,:]b):

  cdef double [:,:,:] c
  c[:,0,0] = a[:,0,0] * b[:,0,0] + a[:,0,1] * b[:,1,0] + a[:,0,2] * b[:,2,0]
  c[:,0,1] = a[:,0,0] * b[:,0,1] + a[:,0,1] * b[:,1,1] + a[:,0,2] * b[:,2,1]
  c[:,0,2] = a[:,0,0] * b[:,0,2] + a[:,0,1] * b[:,1,2] + a[:,0,2] * b[:,2,2]
  c[:,1,0] = a[:,1,0] * b[:,0,0] + a[:,1,1] * b[:,1,0] + a[:,1,2] * b[:,2,0]
  c[:,1,1] = a[:,1,0] * b[:,0,1] + a[:,1,1] * b[:,1,1] + a[:,1,2] * b[:,2,1]
  c[:,1,2] = a[:,1,0] * b[:,0,2] + a[:,1,1] * b[:,1,2] + a[:,1,2] * b[:,2,2]
  c[:,2,0] = a[:,2,0] * b[:,0,0] + a[:,2,1] * b[:,1,0] + a[:,2,2] * b[:,2,0]
  c[:,2,1] = a[:,2,0] * b[:,0,1] + a[:,2,1] * b[:,1,1] + a[:,2,2] * b[:,2,1]
  c[:,2,2] = a[:,2,0] * b[:,0,2] + a[:,2,1] * b[:,1,2] + a[:,2,2] * b[:,2,2]
  
  return c

# def double[:,:,:]inv_dim_3(a):
#   cdef double b[:, :, :] b
#   for i in prange(len(a)):
#     b[i] = np.linalg.inv(a[i])

#   return b

#use type double, not float for whatever reason
cpdef np.ndarray[np.float64_t, ndim = 2] tetra_elasticity_cyth(np.ndarray[np.float64_t, ndim = 3] material_tets, np.ndarray[np.float64_t, ndim = 3] ref_state_tets, np.ndarray[np.float64_t, ndim = 2] Ft, np.ndarray[np.float64_t, ndim = 3] tan_growth_tensor, double bulk_modulus, double k_param, np.ndarray[np.float64_t, ndim = 1] mu, np.ndarray[np.int64_t, ndim = 2] tets, np.ndarray[np.float64_t, ndim = 1] Vn, np.ndarray[np.float64_t, ndim = 1] Vn0, long n_tets, double eps):
  #memory views
  cdef double[:, :, :] material_tets_view = material_tets
  cdef double[:, :, :] ref_state_tets_view = ref_state_tets
  cdef double[:,:] Ft_view
  cdef double[:, :, :] tan_growth_tensor_view = tan_growth_tensor
  cdef double[:] mu_view = mu
  cdef long[:,:] tets_view = tets
  cdef double[:] Vn_view = Vn
  cdef double[:] Vn0_view = Vn0

  
  cdef double[:, :, :] rest_state_growth
  cdef double[:, :, :] deformation_grad
  cdef double[:, :, :] left_cauchy_grad

  cdef double[:] rel_vol_chg
  cdef double[:] rel_vol_chg1
  cdef double[:] rel_vol_chg2
  cdef double[:] rel_vol_chg3
  cdef double[:] rel_vol_chg4
  cdef double[:] rel_vol_chg_av

  # #Apply growth to reference state
  # ref_state_growth = dot_mat_dim_3(tan_growth_tensor, ref_state_tets)
  # #Calculate deformation gradient F //combine relative volume change ?
  # deformation_grad = dot_mat_dim_3(material_tets, inv_dim_3(ref_state_growth))   
  # #Calculate Left-Cauchy-Green gradient B
  # left_cauchy_grad = dot_mat_dim_3(deformation_grad, np.transpose(deformation_grad, (0, 2, 1)))
  # #relative volume change J
  # rel_vol_chg = det_dim_3(deformation_grad)
  #averaged volume change
  rel_vol_chg1 = Vn[tets[:,0]]/Vn0[tets[:,0]]
  rel_vol_chg2 = Vn[tets[:,1]]/Vn0[tets[:,1]]
  rel_vol_chg3 = Vn[tets[:,2]]/Vn0[tets[:,2]]
  rel_vol_chg4 = Vn[tets[:,3]]/Vn0[tets[:,3]]
  for i in range(len(rel_vol_chg_av)):
    rel_vol_chg_av[i] = (rel_vol_chg1[i] + rel_vol_chg2[i] + rel_vol_chg3[i] + rel_vol_chg4[i])/4.0

  return Ft

