#!python
#cython: language_level=3

'''
TODO: memview may not be the ideal tool for array declaration/manipulation. You can use C types instead
At best, use the C structure and code directly from Tallinen for increased prod/reliability
'''

import numpy as np  
from numba import jit, njit, prange 
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
import cProfile
from time import time
from libc.math cimport *


ctypedef np.float64_t dtype_t
cdef double[:,:,:] matmul_dim_3 (double[:,:,:]a, double[:,:,:]b):
  cdef double[:,:,:] c

  
    return Matrix( a*n.a+b*n.d+c*n.g, a*n.b+b*n.e+c*n.h, a*n.c+b*n.f+c*n.i, 
		   d*n.a+e*n.d+f*n.g, d*n.b+e*n.e+f*n.h, d*n.c+e*n.f+f*n.i, 
		   g*n.a+h*n.d+i*n.g, g*n.b+h*n.e+i*n.h, g*n.c+h*n.f+i*n.i );
  return c

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double [:,:] foo(double[:, :, :] material_tets_view, double[:, :, :] ref_state_tets_view, double[:,:] Ft_view, double[:, :, :] tan_growth_tensor_view, double bulk_modulus, double k_param, double[:] mu_view, long[:,:] tets_view, double[:] Vn_view, double[:] Vn0_view):
  #Initialize tetraelasticty variables
  cdef double[:, :, :] ref_state_growth
  cdef double[:, :, :] deformation_grad
  cdef double[:, :, :] left_cauchy_grad
  cdef double[:] rel_vol_chg
  cdef double[:] rel_vol_chg1
  cdef double[:] rel_vol_chg2
  cdef double[:] rel_vol_chg3
  cdef double[:] rel_vol_chg4
  cdef double[:] rel_vol_chg_av

  #Apply growth to reference state
  ref_state_growth = matmul_dim_3(tan_growth_tensor_view, ref_state_tets_view)

  for i in range(len(Ft_view)):
    Ft_view[i][0] += 1

  return Ft_view

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef np.ndarray[np.float64_t, ndim = 2] wrapper(np.ndarray[np.float64_t, ndim = 3] material_tets, np.ndarray[np.float64_t, ndim = 3] ref_state_tets, np.ndarray[np.float64_t, ndim = 2] Ft, np.ndarray[np.float64_t, ndim = 3] tan_growth_tensor, double bulk_modulus, double k_param, np.ndarray[np.float64_t, ndim = 1] mu, np.ndarray[np.int64_t, ndim = 2] tets, np.ndarray[np.float64_t, ndim = 1] Vn, np.ndarray[np.float64_t, ndim = 1] Vn0, long n_tets, double eps):
  #memory views
  cdef double[:, :, :] material_tets_view = material_tets
  cdef double[:, :, :] ref_state_tets_view = ref_state_tets
  cdef double[:,:] Ft_view = Ft
  cdef double[:, :, :] tan_growth_tensor_view = tan_growth_tensor
  cdef double[:] mu_view = mu
  cdef long[:,:] tets_view = tets
  cdef double[:] Vn_view = Vn
  cdef double[:] Vn0_view = Vn0

  return np.asarray(foo(material_tets_view, ref_state_tets_view, Ft_view, tan_growth_tensor_view, bulk_modulus, k_param, mu_view, tets_view, Vn_view, Vn0_view))


#Tester numpy array structure vs c like

#initialize tetraelasticity variables

#apply growth to ref state = cross product of tan_growth_tensor and ref_state_tets

#calculate deformation gradient (cross product of materival tets and the invert of ref state growth)

#calculate left caucgy gradient (cross prodcut of deformation gradient and its transpose)

#calculate relative volume change (determinant of deformation gradient)

#decide if SVD or not

  # return np.asarray(mview)
  # return mview.base
'''


 cdef double* work = <double*>malloc(n*n*sizeof(double))
 try:
     # rest of function
 finally:
     free(work)


'''