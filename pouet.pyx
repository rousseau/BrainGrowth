'''
TODO:
-pure memoryview version
-loop releases the gil
-parallel
'''

import cython
cimport cython
import numpy as np
from mathfunc import det_dim_3

@cython.boundscheck(False)
@cython.wraparound(False)
def transpose_dim_2(array: np.float64) -> np.float64:
    cdef cython.int array_dim_0 = array.shape[0]
    cdef cython.int array_dim_1 = array.shape[1]
    cdef double [:,:] av = array
    cdef double [:,:] transpose = np.zeros(array.shape)
    cdef cython.int i
    cdef cython.int j 
    for i in range(array_dim_0):
        for j in range(array_dim_1):
            transpose[j][i] = av[i][j]
            
    return np.array(transpose)


@cython.boundscheck(False)
@cython.wraparound(False)
def cdet_dim_2(a: np.float64) -> float:
    # pure equivalent of np.linalg.det (a)
    cdef double [:,:] av = a
    cdef double b 
    b = (
          av[0, 0] * av[1, 1] * av[2, 2]
        - av[0, 0] * av[1, 2] * av[2, 1]
        - av[0, 1] * av[1, 0] * av[2, 2]
        + av[0, 1] * av[1, 2] * av[2, 0]
        + av[0, 2] * av[1, 0] * av[2, 1]
        - av[0, 2] * av[1, 1] * av[2, 0]
    )
    return float(b)

@cython.boundscheck(False)
@cython.wraparound(False)
def cdet_dim_3(a: np.float64) -> np.float64:
    '''
    REsults same
    '''
    cdef cython.int a_dim = a.shape[0]
    cdef double [:] b = np.zeros((a_dim))
    cdef double [:,:,:] av = a
    cdef cython.int i

    for i in range(a_dim):
        b[i] = (
          av[i, 0, 0] * av[i, 1, 1] * av[i, 2, 2]
        - av[i, 0, 0] * av[i, 1, 2] * av[i, 2, 1]
        - av[i, 0, 1] * av[i, 1, 0] * av[i, 2, 2]
        + av[i, 0, 1] * av[i, 1, 2] * av[i, 2, 0]
        + av[i, 0, 2] * av[i, 1, 0] * av[i, 2, 1]
        - av[i, 0, 2] * av[i, 1, 1] * av[i, 2, 0]
        )


    return np.array(b)

@cython.boundscheck(False)
@cython.wraparound(False)
def cdot_mat_dim_3(a: np.float64, b: np.float64) -> np.float64:
    cdef int a_dim = a.shape[0]
    cdef double [:,:,:] c = np.zeros((a_dim, 3, 3))
    cdef double [:,:,:] av = a
    cdef double [:,:,:] bv = b
    cdef int i

    for i in range(a_dim):

        c[i, 0, 0] = (
            av[i, 0, 0] * bv[i, 0, 0] + av[i, 0, 1] * bv[i, 1, 0] + av[i, 0, 2] * bv[i, 2, 0]
        )
        c[i, 0, 1] = (
            av[i, 0, 0] * bv[i, 0, 1] + av[i, 0, 1] * bv[i, 1, 1] + av[i, 0, 2] * bv[i, 2, 1]
        )
        c[i, 0, 2] = (
            av[i, 0, 0] * bv[i, 0, 2] + av[i, 0, 1] * bv[i, 1, 2] + av[i, 0, 2] * bv[i, 2, 2]
        )
        c[i, 1, 0] = (
            av[i, 1, 0] * bv[i, 0, 0] + av[i, 1, 1] * bv[i, 1, 0] + av[i, 1, 2] * bv[i, 2, 0]
        )
        c[i, 1, 1] = (
            av[i, 1, 0] * bv[i, 0, 1] + av[i, 1, 1] * bv[i, 1, 1] + av[i, 1, 2] * bv[i, 2, 1]
        )
        c[i, 1, 2] = (
            av[i, 1, 0] * bv[i, 0, 2] + av[i, 1, 1] * bv[i, 1, 2] + av[i, 1, 2] * bv[i, 2, 2]
        )
        c[i, 2, 0] = (
            av[i, 2, 0] * bv[i, 0, 0] + av[i, 2, 1] * bv[i, 1, 0] + av[i, 2, 2] * bv[i, 2, 0]
        )
        c[i, 2, 1] = (
            av[i, 2, 0] * bv[i, 0, 1] + av[i, 2, 1] * bv[i, 1, 1] + av[i, 2, 2] * bv[i, 2, 1]
        )
        c[i, 2, 2] = (
            av[i, 2, 0] * bv[i, 0, 2] + av[i, 2, 1] * bv[i, 1, 2] + av[i, 2, 2] * bv[i, 2, 2]
        )

    return np.array(c)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def ccdet_dim_3(a: np.float64) -> np.float64:
#     '''
#     Pointers direclty on numpy array more efficient than C array or memory view (but more cumbersome) :: This fct does not work yet
#     '''
#     cdef int a_dim = a.shape[0]
#     cdef double b
#     cdef double c[a_dim][3][3]
#     cdef int i
#     # b = (
#     #       a[:, 0, 0] * a[:, 1, 1] * a[:, 2, 2]
#     #     - a[:, 0, 0] * a[:, 1, 2] * a[:, 2, 1]
#     #     - a[:, 0, 1] * a[:, 1, 0] * a[:, 2, 2]
#     #     + a[:, 0, 1] * a[:, 1, 2] * a[:, 2, 0]
#     #     + a[:, 0, 2] * a[:, 1, 0] * a[:, 2, 1]
#     #     - a[:, 0, 2] * a[:, 1, 1] * a[:, 2, 0]
#     # )
#     for i in range(a_dim): #can call shape to speed up things
#         b[i] = (
#           c[i][0][0] * c[i][1][1] * c[i][2][2]
#         - c[i][0][0] * c[i][1][2] * c[i][2][1]
#         - c[i][0][1] * c[i][1][0] * c[i][2][2]
#         + c[i][0][1] * c[i][1][2] * c[i][2][0]
#         + c[i][0][2] * c[i][1][0] * c[i][2][1]
#         - c[i][0][2] * c[i][1][1] * c[i][2][0]
#         )


#     return np.array(b)


