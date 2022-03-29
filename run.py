'''
Runner file to use compiled version of BrainGrowth
'''
# from simulation_cyth import main
from mathfunc import det_dim_3, dot_mat_dim_3
from pouet import *
import numpy as np
import cProfile
from mechanics import tetra_elasticity, tetra_elasticity_np, tetra1, tetra2
from mechanicsC import c_tetra_elasticity

####tetra elasticity test
from numba import jit, njit, prange, vectorize
import cProfile
from mathfunc import det_dim_2, det_dim_3, inv, inv_dim_3, cross_dim_2, transpose_dim_3, dot_mat_dim_3, EV, Eigensystem
from math import sqrt
from time import time
from simulation_cyth import main

n_tets = 35497
n_nodes = 6761
eps = 0.1

material_tets = np.random.rand (n_tets, 3, 3)
ref_state_tets = np.random.rand (n_tets, 3, 3)
Ft = np.random.rand(n_nodes, 3)
Ft_base = Ft.copy()
Ft_2 = Ft.copy()
tan_growth_tensor = np.resize(np.identity(3), (n_tets, 3, 3))
bulk_modulus = 5.0
k_param = 0
mu = np.random.rand(n_tets)
tet_norms = np.random.rand (n_tets, 3)
gm = np.random.rand(n_tets)
at = np.random.rand(n_tets)
#potential pitfall at tets random generation, create function to draw unique if you see problem
tets = np.random.randint(0, 6761, (n_tets, 4))
Vn = np.random.rand(n_nodes)
Vn0 = np.random.rand(n_nodes)

# tetra_elasticity_np(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps)

# c_tetra_elasticity(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps)

# tetra_elasticity(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps)

#tetra1(tets, tan_growth_tensor, ref_state_tets, ref_state_growth, material_tets, Vn, Vn0)

#tetra2(n_tets, tets, Ft, left_cauchy_grad, mu, eps, rel_vol_chg, bulk_modulus,rel_vol_chg_av, deformation_grad, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, k_param, ref_state_growth)

main()

# array = np.random.randn(100000, 3, 3)