# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math
#from sklearn import preprocessing
from vapory import *
import re
import os
import sys
from geometry import *
from growth import *
from normalisation import *
from collision import *
from mechanics import *
from output import *
#from joblib import Parallel, delayed
#import multiprocessing as mp
from numba import prange

# Parameters to change
PATH_DIR = "./data/sphere5_3" # Path of results
THICKNESS_CORTEX = 0.042
GROWTH_RELATIVE = 1.829

# Path of mesh
mesh_path = "/home/x17wang/Bureau/Brain_growth/sphere5.mesh"

# Import mesh, each line as a list
mesh = importMesh(mesh_path)

# Read nodes, get undeformed coordinates (Ut0) and initialize deformed coordinates (Ut) of all nodes
Ut0, Ut, nn = vertex(mesh)

# Read element indices (tets: index of four vertices of tetrahedra) and get number of elements (ne)
tets, ne = tetraVerticesIndices(mesh, nn)

# Read surface triangle indices (faces: index of three vertices of triangles) and get number of surface triangles (nf)
faces, nf = triangleIndices(mesh, nn, ne)

# Determine surface nodes and index maps (nsn: number of nodes at the surface, SN: Nodal index map from surface to full mesh, SNb: Nodal index map from full mesh to surface)
nsn, SN, SNb = numberSurfaceNodes(faces, nn, nf)

# Calculate the total volume of a tetrahedral mesh
Vm = volume_mesh(nn, ne, tets, Ut)
print ('Volume of mesh is ' + str(Vm))

# Parameters
H = THICKNESS_CORTEX  #Thickness of growing layer
Hcp = THICKNESS_CORTEX #Cortical plate thickness for visualization
mug = 1.0 #65.0 Shear modulus of gray matter
muw = 1.167 #75.86 Shear modulus of white matter
K = 5.0 #100.0 Bulk modulus
a = 0.01 #Mesh spacing - set manually based on the average spacing in the mesh
rho = 0.01 #0.0001 #Mass density - adjust to run the simulation faster or slower
gamma = 0.5 #0.1 Damping coefficent
di = 500 #Output data once every di steps

bw = 3.2; #Width of a bounding box, centered at origin, that encloses the whole geometry even after growth ***** TOMODIFY
mw = 8*a; #Width of a cell in the linked cell algorithm for proximity detection
hs = 0.6*a; #Thickness of proximity skin
hc = 0.2*a; #Thickness of repulsive skin
kc = 10.0*K; #100.0*K Contact stiffness
dt = 0.05*np.sqrt(rho*a*a/K) #0.05*np.sqrt(rho*a*a/K) Time step = 1.11803e-05 // 0,000022361
print('dt is: ' + str(dt))
eps = 0.1 #Epsilon
k = 0.0
mpy = -0.004 #Midplane position
t = 0.0 #Current time
step = 0 #Current timestep
zoom = 1.0 #Zoom variable for visualization

csn = np.zeros(nn, dtype = int)  #Nearest surface nodes for all nodes
d2s = np.zeros(nn, dtype = float)  #Distances to nearest surface nodes for all nodes
N0 = np.zeros((nsn,3), dtype = float)  #Normals of surface nodes
Vt = np.zeros((nn,3), dtype = float)  #Velocities
Ft = np.zeros((nn,3), dtype = float)  #Forces
Vn0 = np.zeros(nn, dtype = float) #Nodal volumes in reference state
Vn = np.zeros(nn, dtype = float)  #Deformed nodal volumes
# Ue = 0 #Elastic energy

NNLt = [[] for _ in range(nsn)] #Triangle-proximity lists for surface nodes
Utold = np.zeros((nsn,3), dtype = float)  #Stores positions when proximity list is updated
#ub = vb = wb = 0 #Barycentric coordinates of triangles
G = [np.identity(3)]*ne  # Initial tangential growth tensor
#G = [1.0]*ne
#G = [GROWTH_RELATIVE]*ne
# End of parameters

# Normalize initial mesh coordinates, change mesh information by values normalized
Ut0, Ut = normalisation(Ut0, Ut, nn)   

'''# Initialize deformed coordinates
Ut = Ut0'''

# Finds the nearest surface nodes (csn) to nodes and distances to them (d2s)
csn, d2s = dist2surf(Ut0, tets, SN, nn, nsn, csn, d2s)

# Configuration of tetrahedra at reference state (A0)
A0 = configRefer(Ut0, tets, ne)

# Mark non-growing areas
gr = markgrowth(Ut0, nn)

# Calculate normals of each surface triangle and apply these normals to surface nodes
N0 = normalSurfaces(Ut0, faces, SNb, nf, nsn, N0)

#num_cores = mp.cpu_count()
#pool = mp.Pool(mp.cpu_count())
#H = THICKNESS_CORTEX

# Simulation loop
while t < 1.0:

	# Calculate the relative growth rate
	at = growthRate(GROWTH_RELATIVE, t)

	# Calculate the longitudinal length of the real brain
	L = longitLength(t)

	# Calculate the thickness of growing layer
	H = cortexThickness(THICKNESS_CORTEX, t)

	# Calculate undeformed nodal volume (Vn0) and deformed nodal volume (Vn)
	Vn0, Vn = volumeNodal(G, A0, Vn0, Vn, tets, Ut, ne, nn)
	
	# Initialize elastic energy
	Ue = 0.0
	
	for i in prange(ne):  #......range or prange?
		
		# Calculate normals of each deformed tetrahedron
		#Nt = [pool.apply(tetraNormals, args=(N0, csn, tets, i)) for i in range(ne)]
		#Nt = Parallel(n_jobs=num_cores)(delayed(tetraNormals)(N0, csn, tets, i) for i in range(ne))
		#Nt = tetraNormals(N0, csn, tets, i)

		# Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
		gm, mu = shearModulus(d2s, H, tets, i, muw, mug, gr)

		# Calculate relative tangential growth factor G
		#G[i] = growthTensor_tangen(Nt, gm, at, G, i)
		#G[i] = growthTensor_homo(G, i, GROWTH_RELATIVE, t)  # Calculate homogeneous growth factor G
		#G[i] = growthTensor_relahomo(gm, G, i, GROWTH_RELATIVE, t)  # Calculate relative homogeneous growth factor G

		# Deformed configuration of tetrahedra (At)
		At = configDeform(Ut, tets, i)

		# Calculate elastic forces
		Ft, Ue = tetraElasticity(At, A0[i], Ft, G[i], K, k, mu, tets, Vn, Vn0, i, eps, Ue)

		# Calculate normals of each deformed tetrahedron 
		Nt = tetraNormals(N0, csn, tets, i)

		# Calculate relative tangential growth factor G
		G[i] = growthTensor_tangen(Nt, gm, at, G, i)
		#G[i] = growthTensor_homo_2(G, i, GROWTH_RELATIVE)	
	
	# Calculate contact forces	
	Ft = contactProcess(Ut, Ft, SN, Utold, nsn, NNLt, faces, nf, bw, mw, hs, hc, kc, a, gr)

	# Midplane
	Ft = midPlane(Ut, Ut0, Ft, SN, nsn, mpy, a, hc, K)

	# Output
	if step % di == 0:

		# Obtain zoom parameter by checking the longitudinal length of the brain model
		zoom_pos = paraZoom(Ut, SN, L, nsn)

		# Write .pov files and output mesh in .png files
		writePov2(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom, zoom_pos)
		
		# Write surface mesh output files in .txt files
		writeTXT(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom_pos)

		print ('step: ' + str(step) + ' t: ' + str(t) )
	
	# Newton dynamics
	Ft, Ut, Vt = move(nn, Ft, Vt, Ut, gamma, Vn0, rho, dt)

	t += dt
	step += 1

