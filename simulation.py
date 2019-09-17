# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import math
from geometry import importMesh, vertex, tetraVerticesIndices, triangleIndices, numberSurfaceNodes, edge_length, volume_mesh, markgrowth, configRefer, configDeform, normalSurfaces, tetraNormals, volumeNodal, midPlane, longitLength, paraZoom
from growth import dist2surf, growthRate, cortexThickness, shearModulus, growthTensor_tangen, growthTensor_homo, growthTensor_homo_2, growthTensor_relahomo
from normalisation import normalise_coord
from collision_Tallinen import contactProcess
from mechanics import tetraElasticity, move
from output import area_volume, writePov, writePov2, writeTXT, mesh_to_stl
from mathfunc import make_2D_array
from numba import jit, prange

# Parameters to change
PATH_DIR = "./res/Tallinen_22W_demi_2" # Path of results
THICKNESS_CORTEX = 0.043
GROWTH_RELATIVE = 1.829

# Path of mesh
mesh_path = "./data/Tallinen_22W_demi_2.mesh"  #"./data/Tallinen_22W_demi_2.mesh"  #"/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/week23-3M-tets.mesh" #"/home/x17wang/Codes/BrainGrowth/brain_2.mesh"

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

# Check minimum, maximum and average edge lengths (average mesh spacing) at the surface
mine, maxe, ave = edge_length(Ut, faces, nf)
print ('minimum edge lengths: ' + str(mine) + ' maximum edge lengths: ' + str(maxe) + ' average value of edge length: ' + str(ave))

# Calculate the total volume of a tetrahedral mesh
Vn_init = np.zeros(nn, dtype = np.float64)
Vm = volume_mesh(Vn_init, nn, ne, tets, Ut)
print ('Volume of mesh is ' + str(-Vm))

# Calculate the total surface area of a tetrahedral mesh
Area = 0.0
for i in range(len(faces)):
  Ntmp = np.cross(Ut0[faces[i,1]] - Ut0[faces[i,0]], Ut0[faces[i,2]] - Ut0[faces[i,0]])
  Area += 0.5*np.linalg.norm(Ntmp)
print ('Area of mesh is ' + str(Area))

# Parameters
H = THICKNESS_CORTEX  #Cortical plate thickness
mug = 1.0 #65.0 Shear modulus of gray matter
muw = 1.167 #75.86 Shear modulus of white matter
K = 5.0 #100.0 Bulk modulus
a = 0.01 #0.003 0.01 Mesh spacing - set manually based on the average spacing in the mesh
rho = 0.01 #0.0001 Mass density - adjust to run the simulation faster or slower
gamma = 0.5 #0.1 Damping coefficent
di = 500 #Output data once every di steps

bw = 3.2 #Width of a bounding box, centered at origin, that encloses the whole geometry even after growth ***** TOMODIFY
mw = 8*a #Width of a cell in the linked cell algorithm for proximity detection
hs = 0.6*a #Thickness of proximity skin
hc = 0.2*a #Thickness of repulsive skin
kc = 10.0*K #100.0*K Contact stiffness
dt = 0.01*np.sqrt(rho*a*a/K) #0.05*np.sqrt(rho*a*a/K) Time step = 1.11803e-05 // 0,000022361
print('dt is ' + str(dt))
eps = 0.1 #Epsilon
k = 0.0
mpy = -0.004 #Midplane position
t = 0.0 #Current time
step = 0 #Current time step
zoom = 1.0 #Zoom variable for visualization

csn = np.zeros(nn, dtype = np.int64)  #Nearest surface nodes for all nodes
d2s = np.zeros(nn, dtype = np.float64)  #Distances to nearest surface nodes for all nodes
N0 = np.zeros((nsn,3), dtype = np.float64)  #Normals of surface nodes
Vt = np.zeros((nn,3), dtype = np.float64)  #Velocities
Ft = np.zeros((nn,3), dtype = np.float64)  #Forces
#Vn0 = np.zeros(nn, dtype = float) #Nodal volumes in reference state
#Vn = np.zeros(nn, dtype = float)  #Deformed nodal volumes
# Ue = 0 #Elastic energy

NNLt = [[] for _ in range(nsn)] #Triangle-proximity lists for surface nodes
Utold = np.zeros((nsn,3), dtype = np.float64)  #Stores positions when proximity list is updated
#ub = vb = wb = 0 #Barycentric coordinates of triangles
#G = np.array([np.identity(3)]*ne)  
shape = (ne,3,3)
G = np.zeros(shape, dtype = np.float64)  # Initial tangential growth tensor
G[:,np.arange(3),np.arange(3)] = 1.0
#G = [1.0]*ne
#G = 1.0
# End of parameters

# Normalize initial mesh coordinates, change mesh information by values normalized
Ut0, Ut, cog, maxd = normalise_coord(Ut0, Ut, nn)

'''# Initialize deformed coordinates
Ut = Ut0'''

# Finds the nearest surface nodes (csn) to nodes and distances to them (d2s)
csn, d2s = dist2surf(Ut0, SN, nn, csn, d2s)

# Configuration of tetrahedra at reference state (A0)
A0 = configRefer(Ut0, tets, ne)

# Mark non-growing areas
gr = markgrowth(Ut0, nn)

# Calculate normals of each surface triangle and apply these normals to surface nodes
N0 = normalSurfaces(Ut0, faces, SNb, nf, nsn, N0)

#num_cores = mp.cpu_count()
#pool = mp.Pool(mp.cpu_count())
#H = THICKNESS_CORTEX

# Elastic process
@jit(nopython=True)
def elasticProccess(d2s, H, tets, muw, mug, Ut, A0, Ft, K, k, Vn, Vn0, eps, N0, csn, at, G, ne):

  # Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
  gm, mu = shearModulus(d2s, H, tets, ne, muw, mug)

  # Deformed configuration of tetrahedra (At)
  At = configDeform(Ut, tets, ne)

  # Calculate elastic forces
  Ft = tetraElasticity(At, A0, Ft, G, K, k, mu, tets, Vn, Vn0, ne, eps)

  # Calculate normals of each deformed tetrahedron 
  Nt = tetraNormals(N0, csn, tets, ne)

  # Calculate relative tangential growth factor G
  G = growthTensor_tangen(Nt, gm, at, G, ne)
  #G[i] = growthTensor_homo_2(G, i, GROWTH_RELATIVE)

  return Ft

#myfile = open("/home/x17wang/Codes/BrainGrowth/Force_contact_NNLt.txt", "w")

# Simulation loop
while t < 1.0:

  # Calculate the relative growth rate
  at = growthRate(GROWTH_RELATIVE, t)

  # Calculate the longitudinal length of the real brain
  L = longitLength(t)

  # Calculate the thickness of growing layer
  H = cortexThickness(THICKNESS_CORTEX, t)

  # Calculate undeformed nodal volume (Vn0) and deformed nodal volume (Vn)
  Vn0, Vn = volumeNodal(G, A0, tets, Ut, ne, nn)

  # Initialize elastic energy
  #Ue = 0.0

  # Calculate elastic forces
  #Ft = elasticProccess(d2s, H, tets, muw, mug, Ut, A0, Ft, K, k, Vn, Vn0, eps, N0, csn, at, G, ne)

  # Calculate contact forces
  Ft, NNLt = contactProcess(Ut, Ft, SN, Utold, nsn, NNLt, faces, nf, bw, mw, hs, hc, kc, a, gr)
  #myfile.write("%s\n" % NNLt)
  # Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
  gm, mu = shearModulus(d2s, H, tets, ne, muw, mug, gr)

  # Deformed configuration of tetrahedra (At)
  At = configDeform(Ut, tets, ne)

  # Calculate elastic forces
  Ft = tetraElasticity(At, A0, Ft, G, K, k, mu, tets, Vn, Vn0, ne, eps)

  # Calculate normals of each deformed tetrahedron 
  Nt = tetraNormals(N0, csn, tets, ne)

  # Calculate relative tangential growth factor G
  G = growthTensor_tangen(Nt, gm, at, G, ne)
  #G[i] = growthTensor_homo_2(G, i, GROWTH_RELATIVE)
  #G = 1.0 + GROWTH_RELATIVE*t

  # Midplane
  #Ft = midPlane(Ut, Ut0, Ft, SN, nsn, mpy, a, hc, K)

  # Output
  if step % di == 0:

    # Obtain zoom parameter by checking the longitudinal length of the brain model
    zoom_pos = paraZoom(Ut, SN, L, nsn)

    # Write .pov files and output mesh in .png files
    writePov(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom, zoom_pos)

    # Write surface mesh output files in .txt files
    writeTXT(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom_pos)

	# Convert mesh to .stl format
    mesh_to_stl(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, SN, zoom_pos, cog, maxd, nsn, faces, SNb)

    print ('step: ' + str(step) + ' t: ' + str(t) )

    # Calculate surface area and mesh volume
    Area, Volume = area_volume(Ut, faces, gr, Vn)

    print ('Normalized area: ' + str(Area) + ' Normalized volume: ' + str(Volume) )

  # Newton dynamics
  Ft, Ut, Vt = move(nn, Ft, Vt, Ut, gamma, Vn0, rho, dt)

  t += dt
  step += 1

#myfile.close()
