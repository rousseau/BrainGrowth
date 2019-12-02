# -*- coding: utf-8 -*-
"""
  python simulation.py '-i' './data/sphere5.mesh' '-o' './res/sphere5' '-t' 0.042 '-g' 1.829

"""

from __future__ import division
import argparse
import numpy as np
import math
from geometry import importMesh, vertex, tetraVerticesIndices, triangleIndices, numberSurfaceNodes, edge_length, volume_mesh, markgrowth, configRefer, configDeform, normalSurfaces, tetraNormals, volumeNodal, midPlane, longitLength, paraZoom, tetra_labels_surface, tetra_labels_volume, Curve_fitting
from growth import dist2surf, growthRate, cortexThickness, shearModulus, growthTensor_tangen, growthTensor_homo, growthTensor_homo_2, growthTensor_relahomo, growthRate_2, dist2surf_2
from normalisation import normalise_coord
from collision import contactProcess
from mechanics import tetraElasticity, move
from output import area_volume, writePov, writePov2, writeTXT, mesh_to_stl, point3d_to_voxel, mesh_to_image, stl_to_image
from mathfunc import make_2D_array
from numba import jit, prange

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Dynamic simulations')
  parser.add_argument('-i', '--input', help='Input maillage', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output maillage', type=str, required=True)
  parser.add_argument('-t', '--thickness', help='Cortical thickness', type=float, required=True)
  parser.add_argument('-g', '--growth', help='Relative growth rate', type=float, required=True)
  args = parser.parse_args()

  # Parameters to change
  PATH_DIR = args.output # Path of results
  THICKNESS_CORTEX = args.thickness
  GROWTH_RELATIVE = args.growth

  # Path of mesh
  mesh_path = args.input #"/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/week23-3M-tets.mesh"  # "./data/prm001_25W_Rwhite.mesh" #"/home/x17wang/Bureau/xiaoyu/ Brain_code_and_meshes/week23-3M-tets.mesh" #"/home/x17wang/Codes/BrainGrowth/brain_2.mesh"

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
 
  # Define the label for each surface node
  mesh_file = '/home/x17wang/Exp/Simulations/FSLike_Database/cut_close_matlab_B0/surf/rh.gii'
  n_clusters = 10
  method = 'spectral' #Method of parcellation in lobes
  labels_surface, labels = tetra_labels_surface(mesh_file, method, n_clusters, Ut0, SN, tets)

  # Normalize initial mesh coordinates, change mesh information by values normalized
  Ut0, Ut, cog, maxd, miny = normalise_coord(Ut0, Ut, nn)

  '''# Initialize deformed coordinates
  Ut = Ut0'''

  # Find the nearest surface nodes (csn) to nodes and distances to them (d2s)
  csn, d2s = dist2surf(Ut0, SN)

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

  #myfile = open("/home/x17wang/Codes/BrainGrowth/test.txt", "w")

  ## Parcel brain in lobes
  # Define the label for each tetrahedron
  labels_volume = tetra_labels_volume(Ut0, SN, tets, labels_surface)

  # Curve-fit of temporal growth for each label
  texture_file = '/home/x17wang/Data/GarciaPNAS2018_K65Z/covariateinteraction2.R.noivh.ggdot.func.gii'
  peak, amplitude, latency, multiple = Curve_fitting(texture_file, labels, n_clusters)

  """mesh_file = '/home/x17wang/Exp/Simulations/FSLike_Database/cut_close_matlab_B0/surf/rh.gii'
  mesh = sio.load_mesh(mesh_file)
  #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Ut_barycenter)
  n_clusters = 10
  kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mesh.vertices) 
  tree = spatial.KDTree(mesh.vertices)
  pp = tree.query(Ut0[SN[:]])
  labels_surface = np.zeros(nsn, dtype = np.int64)
  labels_surface = kmeans.labels_[pp[1]]
  labels_volume = np.zeros(ne, dtype = np.int64)
  Ut_barycenter = (Ut0[tets[:,0]] + Ut0[tets[:,1]] + Ut0[tets[:,2]] + Ut0[tets[:,3]])/4.0
  csn_t = np.zeros(ne, dtype = np.int64)
  csn_t = dist2surf_2(Ut_barycenter, Ut0, SN, ne, csn_t)   # Finds the nearest surface nodes to barycenter of tetahedra (csn_t)
  labels_volume = labels_surface[csn_t[:]]
  #labels = kmeans.labels_
  ages=[29, 29, 28, 28.5, 31.5, 32, 31, 32, 30.5, 32, 32, 31, 35.5, 35, 34.5, 35, 34.5, 35, 36, 34.5, 37.5, 35, 34.5, 36, 34.5, 33, 33]
  xdata=np.array(ages)
  tp_model = 6.926*10**(-5)*xdata**3-0.00665*xdata**2+0.250*xdata-3.0189  #time of numerical model

  texture_file = '/home/x17wang/Data/GarciaPNAS2018_K65Z/covariateinteraction2.R.noivh.ggdot.func.gii'
  texture = sio.load_texture(texture_file)

  def func2(x,a,c,sigma):
    return a*np.exp(-(70*x-c)**2/sigma)

  peak=np.zeros((n_clusters,))
  amplitude=np.zeros((n_clusters,))
  latency=np.zeros((n_clusters,))
  for k in range(n_clusters):
    ydata=np.mean(texture.darray[:,np.where(kmeans.labels_ == k)[0]], axis=1)
    popt, pcov = curve_fit(func2, tp_model, ydata, p0=[0.16, 32, 25.])
    peak[k]=popt[1]
    amplitude[k]=popt[0]
    latency[k]=popt[2]"""
  
  #amplitude = np.array([0.14611041,0.15753562,0.155424,0.15821409,0.12946925,0.13027164,0.1469077,0.11374458,0.12887732,0.14332491])
  #peak = np.array([30.8342257,31.35846557,29.7571079,32.24612982,29.56189258,30.49069391,30.13033302,31.2737798,31.7842408,31.96410735])
  #latency = np.array([496.0156081,398.15347548,483.78391847,440.61841579,623.4283619,483.67224817,443.79565242,667.65848266,472.73996025,421.61969477])
  # Tetrahedral indices of lower and upper parts of the objet
  #indices_b = np.where((Ut0[tets[:,0],2]+Ut0[tets[:,1],2]+Ut0[tets[:,2],2]+Ut0[tets[:,3],2])/4 <= -0.1)[0]  #lower part
  #indices_a = np.where((Ut0[tets[:,0],2]+Ut0[tets[:,1],2]+Ut0[tets[:,2],2]+Ut0[tets[:,3],2])/4 >= 0.1)[0]  #upper part

  #filename_nii_reso = "/home/x17wang/Exp/London/London-23weeks/brain_crisp_2_refilled.nii.gz"
  #reso = 0.5

  # Simulation loop
  while t < 1.0:

    # Calculate the relative growth rate
    #at = growthRate(GROWTH_RELATIVE, t, ne, Ut0, tets)
    at = growthRate_2(t, ne, n_clusters, labels_volume, peak, amplitude, latency, multiple)

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
    Ft = midPlane(Ut, Ut0, Ft, SN, nsn, mpy, a, hc, K)

    # Output
    if step % di == 0:

      # Obtain zoom parameter by checking the longitudinal length of the brain model
      zoom_pos = paraZoom(Ut, SN, L, nsn)

      # Write .pov files and output mesh in .png files
      writePov(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom, zoom_pos)

      # Write surface mesh output files in .txt files
      writeTXT(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom_pos)

	  # Convert surface mesh structure (from simulations) to .stl format file
      mesh_to_stl(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, SN, zoom_pos, cog, maxd, nsn, faces, SNb, miny)

      # Convert mesh .stl to image .nii.gz
      #stl_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso)

      # Convert 3d points to image voxel
      #point3d_to_voxel(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, Ut, zoom_pos, maxd, cog, nn, miny)

      # Convert volumetric mesh structure (from simulations) to image .nii.gz of a specific resolution
      #mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso, Ut, zoom_pos, cog, maxd, nn, faces, tets, miny)

      print ('step: ' + str(step) + ' t: ' + str(t) )

      # Calculate surface area and mesh volume
      Area, Volume = area_volume(Ut, faces, gr, Vn)

      print ('Normalized area: ' + str(Area) + ' Normalized volume: ' + str(Volume) )

    # Newton dynamics
    Ft, Ut, Vt = move(nn, Ft, Vt, Ut, gamma, Vn0, rho, dt)

    t += dt
    step += 1

  #myfile.close()
