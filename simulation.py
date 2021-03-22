# -*- coding: utf-8 -*-
"""
  python simulation.py -i './data/Tallinen_22W_demi_anatomist.mesh' -o './res/Tallinen_22W_demi_anatomist' -hc 'half' -t 0.042 -g 1.829 -gm 'regional' -mr './data/rh.gii' -tr './data/covariateinteraction2.R.noivh.GGnorm.func.gii' -lr './data/ATLAS30.R.Fiducial.surf.fineregions.gii' -sc 0.01 -ms 0.01
  python simulation.py -i './data/week23-3M-tets.mesh' -o './res/week23-3M-tets_atlas_Garcia' -hc 'whole' -t 0.042 -g 1.829 -gm 'regional' -mr './data/rh.gii' -ml './data/lh.gii' -tr './data/covariateinteraction2.R.noivh.GGnorm.func.gii' -tl './data/covariateinteraction2.L.noivh.GGnorm.func.gii' -lr './data/ATLAS30.R.Fiducial.surf.fineregions.gii' -ll './data/ATLAS30.L.Fiducial.surf.fineregions.gii' -sc 0.01 -ms 0.01
  python simulation.py -i './data/sphere5.mesh' -o './res/sphere5' -hc 'whole' -t 0.042 -g 1.829 -gm 'global'
  
"""

#global modules
from __future__ import division
import argparse
import numpy as np
import math
import numba as nb
from numba import jit, prange 
import time
import os

#global modules for tracking
import cProfile
import pstats
import io
import sys

#local modules
from geometry import importMesh, vertex, tetraVerticesIndices, triangleIndices, numberSurfaceNodes, edge_length, volume_mesh, markgrowth, configRefer, configDeform, normalSurfaces, tetraNormals, volumeNodal, midPlane, longitLength, paraZoom, tetra_labels_surface_half, tetra_labels_volume_half, Curve_fitting_half, tetra_labels_surface_whole, tetra_labels_volume_whole, Curve_fitting_whole
from growth import dist2surf, growthRate, cortexThickness, shearModulus, growthTensor_tangen, growthTensor_homo, growthTensor_homo_2, growthTensor_relahomo, growthRate_2_half, growthRate_2_whole
from normalisation import normalise_coord
from collision_Tallinen import contactProcess
from mechanics import tetraElasticity, move
from output import area_volume, writePov, writePov2, writeTXT, mesh_to_stl, point3d_to_voxel, mesh_to_image, stl_to_image, writeTex, mesh_to_gifti
from mathfunc import make_2D_array
import slam.io as sio #Slam modification from marseille version including IO

if __name__ == '__main__':
  start_time_initialization = time.time ()
  parser = argparse.ArgumentParser(description='Dynamic simulations')
  parser.add_argument('-i', '--input', help='Input maillage', type=str, default='./data/sphere5.mesh', required=False)
  parser.add_argument('-o', '--output', help='Output maillage', type=str, default='./res/sphere5', required=False)
  parser.add_argument('-hc', '--halforwholebrain', help='Half or whole brain', type=str, default='whole', required=False)
  parser.add_argument('-t', '--thickness', help='Cortical thickness', type=float, default=0.042, required=False)
  parser.add_argument('-g', '--growth', help='Relative growth rate', type=float, default=1.829, required=False)
  parser.add_argument('-gm', '--growthmethod', help='Global or regional growth', type=str, default='Global', required=False) 
  parser.add_argument('-mr', '--registermeshright', help='Mesh of right brain after registration', type=str, required=False)
  parser.add_argument('-ml', '--registermeshleft', help='Mesh of left brain after registration', type=str, required=False)
  parser.add_argument('-tr', '--textureright', help='Texture of template of right brain', type=str, required=False)
  parser.add_argument('-tl', '--textureleft', help='Texture of template of left brain', type=str, required=False)
  parser.add_argument('-lr', '--lobesright', help='User-defined lobes of right brain', type=str, required=False)
  parser.add_argument('-ll', '--lobesleft', help='User-defined lobes of left brain', type=str, required=False)
  parser.add_argument('-sc', '--stepcontrol', help='Step length regulation', type=float, default=0.1, required=False) #increase for speed, 0.01 is default
  parser.add_argument('-ms', '--meshspacing', help='Average spacing in the mesh', type=float, default=0.01, required=False) #default is 0.01
  parser.add_argument('-md', '--massdensity', help='Mass density of brain mesh', type=float, default=0.01, required=False) #increase for speed, too high brings negativ jakobians, default is 0.01
  args = parser.parse_args()
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
  coordinates0, coordinates, n_nodes = vertex(mesh)

  # Read element indices (tets: index of four vertices of tetrahedra) and get number of elements (ne)
  tets, n_tets = tetraVerticesIndices(mesh, n_nodes)

  # Read surface triangle indices (faces: index of three vertices of triangles) and get number of surface triangles (nf)
  faces, n_faces = triangleIndices(mesh, n_nodes, n_tets)

  # Determine surface nodes and index maps  n_surface_nodes: number of nodes at the surface, SN: Nodal index map from surface to full mesh, nodal_idx_b: Nodal index map from full mesh to surface)
  n_surface_nodes, nodal_idx, nodal_idx_b = numberSurfaceNodes(faces, n_nodes, n_faces)

  # Check minimum, maximum and average edge lengths (average mesh spacing) at the surface
  mine, maxe, ave = edge_length(coordinates, faces, n_faces)
  print ('minimum edge lengths: ' + str(mine) + ' maximum edge lengths: ' + str(maxe) + ' average value of edge length: ' + str(ave))

  # Calculate the total volume of a tetrahedral mesh
  Vn_init = np.zeros(n_nodes, dtype = np.float64)
  Vm = volume_mesh(Vn_init, n_nodes, n_tets, tets, coordinates)
  print ('Volume of mesh is ' + str(-Vm))

  # Calculate the total surface area of a tetrahedral mesh
  Area = 0.0
  for i in range(len(faces)):
    Ntmp = np.cross(coordinates0[faces[i,1]] - coordinates0[faces[i,0]], coordinates0[faces[i,2]] - coordinates0[faces[i,0]])
    Area += 0.5*np.linalg.norm(Ntmp)
  print ('Area of mesh is ' + str(Area))

  # Parameters
  H = THICKNESS_CORTEX  #Cortical plate thickness
  mug = 1.0 #65.0 Shear modulus of gray matter
  muw = 1.167 #75.86 Shear modulus of white matter
  K = 5.0 #100.0 Bulk modulus
  mesh_spacing = args.meshspacing #0.003 0.01 Mesh spacing - set manually based on the average spacing in the mesh
  rho = args.massdensity #0.0001 Mass density - adjust to run the simulation faster or slower
  gamma = 0.5 #0.1 Damping coefficent
  di = 500 #Output data once every di steps

  bw = 3.2 #Width of a bounding box, centered at origin, that encloses the whole geometry even after growth ***** TOMODIFY
  mw = 8*mesh_spacing #Width of a cell in the linked cell algorithm for proximity detection
  hs = 0.6*mesh_spacing #Thickness of proximity skin
  hc = 0.2*mesh_spacing #Thickness of repulsive skin
  kc = 10.0*K #100.0*K Contact stiffness
  dt = args.stepcontrol*np.sqrt(rho*mesh_spacing*mesh_spacing/K) #0.05*np.sqrt(rho*a*a/K) Time step = 1.11803e-05 // 0,000022361
  print('dt is ' + str(dt))
  eps = 0.1 #Epsilon
  k = 0.0
  mpy = -0.004 #Midplane position
  t = 0.0 #Current time
  step = 0 #Current time step
  zoom = 1.0 #Zoom variable for visualization

  csn = np.zeros(n_nodes, dtype = np.int64)  #Nearest surface nodes for all nodes
  d2s = np.zeros(n_nodes, dtype = np.float64)  #Distances to nearest surface nodes for all nodes
  N0 = np.zeros((n_surface_nodes,3), dtype = np.float64)  #Normals of surface nodes
  Vt = np.zeros((n_nodes,3), dtype = np.float64)  #Velocities
  Ft = np.zeros((n_nodes,3), dtype = np.float64)  #Forces
  #Vn0 = np.zeros(nn, dtype = float) #Nodal volumes in reference state
  #Vn = np.zeros(nn, dtype = float)  #Deformed nodal volumes
  # Ue = 0 #Elastic energy

  NNLt = [[] for _ in range (n_surface_nodes)] #Triangle-proximity lists for surface nodes
  Utold = np.zeros( (n_surface_nodes,3), dtype = np.float64)  #Stores positions when proximity list is updated
  #ub = vb = wb = 0 #Barycentric coordinates of triangles
  #G = np.array([np.identity(3)]*ne)  
  shape = (n_tets,3,3)
  G = np.zeros(shape, dtype = np.float64)  # Initial tangential growth tensor
  G[:,np.arange(3),np.arange(3)] = 1.0
  #G = [1.0]*ne
  #G = 1.0
  # End of parameters
 
  ## Parcel brain in lobes

  if args.growthmethod.__eq__("regional"):
    n_clusters = 10   #Number of lobes
    method = 'User-defined lobes' #Method of parcellation in lobes
    # Half brain
    if args.halforwholebrain.__eq__("half"):
      # Define the label for each surface node
      mesh_file = args.registermeshright
      lobes_file = args.lobesright
      lobes = sio.load_texture(lobes_file)
      lobes = np.round(lobes.darray[0])
      labels_surface, labels = tetra_labels_surface_half(mesh_file, method, n_clusters, coordinates0, nodal_idx, tets, lobes)

      # Define the label for each tetrahedron
      labels_volume = tetra_labels_volume_half(coordinates0, nodal_idx, tets, labels_surface)

      # Curve-fit of temporal growth for each label
      texture_file = args.textureright
      """texture_file_27 = '/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA28to30/noninjured_ab.L.configincaltrelaxaverage.GGnorm.func.gii'
      texture_file_31 ='/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA30to34/noninjured_bc.L.configincaltrelaxaverage.GGnorm.func.gii'
      texture_file_33 = '/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA34to38/noninjured_cd.L.configincaltrelaxaverage.GGnorm.func.gii'
      texture_file_37 ='/home/x17wang/Data/GarciaPNAS2018_K65Z/PMA30to38/noninjured_bd.L.configincaltrelaxaverage.GGnorm.func.gii'"""
      peak, amplitude, latency = Curve_fitting_half(texture_file, labels, n_clusters, lobes)

    # Whole brain
    else:
      # Define the label for each surface node
      mesh_file = args.registermeshright
      mesh_file_2 = args.registermeshleft
      lobes_file = args.lobesright
      lobes_file_2 = args.lobesleft
      lobes = sio.load_texture(lobes_file)
      lobes = np.round(lobes.darray[0])
      lobes_2 = sio.load_texture(lobes_file_2)
      lobes_2 = np.round(lobes_2.darray[0])
      indices_a = np.where(coordinates0[nodal_idx[:],1] >= (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #right part surface node indices
      indices_b = np.where(coordinates0[nodal_idx[:],1] < (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #left part surface node indices
      indices_c = np.where((coordinates0[tets[:,0],1]+coordinates0[tets[:,1],1]+coordinates0[tets[:,2],1]+coordinates0[tets[:,3],1])/4 >= (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #right part tetrahedral indices
      indices_d = np.where((coordinates0[tets[:,0],1]+coordinates0[tets[:,1],1]+coordinates0[tets[:,2],1]+coordinates0[tets[:,3],1])/4 < (max(coordinates0[:,1]) + min(coordinates0[:,1]))/2.0)[0]  #left part tetrahedral indices
      labels_surface, labels_surface_2, labels, labels_2 = tetra_labels_surface_whole(mesh_file, mesh_file_2, method, n_clusters, coordinates0, nodal_idx, tets, indices_a, indices_b, lobes, lobes_2)

      # Define the label for each tetrahedron
      labels_volume, labels_volume_2 = tetra_labels_volume_whole(coordinates0, nodal_idx, tets, indices_a, indices_b, indices_c, indices_d, labels_surface, labels_surface_2)

      # Curve-fit of temporal growth for each label
      texture_file = args.textureright
      texture_file_2 = args.textureleft
      peak, amplitude, latency, peak_2, amplitude_2, latency_2 = Curve_fitting_whole(texture_file, texture_file_2, labels, labels_2, n_clusters, lobes, lobes_2)

  # Normalize initial mesh coordinates, change mesh information by values normalized
  coordinates0, coordinates, cog, maxd, miny = normalise_coord(coordinates0, coordinates, n_nodes, args.halforwholebrain)

  '''# Initialize deformed coordinates
  Ut = Ut0'''

  # Find the nearest surface nodes (csn) to nodes and distances to them (d2s)
  csn, d2s = dist2surf(coordinates0, nodal_idx)

  # Configuration of tetrahedra at reference state (A0)
  A0 = configRefer(coordinates0, tets, n_tets)

  # Mark non-growing areas
  gr = markgrowth(coordinates0, n_nodes)

  # Calculate normals of each surface triangle and apply these normals to surface nodes
  N0 = normalSurfaces(coordinates0, faces, nodal_idx_b, n_faces, n_surface_nodes, N0)

  #num_cores = mp.cpu_count()
  #pool = mp.Pool(mp.cpu_count())
  #H = THICKNESS_CORTEX

  # Elastic process
  @jit(nopython=True)
  def elasticProccess(d2s, H, tets, muw, mug, coordinates, A0, Ft, K, k, Vn, Vn0, eps, N0, csn, at, G, n_tets):

    # Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
    gm, mu = shearModulus(d2s, H, tets, n_tets, muw, mug)

    # Deformed configuration of tetrahedra (At)
    At = configDeform(coordinates, tets, n_tets)

    # Calculate elastic forces
    Ft = tetraElasticity(At, A0, Ft, G, K, k, mu, tets, Vn, Vn0, n_tets, eps)

    # Calculate normals of each deformed tetrahedron 
    Nt = tetraNormals(N0, csn, tets, n_tets)

    # Calculate relative tangential growth factor G
    G = growthTensor_tangen(Nt, gm, at, G, n_tets)
    #G[i] = growthTensor_homo_2(G, i, GROWTH_RELATIVE)

    return Ft

  #myfile = open("/home/x17wang/Codes/BrainGrowth/test.txt", "w")

  #filename_nii_reso = "/home/x17wang/Exp/London/London-23weeks/brain_crisp_2_refilled.nii.gz"
  #reso = 0.5

  end_time_initialization = time.time () - start_time_initialization
  print ('Time required for initialization : ' + str (end_time_initialization) )

  # Simulation loop
  start_time_simulation = time.time ()
  while t < 1.0:

    # Calculate the relative growth rate
    if args.growthmethod.__eq__("regional"):
      if args.halforwholebrain.__eq__("half"):
        at, bt = growthRate_2_half(t, n_tets, n_surface_nodes, n_clusters, labels_surface, labels_volume, peak, amplitude, latency, lobes)
      else:
        at, bt = growthRate_2_whole(t, n_tets, n_surface_nodes, n_clusters, labels_surface, labels_surface_2, labels_volume, labels_volume_2, peak, amplitude, latency, peak_2, amplitude_2, latency_2, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d)
    else:
      at = growthRate(GROWTH_RELATIVE, t, n_tets)
      
    # Calculate the longitudinal length of the real brain
    L = longitLength(t)

    # Calculate the thickness of growing layer
    H = cortexThickness(THICKNESS_CORTEX, t)

    # Calculate undeformed nodal volume (Vn0) and deformed nodal volume (Vn)
    Vn0, Vn = volumeNodal(G, A0, tets, coordinates, n_tets, n_nodes)

    # Initialize elastic energy
    #Ue = 0.0

    # Calculate elastic forces
    #Ft = elasticProccess(d2s, H, tets, muw, mug, Ut, A0, Ft, K, k, Vn, Vn0, eps, N0, csn, at, G, ne)

    # Calculate contact forces
    Ft, NNLt = contactProcess(coordinates, Ft, nodal_idx, Utold, n_surface_nodes, NNLt, faces, n_faces, bw, mw, hs, hc, kc, mesh_spacing, gr)
    #myfile.write("%s\n" % NNLt)
    # Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
    gm, mu = shearModulus(d2s, H, tets, n_tets, muw, mug, gr)

    # Deformed configuration of tetrahedra (At)
    At = configDeform(coordinates, tets, n_tets)

    # Calculate elastic forces
    Ft = tetraElasticity(At, A0, Ft, G, K, k, mu, tets, Vn, Vn0, n_tets, eps)

    # Calculate normals of each deformed tetrahedron 
    Nt = tetraNormals(N0, csn, tets, n_tets)

    # Calculate relative tangential growth factor G
    G = growthTensor_tangen(Nt, gm, at, G, n_tets)
    #G[i] = growthTensor_homo_2(G, i, GROWTH_RELATIVE)
    #G = 1.0 + GROWTH_RELATIVE*t

    # Midplane
    Ft = midPlane(coordinates, coordinates0, Ft, nodal_idx, n_surface_nodes, mpy, mesh_spacing, hc, K)

    # Output
    if step % di == 0:

      # Write texture of growth in .gii files
      #writeTex(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, bt)

      # Obtain zoom parameter by checking the longitudinal length of the brain model
      zoom_pos = paraZoom(coordinates, nodal_idx, L, n_surface_nodes)

      # Write .pov files and output mesh in .png files
      writePov(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom, zoom_pos)

      # Write surface mesh output files in .txt files
      writeTXT(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom_pos, cog, maxd, miny, args.halforwholebrain)

      # Convert surface mesh structure (from simulations) to .stl format file
      mesh_to_stl(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, zoom_pos, cog, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)
      
      # Convert surface mesh structure (from simulations) to .gii format file
      mesh_to_gifti(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, zoom_pos, cog, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)

      # Convert mesh .stl to image .nii.gz
      #stl_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso)

      # Convert 3d points to image voxel
      #point3d_to_voxel(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, Ut, zoom_pos, maxd, cog, nn, miny)

      # Convert volumetric mesh structure (from simulations) to image .nii.gz of a specific resolution
      #mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso, Ut, zoom_pos, cog, maxd, nn, faces, tets, miny)

      print ('step: ' + str(step) + ' t: ' + str(t) )

      # Calculate surface area and mesh volume
      Area, Volume = area_volume(coordinates, faces, gr, Vn)

      print ('Normalized area: ' + str(Area) + ' Normalized volume: ' + str(Volume) )

      #timestamp for simulation loop
      end_time_simulation = time.time () - start_time_simulation
      print ('Time required for simulation loop : ' + str (end_time_simulation) )
      start_time_simulation = time.time ()

    # Newton dynamics
    Ft, coordinates, Vt = move(n_nodes, Ft, Vt, coordinates, gamma, Vn0, rho, dt)

    t += dt
    step += 1

  #myfile.close()
