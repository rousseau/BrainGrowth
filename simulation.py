# -*- coding: utf-8 -*-
"""
  python simulation.py -i './data/Tallinen_22W_demi_anatomist.mesh' -o './res/Tallinen_22W_demi_anatomist' -hc 'half' -t 0.042 -g 1.829 -gm 'regional' -mr './data/rh.gii' -tr './data/covariateinteraction2.R.noivh.GGnorm.func.gii' -lr './data/ATLAS30.R.Fiducial.surf.fineregions.gii' -sc 0.01 -ms 0.01
  python simulation.py -i './data/week23-3M-tets.mesh' -o './res/week23-3M-tets_atlas_Garcia' -hc 'whole' -t 0.042 -g 1.829 -gm 'regional' -mr './data/rh.gii' -ml './data/lh.gii' -tr './data/covariateinteraction2.R.noivh.GGnorm.func.gii' -tl './data/covariateinteraction2.L.noivh.GGnorm.func.gii' -lr './data/ATLAS30.R.Fiducial.surf.fineregions.gii' -ll './data/ATLAS30.L.Fiducial.surf.fineregions.gii' -sc 0.01 -ms 0.01
  python simulation.py -i './data/sphere5.mesh' -o './res/sphere5' -hc 'whole' -t 0.042 -g 1.829 -gm 'global'
  Env variables for debug:
  NUMBA_DEBUG_ARRAY_OPT_STATS=1
  
"""
#global modules
import argparse
import numpy as np
import time
import slam.io as sio
from scipy.spatial import cKDTree

#global modules for tracking
import cProfile
import pstats
import io
import sys

#local modules
from geometry import netgen_to_array, tetra_normals, get_nodes, get_tetra_indices, get_face_indices, get_nb_surface_nodes, edge_length, volume_mesh, mark_nogrowth, config_refer, config_deform, normals_surfaces, calc_vol_nodal, calc_mid_plane, calc_longi_length, paraZoom, tetra_labels_surface_half, tetra_labels_volume_half, Curve_fitting_half, tetra_labels_surface_whole, tetra_labels_volume_whole, curve_fitting_whole
from growth import growthRate, shear_modulus, growth_tensor_tangen, growthRate_2_half, growthRate_2_whole, calc_growth_filter
from normalisation import normalise_coord
from collision_Tallinen import contact_process
from mechanics import tetra_elasticity, move, tetra1, tetra1_np, tetra2
from output import area_volume, writePov, writeTXT, mesh_to_stl, mesh_to_gifti


if __name__ == '__main__':
  start_time_initialization = time.time ()
  parser = argparse.ArgumentParser(description='Dynamic simulations')
  parser.add_argument('-i', '--input', help='Input mesh', type=str, default='/home/latim/braingrowth_rousseau/FetalAtlas/mesh_to_image_testFolder/template_T2_reduced_z.mesh', required=False)
  parser.add_argument('-o', '--output', help='Output mesh', type=str, default='./res/', required=False)
  parser.add_argument('-vo', '--voutput', help='Raw data required by the "visualization" package', type=str, default='./visualization/', required=False)
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
  parser.add_argument('-md', '--massdensity', help='Mass density of brain mesh', type=float, default=0.1, required=False) #increase for speed, too high brings negativ jakobians, default is 0.01
  args = parser.parse_args()

<<<<<<< HEAD
  ########################
  # PARAMETERS TO CHANGE #
  ########################
  PATH_DIR = args.output # Path of results
=======
  # Parameters to change
  PATH_DIR = args.output
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2
  THICKNESS_CORTEX = args.thickness
  GROWTH_RELATIVE = args.growth

  # Import mesh, each line as a list
  mesh = netgen_to_array(args.input)

<<<<<<< HEAD
  # Read nodes, get undeformed coordinates (Ut0) and initialize deformed coordinates (Ut) of all nodes
  coordinates0, coordinates, n_nodes = get_vertices(mesh) 

  # Read element indices (tets: index of four vertices of tetrahedra) and get number of elements (ne)
  tets, n_tets = get_tetra_vertices_indices(mesh, n_nodes)
=======
  # Read nodes, get undeformed coordinates (Ut0) and initialize deformed coordinates (Ut) of all nodes. #X-Y switch at this point
  coordinates, n_nodes = get_nodes(mesh) 
  coordinates0 = coordinates.copy()

  # Read element indices (tets: index of four vertices of tetrahedra) and get number of elements (ne). #Handness switch at this point
  tets, n_tets = get_tetra_indices(mesh, n_nodes)
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2

  # Read surface triangle indices (faces: index of three vertices of triangles) and get number of surface triangles (nf)
  faces, n_faces = get_triangle_indices(mesh, n_nodes, n_tets)

  # Determine surface nodes and index maps  n_surface_nodes: number of nodes at the surface, Nodal index map (legacy SN) from surface to full mesh, nodal_idx_b: Nodal index map from full mesh to surface)
  n_surface_nodes, nodal_idx, nodal_idx_b = get_nb_surface_nodes(faces, n_nodes)

  # Check minimum, maximum and average edge lengths (average mesh spacing) at the surface
  mine, maxe, ave = edge_length(coordinates, faces, n_faces)
  print ('minimum edge lengths: ' + str(mine) + ' maximum edge lengths: ' + str(maxe) + ' average value of edge length: ' + str(ave))

  # Calculate the total volume of a tetrahedral mesh
  Vm = volume_mesh(n_nodes, n_tets, tets, coordinates)
  print ('Volume of mesh is ' + str(Vm))

  # Calculate the total surface area of a tetrahedral mesh
  Area = 0.0
  for i in range(len(faces)):
    Ntmp = np.cross(coordinates0[faces[i,1]] - coordinates0[faces[i,0]], coordinates0[faces[i,2]] - coordinates0[faces[i,0]])
    Area += 0.5*np.linalg.norm(Ntmp)
  print ('Area of mesh is ' + str(Area))

  ########################
  #    FIX PARAMETERS    #
  ########################
  cortex_thickness = THICKNESS_CORTEX  #Cortical plate thickness
  mug = 1.0 #65.0 Shear modulus of gray matter
  muw = 1.167 #75.86 Shear modulus of white matter
  bulk_modulus = 5.0 
  mesh_spacing = args.meshspacing #0.003 0.01 - set manually based on the average spacing in the mesh
  mass_density = args.massdensity #0.0001 - adjust to run the simulation faster or slower
  damping_coef = 0.5 #0.1 Damping coefficent
  di = 500 #Output data once every di steps

  bounding_box = 3.2 #Width of a bounding box, centered at origin, that encloses the whole geometry even after growth ***** TOMODIFY
  cell_width = 8 * mesh_spacing #Width of a cell in the linked cell algorithm for proximity detection
  prox_skin = 0.6 * mesh_spacing #Thickness of proximity skin
  repuls_skin = 0.2 * mesh_spacing #Thickness of repulsive skin
  contact_stiffness = 10.0 * bulk_modulus #100.0*K Contact stiffness
  dt = args.stepcontrol*np.sqrt(mass_density * mesh_spacing * mesh_spacing / bulk_modulus) #0.05*np.sqrt(rho*a*a/K) Time step = 1.11803e-05 // 0,000022361
  print('dt is ' + str(dt))
  eps = 0.1 #Epsilon
  k_param = 0.0
  midplane_pos = -0.004 #Midplane position
  t = 0.0 #Current time
  step = 0 #Current time step
  zoom = 1.0 #Zoom variable for visualization

  nearest_surf_node = np.zeros(n_nodes, dtype = np.int64)  #Nearest surface nodes for all nodes
  dist_2_surf = np.zeros(n_nodes, dtype = np.float64)  #Distances to nearest surface nodes for all nodes
  surf_node_norms = np.zeros((n_surface_nodes,3), dtype = np.float64)  #Normals of surface nodes
  Vt = np.zeros((n_nodes,3), dtype = np.float64)  #Velocities
  Ft = np.zeros((n_nodes,3), dtype = np.float64)  #Forces
<<<<<<< HEAD
=======
  growth_filter = np.ones(n_tets, dtype = np.float64)
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2
  #Vn0 = np.zeros(nn, dtype = float) #Nodal volumes in reference state
  #Vn = np.zeros(nn, dtype = float)  #Deformed nodal volumes
  # Ue = 0 #Elastic energy

  NNLt = [[] for _ in range (n_surface_nodes)] #Triangle-proximity lists for surface nodes
<<<<<<< HEAD
  coordinates_old = np.zeros( (n_surface_nodes,3), dtype = np.float64)  #Stores positions when proximity list is updated

  shape = (n_tets,3,3)
  tan_growth_tensor = np.zeros(shape, dtype = np.float64)  # Initial tangential growth tensor
  tan_growth_tensor[:,np.arange(3),np.arange(3)] = 1.0
 
  ##############################################################################
  #    PARCEL  BRAIN IN LOBES : MESH GENERATION & PARAMETERS INITIALIZATION    #
  ##############################################################################
=======
  coordinates_old = np.zeros((n_surface_nodes,3), dtype = np.float64)  #Stores positions when proximity list is updated

  tan_growth_tensor = np.ones((n_tets,3,3), dtype = np.float64) * np.identity(3)  # Initial tangential growth tensor

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

  ## Parcel brain in lobes
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2
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
  coordinates0, coordinates, center_of_gravity, maxd, miny = normalise_coord(coordinates0, coordinates, n_nodes, args.halforwholebrain)

  # Find the nearest surface nodes (nearest_surf_node) to nodes and distances to them (dist_2_surf)
  tree = cKDTree(coordinates0[nodal_idx])
  dist_2_surf, nearest_surf_node = tree.query(coordinates0)

  # Configuration of tetrahedra at reference state (ref_state_tets)
  ref_state_tets = config_refer(coordinates0, tets, n_tets)

  # Mark non-growing areas
  gr = mark_nogrowth(coordinates0, n_nodes)

  # Calculate normals of each surface triangle at each node
  surf_node_norms = normals_surfaces(coordinates0, faces, nodal_idx_b, n_faces, n_surface_nodes, surf_node_norms)

  # list to stack all intensities of node resultant deformation at step%di
  node_def_sumVec_stack = np.zeros((n_nodes, 1, 3), dtype=np.float64)

  #optional gaussian filtering on at
  # centroids = calc_tets_center (coordinates, tets)

  # gauss = gaussian_filter (centroids)

  end_time_initialization = time.time () - start_time_initialization
  print ('Time required for initialization : ' + str (end_time_initialization) )

<<<<<<< HEAD
  # Export the initial parameters in npy file
  initial_parameters = np.array([n_nodes, maxd, center_of_gravity], dtype = object)
  np.save('/home/latim/anaconda3/envs/braingrowth/visualization/initial_parameters/parameters.npy', initial_parameters)
  
  ########################
  #    SIMULATION LOOP   #
  ########################
  start_time_simulation = time.time ()
  while t < 1.0:
    
    # Calculate the relative growth rate
=======
  ###### data export required by the "visualization" package - for coordinates denormalization and displacements calculation ######
  # Export the initial parameters required for 'visualization' in npy file
  initial_parameters = np.array([n_nodes, maxd, center_of_gravity], dtype = object)
  np.save(args.ipoutput, initial_parameters)

  # Simulation loop
  start_time_simulation = time.time ()
  while t < 1.0: 
    # Calculate the relative growth rate //bt not used
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2
    if args.growthmethod.__eq__("regional"):
      if args.halforwholebrain.__eq__("half"):
        at, bt = growthRate_2_half(t, n_tets, n_surface_nodes, labels_surface, labels_volume, peak, amplitude, latency, lobes)
      else:
        at, bt = growthRate_2_whole(t, n_tets, n_surface_nodes, labels_surface, labels_surface_2, labels_volume, labels_volume_2, peak, amplitude, latency, lobes, lobes_2, indices_a, indices_b, indices_c, indices_d)
    else:
      at = growthRate(GROWTH_RELATIVE, t, n_tets)
      
    # Calculate the longitudinal length of the real brain
    longi_length = calc_longi_length(t)

<<<<<<< HEAD
    # Calculate the thickness of growing layer
    cortex_thickness = calc_cortex_thickness(THICKNESS_CORTEX, t)
=======
    #update cortex thickness
    cortex_thickness = THICKNESS_CORTEX + 0.01*t
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2

    # Calculate undeformed nodal volume (Vn0) and deformed nodal volume (Vn)
    Vn0, Vn = calc_vol_nodal(tan_growth_tensor, ref_state_tets, tets, coordinates, n_tets, n_nodes)

    # Calculate contact forces
    Ft, NNLt = contact_process(coordinates, Ft, nodal_idx, coordinates_old, n_surface_nodes, NNLt, faces, n_faces, bounding_box, cell_width, prox_skin, repuls_skin, contact_stiffness, mesh_spacing, gr)
  
    # Calculate gray and white matter shear modulus (gm and wm) for a tetrahedron, calculate the global shear modulus
    gm, mu = shear_modulus(dist_2_surf, cortex_thickness, tets, n_tets, muw, mug, gr)

    # Deformed configuration of tetrahedra (At)
    material_tets = config_deform(coordinates, tets, n_tets)

<<<<<<< HEAD
    # Calculate elastic forces & deformation_grad modifAK
    Ft, deformation_grad = tetra_elasticity(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps) 
=======
    # Calculate elastic forces
    Ft = tetra_elasticity(material_tets, ref_state_tets, Ft, tan_growth_tensor, bulk_modulus, k_param, mu, tets, Vn, Vn0, n_tets, eps)

    #Seperate tetraelasticity initialization and calculatin, useful for optimization purposes. 
    #left_cauchy_grad, rel_vol_chg, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, rel_vol_chg_av, deformation_grad, ref_state_growth = tetra1(tets, tan_growth_tensor, ref_state_tets, ref_state_growth, material_tets, Vn, Vn0)
    #Ft = tetra2(n_tets, tets, Ft, left_cauchy_grad, mu, eps, rel_vol_chg, bulk_modulus,rel_vol_chg_av, deformation_grad, rel_vol_chg1, rel_vol_chg2, rel_vol_chg3, rel_vol_chg4, k_param, ref_state_growth)
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2

    # Calculate normals of each deformed tetrahedron 
    tet_norms = tetra_normals(surf_node_norms, nearest_surf_node, tets, n_tets)

    # Calculate relative tangential growth factor G
    tan_growth_tensor = growth_tensor_tangen(tet_norms, gm, at, tan_growth_tensor, n_tets) #tensor growth gets here its G value

    # Midplane
    Ft = calc_mid_plane(coordinates, coordinates0, Ft, nodal_idx, n_surface_nodes, midplane_pos, mesh_spacing, repuls_skin, bulk_modulus)


    ########################
    #        OUTPUTS       #
    ########################
     # Collect the mri values of a native referential nifti image into an array (to later associate each node of the initial simulation mesh to its mri value)
    """
    if step == 0:
      denormalized_coordinates0 = coordinates_denormalisation(coordinates0, n_nodes, center_of_gravity, maxd, miny, args.halforwholebrain)     # mri_deformed_image_array = mesh_to_nifti.deformed_mri_to_nifti(denormalized_coordinates, initial_mri_intensity)
      initial_mri_intensity = np.zeros(n_nodes, dtype=np.float64)
      initial_mri_intensity =  mri_values_to_mesh(denormalized_coordinates0)
      ## Transfer the initial mri grey values to mesh and back to image (control image) 
      #mri_back_array, img_spacing_0, img_origin_0 = mesh_to_nifti.deformed_mri_array(denormalized_coordinates0, initial_mri_intensity)
      #mesh_to_nifti.mri_intensities_to_nifti(step, mri_back_array, img_spacing_0, img_origin_0) 
      mri_deformed_array = griddata_3d.interpoled_values_array(denormalized_coordinates0, initial_mri_intensity)
      griddata_3d.mri_intensities_to_nifti(step, mri_deformed_array)
    """
    if step % di == 0:

      # Write texture of growth in .gii files
      #writeTex(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, bt)

      ###### deformation factor to be operated on the mesh to jibe with reality? #######
      ##################################################################################
      # Obtain zoom parameter by checking the longitudinal length of the brain model //modifAK = longi_length/(xmax - xmin)real
      zoom_pos = paraZoom(coordinates, nodal_idx, longi_length)

      ###### writing and converting files in various formats ###########################
      ##################################################################################
      # Write .pov files and output mesh in .png files
      #writePov(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom, zoom_pos)
     
      # Write surface mesh output files in .txt files
      #writeTXT(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, faces, nodal_idx, nodal_idx_b, n_surface_nodes, zoom_pos, center_of_gravity, maxd, miny, args.halforwholebrain)

      # Convert surface mesh structure (from simulations) to .stl format file
<<<<<<< HEAD
      #mesh_to_stl(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, zoom_pos, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)
      
      # Convert surface mesh structure (from simulations) to .gii format file
      #mesh_to_gifti(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, zoom_pos, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)
=======
      mesh_to_stl(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, zoom_pos, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)

      #creation list deformation
      node_deformation = np.zeros((n_nodes), dtype=np.float64)
      node_deformation = np.linalg.norm(coordinates - coordinates0, axis=1)
      
      # Convert surface mesh structure (from simulations) to .gii format file
      mesh_to_gifti(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, zoom_pos, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain)
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2

      # Convert mesh .stl to image .nii.gz
      #stl_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso)

      # Convert 3d points to image voxel
      #point3d_to_voxel(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, Ut, zoom_pos, maxd, center_of_gravity, nn, miny)

      # Convert volumetric mesh structure (from simulations) to image .nii.gz of a specific resolution
      #mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso, Ut, zoom_pos, center_of_gravity, maxd, nn, faces, tets, miny)

<<<<<<< HEAD
      # Export the coordinates in npy file
      data = np.array([step, coordinates], dtype = object)
      np.save('/home/latim/anaconda3/envs/braingrowth/visualization/coordinates/coordinates_%d.npy'%(step), data)

      # Export the mesh and associated nodal values in npy file
      data_values = np.array([step, coordinates, values], dtype = object)
      np.save('/home/latim/anaconda3/envs/braingrowth/visualization/coordinates/valuesmesh_%d.npy'%(step), data_values)

      ###### parameters display during each simulation step ############################
      ##################################################################################
=======
      ###### data export required by the "visualization" package - for displacements calculation ######
      # Export step and associated coordinates in npy files
      data = np.array([step, coordinates], dtype = object)
      np.save(args.coutput + 'coordinates_%d.npy'%(step), data)
      
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2
      print('step: ' + str(step) + ' t: ' + str(t) )

      # Calculate surface area and mesh volume
      Area, Volume = area_volume(coordinates, faces, gr, Vn)

      #print('Normalized area: ' + str(Area) + ' Normalized volume: ' + str(Volume))
      

      #timestamp for simulation loop
      end_time_simulation = time.time() - start_time_simulation
      print('Time required for simulation loop : ' + str (end_time_simulation))
      start_time_simulation = time.time()
<<<<<<< HEAD

      ##########################################
      #      COORDINATES DENORMALISATION       #  export functions (so to be denormalized)
      ##########################################
      #coordinates have been normalized in the center of gravity referential, x<>y, to better fit reality. zoom_pos removed to compare native mri and deformation_nifti_space. Denormalization required before computing deformations and displaying in the image space.
      """
      denormalized_coordinates = coordinates_denormalisation(coordinates, n_nodes, center_of_gravity, maxd, miny, args.halforwholebrain)
      """
      ##########################################
      #            DEFORMATION NORM            # export functions (so denormalized)
      ##########################################
      """
      # Calculates node deformation intensity for each node at each step%500 
      nodal_deformation = np.zeros(n_nodes, dtype = np.float64)
      nodal_deformation = np.linalg.norm(denormalized_coordinates - denormalized_coordinates0, axis = 1)
      ## mesh_to_ply(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, coordinates, nodal_idx, zoom_pos, center_of_gravity, maxd, n_surface_nodes, faces, nodal_idx_b, miny, args.halforwholebrain, nodal_deformation, n_nodes)
      """
      #################################################
      #   DEFORMATIONS & DEFORMED MRI VALUES NIFTI    # export functions (so denormalized)
      #################################################     
      # 
      ###### TREE QUERY METHOD ####### 
      ## Transfer the nodal deformations to the nifti image space
      #deformations_array_tq, img_spacing_def, img_origin_def = mesh_to_nifti.deformations_array(denormalized_coordinates, nodal_deformation, n_nodes)
      #mesh_to_nifti.deformations_to_nifti(step, deformations_array_tq, img_spacing_def, img_origin_def)   
      ## Transfer the initial mri grey values (on deformed position) to the nifti image space           
      #mri_deformed_array_tq, img_spacing_mri, img_origin_mri = mesh_to_nifti.deformed_mri_array(denormalized_coordinates, initial_mri_intensity)
      #mesh_to_nifti.mri_intensities_to_nifti(step, mri_deformed_array_tq, img_spacing_mri, img_origin_mri)  
      
      #np.savetxt('/home/latim/anaconda3/envs/braingrowth/data/data_ak/denormalized_coordinates.out', denormalized_coordinates, delimiter=',')
      """
      ###### GRIDDATA METHOD #######
      ## Transfer the nodal deformations to the nifti image space
      deformations_array = griddata_3d.interpoled_values_array(denormalized_coordinates, nodal_deformation)
      griddata_3d.deformations_to_nifti(step, deformations_array)       
      ## Transfer the initial mri grey values (on deformed position) to the nifti image space 
      mri_deformed_array = griddata_3d.interpoled_values_array(denormalized_coordinates, initial_mri_intensity)
      griddata_3d.mri_intensities_to_nifti(step, mri_deformed_array)
      """
    ########################
    #    NEWTON DYNAMICS   #
    ########################
    Ft, coordinates, Vt = move(n_nodes, Ft, Vt, coordinates, damping_coef, Vn0, mass_density, dt) #only Ft, coordinates and Vt are updated.
=======
 
    # Newton dynamics
    Ft, coordinates, Vt = move(n_nodes, Ft, Vt, coordinates, damping_coef, Vn0, mass_density, dt)
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2

    t += dt
    step += 1

  #myfile.close()

"""
##############################################
###LEGACY code for reference, safely ignore###
##############################################
#Smoothing using slam
    if step == 0:
        #create filtering for each surface node
        stl_path = "./res/sphere5/pov_H0.042000AT1.829000/B0.stl"
        stl_mesh = trimesh.load (stl_path)
        filtering = np.ones (len(stl_mesh.vertices))
        filtering = laplacian_texture_smoothing (stl_mesh, filtering, 10, dt)
        #Expand filtering via nearest surface node (vectorisable)
        gauss = np.ones (n_nodes, dtype = np.float64)
        for i in range (len(gauss)):
          gauss[i] = filtering[nearest_surf_node[i]]
        
        #conversion from node to tet length, take the average from nodes
        gauss_tets = np.ones (n_tets, dtype = np.float64)
        for i in range (len(tets)):
          x = 0
          for j in tets[i]:
            x += gauss [j]
          gauss_tets [i] = x/4
    
    #apply filter (why not before ? Because stl not available at this point)
    at *= gauss_tets
def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped
<<<<<<< HEAD

=======
>>>>>>> 870cf40bbe8897faf0c263d47d58a5d8f2fe64e2
    """
