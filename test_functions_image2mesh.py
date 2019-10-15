import numpy as np
import os
import nibabel as nib
import trimesh
#From https://github.com/MahsaShk/MeshProcessing
from nii_2_mesh_conversion import nii_2_mesh

# Convert mesh structure (from simulations) to .stl format
def mesh_to_stl(save_path, vertices, f_indices):
    
  """
  Save surface mesh structure into a mesh of type stl.

  save_path          : Path of mesh
  vertices           : Coordinates of all surface nodes
  f_indices          : Indices of nodes of all surface triangle faces
  """

  # Create the .stl mesh par Trimesh and save it
  mesh = trimesh.Trimesh(vertices=vertices, faces=f_indices)
  mesh.export(save_path)


# Convert mesh .stl to image .nii.gz of a specific resolution
def stl_to_image(foldname, stlname, filename_nii_reso, niiname, reso):
    
  """
  Save mesh of type stl to a nifti file including a binary map.

  foldname           : Path of the folder
  stlname            : Input mesh name in stl format
  filename_nii_reso  : Reference nifti of output nifti binary map
  niiname            : Output nifti binary map 
  reso               : Given resolution of output nifti binary map
  """

  file_stl_path = os.path.join(foldname, stlname)
  
  # Load stl mesh
  m = trimesh.load(file_stl_path)

  # Voxelize mesh with the specific edge length of a single voxel
  v = m.voxelized(pitch=0.25)

  # Fill surface mesh
  v = v.fill(method='holes')
  
  # Convert to binary image
  arr_reso = nib.load(filename_nii_reso).get_data()
  outimage = np.zeros(arr_reso.shape)
  for i in range(np.size(v.points, axis=0)):
    outimage[int(np.round(v.points[i,0]/reso)), int(np.round(v.points[i,1]/reso)), int(np.round(v.points[i,2]/reso))] = 1

  # Save binary image in a nifti file  
  file_nii_path = os.path.join(foldname, niiname)
  aff = np.eye(4)
  aff[0,0] = reso
  aff[1,1] = reso
  aff[2,2] = reso
  img = nib.Nifti1Image(outimage, aff)
  nib.save(img, file_nii_path)
  
# Convert image .nii.gz with a specific resolution to mesh .stl
def image_to_stl(foldname, filename_nii, filename_nii_reso, filename_stl, label, reso):
    
  """
  Read a nifti file including a binary map of a segmented organ with label id = label. 
  Convert it to a smoothed mesh of type stl.

  foldname           : Path of the folder
  filename_nii       : Input nifti binary map 
  filename_nii_reso  : Input nifti binary map with a specific resolution
  filename_stl       : Output mesh name in stl format
  label              : segmented label id
  reso               : Given resolution of input nifti binary map
  """
    
  #Remove directions of an image .nii.gz, give a specific resolution
  file_nii_path = os.path.join(foldname, filename_nii)
  arr = nib.load(file_nii_path).get_data()
  reso = 0.5
  mat = np.eye(4)
  mat[0,0] = reso
  mat[1,1] = reso
  mat[2,2] = reso
   
  # Save image .nii.gz with a specific resolution
  file_nii_reso_path = os.path.join(foldname, filename_nii_reso)
  nib.save(nib.Nifti1Image(arr,mat), file_nii_reso_path)

  # Transform nifti image to stl mesh
  file_stl_path = os.path.join(foldname, filename_stl)
  nii_2_mesh(file_nii_reso_path, file_stl_path, label)

# Convert 3d coordinates of volumetric mesh to image .nii.gz of a specific resolution
def mesh_to_image(foldname, vertices, filename_nii_reso, niiname, reso):
   
  """
  Convert 3d coordinates of volumetric mesh to a nifti file including a binary map.
  
  foldname           : Path of the folder
  vertices           : 3D coordinates of volumetric mesh 
  filename_nii_reso  : Reference nifti of output nifti binary map
  niiname            : Output nifti binary map 
  reso               : Given resolution of output nifti binary map
  """
  
  """mesh = pymesh.form_mesh(vertices, faces, voxels)
  grid = pymesh.VoxelGrid(cell_size, mesh.dim)
  grid.insert_mesh(mesh)
  grid.create_grid()
  out_mesh = grid.mesh"""
  
  # Convert to binary image
  arr_reso = nib.load(filename_nii_reso).get_data()
  outimage = np.zeros(arr_reso.shape)
  for i in range(np.size(vertices, axis=0)):
    outimage[int(np.round(vertices[i,0]/reso)), int(np.round(vertices[i,1]/reso)), int(np.round(vertices[i,2]/reso))] = 1

  # Save binary image in a nifti file  
  file_nii_path = os.path.join(foldname, niiname)
  aff = np.eye(4)
  aff[0,0] = reso
  aff[1,1] = reso
  aff[2,2] = reso
  img = nib.Nifti1Image(outimage, aff)
  nib.save(img, file_nii_path)
