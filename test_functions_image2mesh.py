import numpy as np
import os
import nibabel as nib
import trimesh
#From https://github.com/MahsaShk/MeshProcessing
#from nii_2_mesh_conversion import nii_2_mesh
from scipy import spatial
#import pymesh

def point3d_to_voxel(point, matrix_image_to_world, abc):
  
  """
  inputs:
      spatial location of the node
      direction matrix of the reference image (matrix_image_to_world, abc)
      
  outputs:
      spatial location in the image
  """
  vertices_seg = np.zeros((voxel.shape(0), voxel.shape(1), voxel.shape(2), 1), dtype=np.float64)
  #image = matrix_world_to_image.dot(point[:])
  image = np.linalg.inv(matrix_image_to_world).dot(point[:] - abc)
  imgi = max(image[:,0])
  imgj = max(image[:,1])
  imgk = max(image[:,2])
  outimage = np.zeros((imgi, imgj, imgk), dtype=np.int64)
  for i in range(np.size(point, axis=0)):
    outimage[image[i,0], image[i,1], image[i,2]] = 1
    
  return outimage     

def voxel_to_point3d(voxel, matrix_image_to_world, abc):
  
  """
  inputs:
      spatial location in the image
      direction matrix of the reference image (matrix_image_to_world, abc)
      
  outputs:
      spatial location of the node
  """
  vertices_seg = np.zeros(voxel.shape, dtype=np.float64)
  indices = np.argwhere(voxel == 1)
  vertices_seg[:, :, :] = matrix_image_to_world.dot([indices[:,0], indices[:,1], indices[:,2]]) + abc 
  
  return vertices_seg

def voxel_to_world(voxel,matrix): #4x4 matrix
  M = matrix[:3, :3]
  abc = matrix[:3, 3]
  abc = abc.reshape((3,1))
  return M.dot(voxel) + abc

#@jit(parallel=True) #Does not work with Trimesh object 
def stl_to_nii(stl_mesh, outarray, matrix):
  #get indices
  array_index = np.asarray(np.where(outarray==0)) #ugly !
  points = voxel_to_world(array_index,matrix)
  #bool_index = stl_mesh.contains(np.transpose(points)) #using trimesh (slow)
  #Code from https://gist.github.com/LMescheder/b5e03ffd1bf8a0dfbb984cacc8c99532
  bool_index = check_mesh_contains(stl_mesh, np.transpose(points)) #using cython (fast) 
  
  outarray = 1.0*bool_index.reshape(outarray.shape)

m = trimesh.load('/home/x17wang/Codes/BrainGrowth/res/week23-3M-tets_atlas_Garcia/pov_H0.045000AT1.829000/B0.stl')
img = nib.load("/home/x17wang/Exp/London/London-23weeks/brain_crisp_2_refilled.nii.gz")
data = img.get_fdata()
stl_to_nii(m, data, img.affine)

def correspondence_voxel_and_point3d(path, point, tets):

  """
  inputs:
      path of the image
      spatial locations of mesh nodes (mesh coordinates)
      
  outputs:
      correspondence between voxels and mesh points
  """

  img = nib.load(path)
  data = img.get_fdata()
  vertices_seg = np.zeros(data.shape, dtype=np.float64)
  csn = np.zeros(data.shape, dtype=np.int64)
  cst = np.zeros(data.shape, dtype=np.int64)
  matrix_image_to_world = img.affine[:3, :3]
  abc = img.affine[:3, 3]
  tets_barycenter = (point[tets[:,0]] + point[tets[:,1]] + point[tets[:,2]] + point[tets[:,3]])/4.0
  tree_1 = spatial.KDTree(point)
  pp1 = tree_1.query(matrix_image_to_world.dot(np.asarray(np.where(data==1))) + abc)
  tree_2 = spatial.KDTree(tets_barycenter)
  pp2 = tree_2.query(matrix_image_to_world.dot(np.asarray(np.where(data==1))) + abc)
  indices = np.argwhere(data == 1)
  csn[indices[:,0], indices[:,1], indices[:,2]] = pp1[1]
  cst[indices[:,0], indices[:,1], indices[:,2]] = pp2[1]
  
  '''for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      for k in range(data.shape[2]):
        vertices_seg[i, j, k] = matrix_image_to_world.dot([i, j, k]) + abc      
        d2n = dot_vec_dim_3(point[:] - vertices_seg[i, j, k], point[:] - vertices_seg[i, j, k])
        d2t = dot_vec_dim_3(tets_barycenter[:] - vertices_seg[i, j, k], tets_barycenter[:] - vertices_seg[i, j, k])
        csn[i, j, k] = np.argmin(d2n)
        cst[i, j, k] = np.argmin(d2t)'''

  return csn, cst, matrix_image_to_world, abc

def get_closest_node(point, mesh):
  """
  Nearest neighbour
  """
  node = 0
  return node

def voxelization(mesh, vertices_seg):
  output_image = np.zeros(())
  
  '''Loop over all voxels
  For each voxel, find the 3D location with voxel_to_point3d
  Use Kd-Tree for finding adjacent tetrahdron 
  '''
  Ut_barycenter = (Ut[tets[:,0]] + Ut[tets[:,1]] + Ut[tets[:,2]] + Ut[tets[:,3]])/4.0
  d2 = dot_vec_dim_3(point[:] - vertices_seg[i, j, k], point[:] - vertices_seg[i, j, k])
  return output_image    
    

def save_mesh_in_stl():
  """
  inputs:
      
  outputs:
      None
  """    

def convert_stl_to_nifti():
  """
  inputs:
      
  outputs:
      None
  """    
    
  
# Convert surface mesh structure (from simulations) to .stl format
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
  
# Convert volumetric coordinates (from simulations) to image .nii.gz of a specific resolution
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
