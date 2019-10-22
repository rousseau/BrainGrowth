import numpy as np
import math
import os
from vapory import *
from geometry import normalSurfaces
import nibabel as nib
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from stl import mesh, Mode
import trimesh
#import mayavi.mlab
#import pymesh

# Calculate surface area and mesh volume
def area_volume(Ut, faces, gr, Vn):

  Area = 0.0

  for i in range(len(faces)):
    Ntmp = np.cross(Ut[faces[i,1]] - Ut[faces[i,0]], Ut[faces[i,2]] - Ut[faces[i,0]])
    Area += 0.5*np.linalg.norm(Ntmp)     #*(gr[faces[i,0]] + gr[faces[i,1]] + gr[faces[i,2]])/3.0

  Volume = 0.0

  Volume = abs(np.sum(Vn[:]))

  return Area, Volume

# Writes POV-Ray source files and then output in .png files
def writePov(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom, zoom_pos):

  povname = "%s/pov_H%fAT%f/B%d.png"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)

  # Normals in deformed state
  N_init = np.zeros((nsn,3), dtype = float)
  N = normalSurfaces(Ut, faces, SNb, len(faces), nsn, N_init)

  #os.path.dirname(povname)

  camera = Camera('location', [-3*zoom, 3*zoom, -3*zoom], 'look_at', [0, 0, 0], 'sky', [0, 0, -1], 'focal_point', [-0.55, 0.55, -0.55], 'aperture', 0.055, 'blur_samples', 10)
  light = LightSource([-14, 3, -14], 'color', [1, 1, 1])
  background = Background('color', [1, 1, 1])

  vertices = np.zeros((nsn,3), dtype = float)
  normals = np.zeros((nsn,3), dtype = float)
  f_indices = np.zeros((len(faces),3), dtype = int)
  vertices[:,:] = Ut[SN[:],:]*zoom_pos
  normals[:,:] = N[:,:]
  f_indices[:,0] = SNb[faces[:,0]]
  f_indices[:,1] = SNb[faces[:,1]]
  f_indices[:,2] = SNb[faces[:,2]]

  """vertices = []
  normals = []
  f_indices = []
  for i in range(nsn):
    vertex = [Ut[SN[i]][0]*zoom_pos, Ut[SN[i]][1]*zoom_pos, Ut[SN[i]][2]*zoom_pos]
    vertices.append(vertex)
    normal = [N[i][0], N[i][1], N[i][2]]
    normals.append(normal)
  for i in range(len(faces)):
    f_indice = [SNb[faces[i][0]], SNb[faces[i][1]], SNb[faces[i][2]]]
    f_indices.append(f_indice)"""

  Mesh = Mesh2(VertexVectors(nsn, *vertices), NormalVectors(nsn, *normals), FaceIndices(len(faces), *f_indices), 'inside_vector', [0,1,0])
  box = Box([-100, -100, -100], [100, 100, 100])
  pigment = Pigment('color', [1, 1, 0.5])
  normal = Normal('bumps', 0.05, 'scale', 0.005)
  finish = Finish('phong', 1, 'reflection', 0.05, 'ambient', 0, 'diffuse', 0.9)

  intersection = Intersection(Mesh, box, Texture(pigment, normal, finish))

  scene = Scene(camera, objects=[light, background, intersection], included=["colors.inc"])
  #scene.render(povname, width=400, height=300, quality = 9, antialiasing = 1e-5 )
  scene.render(povname, width=800, height=600, quality=9)
  
# Writes POV-Ray source files
def writePov2(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom, zoom_pos):

  povname = "B%d.pov"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)

  completeName = os.path.join(foldname, povname)
  filepov = open(completeName, "w")

  # Normals in deformed state
  N_init = np.zeros((nsn,3), dtype = float)
  N = normalSurfaces(Ut, faces, SNb, len(faces), nsn, N_init)

  """for i in range(len(faces)):
    Ntmp = np.cross(Ut[faces[i][1]] - Ut[faces[i][0]], Ut[faces[i][2]] - Ut[faces[i][0]])
    N[SNb[faces[i][0]]] += Ntmp
    N[SNb[faces[i][1]]] += Ntmp
    N[SNb[faces[i][2]]] += Ntmp

  for i in range(nsn):
    N_norm = np.linalg.norm(N[i])
    N[i] *= 1.0/N_norm"""

  filepov.write("#include \"colors.inc\"\n")
  filepov.write("background { color rgb <1,1,1> }\n")
  filepov.write("camera { location <" + str(-3*zoom) + ", " + str(3*zoom) + ", " + str(-3*zoom) + "> look_at <0, 0, 0> sky <0, 0, -1> focal_point <-0.55, 0.55, -0.55> aperture 0.055 blur_samples 10 }\n")
  filepov.write("light_source { <-14, 3, -14> color rgb <1, 1, 1> }\n")

  filepov.write("intersection {\n")
  filepov.write("mesh2 { \n")
  filepov.write("vertex_vectors { " + str(nsn) + ",\n")
  for i in range(nsn):
    filepov.write("<" + "{0:.5f}".format(Ut[SN[i]][0]*zoom_pos) + "," + "{0:.5f}".format(Ut[SN[i]][1]*zoom_pos) + "," + "{0:.5f}".format(Ut[SN[i]][2]*zoom_pos) + ">,\n")
  filepov.write("} normal_vectors { " + str(nsn) + ",\n")
  for i in range(nsn):
    filepov.write("<" + "{0:.5f}".format(N[i][0]) + "," + "{0:.5f}".format(N[i][1]) + "," + "{0:.5f}".format(N[i][2]) + ">,\n")
  filepov.write("} face_indices { " + str(len(faces)) + ",\n")
  for i in range(len(faces)):
    filepov.write("<" + str(SNb[faces[i][0]]) + "," + str(SNb[faces[i][1]]) + "," + str(SNb[faces[i][2]]) + ">,\n")
  filepov.write("} inside_vector<0,1,0> }\n")
  filepov.write("box { <-100, -100, -100>, <100, 100, 100> }\n")
  filepov.write("pigment { rgb<1,1,0.5> } normal { bumps 0.05 scale 0.005 } finish { phong 1 reflection 0.05 ambient 0 diffuse 0.9 } }\n")

  filepov.close()

# Write surface mesh in .txt files
def writeTXT(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom_pos):

  txtname = "B%d.txt"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)

  completeName = os.path.join(foldname, txtname)
  filetxt = open(completeName, "w")
  filetxt.write(str(nsn) + "\n")
  for i in range(nsn):
    filetxt.write(str(Ut[SN[i]][0]*zoom_pos) + " " + str(Ut[SN[i]][1]*zoom_pos) + " " + str(Ut[SN[i]][2]*zoom_pos) + "\n")
  filetxt.write(str(len(faces)) + "\n")
  for i in range(len(faces)):
    filetxt.write(str(SNb[faces[i][0]]+1) + " " + str(SNb[faces[i][1]]+1) + " " + str(SNb[faces[i][2]]+1) + "\n")
  filetxt.close()

# Convert surface mesh structure (from simulations) to .stl format file
def mesh_to_stl(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, SN, zoom_pos, cog, maxd, nsn, faces, SNb):

  stlname = "B%d.stl"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  save_path = os.path.join(foldname, stlname)

  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((nsn,3), dtype = float)
  f_indices = np.zeros((len(faces),3), dtype = int)
  vertices_seg = np.zeros((nsn,3), dtype = float)

  vertices[:,:] = Ut[SN[:],:]*zoom_pos
  vertices_seg[:,1] = cog[0] - Ut[SN[:],0]*maxd
  #vertices_seg[:,1] = vertices[:,0]*maxd + cog[0]
  vertices_seg[:,0] = Ut[SN[:],1]*maxd + cog[1]
  #vertices_seg[:,0] = cog[1] - vertices[:,1]*maxd
  vertices_seg[:,2] = cog[2] - Ut[SN[:],2]*maxd
  #vertices_seg[:,2] = vertices[:,2]*maxd + cog[2]

  f_indices[:,0] = SNb[faces[:,0]]
  f_indices[:,1] = SNb[faces[:,1]]
  f_indices[:,2] = SNb[faces[:,2]]

  # Create the .stl mesh par Trimesh and save it
  mesh = trimesh.Trimesh(vertices=vertices_seg, faces=f_indices)
  mesh.export(save_path)

  """# Create the .stl mesh
  brain = mesh.Mesh(np.zeros(f_indices.shape[0], dtype=mesh.Mesh.dtype))
  for i, f in enumerate(f_indices):
    for j in range(3):
        brain.vectors[i][j] = vertices_seg[f[j],:]

  # Write the mesh to file ".stl"
  brain.save(save_path, mode=Mode.ASCII)"""

# Convert mesh .stl to image .nii.gz
def stl_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso):

  stlname = "B%d.stl"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

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
  niiname = "B%d_1.nii.gz"%(step)
  file_nii_path = os.path.join(foldname, niiname)
  aff = np.eye(4)
  aff[0,0] = reso
  aff[1,1] = reso
  aff[2,2] = reso
  img = nib.Nifti1Image(outimage, aff)
  nib.save(img, file_nii_path)

# Convert volumetric mesh structure (from simulations) to image .nii.gz of a specific resolution
def mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, filename_nii_reso, reso, Ut, zoom_pos, cog, maxd, nn):

  niiname = "B%d_2.nii.gz"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((nn,3), dtype = float)
  vertices_seg = np.zeros((nn,3), dtype = float)

  vertices[:,:] = Ut[:,:]*zoom_pos
  vertices_seg[:,1] = cog[0] - Ut[:,0]*maxd
  #vertices_seg[:,1] = vertices[:,0]*maxd + cog[0]
  vertices_seg[:,0] = Ut[:,1]*maxd + cog[1]
  #vertices_seg[:,0] = cog[1] - vertices[:,1]*maxd
  vertices_seg[:,2] = cog[2] - Ut[:,2]*maxd
  #vertices_seg[:,2] = vertices[:,2]*maxd + cog[2]

  """mesh = pymesh.form_mesh(vertices_seg, faces, tets)
  grid = pymesh.VoxelGrid(cell_size, mesh.dim)
  grid.insert_mesh(mesh)
  grid.create_grid()
  out_mesh = grid.mesh"""
  #mayavi.mlab.points3d(vertices_seg, mode="cube", scale_factor=1)

  # Convert to binary image
  arr_reso = nib.load(filename_nii_reso).get_data()
  outimage = np.zeros(arr_reso.shape)
  for i in range(np.size(vertices_seg, axis=0)):
    outimage[int(np.round(vertices_seg[i,0]/reso)), int(np.round(vertices_seg[i,1]/reso)), int(np.round(vertices_seg[i,2]/reso))] = 1

  # Save binary image in a nifti file  
  file_nii_path = os.path.join(foldname, niiname)
  aff = np.eye(4)
  aff[0,0] = reso
  aff[1,1] = reso
  aff[2,2] = reso
  img = nib.Nifti1Image(outimage, aff)
  nib.save(img, file_nii_path)
  
'''# Convert mesh to binary .nii.gz image
def mesh_to_image(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, SN, zoom_pos, cog, maxd, nn):

  nifname = "B%d.nii.gz"%(step)

  foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

  # Transform coordinates (because the coordinates are normalized at the beginning)
  vertices = np.zeros((nn,3), dtype = float)
  vertices_seg = np.zeros((nn,3), dtype = float)
  vertices[:,:] = Ut[:,:]*zoom_pos
  vertices_seg[:,1] = cog[0] - vertices[:,0]*maxd
  vertices_seg[:,0] = vertices[:,1]*maxd + cog[1]
  vertices_seg[:,2] = cog[2] - vertices[:,2]*maxd

  # Calculate the center coordinate(x,y,z) of the mesh to define the binary image size
  cog = np.sum(vertices_seg, axis=0)
  cog /= nn 

  # Convert mesh to binary image
  outimage = np.zeros((2*int(round(cog[0]))+1, 2*int(round(cog[1]))+1, 2*int(round(cog[2]))+1), dtype=np.int16)
  for i in range(nn):
    outimage[int(round(vertices_seg[i,0])), int(round(vertices_seg[i,1])), int(round(vertices_seg[i,2]))] = 1

  # Save binary image in a nifti file
  try:
    if not os.path.exists(foldname):
      os.makedirs(foldname)
  except OSError:
    print ('Error: Creating directory. ' + foldname)

  nii = nib.load('/home/x17wang/Exp/London/London-23weeks/brain_crisp_2_refilled.nii.gz')
  #out_inter = ndimage.morphology.binary_fill_holes(out1).astype(out1.dtype)
  #out_inter = ndimage.morphology.binary_dilation(out1, iterations=2).astype(out1.dtype)
  img = nib.Nifti1Image(outimage, nii.affine)
  save_path = os.path.join(foldname, nifname)
  nib.save(img, save_path)'''
