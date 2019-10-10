import trimesh
import nibabel as nib
import numpy as np
#From https://github.com/MahsaShk/MeshProcessing
from nii_2_mesh_conversion import nii_2_mesh


filename_nii = '/home/x17wang/Exp/prm001/prm001_30w_Lwhite_r05.nii.gz'
filename_stl = '/home/x17wang/Exp/prm001/prm001_30w_Lwhite_r05.stl'

#Remove directions
arr = nib.load(filename_nii).get_data()
reso = 0.5
mat = np.eye(4)
mat[0,0] = reso
mat[1,1] = reso
mat[2,2] = reso
nib.save(nib.Nifti1Image(arr,mat), filename_nii)


## Transform nifti image to stl mesh
label = 0
nii_2_mesh(filename_nii, filename_stl, label)

## Transform stl mesh to nifti image
# Path of input mesh
#inputFIlePath = '/home/x17wang/Exp/prm001/prm001_40w_Rwhite.stl'

# Load stl mesh
m = trimesh.load(filename_stl) 

# Voxelize mesh with the specific edge length of a single voxel
v = m.voxelized(pitch=0.25)  #Return a Voxel object representing the current mesh discretized into voxels at the specified pitch, pitch: float, the edge length of a single voxel

# Fill surface mesh
v = v.fill(method='holes')

# Center coordinates
cog = np.sum(v.points, axis=0)
cog /= np.size(v.points, axis=0) 

# Convert mesh to binary image
#print(cog)
#outimage = np.zeros((2*int(np.round(cog[0]/reso))+1, 2*int(np.round(cog[1]/reso))+1, 2*int(np.round(cog[2]/reso))+1), dtype=np.int16)
outimage = np.zeros(arr.shape)
print(outimage.shape)

for i in range(np.size(v.points, axis=0)):
  #Convert stl coordinates (ie world coordinate) to image coordinates  
  #Directions and origin have been reset
  #So we consider only image resolution
  outimage[int(np.round(v.points[i,0]/reso)), int(np.round(v.points[i,1]/reso)), int(np.round(v.points[i,2]/reso))] = 1

# Save binary image in a nifti file
#nii = nib.load('/home/x17wang/Data/prm001/prm001_40w_Rwhite.nii')
save_path = '/home/x17wang/Exp/prm001/test_prm001_30w_Lwhite_r05.nii.gz'

#img = nib.Nifti1Image(outimage, nii.affine)
aff = np.eye(4)
aff[0,0] = reso
aff[1,1] = reso
aff[2,2] = reso

img = nib.Nifti1Image(outimage, aff)
nib.save(img, save_path)
