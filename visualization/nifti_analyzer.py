import nibabel as nib

# nifti analyzer
native_path = '/home/latim/braingrowth_rousseau/FetalAtlas/mesh_to_image_testFolder/template_T2_reduced_z.nii'
native_image = nib.load(native_path)
native_affine = native_image.affine
native_header = native_image.header
print('native affine is: \n' + str(native_affine) + '\n')
print('native header is: \n' + str(native_header)+ '\n')
print('native resolution is: \n' + str(native_header.get_zooms()) + '\n')
print('shape of "template_T2_reduced_z.nii" is: \n' + str(native_image.shape) + '\n')

gd_path = '/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/displacements/griddata/displacements_gd_1000.nii'
gd_image = nib.load(gd_path)
gd_affine = gd_image.affine
gd_header = gd_image.header
print('griddata affine is: \n' + str(gd_affine) + '\n')
print('griddata header is: \n' + str(gd_header)+ '\n')
print('griddata resolution is: \n' + str(gd_header.get_zooms()) + '\n')
print('shape of "displacements_gd_1000.nii" is: \n' + str(gd_image.shape) + '\n')
