import os
import nibabel as nib

# interpolated deformations exported in .nii xx
def values_to_nifti(native_nii_path, values_array, output_path, output_reg_nii_name):
    """Generate a nifti from brain values array, registered in the native mri referential"""
    # native mri parameters 
    native_img = nib.load(native_nii_path)
    header_native = native_img.header
    affine_native = native_img.affine

    # registers output nii with native parameters
    reg_values_img = nib.Nifti1Image(values_array, affine_native, header=header_native)
    save_path = os.path.join(output_path, output_reg_nii_name)
    nib.save(reg_values_img, save_path)
    
