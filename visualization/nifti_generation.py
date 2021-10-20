import os
import itk
import nibabel as nib

# interpoled deformations exported in .nii
def values_to_nifti(values_array, output_path, native_nii_path, output_nii_name, output_reg_nii_name):

    #itk nifti output path
    save_path = os.path.join(output_path, output_nii_name)
    values_img = itk.GetImageFromArray(values_array) 
    itk.imwrite(values_img, save_path) 

    #nib registered nifti output path
    reg_values_img = header_affine_processing(native_nii_path, save_path)
    save_new_path = os.path.join(output_path, output_reg_nii_name)
    nib.save(reg_values_img, save_new_path)

def header_affine_processing(native_path, output_path):

    #getting native affine matrix
    native_img = nib.load(native_path)
    header_native = native_img.header
    affine_native = native_img.affine

    #modifying parameters inside of the output image affine
    registered_img = nib.load(output_path)

    new_img_hdr = registered_img.header 
    new_img_hdr = header_native
    
    new_img_af = registered_img.affine 
    new_img_af = affine_native

    return registered_img