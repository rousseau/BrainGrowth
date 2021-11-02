import os
import nibabel as nib

def nifti_writing(reference_nii_path, values_array, output_path):
    """Generate a nifti from brain values array, registered in the reference mri nifti space. """
    # Get the reference image parameters 
    reference_img = nib.load(reference_nii_path)
    header_reference = reference_img.header
    affine_reference = reference_img.affine

    # Register the output nifti with reference image parameters
    reg_values_img = nib.Nifti1Image(values_array, affine_reference, header=header_reference)
    nib.save(reg_values_img, output_path)
    
