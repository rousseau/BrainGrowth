import numpy as np
import nibabel as nib
from scipy.interpolate import griddata 

def mesh_to_image(coordinates, values, reference_nii_path, method):
    """Interpolate the mesh nodal values onto a 3d reference-image-shape grid"""
    reference_img = nib.load(reference_nii_path)
    reference_header = reference_img.header
    reference_reso = reference_header.get_zooms()[0]

    shape_img = np.shape(reference_img) # 0 < xmax < 117 / 0 < ymax < 159 / 0 < zmax < 126
    
    size_x = shape_img[0]*reference_reso #nibabel convention: no coordinate inversion
    size_y = shape_img[1]*reference_reso
    size_z = shape_img[2]*reference_reso

    grid_X, grid_Y, grid_Z = np.mgrid[0:size_x:reference_reso, 0:size_y:reference_reso, 0:size_z:reference_reso]

    interpolation_array = griddata(coordinates, values, (grid_X, grid_Y, grid_Z), method=method)

    return interpolation_array