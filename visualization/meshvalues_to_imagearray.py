import numpy as np
import nibabel as nib
from scipy.interpolate import griddata 

def interpolated_values_array(coordinates, values, native_nii_path, method):
    """Interpolate the mesh nodal values onto a 3d native-image-shape grid"""
    
    #original_img = itk.imread(native_nii_path)
    native_img = nib.load(native_nii_path)
    native_header = native_img.header
    native_reso = native_header.get_zooms()[0]

    shape_img = np.shape(native_img) # 0 < xmax < 117 / 0 < ymax < 159 / 0 < zmax < 126
    
    size_x = shape_img[0]*native_reso #nibabel convention: no coordinate inversion
    size_y = shape_img[1]*native_reso
    size_z = shape_img[2]*native_reso

    grid_X, grid_Y, grid_Z = np.mgrid[0:size_x:native_reso, 0:size_y:native_reso, 0:size_z:native_reso]

    interpolation_array = griddata(coordinates, values, (grid_X, grid_Y, grid_Z), method=method)

    return interpolation_array