import numpy as np
import itk
from scipy.interpolate import griddata 
from scipy.spatial import cKDTree

def gd_interpoled_values_array(denormalized_coordinates, values, native_nii_path, method):
    '''Interpolation of the mesh nodal values onto a 3d image-shape grid.'''
    
    original_img = itk.imread(native_nii_path)
    shape_img = np.shape(original_img) # 0 < xmax < 117 / 0 < ymax < 159 / 0 < zmax < 126

    size_z = shape_img[0] # first value provided by itk corresponds to Z of the nifti image
    size_y = shape_img[1]
    size_x = shape_img[2] # 117
    grid_X, grid_Y, grid_Z = np.mgrid[0:size_x, 0:size_y, 0:size_z] #grid to be used for interpolation within "griddata" function
    points = np.vstack((grid_X.ravel(), grid_Y.ravel(), grid_Z.ravel())).T #array of all voxels coordinates (array of arrays)

    #if mask_img[points] == 1: (binary mask)   
      # interpolate 
      
    #interpolate nodal deformation values on new grid mesh
    interpolation_array = griddata(denormalized_coordinates, values, (grid_X, grid_Y, grid_Z), method = method) # method = 'nearest', 'linear', ('cubic')/ griddata(points to interpolate, intensities associated to points, interpolation grid, interpolation method)

    interpoled_values_array = np.zeros((size_z, size_y, size_x), dtype=np.float64)
    for i in points:
        #if mask_img[i[2]][i[1]][i[0]] == 0: # x<>z between itk and nifti&array. AS mask_img is an itk object, indentiation follows [z][y][x]
            #continue
        interpoled_values_array[i[2]][i[1]][i[0]] = interpolation_array[i[0]][i[1]][i[2]]
        
    return interpoled_values_array

def tq_interpoled_values_array(denormalized_coordinates, values, native_nii_path, binary_mask_path):
    '''Interpolation of the mesh nodal values onto a 3d image array by tree querying the array physical coordinates. Mask applied during interpolation.'''
        
    original_image = itk.imread(native_nii_path) # (z,y,x) (inversed by itk compared to acquisition)
    shape = np.shape(original_image)
    interpoled_values_array = np.zeros((shape)) 

    mask_image = itk.imread(binary_mask_path) # (z,y,x)

    tree = cKDTree(denormalized_coordinates)

    for i in range(len(interpoled_values_array)):
        for j in range(len(interpoled_values_array[i])):
            for k in range(len(interpoled_values_array[i][j])):
                if mask_image[i,j,k] != 1:
                    interpoled_values_array[i][j][k] = 0
                else:                
                #fetch the closest mesh coordinate from the voxel
                    coord = tree.query(original_image.TransformIndexToPhysicalPoint((k, j, i))) #comparison of z,y,x itk object to x,y,z real mesh object. Need to invert indices.
                    tex = values[coord[1]]
                    #plug coord to the voxel
                    interpoled_values_array[i][j][k] = tex #shape z, y, x

    return interpoled_values_array