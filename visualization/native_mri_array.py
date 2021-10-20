import numpy as np
import itk

#mri initial image values plugged to the mesh nodes
def mri_values_to_mesh(denormalized_coordinates0, native_nii_path):
    '''
    Interpolates the mri initial grey values for each mesh node a t0 (before deformation)
    '''
    itk_image = itk.imread(native_nii_path) 
    mri_intensity = np.zeros(len(denormalized_coordinates0))

    interpolator = itk.BSplineInterpolateImageFunction.New(itk_image) # NearestNeighborInterpolateImageFunction; LinearInterpolateImageFunction; BSplineInterpolateImageFunction'

    for i in range(len(denormalized_coordinates0)):
        index = itk_image.TransformPhysicalPointToContinuousIndex(denormalized_coordinates0[i]) #find the closest pixel to the vertex[i] (continuous index)
        mri_intensity[i] = interpolator.EvaluateAtContinuousIndex(index) #interpolates values of grey around the index et attribute the interpolation value to the associated mesh node

    return mri_intensity  