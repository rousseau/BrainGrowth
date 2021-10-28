"""
The mri grey values from the "reference native nifti" are collected. 
Then plugged to each nodes of each input mesh, as arrays. 
Finally, each "weighted" array is interpolated into the reference nifti space.
"""

import numpy as np
import argparse
import os
import time
import itk

# Local modules
from meshvalues_to_imagearray import interpolated_values_array
from nifti_generation import values_to_nifti

# Generation of a n_nodes array containing mri initial grey values plugged to each mesh node  xx
def mri_values_to_mesh(coordinates0, native_nii_path):
    '''
    Interpolates the mri initial grey values for each mesh node a t0 (before deformation)
    '''
    itk_image = itk.imread(native_nii_path) 
    mri_intensity = np.zeros(len(coordinates0))

    interpolator = itk.BSplineInterpolateImageFunction.New(itk_image) # NearestNeighborInterpolateImageFunction; LinearInterpolateImageFunction; BSplineInterpolateImageFunction'

    for i in range(len(coordinates0)):
        index = itk_image.TransformPhysicalPointToContinuousIndex(coordinates0[i]) #find the closest pixel to the vertex[i] (continuous index)
        mri_intensity[i] = interpolator.EvaluateAtContinuousIndex(index) #interpolates values of grey around the index et attribute the interpolation value to the associated mesh node

    return mri_intensity  

if __name__ == '__main__':
    start_time_initialization = time.time ()
    parser = argparse.ArgumentParser(description='reference mri grey values collected from the reference nifti as a constant. Then plugged to input mesh ')
    parser.add_argument('-c', '--coordinates', help='Input path to folder containing .npy files from braingrowth simulation with [step, coordinates] at each step%500', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/raw_mesh/', required=False)
    parser.add_argument('-nn', '--nativenii', help='Input path to reference native nifti', type=str, default= '/home/latim/braingrowth_rousseau/FetalAtlas/mesh_to_image_testFolder/template_T2_reduced_z.nii', required=False)
    parser.add_argument('-o', '--outputfolder', help='Output path for griddata mri intensities', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/native_mri/', required=False)
    parser.add_argument('-gm', '--gdmethod', help='griddata interpolation method: nearest; linear; cubic?', type=str, default='linear', required=False)
    args = parser.parse_args()

    #####################################
    # VISUALISATION INPUTS AND PARAMETERS
    #####################################
    # Input data 
    native_nii_path = args.nativenii

    # Output paths
    mri_intensities_path = args.outputfolder

    # Methods
    griddata_method = args.gdmethod

    ################
    # MAIN PROGRAM #
    ################
    # Generates the reference mri values (n_nodes) array
    data0 = np.load(args.coordinates + 'coordinates_0.npy', allow_pickle = True) # loading coordinates0 from simulation outputs .npy files
    coordinates0 = data0[1]
    n_nodes = len(coordinates0)
    initial_mri_intensity = np.zeros(n_nodes, dtype=np.float64)
    initial_mri_intensity =  mri_values_to_mesh(coordinates0, native_nii_path) # plugging the mri intensity values of the native nifti to each node of the input mesh 

    # Generates a interpolated nifti of the nodal native mri grey values (at each step%500)   
    for r, d, npyfiles in os.walk(args.valuesmeshf):
        for npyfile in sorted(npyfiles):    

            data = np.load(args.coordinates + str(npyfile), allow_pickle = True) # loading coordinates from simulation outputs .npy files
            step = data[0]
            coordinates = data[1]

            print('Running step ' + str(step) + ' \n')

            reg_mri_output_name = "deformed_mri_%d.nii"%(step)

            mri_deformed_array = interpolated_values_array(coordinates, initial_mri_intensity, native_nii_path, griddata_method)
            values_to_nifti(native_nii_path, mri_deformed_array, mri_intensities_path, reg_mri_output_name) 