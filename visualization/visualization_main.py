"""Generate a nifti image from input mesh values.

Parameters:
-i (default '/home/latim/anaconda3/envs/braingrowth/visualization/values_mesh/')
-nn (default '/home/latim/braingrowth_rousseau/FetalAtlas/mesh_to_image_testFolder/template_T2_reduced_z.nii')
-o (default '/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/displacements/griddata/')
-gm (default 'linear')

Return .nii file containing brain values (e.g. deformed native mri grey values, displacements, deformations, distance to surface, nodal growth rate), registered in the native image space.
""" 

import numpy as np
import argparse
import time
import os

# Local modules
from meshvalues_to_imagearray import interpolated_values_array
from nifti_generation import values_to_nifti

if __name__ == '__main__':
    start_time_initialization = time.time ()
    parser = argparse.ArgumentParser(description='Visualisation of brain values (nifti)')
    parser.add_argument('-i', '--input', help='Input path to folder containing .npy files from braingrowth simulation with [step, coordinates, brain_values] at each step%500', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/values_mesh/displacements/', required=False)
    parser.add_argument('-nn', '--nativenii', help='Input path to reference native nifti', type=str, default= '/home/latim/braingrowth_rousseau/FetalAtlas/mesh_to_image_testFolder/template_T2_reduced_z.nii', required=False)
    parser.add_argument('-o', '--outputfolder', help='Output path for generated brain values nifti', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/displacements/griddata/', required=False)
    parser.add_argument('-gm', '--gdmethod', help='griddata interpolation method: nearest; linear; cubic?', type=str, default='linear', required=False)
    args = parser.parse_args()

    # TO BE PUT IN PERSONALIZED/RAISE ERROR: --input, --nativenii, --outputfolder

    #####################################
    # VISUALISATION INPUTS AND PARAMETERS
    #####################################
    # Input data 
    native_nii_path = args.nativenii
    
    # Output paths
    values_path = args.outputfolder

    # griddata interpolation method
    griddata_method = args.gdmethod

    ################
    # MAIN PROGRAM #
    ################
    # Work on data exported from simulation at different steps%500:
    ##############################################################
    for r, d, npyfiles in os.walk(args.input):
        for npyfile in sorted(npyfiles):

            data = np.load(args.input + str(npyfile), allow_pickle = True) # loading coordinates from simulation outputs .npy files
            step = data[0]
            coordinates = data[1]
            brain_values = data[2]
            print('Running step ' + str(step) + ' \n')

            # Generates a interpolated nifti of the nodal values (at each step%500)    
            reg_output_name = "displacements_%d.nii"%(step) # NAME TO BE PUT IN INPUT ARGUMENTS

            values_array_gd = interpolated_values_array(coordinates, brain_values, native_nii_path, griddata_method)
            values_to_nifti(native_nii_path, values_array_gd, values_path, reg_output_name)  
            
    end_time_simulation = time.time() - start_time_initialization
    print('Nifti generated \n')
    print('Visualization loop time taken was: ' + str(end_time_simulation))