import numpy as np
import argparse
import time
import os
import itk

# Local modules
from denormalization import coordinates_denormalization
from native_mri_array import mri_values_to_mesh
from meshvalues_to_imagearray import tq_interpoled_values_array, gd_interpoled_values_array
from nifti_generation import values_to_nifti

if __name__ == '__main__':
    start_time_initialization = time.time ()
    parser = argparse.ArgumentParser(description='Visualisation of brain values (nifti)')
    parser.add_argument('-p', '--parameters', help='Input path to.npy file containing initial parameters (from simulation)', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/initial_parameters/parameters.npy', required=False)
    parser.add_argument('-c', '--coordinates', help='Input path to folder containing .npy files from braingrowth simulation with [step, coordinates] at each step%500', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/coordinates/', required=False)
    parser.add_argument('-nn', '--nativenii', help='Input path to reference native nifti', type=str, default= '/home/latim/braingrowth_rousseau/FetalAtlas/mesh_to_image_testFolder/template_T2_reduced_z.nii', required=False)
    parser.add_argument('-bm', '--mask', help='Input path to binary mask path for treequery interpolation', type=str, default= '/home/latim/anaconda3/envs/braingrowth/data/data_ak/binary_masking/binary_mask_template_T2_reduced.nii', required=False)
    parser.add_argument('-odt', '--outputdistq', help='Output path for treequery displacements', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/displacements/', required=False)
    parser.add_argument('-omt', '--outputmritq', help='Output path for treequery mri intensities', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/native_mri/', required=False)
    parser.add_argument('-odg', '--outputdisgd', help='Output path for griddata displacements', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/displacements/', required=False)
    parser.add_argument('-omg', '--outputmrigd', help='Output path for griddata mri intensities', type=str, default='/home/latim/anaconda3/envs/braingrowth/visualization/nifti_brain_values/native_mri/', required=False)
    parser.add_argument('-m', '--method', help='Method to plug back values from deformed mesh (coordinates) to the image space : treequery; griddata', type=str, default='griddata', required=False)
    parser.add_argument('-gdm', '--gdmethod', help='griddata interpolation method: nearest; linear; cubic?', type=str, default='linear', required=False)
    args = parser.parse_args()
    ''' 
    Shape of input data: 
    args.parameters = np.array([n_nodes, maxd, center_of_gravity], dtype = object)
    args.coordinates = 'coordinates_step.npy' files for all exported steps, containging np.array([step, coordinates], dtype = object)

    Interpolation method to generate nifti from mesh values has to be choosen : griddata or treequery. 

    Output folders are by default distinguished regarding type of values (e.g. deformed native mri grey values, displacements, deformations, distance to surface, nodal growth rate)
    '''
    print('\n')
    print(' ----- GENERATION OF BRAIN VALUES IMAGES IN THE NATIVE NIFTI FORMAT ----- \n')   
    print(' Parameters are: \n') 
    print('>> The native nifti enabling to generate the image space is: ' + str(args.nativenii) + '\n')
    print('>> The interpolator used to associate a native mri intensity to each mesh node is: itk.BSplineInterpolateImageFunction \n')
    print('>> The method used to plug brain values to the image space is: ' + str(args.method) + '\n')
    print('>> The griddata inherent interpolation method is: ' + str(args.gdmethod) + '\n')
    print('--- \n')

    #####################################
    # VISUALISATION INPUTS AND PARAMETERS
    #####################################
    # Input data 
    native_nii_path = args.nativenii
    binary_mask_path = args.mask #mask required for treequery method only

    # Output paths
    tq_displacements_path = args.outputdistq
    tq_mri_intensities_path = args.outputmritq
    gd_displacements_path = args.outputdisgd
    gd_mri_intensities_path = args.outputmrigd

    # Initial parameters
    parameters = np.load(args.parameters, allow_pickle = True)
    n_nodes = parameters[0]
    maxd = parameters[1]
    center_of_gravity = parameters[2]
    interpolation_method = args.method # 'griddata'; 'treequery'
    griddata_method = args.gdmethod
    initial_mri_intensity = np.zeros(n_nodes, dtype=np.float64)

    ################
    # MAIN PROGRAM #
    ################
    # Transfer the initial mri grey values to undeformed mesh (step = 0) and back to image space ("output control image")
    #####################################################################################################################
    data0 = np.load(args.coordinates + 'coordinates_0.npy', allow_pickle = True) # loading coordinates0 from simulation outputs .npy files
    coordinates0 = data0[1]

    denormalized_coordinates0 = coordinates_denormalization(coordinates0, n_nodes, center_of_gravity, maxd) 
    initial_mri_intensity =  mri_values_to_mesh(denormalized_coordinates0, native_nii_path) # plugging the mri intensity values of the native nifti to each node of the input mesh 
    
    control_image_name = "deformed_mri_0.nii"
    reg_control_image_name = "reg_deformed_mri_0.nii"

    if interpolation_method == 'treequery':
        tq_control_mri_array = tq_interpoled_values_array(denormalized_coordinates0, initial_mri_intensity, native_nii_path, binary_mask_path)
        values_to_nifti(tq_control_mri_array, tq_mri_intensities_path, native_nii_path, control_image_name, reg_control_image_name)

    elif interpolation_method == 'griddata':
        gd_control_mri_array = gd_interpoled_values_array(denormalized_coordinates0, initial_mri_intensity, native_nii_path, griddata_method)
        values_to_nifti(gd_control_mri_array, gd_mri_intensities_path, native_nii_path, control_image_name, reg_control_image_name)

    # Work on data exported from simulation at different steps%500:
    ##############################################################
    for r, d, npyfiles in os.walk(args.coordinates):
        for npyfile in sorted(npyfiles):
            if npyfile == 'coordinates_0.npy':
                continue

            data = np.load(args.coordinates + str(npyfile), allow_pickle = True) # loading coordinates from simulation outputs .npy files
            step = data[0]
            coordinates = data[1]
            print('Running step ' + str(step) + ' \n')

            # Coordinates denormalisation
            denormalized_coordinates = coordinates_denormalization(coordinates, n_nodes, center_of_gravity, maxd)

            # Calculation of nodal displacement
            nodal_displacement = np.zeros(n_nodes, dtype = np.float64)
            nodal_displacement = np.linalg.norm(denormalized_coordinates - denormalized_coordinates0, axis = 1)

            # Generates a interpolated nifti of the nodal displacements (at each step%500)     
            displ_output_name = "displacements_%d.nii"%(step)
            reg_displ_output_name = "reg_displacements_%d.nii"%(step)

            if interpolation_method == 'treequery':
                displacements_array_tq = tq_interpoled_values_array(denormalized_coordinates, nodal_displacement, native_nii_path, binary_mask_path)
                values_to_nifti(displacements_array_tq, tq_displacements_path, native_nii_path, displ_output_name, reg_displ_output_name)  

            elif interpolation_method == 'griddata':
                displacements_array_gd = gd_interpoled_values_array(denormalized_coordinates, nodal_displacement, native_nii_path, griddata_method)
                values_to_nifti(displacements_array_gd, gd_displacements_path, native_nii_path, displ_output_name, reg_displ_output_name)

            # Generates a interpolated nifti of the nodal native mri grey values (at each step%500)   
            mri_output_name = "deformed_mri_%d.nii"%(step)
            reg_mri_output_name = "reg_deformed_mri_%d.nii"%(step)

            if interpolation_method == 'treequery':
                mri_deformed_array_tq = tq_interpoled_values_array(denormalized_coordinates, initial_mri_intensity, native_nii_path, binary_mask_path)
                values_to_nifti(mri_deformed_array_tq, tq_mri_intensities_path, native_nii_path, mri_output_name, reg_mri_output_name)   

            elif interpolation_method == 'griddata':
                mri_deformed_array_gd = gd_interpoled_values_array(denormalized_coordinates, initial_mri_intensity, native_nii_path, griddata_method)
                values_to_nifti(mri_deformed_array_gd, gd_mri_intensities_path, native_nii_path, mri_output_name, reg_mri_output_name)  
            
    end_time_simulation = time.time() - start_time_initialization
    print('Visualization loop time taken was: ' + str(end_time_simulation))


