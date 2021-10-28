import numpy as np
import argparse

# Local modules
from values_interpolation import mesh_to_image
from nii_generation import nifti_writing 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation of brain values (nifti)')
    parser.add_argument('-i', '--input', help='Path to input vtk file (step, coordinates, brain values)', default='./values_mesh/displacements/displacements-mesh_7000.npy', type=str, required=False)
    parser.add_argument('-r', '--reference', help='Reference  nifti', type=str, default='/home/latim/braingrowth_rousseau/FetalAtlas/mesh_to_image_testFolder/template_T2_reduced_z.nii', required=False)
    parser.add_argument('-o', '--output', help='Path to output nifti file', type=str, default='./tests/displacements_7000.nii', required=False)
    parser.add_argument('-m', '--method', help='griddata interpolation method: nearest; linear; cubic', type=str, default='linear', required=False)
    args = parser.parse_args()

    # MAIN PROGRAM 
    # load coordinates and associated values from the input vtk file
    data = np.load(args.input, allow_pickle = True) 
    coordinates = data[1]
    brain_values = data[2]

    # Generate a interpolated nifti of the nodal values  
    values_array_gd = mesh_to_image(coordinates, brain_values, args.reference, args.method)
    nifti_writing(args.reference, values_array_gd, args.output)  
            
    print('\n The output nifti has been generated. \n')