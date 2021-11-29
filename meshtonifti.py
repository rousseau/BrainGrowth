from itk.support.extras import spacing
import numpy as np
import argparse
import nibabel as nib
from scipy.interpolate import griddata
import meshio
from scipy.spatial import cKDTree 
import itk

def mesh_to_image_with_ref(coordinates, values, reference_nii_path, interpolation_method, output_path):
    """Interpolate the mesh nodal values onto a 3d reference-image-shape grid."""
    # Get data from reference image 
    reference_img = nib.load(reference_nii_path)  
    reference_affine = reference_img.affine 
    shape_img = np.shape(reference_img) 
    shape_i, shape_j, shape_k = shape_img[0], shape_img[1], shape_img[2] 

    # Coordinates : mesh coordinates expressed in world coordinates
    matrix_wc_2_img = np.linalg.inv(reference_affine)
    mesh_coordinates = np.concatenate((coordinates,np.ones((coordinates.shape[0],1))),axis=1)

    # Transform coordinates to image coordinates (apply inverse affine matrix)
    mesh_coordinates_in_image_space = (matrix_wc_2_img.dot(mesh_coordinates.T)).T

    # Apply interpolation on the sparse coordinate in image space
    grid_X, grid_Y, grid_Z = np.mgrid[0:shape_i, 0:shape_j, 0:shape_k]
    interpolation_array = griddata(mesh_coordinates_in_image_space[:,0:3], values, (grid_X, grid_Y, grid_Z), method=interpolation_method) 
    reg_values_img = nib.Nifti1Image(interpolation_array, affine=reference_affine)
    nib.save(reg_values_img, output_path)

    #print(np.min(mesh_coordinates_in_image_space))
    #print(np.max(mesh_coordinates_in_image_space))

    return interpolation_array

def mesh_to_image_with_ref_tq(mesh_coordinates, values, reference_nii_path, output_path):
    """Interpolate the mesh nodal values onto a 3d image array by tree querying the array physical coordinates. (for comparison with griddata)"""

    reference_img_itk = itk.imread(reference_nii_path) 
    shape_k_raw, shape_j_raw, shape_i_raw = np.shape(reference_img_itk) #  inversed axis by itk, compared to acquisition : (itk k, j, i) corresponds to (raw nifti i, j, k)
    interpolation_array = np.zeros((shape_k_raw, shape_j_raw, shape_i_raw)) 

    tree = cKDTree(mesh_coordinates)

    for k in range(shape_k_raw):
        for j in range(shape_j_raw):
            for i in range(shape_i_raw):               
                #fetch the closest mesh coordinate from the voxel
                nearest_neighbor = tree.query(reference_img_itk.TransformIndexToPhysicalPoint((i, j, k))) #world system
                tex = values[nearest_neighbor[1]]
                #plug coord to the voxel
                interpolation_array[k][j][i] = tex # itk requires indices inversion before writing
    
    #itk nifti output 
    interpolation_img = itk.GetImageFromArray(interpolation_array)
    interpolation_img.SetSpacing(reference_img_itk.GetSpacing())
    interpolation_img.SetOrigin(reference_img_itk.GetOrigin())
    interpolation_img.SetDirection(reference_img_itk.GetDirection())
    itk.imwrite(interpolation_img, output_path)
   
    return interpolation_array  

def mesh_to_generated_image(mesh_coordinates, values, interpolation_method, output_path):
    """Interpolate the mesh nodal values onto an anisotropic generated 3d grid (Generate an isotropic nifti from sparse mesh coordinates).
    > Provided mesh (fixed lengths)
    > Fixed image resolution -- TO BE CHOOSEN
    > Adaptative image shape in 3dim.
    > image margin -- TO BE CHOOSEN
    """
    # image parameters
    reso = 0.01
    margin = 4 # 2 voxels on both sides

    # Calculate mesh lengths
    length_x = max(mesh_coordinates[:,0]) - min(mesh_coordinates[:,0])
    length_y = max(mesh_coordinates[:,1]) - min(mesh_coordinates[:,1])
    length_z = max(mesh_coordinates[:,2]) - min(mesh_coordinates[:,2])

    # Generate the shape of the anisotropic interpolation grid
    shape_x, shape_y, shape_z = length_x/reso + margin, length_y/reso + margin, length_z/reso + margin

    # Localize the geometrical center of the mesh
    mesh_center_x, mesh_center_y, mesh_center_z = 0.5*(min(mesh_coordinates[:,0]) + max(mesh_coordinates[:,0])), 0.5*(min(mesh_coordinates[:,1]) + max(mesh_coordinates[:,1])), 0.5*(min(mesh_coordinates[:,2]) + max(mesh_coordinates[:,2]))

    # Calculate the translation vector -B between the mesh coordinates and the image space
    coord_img_center_x, coord_img_center_y, coord_img_center_z = 0.5*shape_x*reso, 0.5*shape_y*reso, 0.5*shape_z*reso
    mesh_to_img_vect = [coord_img_center_x - mesh_center_x, coord_img_center_y - mesh_center_y, coord_img_center_z - mesh_center_z]

    # Apply translation vector to mesh coordinates ( (x,y,z) = M@(i,j,k) + B )
    mesh_coord_in_img_space = mesh_coordinates.copy() 
    mesh_coord_in_img_space[:,0] += mesh_to_img_vect[0] # -B[0]
    mesh_coord_in_img_space[:,1] += mesh_to_img_vect[1] # -B[1]
    mesh_coord_in_img_space[:,2] += mesh_to_img_vect[2] # -B[2]
    mesh_coord_in_img_space /= reso # inv(M)

    # Build the nifti affine 
    generated_affine = np.zeros((4,4), dtype=np.float64)
    generated_affine[0,0] = generated_affine[1,1] = generated_affine[2,2] = reso
    generated_affine[0,3] = - mesh_to_img_vect[0]
    generated_affine[1,3] = - mesh_to_img_vect[1]
    generated_affine[2,3] = - mesh_to_img_vect[2]
    generated_affine[3,3] = 1

    # Apply interpolation on the sparse coordinate
    grid_X, grid_Y, grid_Z = np.mgrid[0:shape_x, 0:shape_y, 0:shape_z]
    interpolation_array = griddata(mesh_coord_in_img_space, values, (grid_X, grid_Y, grid_Z), method=interpolation_method) 
    values_img = nib.Nifti1Image(interpolation_array, affine=generated_affine) #Calculates the affine between image and coordinates : translation center of gravity
    nib.save(values_img, output_path)

    return interpolation_array, shape_x, shape_y, shape_z, generated_affine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation of brain values (nifti)')
    parser.add_argument('-i', '--input', help='Path to input vtk file (step, coordinates, brain values)', default='/home/latim/GitHub/BrainGrowth/BrainGrowth/res/dhcpbrain_vtk/brain_dhcp_1500.vtk', type=str, required=False)
    parser.add_argument('-r', '--reference', help='Reference  nifti', type=str, default='/home/latim/Database/dhcp/Vf/dhcp.nii', required=False)
    parser.add_argument('-o', '--output', help='Path to output nifti file', type=str, default='/home/latim/GitHub/BrainGrowth/BrainGrowth/res/dhcpbrain_vtk/dhcpbrain_ras_1500_displ_tq.nii.gz', required=False)
    parser.add_argument('-m', '--method', help='griddata interpolation method: nearest; linear; cubic', type=str, default='nearest', required=False)
    args = parser.parse_args()

    # MAIN PROGRAM 
    # load coordinates and associated values from the input vtk file.
    mesh = meshio.read(args.input)
    mesh_coordinates = mesh.points # list of nodes coordinates
    brain_values = mesh.point_data['Displacement'] # TO BE UDPATED BEFORE RUNNING: 'Displacement'; 'Distance_to_surface'; 'Growth_ponderation' (gr) ; 
    # 'Tangential_growth_wg_term' (gm(y)); 'Tangential_growth' (g(y,t)) 

    """
    # REFERENCE IMAGE + griddata interpolation. Generate a interpolated nifti of the nodal values. 
    mesh_to_image_with_ref(mesh_coordinates, brain_values, args.reference, args.method, args.output)  
    print('\n The nifti "' + str(args.output) + '" has been generated. \n')

    """
    # REFERENCE IMAGE + treequery interpolation. Generate a interpolated nifti of the nodal values.
    #output_path_reg = '/home/latim/GitHub/BrainGrowth/BrainGrowth/res/dhcpbrain_vtk/dhcpbrain_ras_1500_displ_tq_reg.nii.gz' # TO BE UDPATED BEFORE RUNNING
    mesh_to_image_with_ref_tq(mesh_coordinates, brain_values, args.reference, args.output)
    print('\n The nifti "' + str(args.output) + '" has been generated. \n') 
    
    """
    # GENERATED IMAGE + griddata interpolation. Generate a interpolated nifti of the nodal values. 
    mesh_to_generated_image(mesh_coordinates, brain_values, args.method, args.output)
    print('\n The nifti "' + str(args.output) + '" has been generated. \n') 
    """