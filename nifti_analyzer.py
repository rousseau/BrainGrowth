import numpy as np
import nibabel as nib 
import itk

def nii_nibabel(nii_path): #https://nipy.org/nibabel/coordinate_systems.html
    nib_image = nib.load(nii_path)
    affine = nib_image.affine
    header = nib_image.header
    shape = np.shape(nib_image) 
    print('NIBABEL shape : \n' + str(shape) + '\n')
    print('NIBABEL affine : \n' + str(affine) + '\n')
    print('NIBABEL header : \n' + str(header) + '\n')

    return shape, affine, header

def nii_itk(nii_path): #https://discourse.itk.org/t/solved-transformindextophysicalpoint-manually/1031/3
    reference_img_itk = itk.imread(nii_path) 
    print('ITK size ' + str(np.shape(reference_img_itk))) #inverted compared to nibabel (and acquisition truth : i,j,k --> k,j,i)
    direction_cosine_matrix = reference_img_itk.GetDirection().GetVnlMatrix()
    for i in range(3):
        for j in range(3):
            print('dir_mat_' + str(i) + str(j) + ' = ' + str(direction_cosine_matrix.get(i,j)))
    print('ITK ref_origin : \n' + str(reference_img_itk.GetOrigin()))
    print('ITK ref_spacing : \n' + str(reference_img_itk.GetSpacing()))

    #shape_k_raw, shape_j_raw, shape_i_raw = np.shape(reference_img_itk)
    #interpolation_array = np.zeros((shape_k_raw, shape_j_raw, shape_i_raw)) 
    #print((shape_k_raw, shape_j_raw, shape_i_raw))
    #print(np.shape(interpolation_array))
    #print(len(interpolation_array))

    return direction_cosine_matrix

if __name__ == "__main__":
    reference_nii_path = '/home/latim/Database/dhcp/Vf/dhcp.nii' #reference image for meshtonifti tool (from dhcp database)
    meshtonifti_result_nii_gd = '/home/latim/GitHub/BrainGrowth/BrainGrowth/res/dhcpbrain_vtk/brain_dhcp_ras_1500_displ.nii.gz'
    meshtonifti_result_nii_tq_transformed = '/home/latim/GitHub/BrainGrowth/BrainGrowth/res/dhcpbrain_vtk/dhcpbrain_ras_1500_displ_tq4.nii.gz'


    # nibabel analysis
    print('nifti analysis with NIBABEL for' + str(reference_nii_path) + ' : \n')
    nii_nibabel(reference_nii_path)
    print('nifti analysis with NIBABEL for' + str(meshtonifti_result_nii_gd) + ' : \n')
    nii_nibabel(meshtonifti_result_nii_gd)
    print('nifti analysis with NIBABEL for' + str(meshtonifti_result_nii_tq_transformed) + ' : \n')
    nii_nibabel(meshtonifti_result_nii_tq_transformed)

    # itk analysis
    print('nifti analysis with ITK for' + str(reference_nii_path) + ' : \n')
    nii_itk(reference_nii_path)
    print('nifti analysis with ITK for' + str(meshtonifti_result_nii_gd) + ' : \n')
    nii_itk(meshtonifti_result_nii_gd)
    print('nifti analysis with ITK for' + str(meshtonifti_result_nii_tq_transformed) + ' : \n')
    nii_itk(meshtonifti_result_nii_tq_transformed)
