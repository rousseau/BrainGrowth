import numpy as np
import itk

def nifti_itk_analyzer(itk_img):
    """ 
    Return the origin coordinates (displacement vector from ijk space), 
    the resolution and the direction matrix of the input nifti in the LPS+ world system.
    """
    origin = itk_img.GetOrigin()
    spacing = itk_img.GetSpacing()
    direction_cosine_matrix = itk_img.GetDirection().GetVnlMatrix() #to LPS coordinate system (itk)
    d = np.empty([3, 3])
    for i in range(3):
        for j in range(3):
            d[i][j] = direction_cosine_matrix.get(i,j)
            
    return origin, spacing, d

def itk_coordinate_orientation_system_analyzer(itk_img):
    """   
    Return orientation of the input nifti in the LPS+ coordinate system, from reading the direction cosines.
    Convention:  
    dir_mat_00 = 1.0 --> R(1) / L(-1)
    dir_mat_01 = 0.0
    dir_mat_02 = 0.0
    dir_mat_10 = 0.0
    dir_mat_11 = 1.0 --> A(1) / P(-1)
    dir_mat_12 = 0.0
    dir_mat_20 = 0.0
    dir_mat_21 = 0.0
    dir_mat_22 = 1.0 --> I(1) / S(-1)
    """
    orientation = np.empty(3, dtype=object)
    origin, spacing, d = nifti_itk_analyzer(itk_img)
    if d[0][0] == 1:
        orientation[0] = 'R'
    else:
        orientation[0] = 'L'
    if d[1][1] == 1:
        orientation[1] = 'A'
    else:
        orientation[1] = 'P'
    if d[2][2] == 1:
        orientation[2] = 'I'
    else:
        orientation[2] = 'S'
    
    return orientation

def spatial_orientation_adapter(x, y, z): 
    """ 
    Return index of maximum value x,y,z (to be applied to analyze the direction cosines matrix).
    Converted in Python from https://itk.org/Doxygen/html/itkSpatialOrientationAdapter_8h_source.html.
    """
    threshold_cosine_value = 0.001
    
    absX = abs(x)
    absY = abs(y)
    absZ = abs(z)
    
    if ((absX > threshold_cosine_value) and (absX > absY) and (absX > absZ)):
        return 0
    elif ((absY > threshold_cosine_value) and (absY > absX) and (absY > absZ)):
        return 1
    elif ((absZ > threshold_cosine_value) and (absZ > absX) and (absZ > absY)):
        return 2
    else:
        return 0 # if all equal
    
def sign(x):
    if x < 0:
        return -1
    else:
        return 1

def itk_lps_to_ras_transformation(reference_image_itk):
    """
    Transform the direction cosine matrix and origin of the input nifti from LPS+ (itk convention).
    Return itk RAS+ direction matrix and origin. 
    to RAS+ (convention used by TransformIndexToPhysicalPoint method), to be used in itk.
    """
    image_dir = reference_image_itk.GetDirection().GetVnlMatrix()
    origin = reference_image_itk.GetOrigin()
    
    dominant_axis_rl = spatial_orientation_adapter(image_dir.get(0,0),image_dir.get(1,0),image_dir.get(2,0))
    sign_rl = sign(image_dir.get(dominant_axis_rl,0))
    dominant_axis_ap = spatial_orientation_adapter(image_dir.get(0,1),image_dir.get(1,1),image_dir.get(2,1))
    sign_ap = sign(image_dir.get(dominant_axis_ap,1))
    dominant_axis_si = spatial_orientation_adapter(image_dir.get(0,2),image_dir.get(1,2),image_dir.get(2,2))
    sign_si = sign(image_dir.get(dominant_axis_si,2))

    to_ras_transformation_matrix = np.eye(3)
    
    if (sign_rl == 1):
        to_ras_transformation_matrix[dominant_axis_rl][dominant_axis_rl] = -1.0
        origin[dominant_axis_rl] *= -1.0

    if (sign_ap == -1):
        to_ras_transformation_matrix[dominant_axis_ap][dominant_axis_ap] = -1.0
        origin[dominant_axis_ap] *= -1.0

    if (sign_si == 1):
        to_ras_transformation_matrix[dominant_axis_si][dominant_axis_si] = 1.0
        origin[dominant_axis_si] *= 1.0
    
    return to_ras_transformation_matrix, origin

def apply_lps_ras_transformation(reference_image_itk):

    origin0, spacing0, d0 = nifti_itk_analyzer(reference_image_itk)
    to_ras_transformation, origin_to_ras = itk_lps_to_ras_transformation(reference_image_itk)

    direction_ras_array = d0.dot(to_ras_transformation) # LPS+ to RAS+
    reference_image_itk.SetDirection(direction_ras_array)
    reference_image_itk.SetOrigin(origin_to_ras)

    return reference_image_itk

if __name__ == '__main__':
    #input image
    reference_nii_path = '../data/data_anne/dhcp/dhcp.nii'
    reference_image_itk = itk.imread(reference_nii_path)

    #analysis of the initial nifti
    orientation0 = itk_coordinate_orientation_system_analyzer(reference_image_itk)
    origin0, spacing0, d0 = nifti_itk_analyzer(reference_image_itk)
    print(orientation0)
    print(d0)
    print(origin0)
    print(spacing0)

    #transform LPS+ to RAS+ itk direction matrix and origin
    reference_image_itk_ras = apply_lps_ras_transformation(reference_image_itk)

    # analysis of the nifti reoriented to RAS+
    orientation_ras = itk_coordinate_orientation_system_analyzer(reference_image_itk_ras)
    origin_ras, spacing_ras, d_ras = nifti_itk_analyzer(reference_image_itk_ras)
    print('\n')
    print(orientation_ras)
    print(d_ras)
    print(origin_ras)
    print(spacing_ras)

    #itk.imwrite(reference_image_itk, '../data/data_anne/dhcp/dhcp_ras_itk.nii')