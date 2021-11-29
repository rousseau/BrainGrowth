import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import itk
import trimesh as tr
import scipy

"""
TODO: save texture to file to be reused
"""

def mesh_to_nii(coordinates, tex, nii_path, output_path = './res/output.nii.gz', threshold = 0.1):
    tree = scipy.spatial.cKDTree(coordinates)
    image = itk.imread(nii_path)
    array = itk.GetArrayFromImage(image)
    
    output = np.zeros ((np.asarray(image).shape))
    
    dim = [-1, -1, -1]
    
    #for each pixel of the image:
    #passer par iterateur
    for i in array:
        dim[0] += 1
        dim[1] = -1
        for j in i:
            dim[1] += 1
            dim[2] = -1
            for k in j:
                dim[2] += 1
                if k < threshold:
                    pass
                else:
                    coord = image.TransformIndexToPhysicalPoint(dim)
                    #check the closest mesh value, this function can take a whole list of points
                    distance, node_idx = tree.query(coord)
                    value = tex[node_idx] #THIS LINE IS NOT CORRECT, CHANGE IF NECESSARY
                    
                    #THIS LINE MAY EED ADJUSTMENT
                    output[dim[0], dim[1], dim[2]] = value
                    
    #write the result of a np array in a nifti
    ni_img = nb.Nifti1Image(output, affine=np.eye(4))
    
    nb.save(ni_img, output_path)   