import vtk
import argparse


def nii_2_mesh(filename_nii, filename_stl, label):

    """
    Read a nifti file including a binary map of a segmented organ with label id = label. 
    Convert it to a smoothed mesh of type stl.

    filename_nii     : Input nifti binary map 
    filename_stl     : Output mesh name in stl format
    label            : segmented label id 
    """

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()
    
    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, label) # use surf.GenerateValues function if more than one contour is available in the file	
    surf.Update()
    
    #  auto Orient Normals
    surf_cor = vtk.vtkPolyDataNormals()
    surf_cor.SetInputConnection(surf.GetOutputPort())
    surf_cor.ConsistencyOn()
    surf_cor.AutoOrientNormalsOn()
    surf_cor.SplittingOff()
    surf_cor.Update()
    
    #smoothing the mesh
    smoother= vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf_cor.GetOutput())
    else:
        smoother.SetInputConnection(surf_cor.GetOutputPort())
    smoother.SetNumberOfIterations(60)
    smoother.NonManifoldSmoothingOn()
    #smoother.NormalizeCoordinatesOn() #The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()
     
    # save the output
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename_stl)
    writer.Write()

"""if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert Nifti files to stl formats')
    parser.add_argument('-i','--input', help='Input Nifti file', nargs='?', type=str, required=True)
    parser.add_argument('-o','--output', help='Ouput stl mesh', nargs='?', type=str, required=False)
    parser.add_argument('-l','--label', help='Number of label', nargs='?', type=int)
    args = parser.parse_args()
    nii_2_mesh(args.input, args.output, args.label)"""


#filename_nii =  '/home/x17wang/Data/prm001/prm001_40w_Rwhite.nii'
#filename_stl = '/home/x17wang/Exp/prm001/prm001_40w_Rwhite.stl'
#label = 0
