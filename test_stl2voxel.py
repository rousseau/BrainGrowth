#python3 test_stl2voxel.py -i inputFilePath(.stl) -o outputFilePath(.nii) -r referenceFilePath(.nii)

import argparse
import os.path
import numpy as np
import slice
import stl_reader
import perimeter
from util import padVoxelArray
import nibabel as nib

def doExport(inputFilePath, outputFilePath, referenceFilePath, resolution):
    mesh = list(stl_reader.read_stl_verticies(inputFilePath))
    (scale, shift, bounding_box) = slice.calculateScaleAndShift(mesh, resolution)
    mesh = list(slice.scaleAndShiftMesh(mesh, scale, shift))
    #Note: vol should be addressed with vol[z][x][y]
    vol = np.zeros((bounding_box[2],bounding_box[0],bounding_box[1]), dtype=bool)
    for height in range(bounding_box[2]):
        print('Processing layer %d/%d'%(height+1,bounding_box[2]))
        lines = slice.toIntersectingLines(mesh, height)
        prepixel = np.zeros((bounding_box[0], bounding_box[1]), dtype=bool)
        perimeter.linesToVoxels(lines, prepixel)
        vol[height] = prepixel
    vol, bounding_box = padVoxelArray(vol)
    outputFilePattern, outputFileExtension = os.path.splitext(outputFilePath)
    #inputImage = '/home/x17wang/Data/prm001/prm001_30w_Rwhite.nii'  #Path of image corresponding to the starting mesh
    exportNii(vol, originImagePath, outputFilePath)

def exportNii(voxels, originImagePath, outputFilePath):
    voxels = voxels.astype(np.int16)
    nii = nib.load(originImagePath)
    img = nib.Nifti1Image(voxels, nii.affine)
    nib.save(img, outputFilePath)

def file_choices(choices,fname):
    filename, ext = os.path.splitext(fname)
    if ext == '' or ext not in choices:
        if len(choices) == 1:
            parser.error('%s doesn\'t end with %s'%(fname,choices))
        else:
            parser.error('%s doesn\'t end with one of %s'%(fname,choices))
    return fname

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STL files to voxels')
    parser.add_argument('-i','--input', help='Input STL file', nargs='?', type=lambda s:file_choices(('.stl'),s))
    parser.add_argument('-o','--output', help='Ouput Nifti image', nargs='?', type=lambda s:file_choices(('.nii'),s))
    parser.add_argument('-r','--reference', help='Reference Nifti image', nargs='?', type=lambda s:file_choices(('.nii'),s))
    args = parser.parse_args()
    doExport(args.input, args.output, args.reference, 100)
