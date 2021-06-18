import numpy as np
from nii_2_mesh_conversion import nii_2_mesh
import pygalmesh 
import vtk
from vtk.util.numpy_support import vtk_to_numpy


filename_nii = '/home/benjamin/Documents/git_repos/BrainGrowth/cache/template_T2.nii'

filename_stl = '/home/benjamin/Documents/git_repos/BrainGrowth/cache/template_T2.stl'

nii_2_mesh(filename_nii, filename_stl, 0)
#file loading okay, but result crap. Easy visualisation of STL ?

#take a surface mesh and fill it with tetrahedrons
    #mesh formats taken into account by pymesh ? STL okay
    #but pymesh is not the default pip, you have to install it by hand (not the end of the world though). Installation hard
    #installation pymesh cancelled because not standard = clunky
    #meshpy. Only obj object from Meshlab
    #pyglamesh, would do the trick but very heavy


#take the filled volume and feed it back as a numpy array
#from vtk to numpy array ?

filename_mesh = "/home/benjamin/Documents/git_repos/BrainGrowth/data/sphere5.mesh"
inpt = "/home/benjamin/Documents/git_repos/BrainGrowth/data/surf_sphere.stl"

mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
    inpt,
    min_facet_angle = 25.0,
    max_radius_surface_delaunay_ball = 0.15,
    max_facet_distance = 0.008,
    max_circumradius_edge_ratio = 3.0,
    verbose = False
    
    )

mesh.write("output.vtk")

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("/home/benjamin/Documents/git_repos/BrainGrowth/output.vtk")
reader.Update()
data = reader.GetOutput()

def vtk_to_numpy ():
    pass