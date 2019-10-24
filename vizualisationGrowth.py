import slam.io as sio
import trimesh
from curvatureCoarse import curvatureTopologic
import slam.plot as splt

# Visualization of the biomechanical model simulations

folder='/home/INT/lefevre.j/Documents/Codes/Python/BrainGrowth/res/sphere5/pov_H0.042000AT1.829000/'
mesh_file = 'B15000.stl'

mesh = trimesh.load(folder+mesh_file)
curvature = curvatureTopologic(mesh)
#mesh.show()
splt.pyglet_plot(mesh, curvature)