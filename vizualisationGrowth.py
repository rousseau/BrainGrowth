import slam.io as sio
import trimesh
from curvatureCoarse import curvatureTopologic
import slam.plot as splt
import numpy as np
import slam.curvature as scurv
from numba import prange

# Visualization of the biomechanical model simulations

# Curvature computed with the umbrella operator
folder='/home/INT/lefevre.j/Documents/Codes/Python/BrainGrowth/res/sphere5/pov_H0.042000AT1.829000/'
mesh_file = 'B15000.stl'

mesh = trimesh.load(folder+mesh_file)
curvature = curvatureTopologic(mesh)
#mesh.show()
splt.pyglet_plot(mesh, curvature)


# Mean curvature computed by Rusinkiewicz estimation
""" Rusinkiewicz, Szymon. "Estimating curvatures and their derivatives on triangle meshes." Proceedings. 2nd International Symposium 
on 3D Data Processing, Visualization and Transmission, 2004. 3DPVT 2004.. IEEE, 2004."""

folder='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/pov_H0.042000AT1.829000/'
steps = np.arange(0, 45000, 1000)
A = []
for i in steps:
    mesh_file = "B%d.gii"%(i)
    mesh =  sio.load_mesh(folder+mesh_file)

    Area = 0.0
    for j in range(len(mesh.faces)):
      Ntmp = np.cross(mesh.vertices[mesh.faces[j,1]] - mesh.vertices[mesh.faces[j,0]], mesh.vertices[mesh.faces[j,2]] - mesh.vertices[mesh.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)

    mesh.apply_transform(mesh.principal_inertia_transform)
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
    curvature = 0.5 * (PrincipalCurvatures[0, :]*np.sqrt(Area) + PrincipalCurvatures[1, :]*np.sqrt(Area))
    curvature_mean = np.mean(np.absolute(curvature))
    A.append(curvature_mean)


# Sulcal depth
folder='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_realsphpt/pov_H0.046100AT1.829000/'

# Load the initial mesh (.stl and .txt files)
mesh_file_o = 'B0.stl'
mesh_o = trimesh.load(folder+mesh_file_o)
m = []
txt_file_o = "B0.txt"
with open(folder + txt_file_o) as inputfile:
    for line in inputfile:
        m.append(line.strip().split(' '))
    for j in range(len(m)):
        m[j] = list(filter(None, m[j]))
        m[j] = np.array([float(a) for a in m[j]])
vertices = m[1:(int(m[0][0])+1):1]
faces = m[(int(m[0][0])+2):int(m[int(m[0][0])+1][0]+int(m[0][0])+2):1]
vertices = np.array(vertices)
faces = np.array(faces).astype(int) - 1
mesh_o.faces = faces
mesh_o.vertices = vertices/10

steps = np.array([14500])
q = 0

# Create matrices of intersection points and depths (EUD distances)
inters = np.zeros((np.size(mesh_o.vertices, 0), 3), dtype = np.float32)
depth = np.zeros((np.size(steps), np.size(mesh_o.vertices, 0)), dtype = np.float32)

for i in steps:
    # Load the deformed mesh (.stl and .txt files)
    mesh_file = "B%d.stl"%(i)
    mesh1 =  trimesh.load(folder+mesh_file)
    m = []
    txt_file = "B%d.txt"%(i)
    with open(folder + txt_file) as inputfile:
        for line in inputfile:
            m.append(line.strip().split(' '))
        for j in range(len(m)):
            m[j] = list(filter(None, m[j]))
            m[j] = np.array([float(a) for a in m[j]])
    vertices = m[1:(int(m[0][0])+1):1]
    faces = m[(int(m[0][0])+2):int(m[int(m[0][0])+1][0]+int(m[0][0])+2):1]
    vertices = np.array(vertices)
    faces = np.array(faces).astype(int) - 1
    mesh1.faces = faces
    mesh1.vertices = vertices
	
    # Calculate the convex hull of the deformed mesh
    mesh_2 = trimesh.convex.convex_hull(mesh1, qhull_options='QbB Pp Qt')
	
    # Iterate through all vertices of mesh
    for j in prange(np.size(mesh_o.vertices, 0)):
	# Calculate the endpoints of line segment
        endpoints = np.array([mesh_o.vertices[j,:], mesh_o.vertices[j,:]+10000*(mesh1.vertices[j,:]-mesh_o.vertices[j,:])])
        # Iterate through all triangles of convex hull
        for k in prange(np.size(mesh_2.faces, 0)):
            # Calculate the normal of the plane where the triangle lies
            plane_normal = np.cross(mesh_2.vertices[mesh_2.faces[k,1], :]-mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :]-mesh_2.vertices[mesh_2.faces[k,0], :])
            # Calculate the intersection of line segment and plane
            intersections, valid = trimesh.intersections.plane_lines(mesh_2.vertices[mesh_2.faces[k,0], :], plane_normal, endpoints, line_segments=True) # line_segments(bool) if True, only returns intersections as valid if vertices from endpoints are on different sides of the plane
            # Indicate whether a valid intersection exists for input line segment
            if valid == True:
	        # Calculate the area of the intersection point and any two points of the triangle 
                Area1 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,1], :] - intersections))  
                Area2 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                Area3 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                # Calculate the area of the triangle 
                Area4 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :] - mesh_2.vertices[mesh_2.faces[k,0], :]))
                # Determine whether the intersection point is inside the triangle
                if np.absolute(Area1 + Area2 + Area3 - Area4) < 1e-10:
	            # Return the intersection point
                    inters[j, :] = intersections
                    break
		
    # Calculate the EUD distance between the vertices of deformed mesh and the intersection points on the convex hull
    depth[q, :] = np.sqrt((mesh1.vertices[:,0] - inters[:, 0])**2+(mesh1.vertices[:,1] - inters[:, 1])**2+(mesh1.vertices[:,2] - inters[:, 2])**2)

    q += 1


# 3D Gyrification Index (defined as the ratio of the cortical surface area to the area of its smooth convex hull)
folder='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/pov_H0.042000AT1.829000/'
mesh_file_2 = 'B0.gii'
G1 = []
steps = np.arange(0, 45000, 1000)
for i in steps:
    mesh_file = "B%d.gii"%(i)
    mesh =  sio.load_mesh(folder+mesh_file)
    mesh_2 = sio.load_mesh(folder+mesh_file_2)

    L1=(max(mesh.vertices[:,0])-min(mesh.vertices[:,0]))/(max(mesh_2.vertices[:,0])-min(mesh_2.vertices[:,0]))
    L2=(max(mesh.vertices[:,1])-min(mesh.vertices[:,1]))/(max(mesh_2.vertices[:,1])-min(mesh_2.vertices[:,1]))
    L3=(max(mesh.vertices[:,2])-min(mesh.vertices[:,2]))/(max(mesh_2.vertices[:,2])-min(mesh_2.vertices[:,2]))
    mesh_2.vertices[:,0]=L1*mesh_2.vertices[:,0]
    mesh_2.vertices[:,1]=L2*mesh_2.vertices[:,1]
    mesh_2.vertices[:,2]=L3*mesh_2.vertices[:,2]

    Area = 0.0
    
    Ntmp = np.cross(mesh.vertices[mesh.faces[:,1]] - mesh.vertices[mesh.faces[:,0]], mesh.vertices[mesh.faces[:,2]] - mesh.vertices[mesh.faces[:,0]])
    Area += 0.5*np.linalg.norm(Ntmp)
    Area_2 = 0.0
    
    Ntmp_2 = np.cross(mesh_2.vertices[mesh_2.faces[:,1]] - mesh_2.vertices[mesh_2.faces[:,0]], mesh_2.vertices[mesh_2.faces[:,2]] - mesh_2.vertices[mesh_2.faces[:,0]])
    Area_2 += 0.5*np.linalg.norm(Ntmp_2)
      
    GI_1 = Area/Area_2
    
    G1.append(GI_1)
