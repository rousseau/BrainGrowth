import slam.io as sio
import trimesh
from curvatureCoarse import curvatureTopologic
import slam.plot as splt
import numpy as np
import slam.curvature as scurv
import trimesh as tr

# Visualization of the biomechanical model simulations

# Curvature computed with the umbrella operator
folder='/home/benjamin/Documents/git_repos/BrainGrowth/res/sphere5/pov_H0.042000AT1.829000/'
mesh_file = 'B15000.stl'

mesh = trimesh.load(folder+mesh_file)
curvature = curvatureTopologic(mesh)
mesh.show()
splt.pyglet_plot(mesh, curvature)


# Mean curvature computed by Rusinkiewicz estimation NOUVELLE COURBURE à UTILISER, utiliser avec visu slam à visu avec pyglet_plot(maillage, courbure)
folder='/home/benjamin/Documents/git_repos/BrainGrowth/res/sphere5/pov_H0.042000AT1.829000/'
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

splt.pyglet_plot(mesh, curvature_mean)


# Sulcal depth: OLD function ?
folder='/home/benjamin/Documents/git_repos/BrainGrowth/res/sphere5/pov_H0.042000AT1.829000/'
mesh_file_2 = 'B0.stl'
mesh_o = trimesh.load(folder+mesh_file_2)
m = []
txt_file_2 = "B0.txt"
with open(folder + txt_file_2) as inputfile:
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

inters_1 = np.zeros((np.size(mesh_o.vertices, 0), 3), dtype = np.float32)
A = np.zeros((np.size(steps), np.size(mesh_o.vertices, 0)), dtype = np.float32)

for i in steps:
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
    mesh_2 = trimesh.convex.convex_hull(mesh1, qhull_options='QbB Pp Qt')
    for j in range(np.size(mesh_o.vertices, 0)):
        endpoints = np.array([mesh_o.vertices[j,:], mesh_o.vertices[j,:]+10000*(mesh1.vertices[j,:]-mesh_o.vertices[j,:])])
        for k in range(np.size(mesh_2.faces, 0)):
            plane_normal = np.cross(mesh_2.vertices[mesh_2.faces[k,1], :]-mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :]-mesh_2.vertices[mesh_2.faces[k,0], :])
            intersections, valid = trimesh.intersections.plane_lines(mesh_2.vertices[mesh_2.faces[k,0], :], plane_normal, endpoints, line_segments=True)
            if valid == True:
                Area1 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,1], :] - intersections))  
                Area2 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                Area3 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                Area4 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :] - mesh_2.vertices[mesh_2.faces[k,0], :]))
                if np.absolute(Area1 + Area2 + Area3 - Area4) < 1e-10:
                    inters_1[j, :] = intersections   
    
    A[q, :] = np.sqrt((mesh1.vertices[:,0] - inters_1[:, 0])**2+(mesh1.vertices[:,1] - inters_1[:, 1])**2+(mesh1.vertices[:,2] - inters_1[:, 2])**2)

    q += 1


# 3D GI
folder='/home/benjamin/Documents/git_repos/BrainGrowth/res/sphere5/pov_H0.042000AT1.829000/'
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

def stress_visualisation (path, Ft):
    mesh = tr.load(path)
    for i in range(len(mesh.visual.vertex_colors)):
        mesh.visual.vertex_colors[i] = [Ft[i][0]*10000, Ft[i][1]*10000, Ft[i][2]*10000, 1]
    mesh.export('/home/benjamin/Documents/mesh.ply')
    return mesh

def export_displacement(coordinates, coordinates_initial):
    """takes two sets of coordinate and return nothing because I am an ididiot"""
    return np.abs(coordinates - coordinates_initial)

