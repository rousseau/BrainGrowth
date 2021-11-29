# Profondeur
folder='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_realsphpt/pov_H0.046100AT1.829000/'
folder_2='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_22b2/pov_H0.037700AT1.829000/'
folder_3='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_27b2/pov_H0.030700AT1.829000/'
"""folder_4='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/pov_H0.042000AT1.829000/'
folder_5='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/pov_H0.052000AT1.829000/'
folder_6='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/pov_H0.062000AT1.829000/'
folder_7='/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/pov_H0.082000AT1.829000/'"""
G1 = []
G2 = []
G3 = []
G4 = []
G5 = []
G6 = []
G7 = []
depth1 = []
depth2 = []
depth3 = []
depth4 = []
depth5 = []
A_new = []
B_new = []
C_new = []
D_new = []
E_new = []
mesh_file_2 = 'B0.stl'
mesh_ooo = trimesh.load(folder+mesh_file_2)
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
mesh_ooo.faces = faces
mesh_ooo.vertices = vertices/10

mesh_o = trimesh.load(folder_2+mesh_file_2)
m = []
txt_file_2 = "B0.txt"
with open(folder_2 + txt_file_2) as inputfile:
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

mesh_oo = trimesh.load(folder_3+mesh_file_2)
m = []
with open(folder_3 + txt_file_2) as inputfile:
    for line in inputfile:
        m.append(line.strip().split(' '))
    for j in range(len(m)):
        m[j] = list(filter(None, m[j]))
        m[j] = np.array([float(a) for a in m[j]])
vertices = m[1:(int(m[0][0])+1):1]
faces = m[(int(m[0][0])+2):int(m[int(m[0][0])+1][0]+int(m[0][0])+2):1]
vertices = np.array(vertices)
faces = np.array(faces).astype(int) - 1
mesh_oo.faces = faces
mesh_oo.vertices = vertices/10

#steps = np.arange(0, 45000, 1000)
A = np.zeros((5, 30370), dtype = np.float32)
B = np.zeros((5, 30714), dtype = np.float32)
C = np.zeros((5, 29334), dtype = np.float32)
D = np.zeros((5, 30370), dtype = np.float32)
E = np.zeros((5, 30370), dtype = np.float32)
F = np.zeros((5, 30370), dtype = np.float32)
G = np.zeros((5, 30370), dtype = np.float32)
inters_1 = np.zeros((30370, 3), dtype = np.float32)
inters_2 = np.zeros((30714, 3), dtype = np.float32)
inters_3 = np.zeros((29334, 3), dtype = np.float32)
inters_4 = np.zeros((30370, 3), dtype = np.float32)
inters_5 = np.zeros((30370, 3), dtype = np.float32)
inters_6 = np.zeros((30370, 3), dtype = np.float32)
inters_7 = np.zeros((30370, 3), dtype = np.float32)
steps = np.array([44500])
q = 4
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
    for j in range(30370):
        endpoints = np.array([mesh_ooo.vertices[j,:], mesh_ooo.vertices[j,:]+10000*(mesh1.vertices[j,:]-mesh_ooo.vertices[j,:])])
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
                    #if np.size(qq) == 3:
                #inters_1[j, :] = qq
                #break
    """mesh_2 = sio.load_mesh(folder+mesh_file_2)"""
    """L1=(max(mesh_2.vertices[:,0])-min(mesh_2.vertices[:,0]))/(max(mesh_o.vertices[:,0])-min(mesh_o.vertices[:,0]))
    L2=(max(mesh_2.vertices[:,1])-min(mesh_2.vertices[:,1]))/(max(mesh_o.vertices[:,1])-min(mesh_o.vertices[:,1]))
    L3=(max(mesh_2.vertices[:,2])-min(mesh_2.vertices[:,2]))/(max(mesh_o.vertices[:,2])-min(mesh_o.vertices[:,2]))
    mesh_o.vertices[:,0]=L1*mesh_o.vertices[:,0]
    mesh_o.vertices[:,1]=L2*mesh_o.vertices[:,1]
    mesh_o.vertices[:,2]=L3*mesh_o.vertices[:,2]
    tree = spatial.KDTree(mesh_2.vertices)
    csn = tree.query(mesh_o.vertices)"""
    
    A[q, :] = np.sqrt((mesh1.vertices[:,0] - inters_1[:, 0])**2+(mesh1.vertices[:,1] - inters_1[:, 1])**2+(mesh1.vertices[:,2] - inters_1[:, 2])**2)
    """for i in A:
        if i > 10:
            A_new.append(i)
    depth_1 = np.mean(A_new)
    depth1.append(depth_1)
    
    Area = 0.0
    regions_A = np.where(((mesh.vertices[mesh.faces[:,0], 0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-1.0)) & (mesh.vertices[mesh.faces[:,1], 0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-1.0)) & (mesh.vertices[mesh.faces[:,2], 0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-1.0))) | ((mesh.vertices[mesh.faces[:,0], 0] > ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0+1.0)) & (mesh.vertices[mesh.faces[:,1], 0] > ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0+1.0)) & (mesh.vertices[mesh.faces[:,2], 0] > ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0+1.0))))[0]
    Ntmp = np.cross(mesh.vertices[mesh.faces[regions_A,1]] - mesh.vertices[mesh.faces[regions_A,0]], mesh.vertices[mesh.faces[regions_A,2]] - mesh.vertices[mesh.faces[regions_A,0]])
    Area += 0.5*np.linalg.norm(Ntmp)
    
    Area_2 = 0.0
    regions_A = np.where(((mesh_2.vertices[mesh_2.faces[:,0], 0] < ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0-1.0)) & (mesh_2.vertices[mesh_2.faces[:,1], 0] < ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0-1.0)) & (mesh_2.vertices[mesh_2.faces[:,2], 0] < ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0-1.0))) | ((mesh_2.vertices[mesh_2.faces[:,0], 0] > ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0+1.0)) & (mesh_2.vertices[mesh_2.faces[:,1], 0] > ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0+1.0)) & (mesh_2.vertices[mesh_2.faces[:,2], 0] > ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0+1.0))))[0]
    Ntmp_2 = np.cross(mesh_2.vertices[mesh_2.faces[regions_A,1]] - mesh_2.vertices[mesh_2.faces[regions_A,0]], mesh_2.vertices[mesh_2.faces[regions_A,2]] - mesh_2.vertices[mesh_2.faces[regions_A,0]])
    Area_2 += 0.5*np.linalg.norm(Ntmp_2)
      
    GI_1 = Area/Area_2
    
    G1.append(GI_1)"""

    mesh2 = trimesh.load(folder_2+mesh_file)
    m = []
    with open(folder_2 + txt_file) as inputfile:
        for line in inputfile:
            m.append(line.strip().split(' '))
        for j in range(len(m)):
            m[j] = list(filter(None, m[j]))
            m[j] = np.array([float(a) for a in m[j]])
    vertices = m[1:(int(m[0][0])+1):1]
    faces = m[(int(m[0][0])+2):int(m[int(m[0][0])+1][0]+int(m[0][0])+2):1]
    vertices = np.array(vertices)
    faces = np.array(faces).astype(int) - 1
    mesh2.faces = faces
    mesh2.vertices = vertices
    mesh_2 = trimesh.convex.convex_hull(mesh2, qhull_options='QbB Pp Qt')
    for j in range(30714):
        endpoints = np.array([mesh_o.vertices[j,:], mesh_o.vertices[j,:]+10000*(mesh2.vertices[j,:]-mesh_o.vertices[j,:])])
        for k in range(np.size(mesh_2.faces, 0)):
            plane_normal = np.cross(mesh_2.vertices[mesh_2.faces[k,1], :]-mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :]-mesh_2.vertices[mesh_2.faces[k,0], :])
            intersections, valid_2 = trimesh.intersections.plane_lines(mesh_2.vertices[mesh_2.faces[k,0], :], plane_normal, endpoints, line_segments=True)
            if valid_2 == True:
                Area1 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,1], :] - intersections))  
                Area2 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                Area3 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                Area4 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :] - mesh_2.vertices[mesh_2.faces[k,0], :]))
                if np.absolute(Area1 + Area2 + Area3 - Area4) < 1e-10:
                    inters_2[j, :] = intersections
        """for k in range(np.size(mesh_2.faces, 0)):
            qq = intersect_line_triangle(mesh_o.vertices[j,:], mesh_o.vertices[j,:]+1000000*(mesh2.vertices[j,:]-mesh_o.vertices[j,:]), mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,1], :], mesh_2.vertices[mesh_2.faces[k,2], :])
            if np.size(qq) == 3:
                inters_2[j, :] = qq"""
                #break
        """if np.all(inters_2[j, :] == 0):
            inters_2[j, :] = mesh2.vertices[j, :]"""
    """L1=(max(mesh_2.vertices[:,0])-min(mesh_2.vertices[:,0]))/(max(mesh_o.vertices[:,0])-min(mesh_o.vertices[:,0]))
    L2=(max(mesh_2.vertices[:,1])-min(mesh_2.vertices[:,1]))/(max(mesh_o.vertices[:,1])-min(mesh_o.vertices[:,1]))
    L3=(max(mesh_2.vertices[:,2])-min(mesh_2.vertices[:,2]))/(max(mesh_o.vertices[:,2])-min(mesh_o.vertices[:,2]))
    mesh_o.vertices[:,0]=L1*mesh_o.vertices[:,0]
    mesh_o.vertices[:,1]=L2*mesh_o.vertices[:,1]
    mesh_o.vertices[:,2]=L3*mesh_o.vertices[:,2]
    tree = spatial.KDTree(mesh_2.vertices)
    csn = tree.query(mesh_o.vertices)"""
    """L1=(max(mesh.vertices[:,0])-min(mesh.vertices[:,0]))/(max(mesh_2.vertices[:,0])-min(mesh_2.vertices[:,0]))
    L2=(max(mesh.vertices[:,1])-min(mesh.vertices[:,1]))/(max(mesh_2.vertices[:,1])-min(mesh_2.vertices[:,1]))
    L3=(max(mesh.vertices[:,2])-min(mesh.vertices[:,2]))/(max(mesh_2.vertices[:,2])-min(mesh_2.vertices[:,2]))
    mesh_2.vertices[:,0]=L1*mesh_2.vertices[:,0]
    mesh_2.vertices[:,1]=L2*mesh_2.vertices[:,1]
    mesh_2.vertices[:,2]=L3*mesh_2.vertices[:,2]"""

    B[q, :] = np.sqrt((mesh2.vertices[:,0] - inters_2[:, 0])**2+(mesh2.vertices[:,1] - inters_2[:, 1])**2+(mesh2.vertices[:,2] - inters_2[:, 2])**2)
    """for i in B:
        if i > 10:
            B_new.append(i)
    depth_2 = np.mean(B_new)
    depth2.append(depth_2)"""

    """Area = 0.0
    regions_B = np.where(((mesh.vertices[mesh.faces[:,0], 0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-1.0)) & (mesh.vertices[mesh.faces[:,1], 0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-1.0)) & (mesh.vertices[mesh.faces[:,2], 0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-1.0))) | ((mesh.vertices[mesh.faces[:,0], 0] > ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0+1.0)) & (mesh.vertices[mesh.faces[:,1], 0] > ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0+1.0)) & (mesh.vertices[mesh.faces[:,2], 0] > ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0+1.0))))[0]
    Ntmp = np.cross(mesh.vertices[mesh.faces[regions_B,1]] - mesh.vertices[mesh.faces[regions_B,0]], mesh.vertices[mesh.faces[regions_B,2]] - mesh.vertices[mesh.faces[regions_B,0]])
    Area += 0.5*np.linalg.norm(Ntmp)
    
    Area_2 = 0.0
    regions_B = np.where(((mesh_2.vertices[mesh_2.faces[:,0], 0] < ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0-1.0)) & (mesh_2.vertices[mesh_2.faces[:,1], 0] < ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0-1.0)) & (mesh_2.vertices[mesh_2.faces[:,2], 0] < ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0-1.0))) | ((mesh_2.vertices[mesh_2.faces[:,0], 0] > ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0+1.0)) & (mesh_2.vertices[mesh_2.faces[:,1], 0] > ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0+1.0)) & (mesh_2.vertices[mesh_2.faces[:,2], 0] > ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0+1.0))))[0]
    Ntmp_2 = np.cross(mesh_2.vertices[mesh_2.faces[regions_B,1]] - mesh_2.vertices[mesh_2.faces[regions_B,0]], mesh_2.vertices[mesh_2.faces[regions_B,2]] - mesh_2.vertices[mesh_2.faces[regions_B,0]])
    Area_2 += 0.5*np.linalg.norm(Ntmp_2)
      
    GI_2 = Area/Area_2
    
    G2.append(GI_2)"""
    
    mesh3 =  trimesh.load(folder_3+mesh_file)
    m = []
    with open(folder_3 + txt_file) as inputfile:
        for line in inputfile:
            m.append(line.strip().split(' '))
        for j in range(len(m)):
            m[j] = list(filter(None, m[j]))
            m[j] = np.array([float(a) for a in m[j]])
    vertices = m[1:(int(m[0][0])+1):1]
    faces = m[(int(m[0][0])+2):int(m[int(m[0][0])+1][0]+int(m[0][0])+2):1]
    vertices = np.array(vertices)
    faces = np.array(faces).astype(int) - 1
    mesh3.faces = faces
    mesh3.vertices = vertices
    mesh_2 = trimesh.convex.convex_hull(mesh3, qhull_options='QbB Pp Qt')
    for j in range(29334):
        endpoints = np.array([mesh_oo.vertices[j,:], mesh_oo.vertices[j,:]+10000*(mesh3.vertices[j,:]-mesh_oo.vertices[j,:])])
        for k in range(np.size(mesh_2.faces, 0)):
            plane_normal = np.cross(mesh_2.vertices[mesh_2.faces[k,1], :]-mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :]-mesh_2.vertices[mesh_2.faces[k,0], :])
            intersections, valid_3 = trimesh.intersections.plane_lines(mesh_2.vertices[mesh_2.faces[k,0], :], plane_normal, endpoints, line_segments=True)
            if valid_3 == True:
                Area1 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,1], :] - intersections))  
                Area2 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,0], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                Area3 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - intersections, mesh_2.vertices[mesh_2.faces[k,2], :] - intersections))  
                Area4 = 0.5 * np.linalg.norm(np.cross(mesh_2.vertices[mesh_2.faces[k,1], :] - mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,2], :] - mesh_2.vertices[mesh_2.faces[k,0], :]))
                if np.absolute(Area1 + Area2 + Area3 - Area4) < 1e-10:
                    inters_3[j, :] = intersections
        """for k in range(np.size(mesh_2.faces, 0)):
            qq = intersect_line_triangle(mesh_o.vertices[j,:], mesh_o.vertices[j,:]+50*(mesh3.vertices[j,:]-mesh_o.vertices[j,:]), mesh_2.vertices[mesh_2.faces[k,0], :], mesh_2.vertices[mesh_2.faces[k,1], :], mesh_2.vertices[mesh_2.faces[k,2], :])
            if np.size(qq) == 3:
                inters_3[j, :] = qq"""
                #break
        """if np.all(inters_3[j, :] == 0):
            inters_3[j, :] = mesh3.vertices[j, :]"""
    """L1=(max(mesh_2.vertices[:,0])-min(mesh_2.vertices[:,0]))/(max(mesh_o.vertices[:,0])-min(mesh_o.vertices[:,0]))
    L2=(max(mesh_2.vertices[:,1])-min(mesh_2.vertices[:,1]))/(max(mesh_o.vertices[:,1])-min(mesh_o.vertices[:,1]))
    L3=(max(mesh_2.vertices[:,2])-min(mesh_2.vertices[:,2]))/(max(mesh_o.vertices[:,2])-min(mesh_o.vertices[:,2]))
    mesh_o.vertices[:,0]=L1*mesh_o.vertices[:,0]
    mesh_o.vertices[:,1]=L2*mesh_o.vertices[:,1]
    mesh_o.vertices[:,2]=L3*mesh_o.vertices[:,2]
    tree = spatial.KDTree(mesh_2.vertices)
    csn = tree.query(mesh_o.vertices)"""
    """mesh_2 = sio.load_mesh(folder_3+mesh_file_2)
    L1=(max(mesh.vertices[:,0])-min(mesh.vertices[:,0]))/(max(mesh_2.vertices[:,0])-min(mesh_2.vertices[:,0]))
    L2=(max(mesh.vertices[:,1])-min(mesh.vertices[:,1]))/(max(mesh_2.vertices[:,1])-min(mesh_2.vertices[:,1]))
    L3=(max(mesh.vertices[:,2])-min(mesh.vertices[:,2]))/(max(mesh_2.vertices[:,2])-min(mesh_2.vertices[:,2]))
    mesh_2.vertices[:,0]=L1*mesh_2.vertices[:,0]
    mesh_2.vertices[:,1]=L2*mesh_2.vertices[:,1]
    mesh_2.vertices[:,2]=L3*mesh_2.vertices[:,2]"""
    
    C[q, :] = np.sqrt((mesh3.vertices[:,0] - inters_3[:,0])**2+(mesh3.vertices[:,1] - inters_3[:,1])**2+(mesh3.vertices[:,2] - inters_3[:,2])**2)
    """for i in C:
        if i > 10:
            C_new.append(i)
    depth_3 = np.mean(C_new)
    depth3.append(depth_3)"""
    
    """Area = 0.0
    Ntmp = np.cross(mesh.vertices[mesh.faces[:,1]] - mesh.vertices[mesh.faces[:,0]], mesh.vertices[mesh.faces[:,2]] - mesh.vertices[mesh.faces[:,0]])
    Area += 0.5*np.linalg.norm(Ntmp)
    Area_2 = 0.0
    Ntmp_2 = np.cross(mesh_2.vertices[mesh_2.faces[:,1]] - mesh_2.vertices[mesh_2.faces[:,0]], mesh_2.vertices[mesh_2.faces[:,2]] - mesh_2.vertices[mesh_2.faces[:,0]])
    Area_2 += 0.5*np.linalg.norm(Ntmp_2)
      
    GI_3 = Area/Area_2
    
    G3.append(GI_3)"""
    
    """for i in C:
        if i > 10:
            C_new.append(i)
    depth_3 = np.mean(C_new)
    depth3.append(depth_3)"""
    
    """Area = 0.0
    Ntmp = np.cross(mesh.vertices[mesh.faces[:,1]] - mesh.vertices[mesh.faces[:,0]], mesh.vertices[mesh.faces[:,2]] - mesh.vertices[mesh.faces[:,0]])
    Area += 0.5*np.linalg.norm(Ntmp)
    Area_2 = 0.0
    Ntmp_2 = np.cross(mesh_2.vertices[mesh_2.faces[:,1]] - mesh_2.vertices[mesh_2.faces[:,0]], mesh_2.vertices[mesh_2.faces[:,2]] - mesh_2.vertices[mesh_2.faces[:,0]])
    Area_2 += 0.5*np.linalg.norm(Ntmp_2)
      
    GI_3 = Area/Area_2
    
    G3.append(GI_3)"""
np.savetxt('/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/sphere_depth_44500.txt', A[4, :], fmt='%.8f')
np.savetxt('/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/ellip150_depth_44500.txt', B[4, :], fmt='%.8f')
np.savetxt('/home/x17wang/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/ellip225_depth_44500.txt', C[4, :], fmt='%.8f')
