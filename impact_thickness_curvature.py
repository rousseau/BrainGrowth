import slam.io as sio
import trimesh
from curvatureCoarse import curvatureTopologic
import slam.plot as splt
import numpy as np
import matplotlib.pyplot as plt
import math
# Visualization of the biomechanical model simulations

import trimesh
import numpy as np
import slam.io as sio
from trimesh import smoothing as sm
from trimesh.curvature import discrete_mean_curvature_measure as curv_mean
import slam.plot as splt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from curvatureCoarse import graph_laplacian
from scipy.sparse.linalg import eigs
from scipy import spatial
from scipy import pi,sqrt,exp
from scipy.special import erf
from scipy import linspace
from scipy import pi,sqrt,exp
from scipy.special import erf
import scipy.special as sp
from scipy import signal
from Curvature_princip import surface_curvature
import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv

folder='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_172/pov_H0.042000AT1.829000/'
folder_2='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_1011/pov_H0.042000AT1.829000/'
folder_3='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_6833/pov_H0.042000AT1.829000/'
folder_4='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_37955/pov_H0.042000AT1.829000/'
folder_5='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_205699/pov_H0.042000AT1.829000/'
folder_6='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_402133/pov_H0.042000AT1.829000/'
folder_7='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_662403/pov_H0.042000AT1.829000/'
folder_8='/media/x17wang/XiaoyuDisk/Bureau/xiaoyu/Brain_code_and_meshes/data/sphere5_894661/pov_H0.042000AT1.829000/'
steps = np.arange(0, 45000, 1000)
A = []
B = []
C = []
D = []
E = []
F = []
G = []
H = []
dt = 0.05*np.sqrt(0.01*0.01*0.01/5.0)
t = steps*dt
for i in steps:
    mesh_file = "B%d.gii"%(i)
    mesh =  sio.load_mesh(folder+mesh_file)
    mesh_2 =  sio.load_mesh(folder_2+mesh_file)
    mesh_3 =  sio.load_mesh(folder_3+mesh_file)
    mesh_4 =  sio.load_mesh(folder_4+mesh_file)
    mesh_5 =  sio.load_mesh(folder_5+mesh_file)
    mesh_6 =  sio.load_mesh(folder_6+mesh_file)
    mesh_7 =  sio.load_mesh(folder_7+mesh_file)
    mesh_8 =  sio.load_mesh(folder_8+mesh_file)
    #curvature = curvatureTopologic(mesh)
    #curvature = curv_mean(mesh, mesh.vertices, 1)
    #labesa=np.where((mesh.vertices[:,0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-5.0)) | (mesh.vertices[:,0] > ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0+5.0)))[0]
    #labesa=np.where(mesh.vertices[:,0] < ((max(mesh.vertices[:,0]) + min(mesh.vertices[:,0]))/2.0-1.0))[0]
    mesh.apply_transform(mesh.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh.faces)):
      Ntmp = np.cross(mesh.vertices[mesh.faces[j,1]] - mesh.vertices[mesh.faces[j,0]], mesh.vertices[mesh.faces[j,2]] - mesh.vertices[mesh.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
    curvature = 0.5 * (PrincipalCurvatures[0, :]*np.sqrt(Area) + PrincipalCurvatures[1, :]*np.sqrt(Area))
    """carac_length = np.power(mesh.volume, 1/3)
    norm_curvature = curvature*carac_length"""
    curvature_mean = np.mean(np.absolute(curvature))
    A.append(curvature_mean)
    #curvature_2 = curvatureTopologic(mesh_2)
    #curvature_2 = curv_mean(mesh_2, mesh_2.vertices, 1)
    #labesb=np.where(mesh_2.vertices[:,0] < (max(mesh_2.vertices[:,0]) - 1.0))[0]
    #labesb=np.where((mesh_2.vertices[:,0] < ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0-5.0)) | (mesh_2.vertices[:,0] > ((max(mesh_2.vertices[:,0]) + min(mesh_2.vertices[:,0]))/2.0+5.0)))[0]
    mesh_2.apply_transform(mesh_2.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh_2.faces)):
      Ntmp = np.cross(mesh_2.vertices[mesh_2.faces[j,1]] - mesh_2.vertices[mesh_2.faces[j,0]], mesh_2.vertices[mesh_2.faces[j,2]] - mesh_2.vertices[mesh_2.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures_2, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_2)
    curvature_2 = 0.5 * (PrincipalCurvatures_2[0, :]*np.sqrt(Area) + PrincipalCurvatures_2[1, :]*np.sqrt(Area))
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    curvature_mean_2 = np.mean(np.absolute(curvature_2))
    B.append(curvature_mean_2)
    
    mesh_3.apply_transform(mesh_3.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh_3.faces)):
      Ntmp = np.cross(mesh_3.vertices[mesh_3.faces[j,1]] - mesh_3.vertices[mesh_3.faces[j,0]], mesh_3.vertices[mesh_3.faces[j,2]] - mesh_3.vertices[mesh_3.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures_3, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_3)
    curvature_3 = 0.5 * (PrincipalCurvatures_3[0, :]*np.sqrt(Area) + PrincipalCurvatures_3[1, :]*np.sqrt(Area))
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    curvature_mean_3 = np.mean(np.absolute(curvature_3))
    C.append(curvature_mean_3)
    
    mesh_4.apply_transform(mesh_4.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh_4.faces)):
      Ntmp = np.cross(mesh_4.vertices[mesh_4.faces[j,1]] - mesh_4.vertices[mesh_4.faces[j,0]], mesh_4.vertices[mesh_4.faces[j,2]] - mesh_4.vertices[mesh_4.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures_4, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_4)
    curvature_4 = 0.5 * (PrincipalCurvatures_4[0, :]*np.sqrt(Area) + PrincipalCurvatures_4[1, :]*np.sqrt(Area))
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    curvature_mean_4 = np.mean(np.absolute(curvature_4))
    D.append(curvature_mean_4)
    
    mesh_5.apply_transform(mesh_5.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh_5.faces)):
      Ntmp = np.cross(mesh_5.vertices[mesh_5.faces[j,1]] - mesh_5.vertices[mesh_5.faces[j,0]], mesh_5.vertices[mesh_5.faces[j,2]] - mesh_5.vertices[mesh_5.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures_5, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_5)
    curvature_5 = 0.5 * (PrincipalCurvatures_5[0, :]*np.sqrt(Area) + PrincipalCurvatures_5[1, :]*np.sqrt(Area))
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    curvature_mean_5 = np.mean(np.absolute(curvature_5))
    E.append(curvature_mean_5)
    
    mesh_6.apply_transform(mesh_6.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh_6.faces)):
      Ntmp = np.cross(mesh_6.vertices[mesh_6.faces[j,1]] - mesh_6.vertices[mesh_6.faces[j,0]], mesh_6.vertices[mesh_6.faces[j,2]] - mesh_6.vertices[mesh_6.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures_6, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_6)
    curvature_6 = 0.5 * (PrincipalCurvatures_6[0, :]*np.sqrt(Area) + PrincipalCurvatures_6[1, :]*np.sqrt(Area))
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    curvature_mean_6 = np.mean(np.absolute(curvature_6))
    F.append(curvature_mean_6)
    
    mesh_7.apply_transform(mesh_7.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh_7.faces)):
      Ntmp = np.cross(mesh_7.vertices[mesh_7.faces[j,1]] - mesh_7.vertices[mesh_7.faces[j,0]], mesh_7.vertices[mesh_7.faces[j,2]] - mesh_7.vertices[mesh_7.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures_7, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_7)
    curvature_7 = 0.5 * (PrincipalCurvatures_7[0, :]*np.sqrt(Area) + PrincipalCurvatures_7[1, :]*np.sqrt(Area))
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    curvature_mean_7 = np.mean(np.absolute(curvature_7))
    G.append(curvature_mean_7)
    
    mesh_8.apply_transform(mesh_8.principal_inertia_transform)
    Area = 0.0
    for j in range(len(mesh_8.faces)):
      Ntmp = np.cross(mesh_8.vertices[mesh_8.faces[j,1]] - mesh_8.vertices[mesh_8.faces[j,0]], mesh_8.vertices[mesh_8.faces[j,2]] - mesh_8.vertices[mesh_8.faces[j,0]])
      Area += 0.5*np.linalg.norm(Ntmp)
    PrincipalCurvatures_8, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_8)
    curvature_8 = 0.5 * (PrincipalCurvatures_8[0, :]*np.sqrt(Area) + PrincipalCurvatures_8[1, :]*np.sqrt(Area))
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    curvature_mean_8 = np.mean(np.absolute(curvature_8))
    H.append(curvature_mean_8)
    """mesh_5.apply_transform(mesh_5.principal_inertia_transform)
    PrincipalCurvatures_5, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh_5)
    curvature_5 = 0.5 * (PrincipalCurvatures_5[0, :] + PrincipalCurvatures_5[1, :])"""
    """carac_length_2 = np.power(mesh_2.volume, 1/3)
    norm_curvature_2 = curvature_2*carac_length_2"""
    """curvature_mean_5 = np.mean(np.absolute(curvature_5))
    E.append(curvature_mean_5)"""

correlation = np.absolute(np.array(A) - np.array(B))
Dice = np.zeros(t.size, dtype=np.float32)
for i in range(t.size):
    Dice[i] = (min(np.absolute(A[i]), np.absolute(B[i]))*2)/(np.absolute(A[i]) + np.absolute(B[i]))

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
fig, ax = plt.subplots()
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
#plt.plot(np.array([math.log(535),math.log(4280),math.log(34240),math.log(200944),math.log(1181216),math.log(2314240),math.log(3897088),math.log(5248576)]), np.array([A[-1],B[-1],C[-1],D[-1],E[-1],F[-1],G[-1],H[-1]]), color='red', linewidth=0.7, label='md=535')
#plt.scatter(np.array([math.log(535),math.log(4280),math.log(34240),math.log(200944),math.log(1181216),math.log(2314240),math.log(3897088),math.log(5248576)]), np.array([A[-1],B[-1],C[-1],D[-1],E[-1],F[-1],G[-1],H[-1]]), marker = '+', color='red')
#for i in range(4):
#plt.plot(np.array([math.log(535), math.log(4280), math.log(34240), math.log(200944), math.log(1181216),math.log(2314240)]), np.array([A[-1], B[-1], C[-1], D[-1], E[-1], F[-1]]), color='red', linewidth=0.7)
#plt.scatter(np.array([math.log(535), math.log(4280), math.log(34240), math.log(200944), math.log(1181216),math.log(2314240)]), np.array([A[-1], B[-1], C[-1], D[-1], E[-1], F[-1]]), marker = '+', color='red')
#plt.plot(temporal_interval, func(temporal_interval, *popt_1), 'yellow', label='linear_overall', linewidth=1.5)
#plt.annotate('linear_overall', color='black', size=15, xy=(0.2, func(0.2, *popt_1)), xytext=(0.2, 0.7), arrowprops=dict(facecolor='yellow', shrink=0.005))
plt.plot(t, np.array(A), color='red', linewidth=0.9, label='md=214')
plt.scatter(t, np.array(A), marker = '+', color='red')
plt.plot(t, np.array(B), color='orange', linewidth=0.9, label='md=1712')
plt.scatter(t, np.array(B), marker = '+', color='orange')
plt.plot(t, np.array(C), color='gold', linewidth=0.9, label='md=13696')
plt.scatter(t, np.array(C), marker = '+', color='gold')
plt.plot(t, np.array(D), color='green', linewidth=0.9, label='md=80378')
plt.scatter(t, np.array(D), marker = '+', color='green')
plt.plot(t, np.array(E), color='blue', linewidth=0.9, label='md=472486')
plt.scatter(t, np.array(E), marker = '+', color='blue')
plt.plot(t, np.array(F), color='purple', linewidth=0.9, label='md=925696')
plt.scatter(t, np.array(F), marker = '+', color='purple')
plt.plot(t, np.array(G), color='cyan', linewidth=0.9, label='md=1558835')
plt.scatter(t, np.array(G), marker = '+', color='cyan')
plt.plot(t, np.array(H), color='black', linewidth=0.9, label='md=2099430')
plt.scatter(t, np.array(H), marker = '+', color='black')
#plt.ylim(0, 1.5)
"""plt.plot(t, np.array(E), color='blue', linewidth=0.7, label='Hi = 1.23 mm')
plt.scatter(t, np.array(E), marker = '+', color='blue')"""
#plt.plot(temporal_interval, func(temporal_interval, *popt_3), 'pink', label='gompertz_half', linewidth=1.5)
#plt.annotate('gompertz_half', color='blagdb ./BrainsXiaoyuck', size=15, xy=(0.5, func(0.5, *popt_3)), xytext=(0.5, 0.2), arrowprops=dict(facecolor='pink', shrink=0.005))
plt.xlabel('Time of model', fontsize=14, color='black')
plt.ylabel('Mean absolute of curvatures', color='black', fontsize=14)
plt.title('Impact of mesh density on curvatures', color='black', fontsize=16)
plt.gcf().set_facecolor('white') 
#plt.legend(loc=0, numpoints=0.05, fontsize=0.04, prop={'size': 3})
plt.legend(loc=0, numpoints=5, fontsize=5, prop={'size': 5})
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=10, fontweight='bold', color='black')
plt.style.use('dark_background')
plt.style.use('ggplot')
plt.show()
plt.savefig('/home/x17wang/Bureau/materiaux_thèse/mesh_density_ellipsoid_curvatures_new_2.pdf', dpi = fig.dpi, facecolor = 'w', edgecolor = 'w', format='pdf')