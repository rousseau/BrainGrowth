import trimesh
import numpy as np
import slam.io as sio
from trimesh import smoothing as sm
import slam.plot as splt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from curvatureCoarse import graph_laplacian
from scipy.sparse.linalg import eigs

folder='/home/INT/lefevre.j/ownCloud/Documents/Recherche/Data/GarciaPNAS2018_K65Z/'
mesh_file = 'ATLAS30.L.Fiducial.surf.gii'
texture_file = 'covariateinteraction2.L.noivh.ggdot.func.gii'
texture_file2 = 'covariateinteraction2.L.noivh.GGnorm.func.gii'

# Ages

ages = [29, 29, 28, 28.5, 31.5, 32, 31, 32, 30.5, 32, 32, 31, 35.5, 35, 34.5, 35, 34.5, 35, 36, 34.5, 37.5, 35, 34.5, 36, 34.5, 33, 33]
Nsteps=len(ages)

# Visualize growth maps
step=20
mesh = sio.load_mesh(folder+mesh_file)
texture = sio.load_texture2((folder+texture_file))
splt.pyglet_plot(mesh, texture.darray[step], plot_colormap=False)

# Compare the two textures = the same !

texture2 = sio.load_texture2((folder+texture_file2))

Correlations=np.zeros((Nsteps,))
for i in range(Nsteps):
    Correlations[i]=np.corrcoef(texture.darray[i],texture2.darray[i])[0,1]

# Plotting the evolution of growth rate

indice=0

for i in range(Nsteps):
    plt.plot(ages[i],np.mean(texture.darray[i]),'+w')

# Parcellation in lobes

method = 'spectral'
n_clusters = 10
if method.__eq__("Kmeans"):
    # 1) Simple K-means to start simply
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.array(mesh.vertices))
    splt.pyglet_plot(mesh, kmeans.labels_, plot_colormap=False)
else:
    # 2) Another method: spectral clustering
    L, Lsparse = graph_laplacian(mesh)
    evals, evecs = eigs(Lsparse, k=n_clusters - 1, which='SM')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.real(evecs))
    splt.pyglet_plot(mesh, kmeans.labels_, plot_colormap=False)

#splt.pyglet_plot(mesh, np.real(evecs[:,3]), plot_colormap=False)

# One temporal model by parcels
# with Scikit learn

# X=np.array([ages]).transpose()
# order=7
# Xtmp=np.copy(X)
# for o in range(2,order+1):
#     Xtmp=np.concatenate((Xtmp,X**o,),axis=1)
# X=np.copy(Xtmp)
#
# parameters=np.zeros((n_clusters,order+2)) # R^2, intercept, X, X^2
# for k in range(n_clusters):
#     indices_k=kmeans.labels_==k
#     plt.plot(ages,np.mean(texture.darray[:,np.where(kmeans.labels_ == k)[0]],axis=1),'+')
#     # Y: average in each parcel for each time step
#     Y=np.mean(texture.darray[:, np.where(kmeans.labels_ == k)[0]], axis=1)
#     reg = LinearRegression().fit(X, Y)
#     parameters[k,0]=reg.score(X,Y)
#     parameters[k,1]=reg.intercept_
#     parameters[k,2:]=reg.coef_
#
# temporal_interval=np.arange(np.min(ages),np.max(ages),0.1)
# for k in range(n_clusters):
#     y=parameters[k,1]+parameters[k,2]*temporal_interval
#     for o in range(2,order+1):
#         y+=parameters[k,o+1]*temporal_interval**o
#     plt.plot(temporal_interval,y)

# One temporal model by parcel (non linear, obtained with scipy)

xdata=np.array(ages)
#func = lambda x,a,b,sigma: b+a*np.exp(-x**2/sigma)

def func(x,a,b,c,sigma):
    return b+a*np.exp(-(x-c)**2/sigma)

def func2(x,a,c,sigma):
    return a*np.exp(-(x-c)**2/sigma)

temporal_interval=np.arange(np.min(ages),np.max(ages),0.1)

peak=np.zeros((n_clusters,))
amplitude=np.zeros((n_clusters,))
latency=np.zeros((n_clusters,))
for k in range(n_clusters):
    ydata=np.mean(texture.darray[:, np.where(kmeans.labels_ == k)[0]], axis=1)
    #ydata=np.reshape(ydata,(1,len(ydata)))
    #popt, pcov = curve_fit(func, xdata, ydata,p0=[0.16,0.0,32,25.])
    popt, pcov = curve_fit(func2, xdata, ydata, p0=[0.16, 32, 25.])
    plt.plot(ages, ydata, '+')
    plt.plot(temporal_interval,func2(temporal_interval,*popt))
    peak[k]=popt[1]
    amplitude[k]=popt[0]
    latency[k]=popt[2]

# From parcells to texture

peak_texture=np.zeros((texture.darray.shape[1],))
amplitude_texture=np.zeros((texture.darray.shape[1],))
latency_texture=np.zeros((texture.darray.shape[1],))
for k in range(n_clusters):
    peak_texture[np.where(kmeans.labels_ ==k)[0]]=peak[k]
    amplitude_texture[np.where(kmeans.labels_ == k)[0]] = amplitude[k]
    latency_texture[np.where(kmeans.labels_ == k)[0]] = latency[k]

splt.pyglet_plot(mesh, peak_texture, plot_colormap=True)
splt.pyglet_plot(mesh, amplitude_texture, plot_colormap=True)
splt.pyglet_plot(mesh, latency_texture, plot_colormap=True)

