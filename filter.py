import numpy as np
import scipy

def find_index_of_nearest_x(x_array, x_point):
    """
    Returns the nearest point from an array, 1 dimension
    Args:
    x_array (array): array of values to be searched for
    x_point (float): point of interest
    """
    distance =(x_array-x_point)**2
    idx = np.where(distance==distance.min())
    return idx[0][0]

    #alternative ? idx = (np.abs(arr - v)).argmin()

def calc_tets_center (coordinates, tets):
    """
    Calculate centroid of tetrahedres
    Args:
    coordinates (array): cartesian coordinates of points
    tets (array): tetrahedre incides
    Returns:
    Centroids (array): Centroids of tets
    """
    centroids = np.zeros ((len (tets), 3), dtype=np.float64)
    for i in range (len (tets)):
        x = 0
        y = 0
        z = 0
        for j in tets[i]:
            x += coordinates[j][0]
            y += coordinates[j][1]
            z += coordinates [j][2]
        centroids [i] = np.array ([x/4.0, y/4.0, z/4.0])
        
    return centroids


def gaussian_filter (centroids, column = 2, K = 3.0, center = 0.0):
    """
    TODO: Vertical normalization of gaussian curve, so that O = 1 indepedant from K
    Apply gaussian weighting on chosen axe according to: sqrt (K/pi) * exp (-K * x²) + U. 
    Args:
    centroids (array): collection of tetrahedre centroids
    column (int): axe of smoothing
    K (float): Variance, greater means less smoothing
    center (float): Center of weigthing factor
    Return:
    gauss (array): Weighting factor for each tet
    """
    gauss = np.ones (len (centroids))
    
    #extract column of interset
    centroids = centroids[:,column]

    min_coord = np.min (centroids)
    max_coord = np.max (centroids)
    
    #normalize column
    for i in range (len(centroids)):
        centroids[i] = 2*((centroids[i] - min_coord) / (max_coord - min_coord))-1
        gauss[i] = np.sqrt (K/np.pi) * np.exp (-K * centroids[i] * centroids [i]) + center
    
    return gauss






#Create a gaussian pondering deending on x on x axis

#return the gaussian filter to be factored for at

def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = numpy.where(distance==distance.min())
    return idy[0],idx[0]

# Prox updated for speed, ref: https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates?noredirect=1&lq=1
combined_x_y_arrays = numpy.dstack([y_array.ravel(),x_array.ravel()])[0]
points_list = list(points.transpose())


def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return dist, indexes