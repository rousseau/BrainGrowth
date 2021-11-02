import numpy as np

def coordinates_denormalization(coordinates, n_nodes, center_of_gravity, maxd, miny, halforwholebrain): 
    '''
    Operate denormalization and x<>y of the coordinates (all nodes) of the mesh before using the calculated physical values (deformation, stress, etc.)
    Code extracted from initial denormalisation code in "output.py", without considering "zoom_pos" multiplication factor". 
    '''
    coordinates_denorm = np.zeros((n_nodes,3), dtype = float)

    coordinates_denorm[:,1] = center_of_gravity[0] - coordinates[:,0]*maxd
    if halforwholebrain.__eq__("half"):
        coordinates_denorm[:,0] = coordinates[:,1]*maxd + miny
    else:
        coordinates_denorm[:,0] = coordinates[:,1]*maxd + center_of_gravity[1]
    coordinates_denorm[:,2] = center_of_gravity[2] - coordinates[:,2]*maxd

    return coordinates_denorm
