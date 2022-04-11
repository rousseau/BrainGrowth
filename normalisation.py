import numpy as np
from numba import jit, njit, objmode

# Normalize initial mesh coordinates
@njit
def normalise_coord(coordinates0, coordinates, n_nodes, halforwholebrain):
  """
  Normalizes inital mesh coordinates over longest axe
  Args:
  coordinates0 (array): undeformed coordinates of vertices
  coordinates (array): deformed coordinates of vertices
  n_node (int): number of nodes
  halfofwholebrain (string): type of normalisation depending on number of hemispheres
  Returns:
  coordinates0 (array): undeformed coordinates of vertices
  coordinates (array): deformed coordinates of vertices
  center_of_gravity (float): Center in x, y, z of the mesh from min and max values, used for denormalisation of data
  maxd (float): maximum distance from center of gravity in either of the 3 dimensions, used for denormalisation
  miny (float): Minimum coordinate in y direction, used for viewing
  """
  maxx = maxy = maxz = -1e9
  minx = miny = minz = 1e9

  maxx = max(coordinates0[:,0])
  minx = min(coordinates0[:,0])
  maxy = max(coordinates0[:,1])
  miny = min(coordinates0[:,1])
  maxz = max(coordinates0[:,2])
  minz = min(coordinates0[:,2])

  center_of_gravity = np.sum(coordinates0, axis=0)
  center_of_gravity /= n_nodes # The center coordinate(x,y,z)

  with objmode(): 
    print('minx is {}, maxx is {}'.format(minx, maxx))
    print('miny is {}, maxy is {}'.format(miny, maxy))
    print('minz is {}, maxz is {}'.format(minz, maxz))

  if halforwholebrain == "half":
    maxd = max(max(max(abs(maxx-center_of_gravity[0]), abs(minx-center_of_gravity[0])), abs(maxy-miny)), max(abs(maxz-center_of_gravity[2]), abs(minz-center_of_gravity[2])))
    coordinates0[:,0] = -(coordinates[:,0] - center_of_gravity[0])/maxd
    coordinates0[:,1] = (coordinates[:,1] - miny)/maxd
    coordinates0[:,2] = -(coordinates[:,2] - center_of_gravity[2])/maxd
  else:
    maxd = max(max(max(abs(maxx-center_of_gravity[0]), abs(minx-center_of_gravity[0])), max(abs(maxy-center_of_gravity[1]), abs(miny-center_of_gravity[1]))), max(abs(maxz-center_of_gravity[2]), abs(minz-center_of_gravity[2])))
    # new coordinates in barcyenter referential and normalized compared to half maximum coordinates distance to barycenter. 
    coordinates0[:,0] = -(coordinates[:,0] - center_of_gravity[0])/maxd 
    coordinates0[:,1] = (coordinates[:,1] - center_of_gravity[1])/maxd
    coordinates0[:,2] = -(coordinates[:,2] - center_of_gravity[2])/maxd

  with objmode(): 
    print('normalized minx is {}, normalized maxx is {}'.format(min(coordinates0[:,0]), max(coordinates0[:,0])))
    print('normalized miny is {}, normalized maxy is {}'.format(min(coordinates0[:,1]), max(coordinates0[:,1])))
    print('normalized minz is {}, normalized maxz is {}'.format(min(coordinates0[:,2]), max(coordinates0[:,2])))

  coordinates = coordinates0.copy()

  return coordinates0, coordinates, center_of_gravity, maxd, miny

def coordinates_denormalisation(coordinates, n_nodes, center_of_gravity, maxd, miny, halforwholebrain): 
  '''
  Operate denormalization and x<>y of the coordinates (all nodes) of the mesh before using the calculated physical values (deformation, stress, etc.)
  Code extracted from initial denormalisation code in "output.py", without considering "zoom_pos" multiplication factor". 
  Args:
  coordinates (array): deformed coordinates of vertices
  n_nodes (int): number of nodes
  center_of_gravity (array): Center of the mesh in x, y, z
  maxd (float): maximum distance from center of gravity in either of the 3 dimensions, used for denormalisation
  miny (float): Minimum coordinate in y direction, used for viewing
  halfofwholebrain (string): type of normalisation depending on number of hemispheres
  Returns:
  coordinates_denorm (array): denormalized coordinates
  '''
  coordinates_denorm = np.zeros((n_nodes,3), dtype = float)

  coordinates_denorm[:,1] = center_of_gravity[0] - coordinates[:,0]*maxd

  if halforwholebrain == "half":
      coordinates_denorm[:,0] = coordinates[:,1]*maxd + miny

  else:
      coordinates_denorm[:,0] = coordinates[:,1]*maxd + center_of_gravity[1]
  coordinates_denorm[:,2] = center_of_gravity[2] - coordinates[:,2]*maxd

  return coordinates_denorm
