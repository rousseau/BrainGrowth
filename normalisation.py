import numpy as np
from numba import jit

# Normalize initial mesh coordinates
@jit
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
  maxd (float): max value for longest axe, used for denormalisation of data
  mind (float): min value for longest axe, used for denormalisation of data
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

  print ('minx is ' + str(minx) + ' maxx is ' + str(maxx) + ' miny is ' + str(miny) + ' maxy is ' + str(maxy) + ' minz is ' + str(minz) + ' maxz is ' + str(maxz))
  
  if halforwholebrain.__eq__("half"):
    maxd = max(max(max(abs(maxx-center_of_gravity[0]), abs(minx-center_of_gravity[0])), abs(maxy-miny)), max(abs(maxz-center_of_gravity[2]), abs(minz-center_of_gravity[2])))
    coordinates0[:,0] = -(coordinates[:,0] - center_of_gravity[0])/maxd
    coordinates0[:,1] = (coordinates[:,1] - miny)/maxd
    coordinates0[:,2] = -(coordinates[:,2] - center_of_gravity[2])/maxd
  else:
    maxd = max(max(max(abs(maxx-center_of_gravity[0]), abs(minx-center_of_gravity[0])), max(abs(maxy-center_of_gravity[1]), abs(miny-center_of_gravity[1]))), max(abs(maxz-center_of_gravity[2]), abs(minz-center_of_gravity[2])))
    coordinates0[:,0] = -(coordinates[:,0] - center_of_gravity[0])/maxd
    coordinates0[:,1] = (coordinates[:,1] - center_of_gravity[1])/maxd
    coordinates0[:,2] = -(coordinates[:,2] - center_of_gravity[2])/maxd

  print ('normalized minx is ' + str(min(coordinates0[:,0])) + ' normalized maxx is ' + str(max(coordinates0[:,0])) + ' normalized miny is ' + str(min(coordinates0[:,1])) + ' normalized maxy is ' + str(max(coordinates0[:,1])) + ' normalized minz is ' + str(min(coordinates0[:,2])) + ' normalized maxz is ' + str(max(coordinates0[:,2])))

  coordinates = coordinates0.copy()

  return coordinates0, coordinates, center_of_gravity, maxd, miny
