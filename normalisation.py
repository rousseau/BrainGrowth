import numpy as np
import math
from numba import jit, njit, prange

# Normalize initial mesh coordinates
@jit
def normalise_coord(Ut0, Ut, nn):
  # Find center of mass and dimension of the mesh
  maxx = maxy = maxz = -1e9
  minx = miny = minz = 1e9

  maxx = max(Ut0[:,0])
  minx = min(Ut0[:,0])
  maxy = max(Ut0[:,1])
  miny = min(Ut0[:,1])
  maxz = max(Ut0[:,2])
  minz = min(Ut0[:,2])

  cog = np.sum(Ut0, axis=0)
  cog /= nn # The center coordinate(x,y,z)

  print ('minx is ' + str(minx) + ' maxx is ' + str(maxx) + ' miny is ' + str(miny) + ' maxy is ' + str(maxy) + ' minz is ' + str(minz) + ' maxz is ' + str(maxz))
  #print ('center x is ' + str(cog[0]) + ' center y is ' + str(cog[1]) + ' center z is ' + str(cog[2]))

  # Change mesh information by values normalized 
  #maxd = max(max(max(abs(maxx-cog[0]), abs(minx-cog[0])), max(abs(maxy-cog[1]), abs(miny-cog[1]))), max(abs(maxz-cog[2]), abs(minz-cog[2])))
  maxd = max(max(max(abs(maxx-cog[0]), abs(minx-cog[0])), abs(maxy-miny)), max(abs(maxz-cog[2]), abs(minz-cog[2])))  # The biggest value of difference between the coordinate(x, y, z) and center(x, y,z) respectively
  #maxd = max(abs(maxx-minx), abs(maxy-miny), abs(maxz-minz))

  """Ut0[:,0] = (Ut[:,0] - cog[0])/maxd
  Ut0[:,1] = (Ut[:,1] - cog[1])/maxd
  Ut0[:,2] = (Ut[:,2] - cog[2])/maxd"""

  Ut0[:,0] = -(Ut[:,0] - cog[0])/maxd
  Ut0[:,1] = (Ut[:,1] - miny)/maxd
  Ut0[:,2] = -(Ut[:,2] - cog[2])/maxd

  print ('normalized minx is ' + str(min(Ut0[:,0])) + ' normalized maxx is ' + str(max(Ut0[:,0])) + ' normalized miny is ' + str(min(Ut0[:,1])) + ' normalized maxy is ' + str(max(Ut0[:,1])) + ' normalized minz is ' + str(min(Ut0[:,2])) + ' normalized maxz is ' + str(max(Ut0[:,2])))

  Ut = Ut0

  return Ut0, Ut, cog, maxd, miny
