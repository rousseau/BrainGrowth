# BrainGrowth

Execute simulation.py， change the line "mesh_path = " and give the address of the mesh to "mesh_path".

For sphere5.mesh, in simulation.py, a = 0.01, dt = 0.05*np.sqrt(rho*a*a/K);

For prm001_30w_Rwhite_petit_taille_2.mesh, in simulation.py, a = 0.001, dt = 0.01*np.sqrt(rho*a*a/K); in output.py, in the function writePov, vertices[:,:] = - Ut[SN[:],:]*zoom_pos;

