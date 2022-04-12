import numpy as np
from numba import njit, prange 
from mathfunc import det_dim_3, cross_dim_3, dot_mat_dim_3, inv_dim_3, transpose_dim_3, normalize_dim_3, normalize, EV
from math import sqrt, exp

@njit(parallel=True)
def sim(n_tets, GROWTH_RELATIVE, t, THICKNESS_CORTEX, n_nodes, coordinates, tets, tan_growth_tensor, ref_state_tets, dist_2_surf, muw, mug, gm, gr, eps, k_param, bulk_modulus, surf_node_norms, nearest_surf_node, n_surface_nodes, nodal_idx, coordinates0, midplane_pos, mesh_spacing, repuls_skin, damping_coef, mass_density, dt, Vt, Ft):
    for _ in range(1):
        #variables initialisation
        at = np.zeros(n_tets, dtype=np.float64)
        material_tets = np.zeros((n_tets,3,3), dtype=np.float64)
        Vn0 = np.zeros(n_nodes, dtype=np.float64) #Initialize nodal volumes in reference state
        Vn = np.zeros(n_nodes, dtype=np.float64)  #Initialize deformed nodal volumes
        vol0 = np.zeros(n_tets, dtype=np.float64)
        vol = np.zeros(n_tets, dtype=np.float64)
        gm = np.zeros(n_tets, dtype=np.float64)
        mu = np.zeros(n_tets, dtype=np.float64)
        ref_state_growth = np.zeros ((n_tets, 3, 3), dtype=np.float64) #Ar
        deformation_grad = np.zeros((n_tets,3,3), dtype=np.float64)
        left_cauchy_grad = np.zeros((n_tets,3,3), dtype=np.float64)
        rel_vol_chg = np.zeros(n_tets, dtype=np.float64)
        rel_vol_chg1 = np.zeros(n_tets, dtype=np.float64)
        rel_vol_chg2 = np.zeros(n_tets, dtype=np.float64)
        rel_vol_chg3 = np.zeros(n_tets, dtype=np.float64)
        rel_vol_chg4 = np.zeros(n_tets, dtype=np.float64)
        rel_vol_chg_av = np.zeros(n_tets, dtype=np.float64)
        tet_norms = np.zeros((n_tets,3), dtype=np.float64)
        A = np.zeros((n_tets,3,3), dtype=np.float64)

        #Initialize stresses 
        s = np.zeros((n_tets,3,3), dtype=np.float64)
        p = np.zeros((n_tets,3,3), dtype=np.float64)

        at[:] = GROWTH_RELATIVE*t

        longi_length = -0.98153*t**2+3.4214*t+1.993
        cortex_thickness = THICKNESS_CORTEX + 0.01*t
        material_tets[:,0] = coordinates[tets[:,1]] - coordinates[tets[:,0]]
        material_tets[:,1] = coordinates[tets[:,2]] - coordinates[tets[:,0]]
        material_tets[:,2] = coordinates[tets[:,3]] - coordinates[tets[:,0]]
        material_tets[:] = transpose_dim_3(material_tets[:])

        vol0[:] = det_dim_3(dot_mat_dim_3(tan_growth_tensor[:], ref_state_tets[:]))/6.0
        vol[:] = det_dim_3(material_tets[:])/6.0
        
        for i in prange(n_tets):
            for tet in tets[i]:
                Vn0[tet] += vol0[i]/4.0
                Vn[tet] += vol[i]/4.0
            gm[i] = 1.0/(1.0 + exp(10.0*(0.25*(dist_2_surf[tets[i,0]] + dist_2_surf[tets[i,1]] + dist_2_surf[tets[i,2]] + dist_2_surf[tets[i,3]])/cortex_thickness - 1.0)))*0.25*(gr[tets[i,0]] + gr[tets[i,1]] + gr[tets[i,2]] + gr[tets[i,3]])
            mu[i] = muw*(1.0 - gm[i]) + mug*gm[i]  # Global modulus of white matter and gray matter
            
        #Apply growth to reference state
        ref_state_growth = dot_mat_dim_3(tan_growth_tensor, ref_state_tets)
        #Calculate deformation gradient F //combine relative volume change ?
        deformation_grad = dot_mat_dim_3(material_tets, inv_dim_3(ref_state_growth))   
        #Calculate Left-Cauchy-Green gradient B
        left_cauchy_grad = dot_mat_dim_3(deformation_grad, transpose_dim_3(deformation_grad))
        
        #relative volume change J
        rel_vol_chg = det_dim_3(deformation_grad)
        #averaged volume change
        rel_vol_chg1 = Vn[tets[:,0]]/Vn0[tets[:,0]]
        rel_vol_chg2 = Vn[tets[:,1]]/Vn0[tets[:,1]]
        rel_vol_chg3 = Vn[tets[:,2]]/Vn0[tets[:,2]]
        rel_vol_chg4 = Vn[tets[:,3]]/Vn0[tets[:,3]]
        rel_vol_chg_av = (rel_vol_chg1 + rel_vol_chg2 + rel_vol_chg3 + rel_vol_chg4)/4.0 

        identity = np.identity(3)
            
        #decide if need SVD or not
        for i in prange (n_tets):

            N1 = np.zeros(3, dtype=np.float64)
            N2 = np.zeros(3, dtype=np.float64)
            N3 = np.zeros(3, dtype=np.float64)
            N4 = np.zeros(3, dtype=np.float64)
                
            ll1, ll2, ll3 = EV(left_cauchy_grad[i])
            # ll1, ll3, ll2 = np.linalg.eig(left_cauchy_grad[i])
                
            if ll3 >= eps*eps and rel_vol_chg[i] > 0.0:  # No need for SVD
                s[i] = (left_cauchy_grad[i] - identity  * np.trace(left_cauchy_grad[i])/3.0) * mu[i]/(rel_vol_chg[i] * np.power(rel_vol_chg[i], 2.0/3.0)) + identity  * bulk_modulus * (rel_vol_chg_av[i]-1.0)
                p[i] = np.dot(s[i], np.linalg.inv(deformation_grad[i].transpose()))*rel_vol_chg[i] # Piola-Kirchhoff stress
                #W = 0.5*mu[i]*(np.trace(left_cauchy_grad[i])/powJ23 - 3.0) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))*0.25  
                
            else:   #need SVD
                C = np.dot(deformation_grad[i].transpose(), deformation_grad[i])
                w2, v2 = np.linalg.eigh(C)
                v2 = -v2
                
                l1 = sqrt(w2[0])
                l2 = sqrt(w2[1])
                l3 = sqrt(w2[2])
            
                if np.linalg.det(v2) < 0.0:
                    v2[0,0] = -v2[0,0]
                    v2[1,0] = -v2[1,0]
                    v2[2,0] = -v2[2,0]
            
                Fdi = np.identity(3)
                if l1 >= 1e-25:
                    Fdi[0,0] = 1.0/l1
                    Fdi[1,1] = 1.0/l2
                    Fdi[2,2] = 1.0/l3
            
                U = np.dot(deformation_grad[i], np.dot(v2, Fdi))
            
                if l1 < 1e-25:
                    U[0,0] = U[1,1]*U[2,2] - U[2,1]*U[1,2]
                    U[1,0] = U[2,1]*U[0,2] - U[0,1]*U[2,2]
                    U[2,0] = U[0,1]*U[1,2] - U[1,1]*U[0,2]
            
                if np.linalg.det(deformation_grad[i]) < 0.0:
                    l1 = -l1
                    U[0,0] = -U[0,0]
                    U[1,0] = -U[1,0]
                    U[2,0] = -U[2,0]
            
                Pd = identity
                pow23 = np.power(eps*l2*l3, 2.0/3.0)
                Pd[0,0] = mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23 + k_param*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l2*l3
                Pd[1,1] = mu[i]/3.0*(-eps*eps/l2 + 2.0*l2 - l3*l3/l2)/pow23 + mu[i]/9.0*(-4.0*eps/l2 - 4.0/eps*l2 + 2.0/eps/l2*l3*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l3
                Pd[2,2] = mu[i]/3.0*(-eps*eps/l3 - l2*l2/l3 + 2.0*l3)/pow23 + mu[i]/9.0*(-4.0*eps/l3 + 2.0/eps*l2*l2/l3 - 4.0/eps*l3)/pow23*(l1-eps) + bulk_modulus*(rel_vol_chg_av[i]-1.0)*l1*l2
                p[i] = np.dot(U, np.dot(Pd, v2.transpose()))
            #W = 0.5*mu[i]*((eps*eps + l2*l2 + l3*l3)/pow23 - 3.0) + mu[i]/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23*(l1-eps) + 0.5*k_param*(l1-eps)*(l1-eps) + 0.5*bulk_modulus*((rel_vol_chg1[i]-1.0)*(rel_vol_chg1[i]-1.0) + (rel_vol_chg2[i]-1.0)*(rel_vol_chg2[i]-1.0) + (rel_vol_chg3[i]-1.0)*(rel_vol_chg3[i]-1.0) + (rel_vol_chg4[i]-1.0)*(rel_vol_chg4[i]-1.0))/4.0
            
            # Calculate tetra face negative normals (because traction Ft=-P*n)
            xr1 = np.array([ref_state_growth[i,0,0], ref_state_growth[i,1,0], ref_state_growth[i,2,0]])
            xr2 = np.array([ref_state_growth[i,0,1], ref_state_growth[i,1,1], ref_state_growth[i,2,1]])
            xr3 = np.array([ref_state_growth[i,0,2], ref_state_growth[i,1,2], ref_state_growth[i,2,2]])

            # N1 = cross_dim_2(xr3, xr1)  #functionalisation of cross product = loss of performance ~10ms
            # N2 = cross_dim_2(xr2, xr3)
            # N3 = cross_dim_2(xr1, xr2)
            # N4 = cross_dim_2(xr2-xr3, xr1-xr3)

            vec1 = xr3
            vec2 = xr1
            vec3 = xr2
            vec4 = xr2-xr3
            vec5 = xr1-xr3

            N1[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1] # N1 = np.cross(xr3, xr1)
            N1[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
            N1[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
            
            N2[0] = vec3[1] * vec1[2] - vec3[2] * vec1[1] # N2 = np.cross(xr2, xr3)
            N2[1] = vec3[2] * vec1[0] - vec3[0] * vec1[2]
            N2[2] = vec3[0] * vec1[1] - vec3[1] * vec1[0]
            
            N3[0] = vec2[1] * vec3[2] - vec2[2] * vec3[1] # N3 = np.cross(xr1, xr2)
            N3[1] = vec2[2] * vec3[0] - vec2[0] * vec3[2]
            N3[2] = vec2[0] * vec3[1] - vec2[1] * vec3[0]
            
            N4[0] = vec4[1] * vec5[2] - vec4[2] * vec5[1] # N4 = np.cross(xr2-xr3, xr1-xr3) 
            N4[1] = vec4[2] * vec5[0] - vec4[0] * vec5[2]
            N4[2] = vec4[0] * vec5[1] - vec4[1] * vec5[0]
                
            # Distribute forces among tetra vertices, probably not vectorizable. Surprising that its //
            Ft[tets[i,0]] += np.dot(p[i], (N1 + N2 + N3).T)/6.0
            Ft[tets[i,1]] += np.dot(p[i], (N1 + N3 + N4).T)/6.0
            Ft[tets[i,2]] += np.dot(p[i], (N2 + N3 + N4).T)/6.0
            Ft[tets[i,3]] += np.dot(p[i], (N1 + N2 + N4).T)/6.0
        
 
        tet_norms[:] = surf_node_norms[nearest_surf_node[tets[:,0]]] + surf_node_norms[nearest_surf_node[tets[:,1]]] + surf_node_norms[nearest_surf_node[tets[:,2]]] + surf_node_norms[nearest_surf_node[tets[:,3]]]
        tet_norms = normalize(tet_norms)

        
        A[:,0,0] = tet_norms[:,0]*tet_norms[:,0]
        A[:,0,1] = tet_norms[:,0]*tet_norms[:,1]
        A[:,0,2] = tet_norms[:,0]*tet_norms[:,2]
        A[:,1,0] = tet_norms[:,0]*tet_norms[:,1]
        A[:,1,1] = tet_norms[:,1]*tet_norms[:,1]
        A[:,1,2] = tet_norms[:,1]*tet_norms[:,2]
        A[:,2,0] = tet_norms[:,0]*tet_norms[:,2]
        A[:,2,1] = tet_norms[:,1]*tet_norms[:,2]
        A[:,2,2] = tet_norms[:,2]*tet_norms[:,2]

        gm = np.reshape(np.repeat(gm, 9), (n_tets, 3, 3))
        at = np.reshape(np.repeat(at, 9), (n_tets, 3, 3))
        #identity = np.resize(identity, (n_tets, 3, 3)) // not compatible numba, but apparently broacasting similar
        tan_growth_tensor = identity + (identity - A) * gm * at

        for i in prange(n_surface_nodes):
            pt = nodal_idx[i]
            if coordinates0[pt,1] < midplane_pos - 0.5*mesh_spacing and coordinates[pt,1] > midplane_pos:
                Ft[pt,1] -= (midplane_pos - coordinates[pt,1])/repuls_skin*mesh_spacing*mesh_spacing*bulk_modulus
            if coordinates0[pt,1] > midplane_pos + 0.5*mesh_spacing and coordinates[pt,1] < midplane_pos:
                Ft[pt,1] -= (midplane_pos - coordinates[pt,1])/repuls_skin*mesh_spacing*mesh_spacing*bulk_modulus

        vol0 = np.reshape(np.repeat(Vn0, 3), (n_nodes, 3))

        Ft -= Vt * damping_coef * vol0
        Vt += Ft/(vol0*mass_density)*dt
        coordinates += Vt*dt

        Ft = np.zeros((n_nodes, 3), dtype=np.float64)