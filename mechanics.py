from __future__ import division
from mathfunc import *
import numpy as np
import math
import re
import os
import sys
from numba import jit, prange

# Calculate elastic forces
@jit
def tetraElasticity(At, A0, Ft, G, K, k, mu, tets, Vn, Vn0, i, eps, Ue):
	
	# Deformed volume
	vol = np.linalg.det(At)/6.0

	# Apply growth to reference state
	Ar = np.dot(G, A0) 
	#Ar = np.dot(At, G) 
	#Ar = G*np.array(A0)

	# Calculate deformation gradient
	F = np.dot(At, np.linalg.inv(Ar))   # Ar: rest tetra, At: material tetra

	# Calculate left Cauchy-Green strain tensor
	B = np.dot(F, F.transpose())

	# Calculate relative volume change and averaged nodal volume change
	J = np.linalg.det(F) # Relative volume change
	J1 = Vn[tets[i][0]]/Vn0[tets[i][0]]
	J2 = Vn[tets[i][1]]/Vn0[tets[i][1]]
	J3 = Vn[tets[i][2]]/Vn0[tets[i][2]]
	J4 = Vn[tets[i][3]]/Vn0[tets[i][3]]
	Ja = (J1 + J2 + J3 + J4)/4.0   # Averaged nodal volume change

	# Decide if need for SVD or not
	ll1, ll2, ll3 = EV(B)
	if ll3 >= eps**2 and J > 0.0: # No need for SVD

		# Calculate the total stress (shear stress + bulk stress)
		powJ23 = np.power(J, 2.0/3.0)
		S = (B - np.identity(3)*np.trace(B)/3.0)*mu/(J*powJ23) + np.identity(3)*K*(Ja-1.0)
		P = np.dot(S, np.linalg.inv(F.transpose()))*J
		W = 0.5*mu*(np.trace(B)/powJ23 - 3.0) + 0.5*K*((J1-1.0)*(J1-1.0) + (J2-1.0)*(J2-1.0) + (J3-1.0)*(J3-1.0) + (J4-1.0)*(J4-1.0))*0.25

	else:  # Needs SVD
		
		C = np.dot(F.transpose(), F)

		V = np.identity(3) 
		eva = [0.0]*3
		w2, v2 = Eigensystem(3, C, V, eva)
		#u2, w2, v2 = np.linalg.svd(C, full_matrices=True)
		#w2, v2 = np.linalg.eig(C)

		l1 = np.sqrt(w2[0])
		l2 = np.sqrt(w2[1])
		l3 = np.sqrt(w2[2])

		if np.linalg.det(v2) < 0.0:
			v2[0,0] = -v2[0,0]
			v2[1,0] = -v2[1,0]
			v2[2,0] = -v2[2,0]
		#v2 = np.transpose(v2)

		Fdi = np.identity(3)
		if l1 >= 1e-25:
			Fdi[0,0] = 1.0/l1
			Fdi[1,1] = 1.0/l2
			Fdi[2,2] = 1.0/l3

		U = np.dot(F, np.dot(v2, Fdi))

		if l1 < 1e-25:
			U[0,0] = U[1,1]*U[2,2] - U[2,1]*U[1,2]
			U[1,0] = U[2,1]*U[0,2] - U[0,1]*U[2,2]
			U[2,0] = U[0,1]*U[1,2] - U[1,1]*U[0,2]

		if np.linalg.det(F) < 0.0:
			l1 = -l1
			U[0,0] = -U[0,0]
			U[1,0] = -U[1,0]
			U[2,0] = -U[2,0]

		Pd = np.identity(3)
		pow23 = np.power(eps*l2*l3, 2.0/3.0)
		Pd[0,0] = mu/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23 + k*(l1-eps) + K*(Ja-1.0)*l2*l3
		Pd[1,1] = mu/3.0*(-eps*eps/l2 + 2.0*l2 - l3*l3/l2)/pow23 + mu/9.0*(-4.0*eps/l2 - 4.0/eps*l2 + 2.0/eps/l2*l3*l3)/pow23*(l1-eps) + K*(Ja-1.0)*l1*l3
		Pd[2,2] = mu/3.0*(-eps*eps/l3 - l2*l2/l3 + 2.0*l3)/pow23 + mu/9.0*(-4.0*eps/l3 + 2.0/eps*l2*l2/l3 - 4.0/eps*l3)/pow23*(l1-eps) + K*(Ja-1.0)*l1*l2
		P = np.dot(U, np.dot(Pd, v2.transpose()))
		W = 0.5*mu*((eps*eps + l2*l2 + l3*l3)/pow23 - 3.0) + mu/3.0*(2.0*eps - l2*l2/eps - l3*l3/eps)/pow23*(l1-eps) + 0.5*k*(l1-eps)*(l1-eps) + 0.5*K*((J1-1.0)*(J1-1.0) + (J2-1.0)*(J2-1.0) + (J3-1.0)*(J3-1.0) + (J4-1.0)*(J4-1.0))/4.0
	
	# Increment total elastic energy
	if J*J > 1e-50:
		Ue += W*vol/J
	 	
	# Calculate tetra face negative normals (because traction Ft=-P*n)
	xr1 = np.array([Ar[0,0], Ar[1,0], Ar[2,0]])
	xr2 = np.array([Ar[0,1], Ar[1,1], Ar[2,1]])
	xr3 = np.array([Ar[0,2], Ar[1,2], Ar[2,2]])
	N1 = cross(xr3, xr1)
	N2 = cross(xr2, xr3)
	N3 = cross(xr1, xr2)
	N4 = cross(xr2-xr3, xr1-xr3)

	# Distribute forces among tetra vertices
	#Ft[tets[i][0]] += np.array((np.dot(np.array(P), (N1 + N2 + N3)[np.newaxis].T).T/6.0).ravel(), dtype = float)
	Ft[tets[i][0]] += np.dot(np.array(P), (N1 + N2 + N3).T)/6.0
	Ft[tets[i][1]] += np.dot(np.array(P), (N1 + N3 + N4).T)/6.0
	Ft[tets[i][2]] += np.dot(np.array(P), (N2 + N3 + N4).T)/6.0
	Ft[tets[i][3]] += np.dot(np.array(P), (N1 + N2 + N4).T)/6.0
	#Ft[tets[i][0]] += (np.dot(np.array(P), (N1 + N2 + N3)[np.newaxis].T).T/6.0).ravel()
	#Ft[tets[i][1]] += (np.dot(np.array(P), (N1 + N3 + N4)[np.newaxis].T).T/6.0).ravel()
	#Ft[tets[i][2]] += (np.dot(np.array(P), (N2 + N3 + N4)[np.newaxis].T).T/6.0).ravel()
	#Ft[tets[i][3]] += (np.dot(np.array(P), (N1 + N2 + N4)[np.newaxis].T).T/6.0).ravel()

	return Ft, Ue

# Newton dynamics (Integrate velocity into displacement)
@jit
def move(nn, Ft, Vt, Ut, gamma, Vn0, rho, dt):
	for i in prange(nn):
		Ft[i] -= Vt[i]*gamma*Vn0[i]
		Vt[i] += Ft[i]/(Vn0[i]*rho)*dt
		Ut[i] += Vt[i]*dt
		Ft[i] = np.array([0.0,0.0,0.0])

	return Ft, Ut, Vt
