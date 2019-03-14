# Write output .pov files
import numpy as np
import math
import os
#from sklearn import preprocessing
from vapory import *

# Writes POV-Ray source files and output in .png files
def writePov(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom, zoom_pos):

	povname = "%s/pov_H%fAT%f/B%d.png"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step)

	foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

	try:
		if not os.path.exists(foldname):
			os.makedirs(foldname)
	except OSError:
		print ('Error: Creating directory. ' + foldname)

	# Normals in deformed state
	N = np.zeros((nsn,3), dtype = float)

	for i in range(len(faces)):
		Ntmp = np.cross(Ut[faces[i][1]] - Ut[faces[i][0]], Ut[faces[i][2]] - Ut[faces[i][0]])
		N[SNb[faces[i][0]]] += Ntmp
		N[SNb[faces[i][1]]] += Ntmp
		N[SNb[faces[i][2]]] += Ntmp
	#N = preprocessing.normalize(N)
	for i in range(nsn):
		N_norm = np.linalg.norm(N[i])
		N[i] *= 1.0/N_norm

	#os.path.dirname(povname)

	camera = Camera('location', [-3*zoom, 3*zoom, -3*zoom], 'look_at', [0, 0, 0], 'sky', [0, 0, -1], 'focal_point', [-0.55, 0.55, -0.55], 'aperture', 0.055, 'blur_samples', 10)
	light = LightSource([-14, 3, -14], 'color', [1, 1, 1])
	background = Background('color', [1,1,1])

	vertices = []
	normals = []
	f_indices = []
	for i in range(nsn):
	   	vertex = [Ut[SN[i]][0]*zoom_pos, Ut[SN[i]][1]*zoom_pos, Ut[SN[i]][2]*zoom_pos]
		vertices.append(vertex)
		normal = [N[i][0], N[i][1], N[i][2]]
		normals.append(normal)
	for i in range(len(faces)):
		f_indice = [SNb[faces[i][0]], SNb[faces[i][1]], SNb[faces[i][2]]]
		f_indices.append(f_indice)

	Mesh = Mesh2(VertexVectors(nsn, *vertices), NormalVectors(nsn, *normals), FaceIndices(len(faces), *f_indices), 'inside_vector', [0,1,0])
	box = Box([-100, -100, -100], [100, 100, 100])
	pigment = Pigment( 'color', [1, 1, 0.5])
	normal = Normal( 'bumps', 0.05, 'scale', 0.005)
	finish = Finish( 'phong', 1, 'reflection', 0.05, 'ambient', 0, 'diffuse', 0.9)

	intersection = Intersection(Mesh, box, Texture(pigment, normal, finish))

	scene = Scene(camera, objects= [light, background, intersection], included = ["colors.inc"])
	#scene.render(povname, width=400, height=300, quality = 9, antialiasing = 1e-5 )
	scene.render(povname, width=400, height=300, quality = 9)


def writePov2(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom, zoom_pos):

	povname = "B%d.pov"%(step)

	foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

	try:
		if not os.path.exists(foldname):
			os.makedirs(foldname)
	except OSError:
		print ('Error: Creating directory. ' + foldname)

	completeName = os.path.join(foldname, povname) 
	filepov = open(completeName, "w")

	# Normals in deformed state
	N = np.zeros((nsn,3), dtype = float)

	for i in range(len(faces)):
		Ntmp = np.cross(Ut[faces[i][1]] - Ut[faces[i][0]], Ut[faces[i][2]] - Ut[faces[i][0]])
		N[SNb[faces[i][0]]] += Ntmp
		N[SNb[faces[i][1]]] += Ntmp
		N[SNb[faces[i][2]]] += Ntmp

	for i in range(nsn):
		N_norm = np.linalg.norm(N[i])
		N[i] *= 1.0/N_norm

	filepov.write("#include \"colors.inc\"\n")
	filepov.write("background { color rgb <1,1,1> }\n")
	filepov.write("camera { location <" + str(-3*zoom) + ", " + str(3*zoom) + ", " + str(-3*zoom) + "> look_at <0, 0, 0> sky <0, 0, -1> focal_point <-0.55, 0.55, -0.55> aperture 0.055 blur_samples 10 }\n")
	filepov.write("light_source { <-14, 3, -14> color rgb <1, 1, 1> }\n")

	filepov.write("intersection {\n")
	filepov.write("mesh2 { \n")
	filepov.write("vertex_vectors { " + str(nsn) + ",\n")
	for i in range(nsn):
		filepov.write("<" + "{0:.5f}".format(Ut[SN[i]][0]*zoom_pos) + "," + "{0:.5f}".format(Ut[SN[i]][1]*zoom_pos) + "," + "{0:.5f}".format(Ut[SN[i]][2]*zoom_pos) + ">,\n")
	filepov.write("} normal_vectors { " + str(nsn) + ",\n")
	for i in range(nsn):
		filepov.write("<" + "{0:.5f}".format(N[i][0]) + "," + "{0:.5f}".format(N[i][1]) + "," + "{0:.5f}".format(N[i][2]) + ">,\n")
	filepov.write("} face_indices { " + str(len(faces)) + ",\n")
	for i in range(len(faces)):
		filepov.write("<" + str(SNb[faces[i][0]]) + "," + str(SNb[faces[i][1]]) + "," + str(SNb[faces[i][2]]) + ">,\n")
	filepov.write("} inside_vector<0,1,0> }\n")
	filepov.write("box { <-100, -100, -100>, <100, 100, 100> }\n")
	filepov.write("pigment { rgb<1,1,0.5> } normal { bumps 0.05 scale 0.005 } finish { phong 1 reflection 0.05 ambient 0 diffuse 0.9 } }\n")

	filepov.close()


# Write surface mesh in .txt files
def writeTXT(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE, step, Ut, faces, SN, SNb, nsn, zoom_pos):

	txtname = "B%d.txt"%(step)

	foldname = "%s/pov_H%fAT%f/"%(PATH_DIR, THICKNESS_CORTEX, GROWTH_RELATIVE)

	try:
		if not os.path.exists(foldname):
			os.makedirs(foldname)
	except OSError:
		print ('Error: Creating directory. ' + foldname)
	
	completeName = os.path.join(foldname, txtname) 
	filetxt = open(completeName, "w") 
	filetxt.write(str(nsn) + "\n")
	for i in range(nsn):
		filetxt.write(str(Ut[SN[i]][0]*zoom_pos) + " " + str(Ut[SN[i]][1]*zoom_pos) + " " + str(Ut[SN[i]][2]*zoom_pos) + "\n")
	filetxt.write(str(len(faces)) + "\n")
	for i in range(len(faces)):
		filetxt.write(str(SNb[faces[i][0]]+1) + " " + str(SNb[faces[i][1]]+1) + " " + str(SNb[faces[i][2]]+1) + "\n")
	filetxt.close()

