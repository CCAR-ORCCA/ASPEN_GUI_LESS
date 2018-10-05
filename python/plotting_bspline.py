import mayavi
from mayavi import mlab
import numpy as np





points = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBSpline/build/points.txt")
mesh_points = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBSpline/build/control_mesh.txt").T


mlab.points3d(points[0,:],points[1,:],points[2,:],color = (1,1,0.),mode = "point")
mesh_node = mlab.points3d(mesh_points[0,:],mesh_points[1,:],mesh_points[2,:],color = (1,0,0.),
	mode = "point")
mesh_node.actor.property.set(representation='p', point_size=3.5)

for i in range(mesh_points.shape[1]):
	mlab.text3d(mesh_points[0,i], mesh_points[1,i], mesh_points[2,i], str(i),scale = 0.05)

# mlab.axes()
mlab.show()