import mpl_toolkits.mplot3d as a3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy as sp
ax = a3.Axes3D(plt.figure())

xmax = 1
xmin = -1
ymin = -1
ymax = 1
zmin = -1
zmax = 1

n_rays = 100
t_max = 4

for i in range(n_rays):

	u = np.array([1,1e-400,1e-400]) 
	u = u / np.linalg.norm(u)

	origin = np.array([-2,0,0]) + np.random.rand(3)
	endpoint = origin + t_max * u

	ray = np.zeros([2,3])

	ray[0,:] = origin
	ray[1,:] = endpoint


	tmin_x = (xmin - origin[0])/u[0]
	tmax_x = (xmax - origin[0])/u[0]

	tmin_y = (ymin - origin[1])/u[1]
	tmax_y = (ymax - origin[1])/u[1]

	tmin_z = (zmin - origin[2])/u[2]
	tmax_z = (zmax - origin[2])/u[2]


	all_t_sorted = np.sort(np.array([tmin_x,tmax_x,tmin_y,tmax_y,tmin_z,tmax_z]))

	test_t = 0.5 * (all_t_sorted[2] + all_t_sorted[3])

	impact = origin + test_t * u
	hit = False

	if (impact[0] < xmax and impact[0] > xmin):
		if (impact[1] < ymax and impact[1] > ymin):
			if (impact[2] < zmax and impact[2] > zmin):
				hit = True


	

	if hit is True:
		ax.plot(ray[:,0],ray[:,1],ray[:,2],c = 'green')
		
	else:
		ax.scatter(impact[0],impact[1],impact[2],c = 'orange')

		ax.plot(ray[:,0],ray[:,1],ray[:,2],c = 'red')


# Vertex data
verts = [    
      (xmin, ymin, zmin), (xmin, ymin, zmax), (xmin, ymax, zmax), (xmin, ymax, zmin),
      (xmax, ymin, zmin), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmax, ymax, zmin)
        ]

# Face data
faces = np.array([ 
   [0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5],     
   [0, 1, 5, 4], [2, 3, 7, 6]
   ])

ax.dist=30
ax.azim=-140
ax.elev=20#[enter image description here][1]

ax.set_xlim([min(xmin,ymin,zmin),max(xmax,ymax,zmax)])
ax.set_ylim([min(xmin,ymin,zmin),max(xmax,ymax,zmax)])
ax.set_zlim([min(xmin,ymin,zmin),max(xmax,ymax,zmax)])

for i in np.arange(len(faces)):
    square =[ verts[faces[i,0]], verts[faces[i,1]], verts[faces[i, 2]], verts[faces[i, 3]]]
    face = a3.art3d.Poly3DCollection([square])
    face.set_color(colors.rgb2hex(sp.rand(3)))
    face.set_edgecolor('k')
    face.set_alpha(0.5)
    ax.add_collection3d(face)




plt.show()