import numpy as np
import matplotlib.pyplot as plt
import polyhedron
from mpl_toolkits.mplot3d import Axes3D
import mayavi
from mayavi import mlab
import matplotlib.cm as cmx
import matplotlib.colors as colors
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)




def plot_mu_descriptor(path_pc0):
    
    descriptors = np.loadtxt(path_pc0)
    mu_descriptor = np.mean(descriptors,axis = 1)
    distances = np.zeros(descriptors.shape[1])

    l =  range(len(distances))

   

    for k in range(len(distances)):
        for i in range(len(mu_descriptor)):
            distances[k] += (descriptors[i,k] - mu_descriptor[i]) ** 2 /(descriptors[i,k] + mu_descriptor[i])
   


    distances = distances / np.std(distances)
    
    for k in l:
        if (distances[k]>3):
            plt.plot(range(len(mu_descriptor)),descriptors[:,k],color = "lightgrey")

    plt.plot(range(len(mu_descriptor)),mu_descriptor,"-o")

    plt.show()

    plt.hist(distances,bins = 20)
    plt.show()

    
    

def plot_descriptors(path_pc0,path_pc1 = None,point_range = [0]):


    descriptors = np.loadtxt(path_pc0)
    descriptors_1 = None
    if path_pc1 is not None:
	    descriptors_1 = np.loadtxt(path_pc1)


    for i in point_range:
        plt.plot(range(descriptors.shape[0]),descriptors[:,i],'o-')
        if descriptors_1 is not None:
	        plt.plot(range(descriptors_1.shape[0]),descriptors_1[:,i],'x-')

        plt.title("Point " + str(i + 1))
        plt.show()


def plot_matches(pc0_path,pc1_path,
    matches_path,index_limit = None,custom_weights_path = None):



    cmap = 'viridis'

    pc0,f0 = polyhedron.load_shape(pc0_path)
    pc1,f1 = polyhedron.load_shape(pc1_path)

    matches = np.loadtxt(matches_path)
    weights = np.loadtxt(custom_weights_path)

    
    if index_limit is not None:
        matches = matches[0:index_limit + 1,:]
        weights = weights[0:index_limit + 1]

    # print matches.shape
    # indices_active_matches = weights > 0
    # matches = matches[indices_active_matches,:]
    # weights = weights[indices_active_matches]


    print "Weights amplitude : min == " + str(min(weights)) + " , max == " + str(max(weights)) + "\n"

    cmap =  plt.get_cmap(cmap) 
    cNorm  = colors.Normalize(vmin=min(weights), vmax=max(weights))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    # Point clouds
    pc0 = np.vstack(pc0)
    pc1 = np.vstack(pc1)
    mayavi.mlab.figure(bgcolor=(66./255, 134./255, 244./255))
    p0_node = mlab.points3d(pc0[:,0],pc0[:,1],pc0[:,2],color = (1,1,0.),mode = "point")
    p1_node = mlab.points3d(pc1[:,0],pc1[:,1],pc1[:,2],color = (1,0,0),mode = "point")
    p0_node.actor.property.set(representation='p', point_size=3)
    p1_node.actor.property.set(representation='p', point_size=3)


    # Colored matches error
    all_colors = [scalarMap.to_rgba(weights[i])[0:3] for i in range(len(weights))]


    # l = mlab.plot3d(1e-10 * matches[:,0],1e-10 * matches[:,1],1e-10 * matches[:,2],weights,opacity = 0,colormap = 'viridis' )

    for i in range(matches.shape[0]):   

        print("Plotting match " + str(i+1) + " over " + str(matches.shape[0]) + "\n")

        p0 = matches[i,0:3]
        p1 = matches[i,3:6]
        if weights[i] !=0:
            mlab.plot3d([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],color = all_colors[i],tube_radius = None,line_width= 3)
        else:
            mlab.plot3d([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],color = (0.5,0.5,0.5),tube_radius = None,line_width= 3)


    # l = mlab.plot3d(range(len(distances)),distances,opacity = 0)
    # mlab.colorbar(l)
    
    # mlab.colorbar(object=l, title="Match quality", orientation=None, nb_labels=None, nb_colors=None, label_fmt=None)
    mlab.show()

plot_matches(
    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/BenchmarkICP/itokawa_64_scaled_aligned.obj",
    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/BenchmarkICP/source_3_before.obj",
    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/BenchmarkICP/build/ransac_pairs.txt",
    index_limit = 1000,
    custom_weights_path = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/BenchmarkICP/build/all_pairs_weights.txt")

# plot_descriptors(path_pc0 = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/BenchmarkICP/source_descriptors.txt",
# 	path_pc1 = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/BenchmarkICP/destination_descriptors.txt",
# 	point_range= range(100))




