import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
ffmpeg -f image2 -framerate 25 -i computed_prefit_00%04d.png -vcodec libx264 -b:v 800k ../itokawa/computed_itokawa.avi
'''


from matplotlib import rc
rc('text', usetex=True)


def plot_cm_estimate(path = None):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        cm_history = np.loadtxt(path + "cm_time_history_mat.txt")
        P_history = np.loadtxt(path + "P_cm_hat_time_history_mat.txt")
        true_cm = np.loadtxt(path + "true_cm.obj",dtype = str)[1:].astype(float)

    else:
        cm_history = np.loadtxt("cm_time_history_mat.txt")
        P_history = np.loadtxt("P_cm_hat_time_history_mat.txt")
        true_cm = np.loadtxt("true_cm.obj",dtype = str)[1:].astype(float)

    sd_list = []

    for i in range(P_history.shape[1]):
        sd_list += [np.sqrt(np.linalg.eigvalsh([P_history[:,i].reshape(3,3)]))]

    sd_mat = np.vstack(sd_list).T

    indices = range(cm_history.shape[1])

    estimate_error = cm_history.T - true_cm
    
    # Estimate error
    plt.plot(indices,estimate_error[:,0],label = "$\Delta x$")
    plt.plot(indices,estimate_error[:,1],label = "$\Delta y$")
    plt.plot(indices,estimate_error[:,2],label = "$\Delta z$")


    # Covariance
    plt.gca().set_color_cycle(None)

    plt.plot(indices, 3 * sd_mat[0,:],'--')
    plt.plot(indices, 3 * sd_mat[1,:],'--')
    plt.plot(indices, 3 * sd_mat[2,:],'--')

    plt.gca().set_color_cycle(None)

    plt.plot(indices, - 3 * sd_mat[0,:],'--')
    plt.plot(indices, - 3 * sd_mat[1,:],'--')
    plt.plot(indices, - 3 * sd_mat[2,:],'--')

    max_val = np.amax(np.abs(estimate_error))

    plt.ylim([- 1. * max_val,1. * max_val])

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.1),ncol = 3)
    plt.xlabel("Measurement index")
    plt.ylabel("Error (m)")

    plt.title("Center of mass estimation")
    plt.show()
    plt.clf()



def plot_focal_plane():

    ranges = np.loadtxt("../build/ranges.txt")
    cm = plt.cm.get_cmap('RdYlBu')

    y_res = np.amax(ranges[:,0]) + 1;
    z_res = np.amax(ranges[:,1]) + 1;

    min_range = min([ranges[i,-1] for i in range(len(ranges[:,-1])) if ranges[i,-1] < 1e20])
    max_range = max([ranges[i,-1] for i in range(len(ranges[:,-1])) if ranges[i,-1] < 1e20])

    valid_mes = np.vstack([ranges[i,:]for i in range(len(ranges[:,-1])) if ranges[i,-1] < 1e20])
    if len(valid_mes) != len(ranges):
        invalid_mes = np.vstack([ranges[i,:]for i in range(len(ranges[:,-1])) if ranges[i,-1] > 1e20])

    sc = plt.scatter(valid_mes[:,0],valid_mes[:,1],c = valid_mes[:,2], 
        s = 60, vmin = min_range, vmax = max_range, marker = "s",cmap = cm)
    
    plt.colorbar(sc)
    plt.xlim([0,y_res - 1])
    plt.ylim([0,z_res - 1])

    plt.grid()
    plt.show()

def plot_long_lat():

    long_lat = np.loadtxt("../output/long_lat.txt")
    long_lat_rel = np.loadtxt("../output/long_lat_rel.txt")

    plt.scatter(180 / np.pi * long_lat[:,0],180 / np.pi * long_lat[:,1],c= 'b',label = 'Inertial frame')
    plt.scatter(180 / np.pi * long_lat_rel[:,0],180 / np.pi * long_lat_rel[:,1],c= 'r',label = 'Body-fixed frame')
    plt.legend(bbox_to_anchor=(0.5, 1.1),ncol = 2,loc = 'upper center')
    plt.grid()
    plt.xlim([-180,180])
    plt.ylim([-90,90])
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.savefig("long_lat.pdf")
    plt.show()

def plot_diff():
    volume_dif = np.loadtxt("../output/volume_dif.txt")
    surface_dif = np.loadtxt("../output/surface_dif.txt")

    plt.semilogy(range(len(volume_dif)),100 * np.abs(volume_dif),label = 'Volume')
    plt.semilogy(range(len(surface_dif)),100 * np.abs(surface_dif),label = 'Area')
    plt.legend(bbox_to_anchor=(0.5, 1.1),ncol = 2,loc = 'upper center')
    plt.ylabel("Relative difference (%)")
    plt.xlabel("Measurement index")

    plt.grid()
    plt.savefig("dif.pdf")
    plt.clf()


def plot_volume_dif():
    volume_dif = np.loadtxt("../output/volume_dif.txt")

    plt.plot(range(len(volume_dif)),100 * np.abs(volume_dif))
    plt.xlabel("Measurement index")
    plt.ylabel("Relative volume difference (%)")
    plt.savefig("volume_dif.pdf")


    plt.show()

def plot_surface_dif():
    surface_dif = np.loadtxt("../output/surface_dif.txt")

    plt.plot(range(len(surface_dif)),100 * np.abs(surface_dif))
    plt.xlabel("Measurement index")
    plt.ylabel("Relative surface difference (%)")
    plt.savefig("surface_dif.pdf")


    plt.show()

def plot_energy(path = None):
    if path is None:
       energy = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/build/orbit_energy.txt")
    else:
       energy = np.loadtxt(path)

    plt.plot(range(len(energy)),(energy - energy[0])/energy[0] * 100)
    plt.xlabel("Index")
    plt.ylabel("Relative change (%)")

    plt.show()

def plot_timestep(path):
    times = np.loadtxt(path)
    times_diff = np.diff(times)

    plt.plot(range(len(times_diff)),times_diff)
   
    plt.xlabel("Index")
    plt.ylabel("Time step (s)")
    plt.show()


def plot_orbit(path = None):
    
    if path is None:
        orbit = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/build/orbit.txt")
    else:
        orbit = np.loadtxt(path)

    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[0,:],orbit[1,:],orbit[2,:])
    ax.scatter(orbit[0,0],orbit[1,0],orbit[2,0],'*',color = 'g')
    ax.scatter(orbit[0,-1],orbit[1,-1],orbit[2,-1],'*',color = 'r')



    coefs = (1e-6, 1e-6 , 1e-6 )  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
    
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')


    max_radius = 3 * max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))




    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


    plt.show()
    plt.clf()






