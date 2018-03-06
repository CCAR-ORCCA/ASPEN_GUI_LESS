import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from polyhedron import plot_shape

rc('text', usetex=True)


def plot_all_results(path = "",save = False):

    # plot_residuals(path,save)
    # plot_orbit(path,save)
    plot_state_error(path,save)

def plot_residuals(path = "",save = False):

    residuals = np.loadtxt(path + "residuals.txt")
    
    for mes in range(residuals.shape[0]):
        plt.scatter(range(residuals.shape[1]),residuals[mes,:])

    if save is False:
        plt.show()
    else:
        plt.savefig("residuals.pdf")
    plt.clf()


def plot_orbit(path,save = False):

    X_true = np.loadtxt(path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_true[0,:]/1000.,X_true[1,:]/1000.,X_true[2,:]/1000.)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Y (km)")


    plot_shape("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_64_scaled_aligned.obj",
        already_in_body_frame = True,ax = ax,scale_factor = 1)


    if save is False :
        plt.show()
    else:
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/conferences/GNSKi_2018/paper/Figures/traj_body_frame.pdf")



def plot_state_error(path = "",save = False):

    X_true = np.loadtxt(path + "X_true.txt")
    X_hat = np.loadtxt(path + "X_hat.txt")
    P = np.loadtxt(path + "covariances.txt")
    T_obs = np.loadtxt(path + "nav_times.txt") / 60


    sd = []

    for i in range(X_hat.shape[1]):
        sd += [np.sqrt(np.diag(P[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]))]

    sd = np.vstack(sd).T
  
    # Position
    plt.plot(T_obs,X_true[0,:] - X_hat[0,:],'-o',label = "radial")
    plt.plot(T_obs,X_true[1,:] - X_hat[1,:],'-o',label = "in-track")
    plt.plot(T_obs,X_true[2,:] - X_hat[2,:],'-o',label = "cross-track")

    plt.gca().set_color_cycle(None)

    plt.scatter(T_obs,3 * sd[0,:])
    plt.scatter(T_obs,3 * sd[1,:])
    plt.scatter(T_obs,3 * sd[2,:])

    plt.gca().set_color_cycle(None)

    plt.scatter(T_obs,- 3 * sd[0,:])
    plt.scatter(T_obs,- 3 * sd[1,:])
    plt.scatter(T_obs,- 3 * sd[2,:])
    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.1),ncol = 3)
    plt.xlabel("Time (min)")
    plt.ylabel("Position error (m)")


    if save is False:
        plt.show()
    else:
        plt.savefig("position_error.pdf")

    plt.clf()
    
    # Velocity
    plt.plot(T_obs,100 * (X_true[3,:] - X_hat[3,:]),'-o',label = "radial")
    plt.plot(T_obs,100 * (X_true[4,:] - X_hat[4,:]),'-o',label = "in-track")
    plt.plot(T_obs,100 * (X_true[5,:] - X_hat[5,:]),'-o',label = "cross-track")

    plt.gca().set_color_cycle(None)

    plt.scatter(T_obs,3 * sd[3,:] * 100)
    plt.scatter(T_obs,3 * sd[4,:] * 100)
    plt.scatter(T_obs,3 * sd[5,:] * 100)

    plt.gca().set_color_cycle(None)

    plt.scatter(T_obs,- 3 * sd[3,:] * 100)
    plt.scatter(T_obs,- 3 * sd[4,:] * 100)
    plt.scatter(T_obs,- 3 * sd[5,:] * 100)

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.1),ncol = 3)
    plt.xlabel("Time (min)")
    plt.ylabel("Velocity error (cm/s)")
    plt.ylim([-1e-3,1e-3])

    if save is False:
        plt.show()
    else:
        plt.savefig("velocity_error.pdf")

plot_all_results("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/filter/",save = False)
