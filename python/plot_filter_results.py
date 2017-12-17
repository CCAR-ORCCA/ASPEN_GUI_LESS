import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
rc('text', usetex=True)


def plot_all_results(path = "",save = False):

    plot_residuals(path,save)
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

def plot_state_error(path = "",save = False):

    X_true = np.loadtxt(path + "X_true.txt")
    X_hat = np.loadtxt(path + "X_hat.txt")
    P = np.loadtxt(path + "covariances.txt")

    sd = []

    for i in range(X_hat.shape[1]):
        sd += [np.sqrt(np.diag(P[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]))]

    sd = np.vstack(sd).T
  
    # Position
    plt.plot(range(sd.shape[1]),X_true[0,:] - X_hat[0,:])
    plt.plot(range(sd.shape[1]),X_true[1,:] - X_hat[1,:])
    plt.plot(range(sd.shape[1]),X_true[2,:] - X_hat[2,:])

    plt.gca().set_color_cycle(None)

    plt.plot(range(sd.shape[1]),3 * sd[0,:],'--')
    plt.plot(range(sd.shape[1]),3 * sd[1,:],'--')
    plt.plot(range(sd.shape[1]),3 * sd[2,:],'--')

    plt.gca().set_color_cycle(None)

    plt.plot(range(sd.shape[1]),- 3 * sd[0,:],'--')
    plt.plot(range(sd.shape[1]),- 3 * sd[1,:],'--')
    plt.plot(range(sd.shape[1]),- 3 * sd[2,:],'--')

    if save is False:
        plt.show()
    else:
        plt.savefig("position_error.pdf")

    plt.clf()
    
    # Velocity
    plt.plot(range(X_true.shape[1]),X_true[3,:] - X_hat[3,:])
    plt.plot(range(X_true.shape[1]),X_true[4,:] - X_hat[4,:])
    plt.plot(range(X_true.shape[1]),X_true[5,:] - X_hat[5,:])

    plt.gca().set_color_cycle(None)

    plt.plot(range(sd.shape[1]),3 * sd[3,:],'--')
    plt.plot(range(sd.shape[1]),3 * sd[4,:],'--')
    plt.plot(range(sd.shape[1]),3 * sd[5,:],'--')

    plt.gca().set_color_cycle(None)

    plt.plot(range(sd.shape[1]),- 3 * sd[3,:],'--')
    plt.plot(range(sd.shape[1]),- 3 * sd[4,:],'--')
    plt.plot(range(sd.shape[1]),- 3 * sd[5,:],'--')

    if save is False:
        plt.show()
    else:
        plt.savefig("velocity_error.pdf")

plot_all_results("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestFilter/build/",save = False)
