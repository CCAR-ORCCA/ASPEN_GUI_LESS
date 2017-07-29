import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
ffmpeg -f image2 -framerate 25 -i computed_prefit_00%04d.png -vcodec libx264 -b:v 800k ../itokawa/computed_itokawa.avi
'''


from matplotlib import rc
rc('text', usetex=True)



def plot_results(path = None,save = False):


    plot_cm_estimate(path ,save )
    plot_mrp_histories(path,save)
    plot_angle_error(path,save)
    plot_cm_eigen(path,save )
    plot_omega_histories(path,save)
    plot_omega_norm_histories(path,save)
    plot_spin_axis_histories(path,save)

def plot_cm_estimate(path = None,save = True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        cm_history = np.loadtxt(path + "cm_time_history_mat.txt")
        P_history = np.loadtxt(path + "P_cm_hat_time_history_mat.txt")
        true_cm = np.loadtxt(path + "true_cm.obj",dtype = str)[1:].astype(float)
        time = np.loadtxt(path + "time_history.txt")


    else:
        cm_history = np.loadtxt("cm_time_history_mat.txt")
        P_history = np.loadtxt("P_cm_hat_time_history_mat.txt")
        true_cm = np.loadtxt("true_cm.obj",dtype = str)[1:].astype(float)
        time = np.loadtxt("time_history.txt")


    sd_list = []

    for i in range(P_history.shape[1]):
        sd = np.sqrt(np.diag(P_history[:,i].reshape(3,3)))

        sd_list += [sd]


    sd_mat = np.vstack(sd_list).T

    estimate_error = cm_history.T - true_cm
    print "RMS: " + str(np.std(np.linalg.norm(estimate_error[int(len(estimate_error) / 2):,:],axis = 1))) + " m"
    
    # Estimate error
    plt.plot(time,estimate_error[1:,0],"-o",label = "$\Delta x$")
    plt.plot(time,estimate_error[1:,1],"-o",label = "$\Delta y$")
    plt.plot(time,estimate_error[1:,2],"-o",label = "$\Delta z$")

    # Covariance
    plt.gca().set_color_cycle(None)

    plt.plot(time, 3 * sd_mat[0,1:],'--')
    plt.plot(time, 3 * sd_mat[1,1:],'--')
    plt.plot(time, 3 * sd_mat[2,1:],'--')

    plt.gca().set_color_cycle(None)

    plt.plot(time, - 3 * sd_mat[0,1:],'--')
    plt.plot(time, - 3 * sd_mat[1,1:],'--')
    plt.plot(time, - 3 * sd_mat[2,1:],'--')

    max_val = np.amax(np.abs(estimate_error[2:]))

    plt.ylim([- 0.1 * max_val,0.1 * max_val])

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.1),ncol = 3)
    plt.xlabel("Measurement time (s)")
    plt.ylabel("Error (m)")

    plt.title("Center of mass estimation")
    if (save is True):
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/ASPEN_progress/Figures/com_pos.pdf")
    else:
        plt.show()
    plt.clf()




def plot_cm_eigen(path = None,save = False):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        P_history = np.loadtxt(path + "P_cm_hat_time_history_mat.txt")
        time = np.loadtxt(path + "time_history.txt")


    else:
        P_history = np.loadtxt("P_cm_hat_time_history_mat.txt")
        time = np.loadtxt("time_history.txt")


    sd_list = []

    for i in range(P_history.shape[1]):
        sd_list += [np.sqrt(np.linalg.eigvalsh([P_history[:,i].reshape(3,3)]))]

    sd_mat = np.vstack(sd_list).T


    # Covariance

    plt.semilogy(time, sd_mat[0,1:],'--',label = r"$\sigma_1$")
    plt.semilogy(time, sd_mat[1,1:],'--',label = r"$\sigma_2$")
    plt.semilogy(time, sd_mat[2,1:],'--',label = r"$\sigma_3$")


    plt.legend(loc = "best")
    plt.xlabel("Measurement time (s)")
    plt.ylabel("Eigenvalue (m)")

    plt.title("Center of mass estimated covariance eigenvalues")
    if (save is True):
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/ASPEN_progress/Figures/eigen_pos.pdf")
    else:
        plt.show()
    plt.clf()





def plot_omega_histories(path = None,save = True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        omega_mes_history = np.loadtxt(path + "omega_mes_time_history_mat.txt").T
        omega_true_history = np.loadtxt(path + "omega_true_time_history_mat.txt").T

        time = np.loadtxt(path + "time_history.txt")


    else:
        omega_mes_history = np.loadtxt("omega_mes_time_history_mat.txt").T
        omega_true_history = np.loadtxt("omega_true_time_history_mat.txt").T
        time = np.loadtxt("time_history.txt")



    # Truth 
    plt.plot(time,omega_true_history[:,0],"-o",label = "$\omega_x$")
    plt.plot(time,omega_true_history[:,1],"-o",label = "$\omega_y$")
    plt.plot(time,omega_true_history[:,2],"-o",label = "$\omega_z$")
    plt.gca().set_color_cycle(None)

    # Estimate 
    plt.plot(time,omega_mes_history[:,0],"-x",label = r"$\tilde{\omega}_x$")
    plt.plot(time,omega_mes_history[:,1],"-x",label = r"$\tilde{\omega}_y$")
    plt.plot(time,omega_mes_history[:,2],"-x",label = r"$\tilde{\omega}_z$")



    plt.legend(loc = "lower center",bbox_to_anchor = (0.5,-0.13),ncol = 6)
    plt.xlabel("Measurement time (s)")
    plt.ylabel("Angular velocity (rad/s)")

    plt.title("Angular velocity time histories")
    if (save is True):
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/ASPEN_progress/Figures/omega_histories.pdf")
    else:
        plt.show()
    plt.clf()


def plot_angle_error(path = None,save = True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        mrp_mes_history = np.loadtxt(path + "mrp_mes_time_history_mat.txt").T
        mrp_true_history = np.loadtxt(path + "mrp_true_time_history_mat.txt").T

        time = np.loadtxt(path + "time_history.txt")


    else:
        mrp_mes_history = np.loadtxt("mrp_mes_time_history_mat.txt").T
        mrp_true_history = np.loadtxt("mrp_true_time_history_mat.txt").T

        time = np.loadtxt("time_history.txt")


    angle_error =[]

    for i in range(len(time)):

        dcm_mes = mrp_to_dcm(mrp_mes_history[i,:])
        dcm_true = mrp_to_dcm(mrp_true_history[i,:])
       


        angle_error += [ np.arccos((np.trace(dcm_mes.T .dot( dcm_true)) - 1 )/2)]

    angle_error = 180. / np.pi * np.array(angle_error)


    # Truth 
    plt.plot(time,angle_error,"-o")
    
    plt.xlabel("Measurement time (s)")
    plt.ylabel("Angle error (deg)")

    plt.title("Angle error history")
    if (save is True):
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/ASPEN_progress/Figures/angle_error.pdf")
    else:
        plt.show()
    plt.clf()



def plot_mrp_histories(path = None,save = True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        mrp_mes_history = np.loadtxt(path + "mrp_mes_time_history_mat.txt").T
        mrp_true_history = np.loadtxt(path + "mrp_true_time_history_mat.txt").T

        time = np.loadtxt(path + "time_history.txt")


    else:
        mrp_mes_history = np.loadtxt("mrp_mes_time_history_mat.txt").T
        mrp_true_history = np.loadtxt("mrp_true_time_history_mat.txt").T

        time = np.loadtxt( "time_history.txt")

    # Truth 
    plt.plot(time,mrp_true_history[:,0],"-o",label = "$\sigma_x$")
    plt.plot(time,mrp_true_history[:,1],"-o",label = "$\sigma_y$")
    plt.plot(time,mrp_true_history[:,2],"-o",label = "$\sigma_z$")
    plt.gca().set_color_cycle(None)

    # Estimate 
    plt.plot(time,mrp_mes_history[:,0],"-x",label = r"$\tilde{\sigma}_x$")
    plt.plot(time,mrp_mes_history[:,1],"-x",label = r"$\tilde{\sigma}_y$")
    plt.plot(time,mrp_mes_history[:,2],"-x",label = r"$\tilde{\sigma}_z$")



    plt.legend(loc = "lower center",bbox_to_anchor = (0.5,-0.13),ncol = 6)
    plt.xlabel("Measurement time (s)")
    plt.ylabel("MRP")

    plt.title("MRP time histories")
    if (save is True):
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/ASPEN_progress/Figures/mrp_histories.pdf")
    else:
        plt.show()
    plt.clf()


def plot_omega_norm_histories(path = None,save = True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        omega_mes_history = np.loadtxt(path + "omega_mes_time_history_mat.txt").T
        omega_true_history = np.loadtxt(path + "omega_true_time_history_mat.txt").T

        time = np.loadtxt(path + "time_history.txt")

    else:
        omega_mes_history = np.loadtxt("omega_mes_time_history_mat.txt").T
        omega_true_history = np.loadtxt("omega_true_time_history_mat.txt").T
        time = np.loadtxt("time_history.txt")


    # Truth 
    plt.plot(time,np.linalg.norm(omega_true_history,axis = 1),"-o",label = "Truth")

    # Estimate 
    plt.plot(time,np.linalg.norm(omega_mes_history,axis = 1),"-x",label = "Measured")


    plt.legend(loc = "lower center",bbox_to_anchor = (0.5,-0.13),ncol = 6)
    plt.xlabel("Measurement time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.ylim([0,1.1 * (max(np.amax(np.linalg.norm(omega_true_history,axis = 1)),
        np.amax(np.linalg.norm(omega_mes_history,axis = 1))))])

    plt.title("Angular velocity norm time histories")
    if (save is True):
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/ASPEN_progress/Figures/omega_norm_histories.pdf")
    else:
        plt.show()
    plt.clf()



def plot_spin_axis_histories(path = None,save = True):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if path is not None:
        omega_mes_history = np.loadtxt(path + "omega_mes_time_history_mat.txt").T
        omega_true_history = np.loadtxt(path + "omega_true_time_history_mat.txt").T

        time = np.loadtxt(path + "time_history.txt")


    else:
        omega_mes_history = np.loadtxt("omega_mes_time_history_mat.txt").T
        omega_true_history = np.loadtxt("omega_true_time_history_mat.txt").T
        time = np.loadtxt("time_history.txt")



    # Truth 
    plt.plot(time,omega_true_history[:,0]/np.linalg.norm(omega_true_history,axis = 1),"-o",label = "$s_x$")
    plt.plot(time,omega_true_history[:,1]/np.linalg.norm(omega_true_history,axis = 1),"-o",label = "$s_y$")
    plt.plot(time,omega_true_history[:,2]/np.linalg.norm(omega_true_history,axis = 1),"-o",label = "$s_z$")
    plt.gca().set_color_cycle(None)

    # Estimate 
    plt.plot(time,omega_mes_history[:,0]/np.linalg.norm(omega_mes_history,axis = 1),"-x",label = r"$\tilde{s}_x$")
    plt.plot(time,omega_mes_history[:,1]/np.linalg.norm(omega_mes_history,axis = 1),"-x",label = r"$\tilde{s}_y$")
    plt.plot(time,omega_mes_history[:,2]/np.linalg.norm(omega_mes_history,axis = 1),"-x",label = r"$\tilde{s}_z$")



    plt.legend(loc = "lower center",bbox_to_anchor = (0.5,-0.13),ncol = 6)
    plt.xlabel("Measurement time (s)")
    plt.ylabel("Spin axis component")

    plt.title("Spin axis time histories")
    if (save is True):
        plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/ASPEN_progress/Figures/spin_axis_histories.pdf")
    else:
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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    volume_dif = np.loadtxt("./volume_dif.txt")
    surface_dif = np.loadtxt("./surface_dif.txt")

    plt.semilogy(range(len(volume_dif)),100 * np.abs(volume_dif),label = 'Volume')
    plt.semilogy(range(len(surface_dif)),100 * np.abs(surface_dif),label = 'Area')
    plt.legend(bbox_to_anchor=(0.5, 1.1),ncol = 2,loc = 'upper center')
    plt.ylabel(r"Relative difference $(\%)$")
    plt.xlabel(r"Measurement index")

    plt.grid()
    plt.savefig("dif.pdf")
    plt.clf()


def plot_volume_dif():

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    volume_dif = np.loadtxt("./volume_dif.txt")

    plt.semilogy(range(len(volume_dif)),100 * np.abs(volume_dif))
    plt.xlabel(r"Measurement index")
    plt.ylabel(r"Relative volume difference $(\%)$")
    plt.savefig("volume_dif.pdf")


    plt.show()

def plot_surface_dif():

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    surface_dif = np.loadtxt("./surface_dif.txt")

    plt.semilogy(range(len(surface_dif)),100 * np.abs(surface_dif))
    plt.xlabel(r"Measurement index")
    plt.ylabel(r"Relative surface difference $(\%)$")
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



def mrp_to_dcm(sigma):
  '''
  MRP to DCM
  Inputs:
  ------
  - sigma : (3-by-1) MRP
  Outputs:
  ------
  - dcm : (3-by-3 ) DCM
  '''
  dcm = np.eye(3) + (8 * tilde(sigma).dot(tilde(sigma)) - 4 * (1 - np.linalg.norm(sigma) ** 2) * tilde(sigma))/(1 + np.linalg.norm(sigma)**2)**2
  return dcm





def tilde(a):

    '''
    Returns the skew-symmetric matrix corresponding to the linear mapping cross(a,.)
    Inputs : 
    ------
    a : (3-by-1 np.array) vector
    Outputs:
    ------
    atilde : (3-by-3 np.array) linear mapping matrix
    '''
    atilde = np.array([[0,-a[2],a[1]],
        [a[2],0,-a[0]],
        [-a[1],a[0],0]])
    return atilde
