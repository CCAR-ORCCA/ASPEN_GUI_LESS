import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from polyhedron import plot_shape
import RigidBodyKinematics as RBK
import matplotlib.ticker as mtick

rc('text', usetex=True)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def plot_all_results(path = "",savepath = None):

    # plot_residuals(path,save)
    if savepath is not None:
        save = True
    plot_orbit(path)
    plot_cart_state_error_inertial(path,savepath)
    plot_state_error_RIC(path,savepath)
    plot_attitude_state_inertial(path,savepath)


def plot_orbit_planar(path,savepath = "/Users/bbercovici/GDrive/CUBoulder/Research/papers/UQ_NAV_JGCD/R0/Figures/trajectory_BF.pdf"):

    X_true = np.loadtxt(path + "trajectory_harmo.txt")



    plt.plot(X_true[0,:]/1000,X_true[1,:]/1000)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.show()

    plt.clf()
    plt.plot(X_true[0,:]/1000,X_true[2,:]/1000)
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.axis('equal')
    plt.show()

    plt.clf()
    plt.plot(X_true[1,:]/1000,X_true[2,:]/1000)
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.axis('equal')
    plt.show()


    plt.clf()






def plot_orbit(path,savepath = "/Users/bbercovici/GDrive/CUBoulder/Research/papers/UQ_NAV_JGCD/R0/Figures/trajectory_BF.pdf"):

    X_true = np.loadtxt(path + "trajectory_harmo.txt")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_true[0,:]/1000,X_true[1,:]/1000,X_true[2,:]/1000)


    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    # plt.show()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


    plt.tight_layout()
    plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/papers/UQ_NAV_JGCD/R0/Figures/trajectory_inertial.pdf")
    plt.cla()
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X_true_BF = np.loadtxt(path + "trajectory_harmo.txt")
    for i in range(X_true_BF.shape[1]):
        BN = RBK.mrp_to_dcm(X_true[6:9,i])
        X_true_BF[0:3,i] = BN.dot(X_true_BF[0:3,i])

    ax.plot(X_true_BF[0,:]/1000,X_true_BF[1,:]/1000,X_true_BF[2,:]/1000)
    

    # plot_shape("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_8_scaled.obj",ax = ax,scale_factor = 0.001,
    #     show = False)


    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.grid(False)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


    # plt.show()
    plt.tight_layout()


    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)

    plt.clf()
    plt.cla()


def plot_state_error_RIC(path = "",savepath= None):

    X_true = np.loadtxt(path + "X_true.txt")
    X_hat = np.loadtxt(path + "X_hat.txt")
    P = np.loadtxt(path + "covariances.txt")
    T_obs = np.loadtxt(path + "nav_times.txt") / 60


    # The states must be converted to RIC frame

    X_true_RIC = np.zeros(X_true.shape)
    X_hat_RIC = np.zeros(X_hat.shape)
    P_RIC = np.zeros(P.shape)


    for i in range(len(T_obs)):

        # This way the body states are set
        X_true_RIC[:,i] = X_true[:,i]
        X_hat_RIC[:,i] = X_hat[:,i]
        P_RIC[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]] = P[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]

        # True state
        r1_N_true = normalized(X_true[0:3,i])
        r3_N_true = normalized(np.cross(X_true[0:3,i],X_true[3:6,i]))
        r2_N_true = np.cross(r3_N_true,r1_N_true)
        RN_true = np.zeros([3,3])
        RN_true[0,:] = r1_N_true
        RN_true[1,:] = r2_N_true
        RN_true[2,:] = r3_N_true
        X_true_RIC[0:3,i] = RN_true.dot(X_true[0:3,i])

        # Estimated state
        X_hat_RIC[0:3,i] = RN_true.dot(X_hat[0:3,i])
        P_RIC[0:3,i * P.shape[0] : i * P.shape[0] + 3] = RN_true.dot(P_RIC[0:3,i * P.shape[0] : i * P.shape[0] + 3]).dot(RN_true.T)
        P_RIC[3:6,i * P.shape[0] + 3 : i * P.shape[0] + 6] = RN_true.dot(P_RIC[3:6,i * P.shape[0] + 3 : i * P.shape[0] + 6]).dot(RN_true.T)


    sd = []

    for i in range(X_hat.shape[1]):
        sd += [np.sqrt(np.diag(P_RIC[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]))]

    sd = np.vstack(sd).T
  
    # Position
    plt.plot(T_obs,X_true_RIC[0,:] - X_hat_RIC[0,:],'-o',label = "radial")
    plt.plot(T_obs,X_true_RIC[1,:] - X_hat_RIC[1,:],'-o',label = "in-track")
    plt.plot(T_obs,X_true_RIC[2,:] - X_hat_RIC[2,:],'-o',label = "cross-track")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[0,:],"--+")
    plt.plot(T_obs,3 * sd[1,:],"--+")
    plt.plot(T_obs,3 * sd[2,:],"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[0,:],"--+")
    plt.plot(T_obs,- 3 * sd[1,:],"--+")
    plt.plot(T_obs,- 3 * sd[2,:],"--+")
    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)
    plt.xlabel("Time (min)")
    plt.ylabel("Position error (m)")
    plt.tight_layout()

    if savepath is None:
        plt.show()
    else:
        plt.savefig( savepath + "/position_error_RIC.pdf")

    plt.clf()
    plt.cla()

    
    # Velocity
    plt.plot(T_obs,100 * (X_true_RIC[3,:] - X_hat_RIC[3,:]),'-o',label = "radial")
    plt.plot(T_obs,100 * (X_true_RIC[4,:] - X_hat_RIC[4,:]),'-o',label = "in-track")
    plt.plot(T_obs,100 * (X_true_RIC[5,:] - X_hat_RIC[5,:]),'-o',label = "cross-track")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[5,:] * 100,"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[5,:] * 100,"--+")

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)
    plt.xlabel("Time (min)")
    plt.ylabel("Velocity error (cm/s)")

    plt.ylim([- 5 * sd[3,2] * 100,5 * sd[3,2] * 100])
    plt.tight_layout()


    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "/velocity_error_RIC.pdf")

    plt.clf()
    plt.cla()


def plot_attitude_state_inertial(path = "",savepath = None):

    X_true = np.loadtxt(path + "X_true.txt")
    X_hat = np.loadtxt(path + "X_hat.txt")
    P = np.loadtxt(path + "covariances.txt")
    T_obs = np.loadtxt(path + "nav_times.txt") / 60

    mrp_error = np.zeros([3,len(T_obs)])

    sd = []

    for i in range(X_hat.shape[1]):
        sd += [np.sqrt(np.diag(P[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]))]
        mrp_error[:,i] = RBK.dcm_to_mrp(RBK.mrp_to_dcm(X_true[6:9,i]).T.dot(RBK.mrp_to_dcm(X_hat[6:9,i])))

    sd = np.vstack(sd).T

    # estimated MRP 
    plt.plot(T_obs,X_hat[6,:],'-o',label = r"$\sigma_1$")
    plt.plot(T_obs,X_hat[7,:],'-o',label = r"$\sigma_2$")
    plt.plot(T_obs,X_hat[8,:],'-o',label = r"$\sigma_3$")

    plt.xlabel("Time (min)")
    plt.ylabel("MRP (estimated)")
    plt.gcf().tight_layout()

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)
    plt.tight_layout()
    
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "/attitude.pdf")

    plt.clf()
    plt.gca().set_color_cycle(None)


    # Position error
    plt.plot(T_obs,mrp_error[0,:],'-o',label = r"$\sigma_1$")
    plt.plot(T_obs,mrp_error[1,:],'-o',label = r"$\sigma_2$")
    plt.plot(T_obs,mrp_error[2,:],'-o',label = r"$\sigma_3$")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[6,:],"--+")
    plt.plot(T_obs,3 * sd[7,:],"--+")
    plt.plot(T_obs,3 * sd[8,:],"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[6,:],"--+")
    plt.plot(T_obs,- 3 * sd[7,:],"--+")
    plt.plot(T_obs,- 3 * sd[8,:],"--+")

    plt.xlabel("Time (min)")
    plt.ylabel("MRP error")
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

    plt.gcf().tight_layout()

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)
    plt.tight_layout()
    
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "/error_attitude.pdf")

    plt.clf()
    plt.gca().set_color_cycle(None)

    # Velocity error

    r2d = 180./np.pi
    plt.plot(T_obs,r2d * (X_true[9,:] - X_hat[9,:]),'-o',label = r"$\omega_1$")
    plt.plot(T_obs,r2d * (X_true[10,:] - X_hat[10,:]),'-o',label = r"$\omega_2$")
    plt.plot(T_obs,r2d * (X_true[11,:] - X_hat[11,:]),'-o',label = r"$\omega_3$")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,r2d * 3 * sd[9,:],"--+")
    plt.plot(T_obs,r2d * 3 * sd[10,:],"--+")
    plt.plot(T_obs,r2d * 3 * sd[11,:],"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * r2d* sd[9,:],"--+")
    plt.plot(T_obs,- 3 * r2d* sd[10,:],"--+")
    plt.plot(T_obs,- 3 * r2d* sd[11,:],"--+")

    plt.xlabel("Time (min)")
    plt.ylabel("Omega error (deg/s)")
    plt.gcf().tight_layout()

    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    plt.ylim([- 5 * r2d * sd[9,2],5 * r2d * sd[9,2]])

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "/error_omega.pdf")

    plt.clf()

    # angle error
    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,4 * np.arctan(np.linalg.norm(mrp_error,axis = 0)) * 180./np.pi,'-o')
   
    plt.xlabel("Time (min)")
    plt.ylabel("Angle error (deg)")

    plt.gcf().tight_layout()

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "/error_angle.pdf")

    plt.clf()








    

def plot_cart_state_error_inertial(path = "",savepath = None):

    X_true = np.loadtxt(path + "X_true.txt")
    X_hat = np.loadtxt(path + "X_hat.txt")
    P = np.loadtxt(path + "covariances.txt")
    T_obs = np.loadtxt(path + "nav_times.txt") / 60

    X_true_augmented = np.zeros([4,len(T_obs)])
    X_hat_augmented = np.zeros([4,len(T_obs)])

    X_true_augmented[0,:] = T_obs
    X_hat_augmented[0,:] = T_obs

    X_hat_augmented[1:,:] = X_hat[0:3,:]
    X_true_augmented[1:,:] = X_true[0:3,:]

    np.savetxt(path + "X_hat_augmented.txt",X_hat_augmented)
    np.savetxt(path + "X_true_augmented.txt",X_true_augmented)

    sd = []

    for i in range(X_hat.shape[1]):
        sd += [np.sqrt(np.diag(P[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]))]

    sd = np.vstack(sd).T
  
    # Position
    plt.plot(T_obs,X_true[0,:] - X_hat[0,:],'-o',label = "X")
    plt.plot(T_obs,X_true[1,:] - X_hat[1,:],'-o',label = "Y")
    plt.plot(T_obs,X_true[2,:] - X_hat[2,:],'-o',label = "Z")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[0,:],"--+")
    plt.plot(T_obs,3 * sd[1,:],"--+")
    plt.plot(T_obs,3 * sd[2,:],"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[0,:],"--+")
    plt.plot(T_obs,- 3 * sd[1,:],"--+")
    plt.plot(T_obs,- 3 * sd[2,:],"--+")
    plt.xlabel("Time (min)")
    plt.ylabel("Position error (m)")
    plt.gcf().tight_layout()

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "/error_pos.pdf")

    plt.clf()
    
    # Velocity
    plt.plot(T_obs,100 * (X_true[3,:] - X_hat[3,:]),'-o',label = "X")
    plt.plot(T_obs,100 * (X_true[4,:] - X_hat[4,:]),'-o',label = "Y")
    plt.plot(T_obs,100 * (X_true[5,:] - X_hat[5,:]),'-o',label = "Z")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[5,:] * 100,"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[5,:] * 100,"--+")

    plt.ylim([- 5 * sd[3,2] * 100,5 * sd[3,2] * 100])

    plt.xlabel("Time (min)")
    plt.ylabel("Velocity error (cm/s)")
    plt.gcf().tight_layout()
    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "/error_vel.pdf")


    plt.clf()
    plt.cla()


# plot_all_results("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/filter/",
#     savepath = "/Users/bbercovici/GDrive/CUBoulder/Research/papers/UQ_NAV_JGCD/R0/Figures" )

# plot_all_results("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/filter/",
#     savepath = None )

plot_orbit_planar("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/IOD/output/traj/",savepath = None)
