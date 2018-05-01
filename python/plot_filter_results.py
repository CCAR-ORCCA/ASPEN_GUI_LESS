import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from polyhedron import plot_shape
import RigidBodyKinematics as RBK

rc('text', usetex=True)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def plot_all_results(path = "",savepath = None):

    # plot_residuals(path,save)
    # plot_orbit(path,save)
    plot_cart_state_error_inertial(path,savepath)
    plot_attitude_state_inertial(path)
    plot_state_error_RIC(path)

# def plot_residuals(path = "",save = False):

#     residuals = np.loadtxt(path + "residuals.txt")
    
#     for mes in range(residuals.shape[0]):
#         plt.scatter(range(residuals.shape[1]),residuals[mes,:])

#     if save is False:
#         plt.show()
#     else:
#         plt.savefig("residuals.pdf")
#     plt.clf()


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


def plot_state_error_RIC(path = "",save = False):

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
    plt.plot(T_obs,X_true_RIC[0,:] - X_hat_RIC[0,:],'.',label = "radial")
    plt.plot(T_obs,X_true_RIC[1,:] - X_hat_RIC[1,:],'.',label = "in-track")
    plt.plot(T_obs,X_true_RIC[2,:] - X_hat_RIC[2,:],'.',label = "cross-track")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[0,:],"-+")
    plt.plot(T_obs,3 * sd[1,:],"-+")
    plt.plot(T_obs,3 * sd[2,:],"-+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[0,:],"-+")
    plt.plot(T_obs,- 3 * sd[1,:],"-+")
    plt.plot(T_obs,- 3 * sd[2,:],"-+")
    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.1),ncol = 3)
    plt.xlabel("Time (min)")
    plt.ylabel("Position error (m)")

    if save is False:
        plt.show()
    else:
        plt.savefig("position_error_RIC.pdf")

    plt.clf()
    
    # Velocity
    plt.plot(T_obs,100 * (X_true_RIC[3,:] - X_hat_RIC[3,:]),'.',label = "radial")
    plt.plot(T_obs,100 * (X_true_RIC[4,:] - X_hat_RIC[4,:]),'.',label = "in-track")
    plt.plot(T_obs,100 * (X_true_RIC[5,:] - X_hat_RIC[5,:]),'.',label = "cross-track")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[3,:] * 100,"-+")
    plt.plot(T_obs,3 * sd[4,:] * 100,"-+")
    plt.plot(T_obs,3 * sd[5,:] * 100,"-+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[3,:] * 100,"-+")
    plt.plot(T_obs,- 3 * sd[4,:] * 100,"-+")
    plt.plot(T_obs,- 3 * sd[5,:] * 100,"-+")

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.1),ncol = 3)
    plt.xlabel("Time (min)")
    plt.ylabel("Velocity error (cm/s)")

    plt.ylim([- 5 * sd[3,2] * 100,5 * sd[3,2] * 100])


    if save is False:
        plt.show()
    else:
        plt.savefig("velocity_error_RIC.pdf")

    plt.clf()


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

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "_attitude.pdf")

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
    plt.gcf().tight_layout()

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "_error_attitude.pdf")

    plt.clf()
    plt.gca().set_color_cycle(None)

    # Velocity error
    plt.plot(T_obs,X_true[9,:] - X_hat[9,:],'-o',label = r"$\omega_1$")
    plt.plot(T_obs,X_true[10,:] - X_hat[10,:],'-o',label = r"$\omega_2$")
    plt.plot(T_obs,X_true[11,:] - X_hat[11,:],'-o',label = r"$\omega_3$")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[9,:],"--+")
    plt.plot(T_obs,3 * sd[10,:],"--+")
    plt.plot(T_obs,3 * sd[11,:],"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[9,:],"--+")
    plt.plot(T_obs,- 3 * sd[10,:],"--+")
    plt.plot(T_obs,- 3 * sd[11,:],"--+")

    plt.xlabel("Time (min)")
    plt.ylabel("Omega error (rad/s)")
    plt.gcf().tight_layout()

    plt.ylim([- 5 * sd[9,2],5 * sd[9,2]])

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "_error_omega.pdf")

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



    # plt.plot(T_obs,np.mean(X_true[0,:] - X_hat[0,:]) * np.ones(len(T_obs)),'--')
  
    # plt.plot(T_obs,np.mean(X_true[1,:] - X_hat[1,:]) * np.ones(len(T_obs)),'--')

    # plt.plot(T_obs,np.mean(X_true[2,:] - X_hat[2,:]) * np.ones(len(T_obs)),'--')
    # plt.show()
    # plt.clf()

    # plt.gca().set_color_cycle(None)

  
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
        plt.savefig(savepath + "_error_pos.pdf")

    plt.clf()

    # Position
    plt.plot(range(len(T_obs)),X_true[0,:] - X_hat[0,:],'-o',label = "X")
    plt.plot(range(len(T_obs)),X_true[1,:] - X_hat[1,:],'-o',label = "Y")
    plt.plot(range(len(T_obs)),X_true[2,:] - X_hat[2,:],'-o',label = "Z")

    plt.gca().set_color_cycle(None)

    plt.plot(range(len(T_obs)),3 * sd[0,:],"--+")
    plt.plot(range(len(T_obs)),3 * sd[1,:],"--+")
    plt.plot(range(len(T_obs)),3 * sd[2,:],"--+")

    plt.gca().set_color_cycle(None)

    plt.plot(range(len(T_obs)),- 3 * sd[0,:],"--+")
    plt.plot(range(len(T_obs)),- 3 * sd[1,:],"--+")
    plt.plot(range(len(T_obs)),- 3 * sd[2,:],"--+")

    plt.xlabel("Index ")
    plt.ylabel("Position error (m)")
    plt.gcf().tight_layout()

    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)
    plt.show()

    plt.clf()
    
    # Velocity
    plt.plot(T_obs,100 * (X_true[3,:] - X_hat[3,:]),'.',label = "X")
    plt.plot(T_obs,100 * (X_true[4,:] - X_hat[4,:]),'.',label = "Y")
    plt.plot(T_obs,100 * (X_true[5,:] - X_hat[5,:]),'.',label = "Z")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,3 * sd[3,:] * 100,"-+")
    plt.plot(T_obs,3 * sd[4,:] * 100,"-+")
    plt.plot(T_obs,3 * sd[5,:] * 100,"-+")

    plt.gca().set_color_cycle(None)

    plt.plot(T_obs,- 3 * sd[3,:] * 100,"-+")
    plt.plot(T_obs,- 3 * sd[4,:] * 100,"-+")
    plt.plot(T_obs,- 3 * sd[5,:] * 100,"-+")

    plt.ylim([- 5 * sd[3,2] * 100,5 * sd[3,2] * 100])

    plt.xlabel("Time (min)")
    plt.ylabel("Velocity error (cm/s)")
    plt.gcf().tight_layout()
    plt.legend(loc = "upper center",bbox_to_anchor = (0.5,1.05),ncol = 3,framealpha = 1)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + "_error_vel.pdf")
 
path_no_icp = "/Users/bbercovici/GDrive/CUBoulder/Research/conferences/snowbird_2018/paper/Figures/no_ICP"
path_with_icp = "/Users/bbercovici/GDrive/CUBoulder/Research/conferences/snowbird_2018/paper/Figures/with_ICP"

plot_all_results("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/filter/",savepath = None )

