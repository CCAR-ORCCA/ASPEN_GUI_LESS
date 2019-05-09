import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import polyhedron
import RigidBodyKinematics as RBK
import matplotlib.ticker as mtick
import os
from pprint import pprint
from IOD_results_plots import draw_sphere
from multiprocessing import Pool
from np_array_to_latex import np_array_to_latex

import json
rc('text', usetex=True)


def create_input_tables(all_results_dirs,save_path,table_size = 10):
    n_tables = int(float(len(all_results_dirs)) / table_size)
    size_last_table = len(all_results_dirs) - n_tables * table_size
    
    print((len(all_results_dirs),n_tables,size_last_table))
    for table_index in range(n_tables):

        table = np.zeros([5,table_size])


        for i in range(table_size):
            data = {}
            with open(all_results_dirs[table_index * table_size + i] + "/input_file.json") as f:
                data = json.load(f)


            table[0,i] = data["LOS_NOISE_SD_BASELINE"]
            table[1,i] = 1e9 * data["PROCESS_NOISE_SIGMA_VEL"]
            table[2,i] = 1e9 * data["PROCESS_NOISE_SIGMA_OMEG"]
            table[3,i] = int(1./data["INSTRUMENT_FREQUENCY_NAV"])
            table[4,i] = str(int(data["SHAPE_RECONSTRUCTION_OUTPUT_DIR"].split("_")[-1][0]) + 1 + (6 if "robustness" in str(data["SHAPE_RECONSTRUCTION_OUTPUT_DIR"]) else 0))

            print([table_index * table_size + i,table[4,i]])

        np_array_to_latex(table,save_path + "/input_table_case_" + str(table_index * table_size + 1)  + "_to_" + str(table_index * table_size + table_size) ,
        row_headers = [r"$\sigma_{\rho}$ ($\mathrm{m}$)",
        r"$\sigma_{\ddot{\mathbf{r}}}$ ($\mathrm{nm/s^2}$)",
        r"$\sigma_{\dot{\boldsymbol{\omega}}}$ ($\mathrm{nrad/s^2}$)",
        r"$T_{\mathrm{obs}}$ ($\mathrm{s}$)",
        "Input Case"],
        column_headers = ["Case " + str(i) for i in range(table_index * table_size + 1,table_index * table_size + 1 + table_size + 1)],
     column = True,
     type = 'f', 
     decimals = 2, 
     ante_decimals = 6,
     is_symmetric = "no",
     pretty = True,
     integer_row = [3,4])

    table = np.zeros([5,size_last_table])
    if (size_last_table > 0):
        for i in range(size_last_table):
            data = {}
            with open(all_results_dirs[n_tables * table_size + i] + "/input_file.json") as f:
                data = json.load(f)


            table[0,i] = data["LOS_NOISE_SD_BASELINE"]
            table[1,i] = 1e9 * data["PROCESS_NOISE_SIGMA_VEL"]
            table[2,i] = 1e9 * data["PROCESS_NOISE_SIGMA_OMEG"]
            table[3,i] = int(1./data["INSTRUMENT_FREQUENCY_NAV"])
            table[4,i] = str(int(data["SHAPE_RECONSTRUCTION_OUTPUT_DIR"].split("_")[-1][0]) + 1 + (6 if "robustness" in str(data["SHAPE_RECONSTRUCTION_OUTPUT_DIR"]) else 0))
            


        np_array_to_latex(table,save_path + "/input_table_case_" + str(n_tables * table_size + 1)  + "_to_" + str((n_tables) * table_size + size_last_table ) ,
            row_headers = [r"$\sigma_{\rho}$ ($\mathrm{m}$)",
            r"$\sigma_{\ddot{\mathbf{r}}}$ ($\mathrm{nm/s^2}$)",
            r"$\sigma_{\dot{\boldsymbol{\omega}}}$ ($\mathrm{nrad/s^2}$)",
            r"$T_{\mathrm{obs}}$ ($\mathrm{s}$)",
            "Input Case"],
            column_headers = ["Case " + str(i) for i in range((n_tables ) * table_size + 1,(n_tables ) * table_size + 1 + size_last_table )],
         column = True,
         type = 'f', 
         decimals = 2, 
         ante_decimals = 6,
         is_symmetric = "no",
         pretty = True,
         integer_row = [3,4])
        


def create_output_tables(all_results_dirs,save_path,table_size = 10):
    n_tables = int(float(len(all_results_dirs)) / table_size)
    size_last_table = len(all_results_dirs) - n_tables * table_size


    full_output_table = np.zeros([6,len(all_results_dirs)])
    
    row_headers = [
        r"$\sigma_{\mathbf{r}}$ ($\mathrm{m}$)",
        r"$\sigma_{\dot{\mathbf{r}}}$ ($\mathrm{mm/s}$)",
        r"$\sigma_{\boldsymbol{\sigma}}$ ($\mathrm{deg}$)",
        r"$\sigma_{\boldsymbol{\omega}}$ ($\mathrm{\mu deg/s}$)",
        r"$\sigma_{\mu}$ ($\mathrm{cm^3/s^2}$)",
        r"$\sigma_{C_r}$"]


    for table_index in range(n_tables):

        table = np.zeros([6,table_size])
        table[:] = np.nan


        for i in range(table_size):
            
            input_path =  all_results_dirs[table_index * table_size + i]
            
            try:
                X_true = np.loadtxt(input_path + "/X_true.txt")
                X_hat = np.loadtxt(input_path + "/X_hat.txt")
            except IOError:
                print("Skipping " + str(input_path) + ", can't find outputs")
                continue

            dX = X_hat - X_true

            mrp_error = np.zeros([3,dX.shape[1]])
            principal_angle_error = np.zeros(dX.shape[1])
            for t in range(dX.shape[1]):
                mrp_error[:,t] = RBK.dcm_to_mrp(RBK.mrp_to_dcm(X_true[6:9,t]).T.dot(RBK.mrp_to_dcm(X_hat[6:9,t])))
                principal_angle_error[t] = 4 * np.arctan(np.linalg.norm(mrp_error[:,t]))

            mean_dX_position = np.mean(dX[0:3,:],axis = 1)
            mean_dX_velocity = np.mean(dX[3:6,:],axis = 1)
            mean_dX_principal_angle = np.mean(principal_angle_error)
            mean_dX_omega = np.mean(dX[9:12,:],axis = 1)
            mean_dX_Mu = np.mean(dX[12,:])
            mean_dX_Cr = np.mean(dX[13,:])

            sd_position = 0
            sd_velocity = 0
            sd_mrp = 0
            sd_omega = 0
            sd_mu = 0
            sd_Cr = 0

            for t in range(dX.shape[1]):
                sd_position += 1./(dX.shape[1] - 1) * np.linalg.norm(dX[0:3,t] - mean_dX_position) ** 2
                sd_velocity += 1./(dX.shape[1] - 1) * np.linalg.norm(dX[3:6,t] - mean_dX_velocity) ** 2
                sd_mrp += 1./(dX.shape[1] - 1) * (principal_angle_error[t] - mean_dX_principal_angle) ** 2
                sd_omega += 1./(dX.shape[1] - 1) * np.linalg.norm(dX[9:12,t] - mean_dX_omega) ** 2
                sd_mu += 1./(dX.shape[1] - 1) * (dX[12,t] - mean_dX_Mu) ** 2
                sd_Cr += 1./(dX.shape[1] - 1) * (dX[13,t] - mean_dX_Cr) ** 2

            sd_position = np.sqrt(sd_position)
            sd_velocity = 1e3 * np.sqrt(sd_velocity)
            sd_mrp = 180./np.pi * np.sqrt(sd_mrp)
            sd_omega = 1e6 * 180./np.pi * np.sqrt(sd_omega)
            sd_mu = 1e6 * np.sqrt(sd_mu)
            sd_Cr = np.sqrt(sd_Cr)
           

            table[0,i] = sd_position
            table[1,i] = sd_velocity
            table[2,i] = sd_mrp
            table[3,i] = sd_omega
            table[4,i] = sd_mu
            table[5,i] = sd_Cr


        full_output_table[:,table_index * table_size:table_index * table_size + table_size] = table
        np_array_to_latex(table,save_path + "/output_table_case_" + str(table_index * table_size + 1)  + "_to_" + str(table_index * table_size + table_size) ,
        row_headers = row_headers,
        column_headers = ["Case " + str(i) for i in range(table_index * table_size + 1,table_index * table_size + 1 + table_size + 1)],
     column = True,
     type = 'f', 
     decimals = 3, 
     ante_decimals = 6,
     is_symmetric = "no",
     pretty = True)

    table = np.zeros([6,size_last_table])
    table[:] = np.nan

    if (size_last_table > 0):
        for i in range(size_last_table):
            
            input_path =  all_results_dirs[(n_tables) * table_size + i]
            
            try:
                X_true = np.loadtxt(input_path + "/X_true.txt")
                X_hat = np.loadtxt(input_path + "/X_hat.txt")
            except IOError:
                print("Skipping " + str(input_path) + ", can't find outputs")

                continue


            dX = X_hat - X_true

            mrp_error = np.zeros([3,dX.shape[1]])
            principal_angle_error = np.zeros(dX.shape[1])
            for t in range(dX.shape[1]):
                mrp_error[:,t] = RBK.dcm_to_mrp(RBK.mrp_to_dcm(X_true[6:9,t]).T.dot(RBK.mrp_to_dcm(X_hat[6:9,t])))
                principal_angle_error[t] = 4 * np.arctan(np.linalg.norm(mrp_error[:,t]))

            mean_dX_position = np.mean(dX[0:3,:],axis = 1)
            mean_dX_velocity = np.mean(dX[3:6,:],axis = 1)
            mean_dX_omega = np.mean(dX[9:12,:],axis = 1)
            mean_dX_principal_angle = np.mean(principal_angle_error)
            mean_dX_Mu = np.mean(dX[12,:])
            mean_dX_Cr = np.mean(dX[13,:])

            sd_position = 0
            sd_velocity = 0
            sd_mrp = 0
            sd_omega = 0
            sd_mu = 0
            sd_Cr = 0

            for t in range(dX.shape[1]):
                sd_position += 1./(dX.shape[1] - 1) * np.linalg.norm(dX[0:3,t] - mean_dX_position) ** 2
                sd_velocity += 1./(dX.shape[1] - 1) * np.linalg.norm(dX[3:6,t] - mean_dX_velocity) ** 2
                sd_mrp += 1./(dX.shape[1] - 1) * (principal_angle_error[t] - mean_dX_principal_angle) ** 2
                sd_omega += 1./(dX.shape[1] - 1) * np.linalg.norm(dX[9:12,t] - mean_dX_omega) ** 2
                sd_mu += 1./(dX.shape[1] - 1) * (dX[12,t] - mean_dX_Mu) ** 2
                sd_Cr += 1./(dX.shape[1] - 1) * (dX[13,t] - mean_dX_Cr) ** 2

            sd_position = np.sqrt(sd_position)
            sd_velocity = 1e3 * np.sqrt(sd_velocity)
            sd_mrp = 180./np.pi * np.sqrt(sd_mrp)
            sd_omega = 1e6 * 180./np.pi * np.sqrt(sd_omega)
            sd_mu = 1e6 * np.sqrt(sd_mu)
            sd_Cr = np.sqrt(sd_Cr)

            table[0,i] = sd_position
            table[1,i] = sd_velocity
            table[2,i] = sd_mrp
            table[3,i] = sd_omega
            table[4,i] = sd_mu
            table[5,i] = sd_Cr


        np_array_to_latex(table,save_path + "/output_table_case_" + str(n_tables * table_size + 1)  + "_to_" + str((n_tables) * table_size + size_last_table ) ,
            row_headers = row_headers,
            column_headers = ["Case " + str(i) for i in range((n_tables ) * table_size + 1,(n_tables ) * table_size + 1 + size_last_table )],
         column = True,
         type = 'f', 
         decimals = 3, 
         ante_decimals = 6,
         is_symmetric = "no",
         pretty = True)


        full_output_table[:,n_tables * table_size:(n_tables) * table_size + size_last_table] = table

        np.savetxt(save_path + "/all_nav_results_table.txt",full_output_table)

def list_results(save_path,
    mainpath = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Navigation/output"):
    

    print("Searching " + mainpath)
    all_results_dirs  = [x[0] for x in os.walk(mainpath)][1:]

    all_results_dirs_tag = np.array([int(all_results_dirs[i].split("_")[-1]) for i in range(len(all_results_dirs))])

    sorted_order = np.argsort(all_results_dirs_tag)

    all_results_dirs = [all_results_dirs[sorted_order[i]] for i in range(len(sorted_order))]

    print("\t Found " + str(len(all_results_dirs)) + " result directories\n\n")

    for i in range(len(all_results_dirs)):
        print(str(i) + " : " + all_results_dirs[i] + "\n")

        with open(all_results_dirs[i] + "/input_file.json") as f:
            data = json.load(f)

        pprint(data)
        print("\n")

    index_str = input(" Which one should be processed ? Pick a number or enter 'all'\n")
    save_str = input(" Should results be saved? (y/n) ?\n")



    if index_str != "all":
        if save_str is "y":
            plot_all_results(all_results_dirs[int(index_str)],save_path)
        elif save_str is "n":
            plt.switch_backend('Qt5Agg')

            plot_all_results(all_results_dirs[int(index_str)])
        else:
            raise(TypeError("Unrecognized input: " + str(save_str)))
    else:   

        if save_str is "y":

            create_input_tables(all_results_dirs,save_path)
            create_output_tables(all_results_dirs,save_path)
            

            plt.switch_backend('PDF')

            all_results_pooled = [[all_results_dirs[i],save_path] for i in range(len(all_results_dirs))]

            p = Pool(10)
            p.map(plot_all_results_pool, all_results_pooled)

        elif save_str is "n":
            plt.switch_backend('Qt5Agg')

            for directory in all_results_dirs:
                plot_all_results(directory)
        else:
            raise(TypeError("Unrecognized input: " + str(save_str)))
 
        

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def plot_all_results(path,save_path = ""):
    print("\t Plotting from " + path + "\n")
    try:

        plot_orbit_planar(path,save_path)
        plot_orbit(path,save_path)
        plot_cart_state_error_inertial(path,save_path)
        plot_state_error_RIC(path,save_path)
        plot_attitude_state_inertial(path,save_path)
    except IOError:

        pass

def plot_all_results_pool(inputs):
    path = inputs[0]
    save_path = inputs[1]

    print("\t Plotting from " + path + "\n")
    try:
        
        plot_orbit_planar(path,save_path)
        plot_orbit(path,save_path)
        plot_cart_state_error_inertial(path,save_path)
        plot_state_error_RIC(path,save_path)
        plot_attitude_state_inertial(path,save_path)
    except IOError:

        pass




def plot_orbit_planar(path,save_path = ""):

    X_true = np.loadtxt(path + "/state_true_orbit.txt")
    case = int(path.split("_")[-1]) + 1
    plt.plot(X_true[0,:]/1000,X_true[1,:]/1000)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.title("Z-plane orbit, Case " + str(case))


    if save_path != "":
        # plt.tight_layout()

        plt.savefig(save_path + "/orbit_planar_z_" + str(case) + ".pdf",bbox_to_inches = "tight")
    else:

        plt.show()

    plt.clf()
    plt.plot(X_true[0,:]/1000,X_true[2,:]/1000)
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.axis('equal')
    plt.title("Y-plane orbit, Case " + str(case))

    if save_path != "":
        # plt.tight_layout()

        plt.savefig(save_path + "/orbit_planar_y_" + str(case) + ".pdf",bbox_to_inches = "tight")
    else:
        plt.show()

    plt.clf()
    plt.plot(X_true[1,:]/1000,X_true[2,:]/1000)
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.title("X-plane orbit, Case " + str(case))
    plt.axis('equal')
    
    if save_path != "":
        # plt.tight_layout()

        plt.savefig(save_path + "/orbit_planar_x_" + str(case) + ".pdf",bbox_to_inches = "tight")
    else:
        plt.show()


    plt.clf()


def plot_orbit(path,save_path = ""):

    X_true = np.loadtxt(path + "/state_true_orbit_dense.txt")
    case = int(path.split("_")[-1]) + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_true[0,:]/1000,X_true[1,:]/1000,X_true[2,:]/1000)
    ax.scatter(X_true[0,0]/1000,X_true[1,0]/1000,X_true[2,0]/1000,color = 'green')
    ax.scatter(X_true[0,-1]/1000,X_true[1,-1]/1000,X_true[2,-1]/1000,color = 'red')


    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    draw_sphere(535./2000)

    # plt.show()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    plt.title("Inertial trajectory, Case " + str(case))

    if save_path != "":
        # plt.tight_layout()

        plt.savefig(save_path + "/trajectory_inertial_case_" + str(case) + ".pdf",
            bbox_to_inches = "tight")
    else:
        plt.tight_layout()

        plt.show()
    plt.cla()
    plt.clf()
    plt.close(fig)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X_true_BF = np.loadtxt(path + "/state_true_orbit_dense.txt")
    for i in range(X_true_BF.shape[1]):
        
        BN = RBK.mrp_to_dcm(X_true_BF[6:9,i])

        X_true_BF[0:3,i] = BN.dot(X_true_BF[0:3,i])

    ax.plot(X_true_BF[0,:]/1000,X_true_BF[1,:]/1000,X_true_BF[2,:]/1000)
    ax.scatter(X_true_BF[0,0]/1000,X_true_BF[1,0]/1000,X_true_BF[2,0]/1000,color = 'green')
    ax.scatter(X_true_BF[0,-1]/1000,X_true_BF[1,-1]/1000,X_true_BF[2,-1]/1000,color = 'red')
    

    vertices,facets = polyhedron.load_shape("../resources/shape_models/itokawa_8.obj")
    polyhedron.draw_shape(vertices,facets)


    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.grid(False)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    plt.title("Body-frame trajectory, Case " + str(case))


    if save_path == "":
        plt.tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/orbit_body_frame_case_" + str(case) + ".pdf",bbox_to_inches = "tight")

    plt.clf()
    plt.cla()
    plt.close(fig)


def plot_state_error_RIC(path,save_path= ""):

    X_true = np.loadtxt(path + "/X_true.txt")
    X_hat = np.loadtxt(path + "/X_hat.txt")
    P = np.loadtxt(path + "/covariances.txt")
    T_obs = np.loadtxt(path + "/nav_times.txt") / 60

    case = int(path.split("_")[-1]) + 1
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

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,3 * sd[0,:],"--+")
    plt.plot(T_obs,3 * sd[1,:],"--+")
    plt.plot(T_obs,3 * sd[2,:],"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * sd[0,:],"--+")
    plt.plot(T_obs,- 3 * sd[1,:],"--+")
    plt.plot(T_obs,- 3 * sd[2,:],"--+")
    plt.legend(loc = "upper right")
    plt.xlabel("Time (min)")
    plt.ylabel(r"Error ($\mathrm{m}$)")
    plt.title("RIC frame position error, Case " + str(case))

    if save_path == "":
        plt.tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig( save_path + "/position_error_RIC_" + str(case) +  ".pdf",bbox_to_inches = "tight")

    plt.clf()
    plt.cla()

    
    # Velocity
    plt.plot(T_obs,100 * (X_true_RIC[3,:] - X_hat_RIC[3,:]),'-o',label = "radial")
    plt.plot(T_obs,100 * (X_true_RIC[4,:] - X_hat_RIC[4,:]),'-o',label = "in-track")
    plt.plot(T_obs,100 * (X_true_RIC[5,:] - X_hat_RIC[5,:]),'-o',label = "cross-track")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[5,:] * 100,"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[5,:] * 100,"--+")

    plt.legend(loc = "upper right")
    plt.xlabel("Time (min)")
    plt.ylabel(r"Error ($\mathrm{cm/s}$)")
    plt.title("RIC frame velocity error, Case " + str(case))


    plt.ylim([- 5 * sd[3,2] * 100,5 * sd[3,2] * 100])


    if save_path == "":
        plt.tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/velocity_error_RIC_" + str(case) +  ".pdf",bbox_to_inches = "tight")

    plt.clf()
    plt.cla()

    # mu

    plt.plot(T_obs,(X_true_RIC[-2,:] - X_hat_RIC[-2,:]),'-o')

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,3 * sd[-2,:],"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * sd[-2,:],"--+")

    plt.xlabel("Time (min)")
    plt.ylabel(r" Error $(\mathrm{kg\cdot m^3 / s^2})$")
    plt.title("Standard gravitational error, Case " + str(case))
    plt.ylim([- 5 * sd[-2,2] ,5 * sd[-2,2] ])
    


    if save_path == "":
        plt.tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/mu_error_RIC_" + str(case) + ".pdf",bbox_to_inches = "tight")

    plt.clf()
    plt.cla()

    # Cr

    plt.plot(T_obs,(X_true_RIC[-1,:] - X_hat_RIC[-1,:]),'-o')

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,3 * sd[-1,:],"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * sd[-1,:],"--+")

    plt.xlabel("Time (min)")
    plt.ylabel(r"Error")
    plt.title("SRP cannonball coefficient error, Case " + str(case))
    plt.ylim([- 5 * sd[-1,2] ,5 * sd[-1,2] ])
    


    if save_path == "":
        plt.tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/Cr_error_RIC_" + str(case) +  ".pdf",bbox_to_inches = "tight")

    plt.cla()
    plt.clf()


def plot_attitude_state_inertial(path,save_path = ""):
    case = int(path.split("_")[-1]) + 1

    X_true = np.loadtxt(path + "/X_true.txt")
    X_hat = np.loadtxt(path + "/X_hat.txt")
    P = np.loadtxt(path + "/covariances.txt")
    T_obs = np.loadtxt(path + "/nav_times.txt") / 60

    mrp_error = np.zeros([3,len(T_obs)])

    sd = []

    for i in range(X_hat.shape[1]):
        sd += [np.sqrt(np.diag(P[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]))]
        mrp_error[:,i] = RBK.dcm_to_mrp(RBK.mrp_to_dcm(X_true[6:9,i]).T.dot(RBK.mrp_to_dcm(X_hat[6:9,i])))

    sd = np.vstack(sd).T

    # estimated MRP 
    plt.plot(T_obs,X_true[6,:],'-o',label = r"$\sigma_1$")
    plt.plot(T_obs,X_true[7,:],'-o',label = r"$\sigma_2$")
    plt.plot(T_obs,X_true[8,:],'-o',label = r"$\sigma_3$")

    plt.xlabel("Time (min)")
    plt.ylabel("MRP (estimated)")
    plt.title("Attitude MRP, Case " + str(case))

    plt.legend(loc = "best")
    
    if save_path == "":
        plt.tight_layout()
        plt.show()
    else:
        # plt.tight_layout()


        plt.savefig(save_path + "/attitude_" + str(case) + ".pdf",bbox_to_inches = "tight")

    plt.cla()
    plt.clf()
    plt.gca().set_prop_cycle(None)


    # Position error
    plt.plot(T_obs,mrp_error[0,:],'-o',label = r"$\sigma_1$")
    plt.plot(T_obs,mrp_error[1,:],'-o',label = r"$\sigma_2$")
    plt.plot(T_obs,mrp_error[2,:],'-o',label = r"$\sigma_3$")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,3 * sd[6,:],"--+")
    plt.plot(T_obs,3 * sd[7,:],"--+")
    plt.plot(T_obs,3 * sd[8,:],"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * sd[6,:],"--+")
    plt.plot(T_obs,- 3 * sd[7,:],"--+")
    plt.plot(T_obs,- 3 * sd[8,:],"--+")

    plt.xlabel("Time (min)")
    plt.ylabel("Error")
    plt.title("Attitude MRP error, Case " + str(case))

    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))


    plt.legend(loc = "upper right")
    
    if save_path == "":
        plt.tight_layout()

        plt.show()

    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/error_attitude_" +  str(case) + ".pdf",bbox_to_inches = "tight")

    plt.clf()
    plt.gca().set_prop_cycle(None)

    # Velocity error

    r2d = 180./np.pi
    plt.plot(T_obs,r2d * (X_true[9,:] - X_hat[9,:]),'-o',label = r"$\omega_1$")
    plt.plot(T_obs,r2d * (X_true[10,:] - X_hat[10,:]),'-o',label = r"$\omega_2$")
    plt.plot(T_obs,r2d * (X_true[11,:] - X_hat[11,:]),'-o',label = r"$\omega_3$")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,r2d * 3 * sd[9,:],"--+")
    plt.plot(T_obs,r2d * 3 * sd[10,:],"--+")
    plt.plot(T_obs,r2d * 3 * sd[11,:],"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * r2d* sd[9,:],"--+")
    plt.plot(T_obs,- 3 * r2d* sd[10,:],"--+")
    plt.plot(T_obs,- 3 * r2d* sd[11,:],"--+")

    plt.xlabel("Time (min)")
    plt.ylabel(r"Error ($\mathrm{deg/s}$)")
    plt.title("Angular velocity error, Case " + str(case))

    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    plt.ylim([- 5 * r2d * sd[9,2],5 * r2d * sd[9,2]])

    plt.legend(loc = "upper right")

    if save_path == "":
        plt.gcf().tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/error_omega_"+ str(case) +  ".pdf",bbox_to_inches = "tight")

    plt.clf()

    # angle error
    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,4 * np.arctan(np.linalg.norm(mrp_error,axis = 0)) * 180./np.pi,'-o')
   
    plt.xlabel("Time (min)")
    plt.ylabel("Error (deg)")
    plt.title("Principal angle error, Case " + str(case))


    if save_path == "":
        plt.gcf().tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/error_angle_" + str(case) + ".pdf",bbox_to_inches = "tight")

    plt.clf()








    

def plot_cart_state_error_inertial(path = "",save_path = ""):
    case = int(path.split("_")[-1]) + 1

    X_true = np.loadtxt(path + "/X_true.txt")
    X_hat = np.loadtxt(path + "/X_hat.txt")
    P = np.loadtxt(path + "/covariances.txt")
    T_obs = np.loadtxt(path + "/nav_times.txt") / 60

    X_true_augmented = np.zeros([4,len(T_obs)])
    X_hat_augmented = np.zeros([4,len(T_obs)])

    X_true_augmented[0,:] = T_obs
    X_hat_augmented[0,:] = T_obs

    X_hat_augmented[1:,:] = X_hat[0:3,:]
    X_true_augmented[1:,:] = X_true[0:3,:]


    sd = []

    for i in range(X_hat.shape[1]):
        sd += [np.sqrt(np.diag(P[:,i * P.shape[0] : i * P.shape[0] + P.shape[0]]))]

    sd = np.vstack(sd).T
  
    # Position
    plt.plot(T_obs,X_true[0,:] - X_hat[0,:],'-o',label = "X")
    plt.plot(T_obs,X_true[1,:] - X_hat[1,:],'-o',label = "Y")
    plt.plot(T_obs,X_true[2,:] - X_hat[2,:],'-o',label = "Z")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,3 * sd[0,:],"--+")
    plt.plot(T_obs,3 * sd[1,:],"--+")
    plt.plot(T_obs,3 * sd[2,:],"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * sd[0,:],"--+")
    plt.plot(T_obs,- 3 * sd[1,:],"--+")
    plt.plot(T_obs,- 3 * sd[2,:],"--+")
    plt.xlabel("Time (min)")
    plt.ylabel(r"Error ($\mathrm{m}$)")
    plt.title("Inertial frame position error, Case " + str(case))

    plt.legend(loc = "upper right")

    if save_path == "":
        plt.gcf().tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/error_pos_" + str(case) + ".pdf",bbox_to_inches = "tight")


    plt.clf()
    
    # Velocity
    plt.plot(T_obs,100 * (X_true[3,:] - X_hat[3,:]),'-o',label = "X")
    plt.plot(T_obs,100 * (X_true[4,:] - X_hat[4,:]),'-o',label = "Y")
    plt.plot(T_obs,100 * (X_true[5,:] - X_hat[5,:]),'-o',label = "Z")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,3 * sd[5,:] * 100,"--+")

    plt.gca().set_prop_cycle(None)

    plt.plot(T_obs,- 3 * sd[3,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[4,:] * 100,"--+")
    plt.plot(T_obs,- 3 * sd[5,:] * 100,"--+")

    plt.ylim([- 5 * sd[3,2] * 100,5 * sd[3,2] * 100])

    plt.xlabel("Time (min)")
    plt.ylabel(r"Error ($\mathrm{cm/s}$)")
    plt.title("Inertial frame velocity error, Case " + str(case))
    plt.legend(loc = "upper right")

    if save_path == "":
        plt.tight_layout()

        plt.show()
    else:
        # plt.tight_layout()

        plt.savefig(save_path + "/error_vel_" + str(case) +  ".pdf",bbox_to_inches = "tight")


    plt.clf()
    plt.cla()



save_path = "/Users/bbercovici/GDrive/CUBoulder/Research/thesis/figs/navigation"

list_results(save_path,mainpath = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/Navigation/output")
