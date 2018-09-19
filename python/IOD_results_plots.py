import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from np_array_to_latex import np_array_to_latex
from mpl_toolkits.mplot3d import Axes3D
import RigidBodyKinematics as RBK
import fnmatch
import os
import polyhedron
from pprint import pprint
import json

plt.rc('text', usetex=True)
plt.rc('font', family='serif')




def create_consistency_matrix(all_results_dirs,savepath):


    consistency_matrix = np.zeros([2,len(all_results_dirs)])
    row_labels = [r"$\vert r_p - \hat{r}_p\vert \leq 3 \sigma_{rp}$ (\%)",
    r"$\vert r_p - \hat{r}_p\vert > 3 \sigma_{rp}$ (\%)"]
    column_labels = [ "Case " + str(int(all_results_dirs[i].split("_")[-1]) + 1) for i in range(len(all_results_dirs)) ] 

    for i in range(len(all_results_dirs)):

        
        all_rps = np.loadtxt(all_results_dirs[i] + "/all_rps.txt")
        all_rps_sds = np.loadtxt(all_results_dirs[i] + "/all_rps_sds.txt")
        true_keplerian_state = np.loadtxt(all_results_dirs[i] + "/true_keplerian_state.txt")
        true_rp = true_keplerian_state[0] * (1 - true_keplerian_state[1])

        N = len(all_rps)

        for j in range(N):

            if (np.abs(all_rps[j] - true_rp) > 3 * all_rps_sds[j]):
                consistency_matrix[1,i] += 100./N
            else:
                consistency_matrix[0,i] += 100./N

    np_array_to_latex(consistency_matrix,savepath,row_headers = row_labels,column_headers = column_labels,
  column = True,type = 'f', decimals = 3, ante_decimals = 6)





def create_output_table_mc(all_results_dirs,savepath):

    column_labels = [ "Case " + str(int(all_results_dirs[i].split("_")[-1]) + 1) for i in range(len(all_results_dirs)) ] 
    row_labels = [r"$\sigma_X$ $\mathrm{(m)}$",
    r"$\sigma_Y$ $\mathrm{(m)}$",
    r"$\sigma_Z$ $\mathrm{(m)}$",
    r"$\sigma_{\dot{X}}$ $\mathrm{(mm/s)}$",
    r"$\sigma_{\dot{Y}}$ $\mathrm{(mm/s)}$",
    r"$\sigma_{\dot{Z}}$ $\mathrm{(mm/s)}$",
    r"$\sigma_{\mu}$ $(\mathrm{m^3/s^2})$"]




    res = np.zeros([7,len(all_results_dirs)])

    for i in range(len(all_results_dirs)):
        cov_pos_mc = np.loadtxt(all_results_dirs[i] + "/cov_mc.txt")

        res[0,i] = np.sqrt(cov_pos_mc[0,0])
        res[1,i] = np.sqrt(cov_pos_mc[1,1])
        res[2,i] = np.sqrt(cov_pos_mc[2,2])
        res[3,i] = 1000 * np.sqrt(cov_pos_mc[3,3])
        res[4,i] = 1000 * np.sqrt(cov_pos_mc[4,4])
        res[5,i] = 1000 * np.sqrt(cov_pos_mc[5,5])
        res[6,i] = np.sqrt(cov_pos_mc[6,6])

    np_array_to_latex(res,savepath,row_headers = row_labels,column_headers = column_labels,
  column = True,type = 'f', decimals = 3, ante_decimals = 6)



def create_output_table_model(all_results_dirs,savepath):

    column_labels = [ "Case " + str(int(all_results_dirs[i].split("_")[-1]) + 1) for i in range(len(all_results_dirs)) ] 
    row_labels = [r"$\Delta\sigma_X$ (\%)",
    r"$\Delta\sigma_Y$ (\%)",
    r"$\Delta\sigma_Z$ (\%)",
    r"$\Delta\sigma_{\dot{X}}$ (\%)",
    r"$\Delta\sigma_{\dot{Y}}$ (\%)",
    r"$\Delta\sigma_{\dot{Z}}$ (\%)",
    r"$\Delta\sigma_{\mu}$ $(\%)$"]

    res = np.zeros([7,len(all_results_dirs)])

    for i in range(len(all_results_dirs)):
        all_covs = np.loadtxt(all_results_dirs[i] + "/all_covs.txt")
        cov_pos_mc = np.loadtxt(all_results_dirs[i] + "/cov_mc.txt")

        sigma_mc = np.sqrt(np.diag(cov_pos_mc))
        average_sd_deviation_percentage = np.zeros(sigma_mc.shape)

        N = int(len(all_covs)/7.)

        for j in range(N):

            sigma_model = np.sqrt(np.diag(all_covs[7 * j:7 * j + 7, :]))
            average_sd_deviation_percentage += 100./N * np.abs(sigma_mc - sigma_model)/sigma_mc


        res[:,i] = average_sd_deviation_percentage
        

    np_array_to_latex(res,savepath,row_headers = row_labels,column_headers = column_labels,
  column = True,type = 'f', decimals = 3, ante_decimals = 6)


def create_input_table(all_results_dirs,savepath):

    column_labels = [ "Case " + str(int(all_results_dirs[i].split("_")[-1]) + 1) for i in range(len(all_results_dirs)) ] 
    row_labels = ["$e$","Orbit fraction","$N$"]

    res = np.zeros([3,len(all_results_dirs)])

    for i in range(len(all_results_dirs)):
        with open(all_results_dirs[i] + "/input_file.json") as f:
            data = json.load(f)
        res[0,i] = data["E"]
        res[1,i] = data["ORBIT_FRACTION"]
        res[2,i] = int(data["OBSERVATION_TIMES"])


    np_array_to_latex(res,savepath,row_headers = row_labels,column_headers = column_labels,
  column = True,type = 'f', decimals = 2, ante_decimals = 6)








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



def draw_sphere(radius):
    theta = np.linspace(0,np.pi,100,endpoint=True)
    phi   = np.linspace(0,np.pi*2,100,endpoint=True)

    X = radius * np.outer(np.sin(theta),np.cos(phi))
    Y = radius * np.outer(np.sin(theta),np.sin(phi))
    Z = radius * np.outer(np.cos(theta),np.ones(100))

  
    plt.gca().plot_surface(X,Y,Z)





def draw_orbit_inertial(path,
    position_estimates = None, 
    crude_guesses = None,
    center = False,
    savepath = None,
    prefix = None,
    suffix = None,
    sphere_radius = None):
    
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    traj = np.loadtxt(path + "/state_obs_point_mass.txt")
    full_traj = np.loadtxt(path + "/state_full_orbit_point_mass.txt")


    ax.plot(full_traj[0,:],full_traj[1,:],full_traj[2,:],"--")
    ax.scatter(traj[0,:],traj[1,:],traj[2,:],color = 'black',marker = '*')

    ax.scatter(traj[0,0],traj[1,0],traj[2,0],color = 'green' , marker = 'o')
    ax.scatter(traj[0,-1],traj[1,-1],traj[2,-1],color = 'red' , marker = 'o')

    if position_estimates is not None:
        ax.scatter(position_estimates[0,:],position_estimates[1,:],position_estimates[2,:],marker = 'v')
    if crude_guesses is not None:
        ax.scatter(crude_guesses[0,:],crude_guesses[1,:],crude_guesses[2,:],color = "red", marker = 'x')
    
    if center is True:


        min_lim = -50
        max_lim = 50


        ax.set_xlim([min_lim + traj[0,0],max_lim + traj[0,0]])
        ax.set_ylim([min_lim + traj[1,0],max_lim + traj[1,0]])
        ax.set_zlim([min_lim + traj[2,0],max_lim + traj[2,0]])

    if sphere_radius is not None:
        draw_sphere(sphere_radius)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(False)

    set_axes_equal(ax)  
    
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + prefix + suffix , bbox_inches='tight')

    plt.clf()
    plt.cla()





def draw_orbit_body_frame(path,
    states,
    position_estimates = None,
    crude_guesses = None,
    savepath = None,
    prefix = None,
    suffix = None,
    vertices = None,
    facets = None):

    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    traj = np.loadtxt(path + "/state_obs_point_mass.txt")
    full_traj = np.loadtxt(path + "/state_full_orbit_point_mass.txt")


    for i in range(full_traj.shape[1]):
        full_traj[0:3,i] = RBK.mrp_to_dcm(full_traj[6:9,i]).dot(full_traj[0:3,i])
    
    for i in range(traj.shape[1]):
        traj[0:3,i] = RBK.mrp_to_dcm(traj[6:9,i]).dot(traj[0:3,i])


    ax.plot(full_traj[0,:],full_traj[1,:],full_traj[2,:],"--")
    ax.scatter(traj[0,:],traj[1,:],traj[2,:],color = 'black',marker = '*')

    ax.scatter(traj[0,0],traj[1,0],traj[2,0],color = 'green' , marker = 'o')
    ax.scatter(traj[0,-1],traj[1,-1],traj[2,-1],color = 'red' , marker = 'o')


    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(False)

    if position_estimates is not None:
        for j in range(states.shape[1]):
            position_estimates[0:3,j] = RBK.mrp_to_dcm(traj[6:9,0]).dot(position_estimates[0:3,j])

        ax.scatter(position_estimates[0,:],position_estimates[1,:],position_estimates[2,:],'v')
    
    if crude_guesses is not None:
        for j in range(crude_guesses.shape[1]):
            crude_guesses[0:3,j] = RBK.mrp_to_dcm(traj[6:9,0]).dot(crude_guesses[0:3,j])
        ax.scatter(crude_guesses[0,:],crude_guesses[1,:],crude_guesses[2,:],color = "red", marker = 'x')

    if vertices and facets is not None:
        polyhedron.draw_shape(vertices,facets)

    set_axes_equal(ax)  
    if savepath is None:
        plt.show()
    else:

        plt.savefig(savepath + prefix + suffix, bbox_inches='tight')

    plt.clf()
    plt.cla()

def draw_2d_covariance(axis,cov,color,alpha = 1):

    cov_2d = np.zeros([2,2])

    if (axis == 0):

        cov_2d[0,0] = cov[1,1]
        cov_2d[1,1] = cov[2,2]
        cov_2d[1,0] = cov[1,2]
        cov_2d[0,1] = cov[2,1]

    elif (axis == 1):

        cov_2d[0,0] = cov[0,0]
        cov_2d[1,1] = cov[2,2]
        cov_2d[1,0] = cov[0,2]
        cov_2d[0,1] = cov[2,0]

    elif (axis == 2):

        cov_2d[0,0] = cov[0,0]
        cov_2d[1,1] = cov[1,1]
        cov_2d[1,0] = cov[0,1]
        cov_2d[0,1] = cov[1,0]

    dims,M = np.linalg.eigh(cov_2d)

    if (np.linalg.det(M)< 0):
        M[:,0] = - M[:,0]
    
    # the axes matrix is of the form 

    # M = [cos t - sin t ]
    #     [ sin t cos t]

    angle = np.arctan2(M[1,0],M[0,0]) * 180./np.pi + 90


    # the horizontal axis is along the largest uncertainty

    width = 2 * 3 * np.sqrt(dims)[1]
    height = 2 * 3 * np.sqrt(dims)[0]

    e = Ellipse((0, 0), width, height, angle,
        facecolor = None,
        edgecolor = color,
        fill = False,
        alpha = alpha)
    

    plt.gca().add_artist(e)

def draw_dispersions(name,
    axis,
    parameters,
    crude_guesses_minus_mean,
    cov_mc,
    cov_model,
    labels,
    savepath = None,
    prefix = None,
    suffix = None):


    cut_names = ["Y-Z","X-Z","X-Y"]

    print "- Plotting " + name + " in " + cut_names[axis] + " slice " 

    x_max = -1
    y_max = -1

    for s in range(parameters.shape[1]):
        if axis == 0:

            plt.scatter(parameters[1,s ],parameters[2,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
            if (crude_guesses_minus_mean is not None):
                plt.scatter(crude_guesses_minus_mean[1,s ],crude_guesses_minus_mean[2,s ],marker = ".",color = "red" ,alpha = 0.7)
            
            x_max= max(np.abs(parameters[1,s ]),x_max)
            y_max= max(np.abs(parameters[2,s ]),y_max)




        elif axis == 1:
            plt.scatter(parameters[0,s ],parameters[2,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
            if (crude_guesses_minus_mean is not None):
                plt.scatter(crude_guesses_minus_mean[0,s ],crude_guesses_minus_mean[2,s ],marker = ".",color = "red" ,alpha = 0.7)
            
            x_max= max(np.abs(parameters[0,s ]),x_max)
            y_max= max(np.abs(parameters[2,s ]),y_max)

        
        elif axis == 2:
            plt.scatter(parameters[0,s ],parameters[1,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
            if (crude_guesses_minus_mean is not None):
                plt.scatter(crude_guesses_minus_mean[0,s ],crude_guesses_minus_mean[1,s ],marker = ".",color = "red" ,alpha = 0.7)
            x_max= max(np.abs(parameters[0,s ]),x_max)
            y_max= max(np.abs(parameters[1,s ]),y_max)

    draw_2d_covariance(axis,cov_mc,color = "lightblue")

    for i in range(max(parameters.shape)):
        if name is "pos":
            draw_2d_covariance(axis,cov_model[7 * i: 7 * i + 3,0:3],color = "red",alpha = 0.1)
        elif name is "vel":
            draw_2d_covariance(axis,cov_model[7 * i + 3: 7 * i + 6, 3:6],color = "red",alpha = 0.1)

    if axis == 0:
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
    elif axis == 1:
        plt.xlabel(labels[0])
        plt.ylabel(labels[2])
    elif axis == 2:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.scatter(0,0,marker = ".",color = "black" )
    plt.tight_layout()

    plt.gca().set_xlim([-1.2 * max(x_max,y_max),1.2 * max(x_max,y_max)])
    plt.gca().set_ylim([-1.2 * max(x_max,y_max),1.2 * max(x_max,y_max)])
    



    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + prefix + suffix,bbox_inches='tight')
    plt.cla()
    plt.clf()




def list_results(graphics_path,mainpath = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/IOD/output"):

    all_results_dirs  = [x[0] for x in os.walk(mainpath)][1:]

    all_results_dirs_tag = np.array([int(all_results_dirs[i].split("_")[-1]) for i in range(len(all_results_dirs))])

    sorted_order = np.argsort(all_results_dirs_tag)

    all_results_dirs = [all_results_dirs[sorted_order[i]] for i in range(len(sorted_order))]

    print "\t Found " + str(len(all_results_dirs)) + " result directories\n\n"

    for i in range(len(all_results_dirs)):
        print str(i) + " : " + all_results_dirs[i] + "\n"

        with open(all_results_dirs[i] + "/input_file.json") as f:
            data = json.load(f)

        pprint(data)
        print "\n"

    index_str = raw_input(" Which one should be processed ? Pick a number or enter 'all'\n")
    save_str = raw_input(" Should results be saved? (y/n) ?\n")

    if index_str != "all":
        if save_str is "y":
            plot_all_results(all_results_dirs[int(index_str)],graphics_path)
        elif save_str is "n":
            plt.switch_backend('Qt5Agg')
            plot_all_results(all_results_dirs[int(index_str)])
        else:
            raise(TypeError("Unrecognized input: " + str(save_str)))
    else:


        create_input_table(all_results_dirs[0:6],graphics_path + "/input_table_1")
        create_input_table(all_results_dirs[6:12],graphics_path + "/input_table_2")
        create_input_table(all_results_dirs[12:18],graphics_path + "/input_table_3")

        create_output_table_mc(all_results_dirs[0:6],graphics_path + "/output_table_mc_1")
        create_output_table_mc(all_results_dirs[6:12],graphics_path + "/output_table_mc_2")
        create_output_table_mc(all_results_dirs[12:18],graphics_path + "/output_table_mc_3")

        create_output_table_model(all_results_dirs[0:6],graphics_path + "/output_table_model_1")
        create_output_table_model(all_results_dirs[6:12],graphics_path + "/output_table_model_2")
        create_output_table_model(all_results_dirs[12:18],graphics_path + "/output_table_model_3")


        create_consistency_matrix(all_results_dirs[0:6],graphics_path + "/rp_consistency_1")
        create_consistency_matrix(all_results_dirs[6:12],graphics_path + "/rp_consistency_2")
        create_consistency_matrix(all_results_dirs[12:18],graphics_path + "/rp_consistency_3")

        
        if save_str is "y":
            for i in range(len(all_results_dirs)):
                plot_all_results(all_results_dirs[i],graphics_path)

        elif save_str is "n":
            plt.switch_backend('Qt5Agg')
            for i in range(len(all_results_dirs)):
                plot_all_results(all_results_dirs[i])

        else:
            raise(TypeError("Unrecognized input: " + str(save_str)))






def draw_dispersions_1d(y_label,
    values,
    predicted_sds,
    savepath = None,prefix = None,suffix = None,
    sphere_radius = None,
    true_rp = None):

    values = np.copy(values - true_rp)

    sort_order = np.argsort(np.abs(values))
    correct_prediction_count = 0

    for i in range(len(values)):
    

        if np.abs(values[sort_order][i]) > 3 * predicted_sds[sort_order][i]:
            plt.scatter(i,np.abs(values[sort_order][i]),marker = ".",color = "orange",zorder = 2)
            plt.scatter(i,3 * predicted_sds[sort_order][i],marker = "x",color = "orange",zorder = 2)

        else:
            plt.scatter(i,np.abs(values[sort_order][i]),marker = ".",color = "green",zorder = 2)
            plt.scatter(i,3 * predicted_sds[sort_order][i],marker = "x",color = "green",zorder = 2)
            correct_prediction_count += 1.

    plt.xlabel("Outcome")
    plt.ylabel(y_label)

    plt.title(str(round(correct_prediction_count / len(values) * 100,2)) + r" \% of $\delta r_p$ outcomes correctly predicted")

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath + prefix + suffix,bbox_inches='tight')


def plot_all_results(path,graphics_path = None):

    cov_pos_mc = np.loadtxt(path + "/cov_mc.txt")
    all_covs = np.loadtxt(path + "/all_covs.txt")
    states = np.loadtxt(path + "/results.txt")
    crude_guesses = np.loadtxt(path + "/crude_guesses.txt")
    all_rps = np.loadtxt(path + "/all_rps.txt")
    all_rps_sds = np.loadtxt(path + "/all_rps_sds.txt")
    true_cartesian_state = np.loadtxt(path + "/true_state.txt")
    true_keplerian_state = np.loadtxt(path + "/true_keplerian_state.txt")
    true_mu = np.loadtxt(path + "/true_mu.txt")
    true_rp = true_keplerian_state[0] * (1 - true_keplerian_state[1])


    states_m_mean = (states.T - np.array(list(true_cartesian_state[0:6]) + [true_mu])).T
    all_rps_m_mean = all_rps - true_rp
    prefix = path.split("/")[-1] + "_"

    print "\t Plotting case " + str(prefix)


    vertices,facets = polyhedron.load_shape("../resources/shape_models/itokawa_8_scaled.obj")
    max_radius = polyhedron.get_circumscribing_radius(vertices)

    draw_dispersions_1d(r"$\delta r_p$ (m)",
        all_rps,
        all_rps_sds,
        graphics_path,
        prefix = prefix,
        suffix = "drp.pdf",
        sphere_radius = max_radius,
        true_rp = true_rp)

    draw_orbit_inertial(path,
        position_estimates = None,
        crude_guesses = None,
        savepath = graphics_path,
        prefix = prefix,
        suffix = "inertial_orbit.pdf",
        sphere_radius = max_radius)
    draw_orbit_inertial(path,
        position_estimates = states[0:3,:],
        center = True,
        crude_guesses = crude_guesses,
        savepath = graphics_path,
        prefix = prefix,
        suffix = "inertial_orbit_center.pdf",
        sphere_radius = max_radius)

    draw_orbit_body_frame(path,states,
        position_estimates = None,
        crude_guesses = None,
        savepath = graphics_path,
        prefix = prefix,
        suffix = "body_fixed_orbit.pdf",
        vertices = vertices,
        facets = facets)
        
    draw_dispersions(name = "pos",
        axis = 0,
        parameters = states_m_mean[0:3,:],
        crude_guesses_minus_mean = None,
        cov_mc = cov_pos_mc[0:3,0:3],
        cov_model = all_covs,
        labels = [r"$\delta X (m)$",r"$\delta Y (m)$",r"$\delta Z (m)$"],
        savepath = graphics_path,
        prefix = prefix,
        suffix = "YZ_pos.pdf")
    draw_dispersions(name = "pos",
        axis = 1,
        parameters = states_m_mean[0:3,:],
        crude_guesses_minus_mean = None,
        cov_mc = cov_pos_mc[0:3,0:3],
        cov_model = all_covs,
        labels = [r"$\delta X (m)$",r"$\delta Y (m)$",r"$\delta Z (m)$"],
        savepath = graphics_path,
        prefix = prefix,
        suffix = "XZ_pos.pdf")
    draw_dispersions(name = "pos",
        axis = 2,
        parameters = states_m_mean[0:3,:],
        crude_guesses_minus_mean = None,
        cov_mc = cov_pos_mc[0:3,0:3],
        cov_model = all_covs,
        labels = [r"$\delta X (m)$",r"$\delta Y (m)$",r"$\delta Z (m)$"],
        savepath = graphics_path,
        prefix = prefix,
        suffix = "XY_pos.pdf")
    draw_dispersions("vel",0,1000 * states_m_mean[3:6,:],None,1e6 * cov_pos_mc[3:6,3:6],1e6 * all_covs,
        [r"$\delta \dot{X} (m/s)$",r"$\delta \dot{Y} (m/s)$",r"$\delta \dot{Z} (m/s)$"],
        savepath = graphics_path,
        prefix = prefix,
        suffix = "YZ_vel.pdf")
    draw_dispersions("vel",1,1000 * states_m_mean[3:6,:],None,1e6 * cov_pos_mc[3:6,3:6],1e6 * all_covs,
        [r"$\delta \dot{X} (mm/s)$",r"$\delta \dot{Y} (mm/s)$",r"$\delta \dot{Z} (mm/s)$"],
        savepath = graphics_path,
        prefix = prefix,
        suffix = "XZ_vel.pdf")
    draw_dispersions("vel",2,1000 * states_m_mean[3:6,:],None,1e6 * cov_pos_mc[3:6,3:6],1e6 * all_covs,
        [r"$\delta \dot{X} (mm/s)$",r"$\delta \dot{Y} (mm/s)$",r"$\delta \dot{Z} (mm/s)$"],
        savepath = graphics_path,
        prefix = prefix,
        suffix = "XY_vel.pdf")


graphics_path = "/Users/bbercovici/GDrive/CUBoulder/Research/conferences/hawai_2019/paper/Figures/"


list_results(graphics_path = graphics_path)
