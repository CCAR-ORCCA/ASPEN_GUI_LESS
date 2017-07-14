import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from nparray2ltx import np_array_to_latex
'''
ffmpeg -f image2 -framerate 25 -i computed_prefit_00%04d.png -vcodec libx264 -b:v 800k ../itokawa/computed_itokawa.avi
'''



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





def plot_exposure_time_vs_angular_drift():
    plt.clf()

    rc('text', usetex=True)
    plt.rc('font', family='serif')
    t = np.array([5.44e-3, 8.20e-3, 10.9e-3, 16.4e-3,
21.8e-3, 32.8e-3, 43.5e-3, 65.6e-3,
87.0e-3, 131e-3, 174e-3, 262e-3,
348e-3, 525e-3, 696e-3, 1.05, 1.39,
2.10, 2.79, 4.20, 5.57, 8.40,
11.1, 16.8, 22.3, 33.6, 44.6,
67.2, 89.1, 134, 178])


    R_T = 50 # maybe 50?
    omega_T = 2 * np.pi /(27 * 60)

    r = np.array([350,5e2,1e3,2.5e3,5e3,7.5e3,1e4]) # (meters)

    alpha = 1 # pixel drift (less than 1)
    f = 0.120 # focal length(meters)
    h = 12e-6 # pixel size(meters)

    T = np.linspace(t[0],t[-1],1e5) # interpolated exposure times

    cmap = plt.cm.get_cmap("RdYlGn", len(r))


    omega_D_list_alpha = []

    for i in range(len(r)):

        omega_D_star = - omega_T * R_T / r[i]
        omega_D_1 = omega_D_star + alpha * h / (f * T)
        omega_D_2 = omega_D_star - alpha * h / (f * T)

        omega_D_list_alpha += [omega_D_star]

        plt.loglog(T, 180./np.pi * (omega_D_1 - omega_D_2)/2,color = cmap(i),label = "r = " + str(r[i]/1000) + " km",linewidth = 1.5)


    plt.xlabel(r"Exposure time (s)")
    plt.ylabel(r"$\omega_D$ error(deg/s)")
    plt.title(r"Maximum $\omega_D$ error from nominal slewing rate $\omega_D^*$ , $\alpha =$" + str(alpha))
    plt.legend(loc = "best")
    plt.grid(True, which="both")

    plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/omega_envelopes.pdf")
    plt.clf()

    r_omega_D_alpha = np.zeros([len(r),2])
    r_omega_D_alpha[:,0] = np.copy(r) / 1000
    r_omega_D_alpha[:,1] = np.array(omega_D_list_alpha) * 180. / np.pi



    np_array_to_latex(r_omega_D_alpha,"/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/r_omega_D_alpha",
        row_headers = None,column_headers = [r"$r$ (km)",r"$\omega_D^*$ (deg/s)"],
 column = True,type = 'e', decimals = 3, ante_decimals = 6)



    # alphas = np.linspace(0,1)


    # plt.plot(alphas,180. / np.pi * alphas * h / f)

    # plt.xlabel(r"Pixel fraction $\alpha$")
    # plt.ylabel(r"$\epsilon^*$ (deg)")
    # plt.title(r"Maximum pointing error assuming regulated $\omega_D$")
    # plt.legend(loc = "best")
    # plt.grid()
    # plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/epsilon_nominal.pdf")
    
    plt.clf()



    # for i in range(len(r)):

    #     omega_D_star = - omega_T * R_T / r[i]
    #     omega_D_1 = omega_D_star + alpha * h / (f * T)

    #     plt.loglog(T,180./np.pi * omega_D_1,label = "r = " + str(r[i]/1000) + " km",linewidth = 1.5)
    # plt.grid()

    # plt.xlabel(r"Exposure time (s)")
    # plt.ylabel(r"$\omega_D$ (deg/s)")
    # plt.title(r"Maximum pointing drift rate assuming $\omega_D = 0$ rad/s, $\alpha = $" + str(alpha))
    # plt.legend(loc = "center right",bbox_to_anchor = (0.93,0.88),ncol = 3,prop = {'size':10})


    # plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/omega_still.pdf")

    plt.clf()



    plt.loglog(r/1000,r * h / f)

    plt.xlabel("Altitude (km)")
    plt.ylabel("Projected pixel size (m)")
    plt.grid(True, which="both")

    plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/pixel_size.pdf")


    plt.clf()


    # for i in range(len(r)):

    #     omega_D_star = - omega_T * R_T / r[i]
    #     omega_D_1 = omega_D_star + alpha * h / (f * T)
    #     epsilon = omega_D_1 * T
    #     plt.loglog(T, 180./np.pi * epsilon,label = "r = " + str(r[i]/1000) + " km",linewidth = 1.5)

    # plt.grid()

    # plt.xlabel(r"Exposure time (s)")
    # plt.ylabel(r"$\epsilon$ (deg)")
    # plt.title(r"Maximum pointing error assuming $\omega_D = 0$ rad/s, $\alpha = $" + str(alpha))
    # plt.legend(loc = "center right",bbox_to_anchor = (0.33,0.25))

    # plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/epsilon_still.pdf")

    plt.clf()

    Delta = 5

    omega_D_list_Delta = []

    for i in range(len(r)):

        omega_D_star = - omega_T * R_T / r[i]
        omega_D_1 = omega_D_star + Delta / (r[i] * T)
        omega_D_2 = omega_D_star - Delta / (r[i] * T)

        plt.loglog(T, 180./np.pi * (omega_D_1 - omega_D_2)/2,color = cmap(i),label = "r = " + str(r[i]/1000) + " km",linewidth = 1.5)

        omega_D_list_Delta += [omega_D_star]


    r_omega_D_Delta = np.zeros([len(r),2])
    r_omega_D_Delta[:,0] = np.copy(r) / 1000
    r_omega_D_Delta[:,1] = np.array(omega_D_list_Delta) * 180. / np.pi



    np_array_to_latex(r_omega_D_Delta,"/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/r_omega_D_Delta",
        row_headers = None,column_headers = [r"$r$ (km)",r"$\omega_D^*$ (deg/s)"],
 column = True,type = 'e', decimals = 3, ante_decimals = 6)


    plt.xlabel(r"Exposure time (s)")
    plt.ylabel(r"$\omega_D$ error (deg/s)")
    plt.title(r"Maximum $\omega_D$ error from nominal $\omega_D^*$, $\Delta =\ $" + str(Delta) + " m")
    plt.legend(loc = "best")
    plt.grid(True, which="both")

    plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/omega_envelopes_Delta.pdf")
    plt.clf()


    # for i in range(len(r)):

    #     omega_D_star = - omega_T * R_T / r[i]
    #     omega_D_1 = omega_D_star + Delta / (r[i] * T)
    #     omega_D_2 = omega_D_star - Delta / (r[i] * T)

    #     epsilon = omega_D_1 * T

    #     plt.loglog(T,180./np.pi * epsilon,label = "r = " + str(r[i]/1000) + " km",linewidth = 1.5)
    # plt.grid()

    # plt.xlabel(r"Exposure time (s)")
    # plt.ylabel(r"$\epsilon$ (deg)")
    # plt.title(r"Maximum pointing error assuming $\omega_D = 0$ rad/s, $\Delta =\ $" + str(Delta) + " m")
    # plt.legend(loc = "center right",bbox_to_anchor = (0.95,0.16),ncol = 3)

    # plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/epsilon_still_Delta.pdf")

    plt.clf()

    # for i in range(len(r)):

    #     omega_D_star = - omega_T * R_T / r[i]
    #     omega_D_1 = omega_D_star + Delta / (r[i] * T)
    #     omega_D_2 = omega_D_star - Delta / (r[i] * T)

    #     l = plt.loglog(T,180./np.pi * omega_D_1,label = "r = " + str(r[i]/1000) + " km",linewidth = 1.5)
    #     plt.loglog(T,180./np.pi * omega_D_1,label = "r = " + str(r[i]/1000) + " km",linewidth = 1.5,
    #         color = l[0].get_color())

    # plt.grid()
    # plt.grid(True, which="both")

    # plt.xlabel(r"Exposure time (s)")
    # plt.ylabel(r"$\omega_D$ (deg/s)")
    # plt.title(r"Maximum pointing drift rate assuming $\omega_D = 0$ rad/s, $\Delta =\ $" + str(Delta) + " m")
    # plt.legend(loc = "center right",bbox_to_anchor = (0.70,0.08),ncol = 3,prop = {'size':10})


    # plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/reports/HO3/omega_still_Delta.pdf")

    plt.clf()


   







    




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






