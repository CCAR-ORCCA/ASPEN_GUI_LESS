import mpl_toolkits.mplot3d as a3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from matplotlib import rc


rc('text', usetex=True)

def compute_volume(verts,facets):
  volume = 0
  for i in range(len(facets)):

    r0 = np.array(verts[facets[i,0]])
    r1 = np.array(verts[facets[i,1]])
    r2 = np.array(verts[facets[i,2]])

    dv = np.inner(r0,np.cross(r1 - r0,r2 - r0))/6
    volume += dv
  return volume

def compute_inertia(verts,facets):
  P_xx = 0
  P_yy = 0
  P_zz = 0
  P_xy = 0
  P_xz = 0
  P_yz = 0

  for i in range(len(facets)):

    r0 = np.array(verts[facets[i,0]])   
    r1 = np.array(verts[facets[i,1]])   
    r2 = np.array(verts[facets[i,2]])    
    dv = np.inner(r0,np.cross(r1 - r0,r2 - r0)) / 6

    P_xx += dv / 20. * (2 * r0[0] * r0[0]
                       + 2 * r1[0] * r1[0]
                       + 2 * r2[0] * r2[0]
                       + r0[0] * r1[0]
                       + r0[0] * r1[0]
                       + r0[0] * r2[0]
                       + r0[0] * r2[0]
                       + r1[0] * r2[0]
                       + r1[0] * r2[0])

    P_yy += dv / 20 * (2 * r0[1] * r0[1]
                       + 2 * r1[1] * r1[1]
                       + 2 * r2[1] * r2[1]
                       + r0[1] * r1[1]
                       + r0[1] * r1[1]
                       + r0[1] * r2[1]
                       + r0[1] * r2[1]
                       + r1[1] * r2[1]
                       + r1[1] * r2[1])

    P_zz += dv / 20 * (2 * r0[2] * r0[2]
                       + 2 * r1[2] * r1[2]
                       + 2 * r2[2] * r2[2]
                       + r0[2] * r1[2]
                       + r0[2] * r1[2]
                       + r0[2] * r2[2]
                       + r0[2] * r2[2]
                       + r1[2] * r2[2]
                       + r1[2] * r2[2])

    P_xy += dv / 20 * (2 * r0[0] * r0[1]
                       + 2 * r1[0] * r1[1]
                       + 2 * r2[0] * r2[1]
                       + r0[0] * r1[1]
                       + r0[1] * r1[0]
                       + r0[0] * r2[1]
                       + r0[1] * r2[0]
                       + r1[0] * r2[1]
                       + r1[1] * r2[0])

    P_xz += dv / 20 * (2 * r0[0] * r0[2]
                       + 2 * r1[0] * r1[2]
                       + 2 * r2[0] * r2[2]
                       + r0[0] * r1[2]
                       + r0[2] * r1[0]
                       + r0[0] * r2[2]
                       + r0[2] * r2[0]
                       + r1[0] * r2[2]
                       + r1[2] * r2[0])

    P_yz += dv / 20 * (2 * r0[1] * r0[2]
                       + 2 * r1[1] * r1[2]
                       + 2 * r2[1] * r2[2]
                       + r0[1] * r1[2]
                       + r0[2] * r1[1]
                       + r0[1] * r2[2]
                       + r0[2] * r2[1]
                       + r1[1] * r2[2]
                       + r1[2] * r2[1])

  I = np.array([[P_yy + P_zz, - P_xy, - P_xz],
                [- P_xy,P_xx + P_zz , - P_yz],
                [- P_xz,- P_yz, P_xx + P_yy]])

  return I





def compute_bounding_box(verts):

  xlim = - 1
  ylim = - 1
  zlim = - 1


  for i in range(len(verts)):

    r = np.array(verts[i])

    if abs(r[0]) > xlim:
      xlim = abs(r[0])

    if abs(r[1]) > ylim:
      ylim = abs(r[1])

    if abs(r[2]) > zlim:
      zlim = abs(r[2])

  return xlim, ylim, zlim

def compute_center_of_mass(verts,facets):

  cm = np.zeros(3)
  volume = compute_volume(verts, facets)

  for i in range(len(facets)):

    r0 = np.array(verts[facets[i,0]])
    r1 = np.array(verts[facets[i,1]])
    r2 = np.array(verts[facets[i,2]])

    dv = np.inner(r0,np.cross(r1 - r0,r2 - r0))/6

    dr = 0.25 * (r0 + r1 + r2)

    cm += dr / volume
  return cm


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


def mrp_to_DCM(sigma):
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



def convert_to_body_frame(inertial_state,mrp):

  orbit = np.zeros([3,mrp.shape[1]])

  for i in range(orbit.shape[1]):

    orbit[:,i] = mrp_to_DCM(mrp[:,i]).dot(inertial_state[:,i])

  return orbit





def plot_facet_seen_count_vs_time(path):

  paths = os.listdir(path)
  read = 0

  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  print paths
  for directory in paths:

    if ("pole" or "hovering" in directory):


      try:
        facet_seen_count_vs_time = np.loadtxt(directory + "/facets_seen_count.txt")
      except IOError:
        print "Incomplete"

      else:
        plt.plot(facet_seen_count_vs_time[:,0] / (86400 * 30.5),facet_seen_count_vs_time[:,1] * 100)
        read = read + 1

  print read
  plt.xlabel(r"Time (months)")
  plt.ylabel(r"Percentage of facets seen")
  plt.title(r"Percentage of facets seen over time",y = 1.04)

  plt.savefig(path + "/facet_visi_vs_time.pdf")
  plt.clf()


def plot_longitude_latitude_impact_count_all(path):

  paths = os.listdir(path)

  plt.rc('text', usetex=True)
  plt.rc('font', family='serif')
  for directory in paths:

    if ("pole" in directory):
      print directory
      plot_longitude_latitude_impact_count(directory)
  plt.clf()
     



def plot_azimuth_elevation_facet_seen_count(rootpath):


  paths = os.listdir(rootpath)

  min_visibility = 1e6
  max_visibility = -1e6

  angles_list = []
  colors = []


  for directory in paths:

    if ("pole" in directory):

      angles = np.loadtxt(directory + "/Trajectory_BodyFixed.txt")[0,7:]
      visibility = np.loadtxt(directory + "/facets_seen_count.txt")

      min_visibility = min(min_visibility,visibility[-1,1])
      max_visibility = max(max_visibility,visibility[-1,1])


      angles_list += [angles[0:2]]
      colors += [100 * visibility[-1,1]]

  angles_mat = np.vstack(angles_list)
  min_visibility = min_visibility * 100
  max_visibility = max_visibility * 100


  sc = plt.scatter(180./np.pi * angles_mat[:,0] - 180,180./np.pi * angles_mat[:,1], 
  c = colors, vmin = min_visibility, vmax = max_visibility, 
  s = 50, edgecolors = 'none')


  plt.colorbar(sc)

  plt.xlim([-180,180])
  plt.ylim([0,180])
  plt.xlabel("Azimuth (deg)")
  plt.ylabel("Elevation (deg)")
  plt.title("Percentage of facets seen after 6 months vs pole orientation",y =1.04)

 
  plt.savefig(rootpath + "visibility.pdf")
  plt.clf()


def plot_histogram_facet_seen_count(rootpath):

  paths = os.listdir(rootpath)
  colors = []

  visibility = []

  for directory in paths:

    if ("pole" in directory):

      visibility += [np.loadtxt(directory + "/facets_seen_count.txt")[-1,1]]

  visibility = np.array(visibility)

  plt.hist(100 * visibility,  bins=[0, 20, 40, 60, 80,100])

  plt.title(r"Final surface visibility distribution")
  plt.xlabel(r"Final surface visibility (percentage)")
  plt.ylabel(r"Occurences")

  # plt.show()

  plt.savefig(rootpath + "visibility_histogram.pdf")
  plt.clf()


def plot_longitude_latitude_impact_count(path_to_impact_count):


  impact_counts = np.loadtxt(path_to_impact_count + "/lat_long_impacts.txt")

  zero_visibility = impact_counts[:,2] == 0
  visibility = impact_counts[:,2] > 0

  min_impact = np.amin(impact_counts[visibility,2])
  max_impact = np.amax(impact_counts[visibility,2])

  sc = plt.scatter(impact_counts[visibility,0], impact_counts[visibility,1], 
    c = impact_counts[visibility,2], vmin = min_impact, vmax = max_impact, 
    s = 10, edgecolors = 'none')

  plt.scatter(impact_counts[zero_visibility,0], impact_counts[zero_visibility,1], 
    color = 'k',s = 10, edgecolors = None, marker = 'd')

  plt.colorbar(sc)

  plt.xlim([-180,180])
  plt.ylim([-90,90])
  plt.xlabel(r"Longitude (deg)")
  plt.ylabel(r"Latitude (deg)")
  plt.title(r"Visibility occurence against longitude/latitude")

 
  plt.savefig(path_to_impact_count + "/visibility.pdf")
  plt.clf()





def plot_body_frame_traj(path_to_traj,path_to_shape,scale_factor,is_nicolas, path_to_interpolated_mrp = None , already_in_body_frame = True):
  if (is_nicolas is False):
    orbit = np.loadtxt(path_to_traj)[0:3,:]
  else:
    orbit = np.loadtxt(path_to_traj)[:,1:4].T



  
  if already_in_body_frame is False:
    mrp = np.loadtxt(path_to_interpolated_mrp)[0:3,:]

    body_frame_orbit = convert_to_body_frame(orbit,mrp)

  else:
    body_frame_orbit = np.copy(orbit)
    
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  

  ax.plot(body_frame_orbit[0,:],body_frame_orbit[1,:],body_frame_orbit[2,:])
  ax.scatter(body_frame_orbit[0,0],body_frame_orbit[1,0],body_frame_orbit[2,0],'*',color = 'g')
  ax.scatter(body_frame_orbit[0,-1],body_frame_orbit[1,-1],body_frame_orbit[2,-1],'*',color = 'r')

  plot_shape(path_to_shape,ax = ax,scale_factor = scale_factor)



def plot_jacobi(path):
    if path is None:
       energy = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/build/orbit_energy.txt")
    else:
       energy = np.loadtxt(path)

    plt.plot(range(len(energy)),(energy - energy[0])/energy[0] * 100)
    plt.xlabel("Index")
    plt.ylabel("Relative change (%)")

    plt.show()



def plot_shape(path,already_in_body_frame = True,ax = None,scale_factor = 1):

  # The obj file is read
  read_obj = np.loadtxt(path,dtype = 'string')

  verts = []
  facets = []

  for line_index in range(len(read_obj)):
      if read_obj[line_index,0] == 'v':
          verts += [tuple(scale_factor * np.array(read_obj[line_index,1:],dtype = 'float'))]
      if read_obj[line_index,0] == 'f':
          facets += [np.array(read_obj[line_index,1:],dtype = 'int') - 1]

  facets = np.vstack(facets)

  # The barycenter is computed
  cm = compute_center_of_mass(verts,facets)

  if already_in_body_frame is False:
    
    # The direction of the principal axes is also computed
    inertia = compute_inertia(verts,facets)
    moments, axes = np.linalg.eigh(inertia)


    if (np.linalg.det(axes) < 0):
      axes[:,0] = - axes[:,0]

    # The shape is centered at its barycenter and oriented along its principal axes
    for i in range(len(verts)):
      verts[i] = axes.T.dot(verts[i] - cm);

  # The limits of the bounding box are found
  x_lim,y_lim,z_lim = compute_bounding_box(verts)

  lim = max(x_lim,y_lim,z_lim)
  if ax is None:
    ax = a3.Axes3D(plt.figure())
    ax.dist = 30
    ax.azim = - 140
    ax.elev = 20

  ax.set_xlim([-2 * lim,2 * lim])
  ax.set_ylim([-2 * lim,2 * lim])
  ax.set_zlim([-2 * lim,2 * lim])

  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_zlabel("Z (m)")

  for i in np.arange(len(facets)):

      trig = [ verts[facets[i,0]], verts[facets[i,1]], verts[facets[i, 2]] ]
      face = a3.art3d.Poly3DCollection([trig])
      face.set_color('grey')
      face.set_edgecolor('k')
      face.set_alpha(1.)
      ax.add_collection3d(face)

  plt.show()