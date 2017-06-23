import mpl_toolkits.mplot3d as a3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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







def plot_longitude_latitude_impact_count(path_to_impact_count):
  impact_counts = np.loadtxt(path_to_impact_count)

  # the np array is turned into a list in preparation for the 
  # use of the imshow function

  impact_counts_list = []
  for i in range(impact_counts.shape[0]):

    if (int(impact_counts[i,-1]) > 0):
      new_list = [ impact_counts[i,0:2] ] * int(impact_counts[i,-1])
      impact_counts_list += [np.vstack(new_list)]

  impact_counts_formatted = np.vstack(impact_counts_list)


  plt.hist2d(impact_counts_formatted[:,0],impact_counts_formatted[:,1],bins = 64,norm = colors.LogNorm())
  plt.colorbar()
  plt.xlim([-180,180])
  plt.ylim([-90,90])
  plt.xlabel("Longitude (deg)")
  plt.ylabel("Latitude (deg)")

  plt.title("Visibility occurence against longitude/latitude")


  plt.savefig("visibility_128.pdf")
  plt.clf()





def plot_body_frame_traj(path_to_traj,path_to_interpolated_mrp,path_to_shape,scale_factor,already_in_body_frame = True):

  orbit = np.loadtxt(path_to_traj)[0:3,:]

  
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