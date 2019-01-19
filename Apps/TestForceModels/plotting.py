import numpy as np
import matplotlib.pyplot as plt

SRP_forces = np.loadtxt("output/case_0/SRP_forces.txt")

SH_gravity_forces = np.loadtxt("output/case_0/SH_gravity_forces.txt")

point_mass_gravity_forces = np.loadtxt("output/case_0/point_mass_gravity_forces.txt")

third_body_forces = np.loadtxt("output/case_0/third_body_forces.txt")

time = np.loadtxt("output/case_0/time_full_orbit_point_mass.txt")
state = np.loadtxt("output/case_0/state_full_orbit_point_mass.txt")

plt.plot(time/3600,np.linalg.norm(state[0:3,:],axis = 0),label = "r")

plt.xlabel("Time (hours)")

plt.ylabel("Radius (m)")

plt.legend(loc = "best")

plt.show()


plt.semilogy(time/3600,np.linalg.norm(third_body_forces,axis = 0),label = "third-body")

plt.semilogy(time/3600,np.linalg.norm(point_mass_gravity_forces - SH_gravity_forces,axis = 0),label = "SH")

plt.semilogy(time/3600,np.linalg.norm(point_mass_gravity_forces,axis = 0),label = "point-mass")

plt.semilogy(time/3600,np.linalg.norm(SRP_forces,axis = 0),label = "SRP")

plt.xlabel("Time (hours)")

plt.ylabel("Acceleration magnitude (m/s^2)")

plt.legend(loc = "best",ncol = 4)

plt.show()

