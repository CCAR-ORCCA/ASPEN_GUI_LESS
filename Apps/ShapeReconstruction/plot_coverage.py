import mayavi
import numpy as np

import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import json
import np_array_to_latex
import fnmatch

from matplotlib.colors import LogNorm

plt.switch_backend('Qt5Agg')


base_loc = '/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output'
dir_list = next(os.walk(base_loc))[1]

case_number_list = [int(dir_list[e].split("_")[1]) for e in range(len(dir_list))]
case_number_list.sort()
dir_list = ["case_" + str(case_number_list[e]) for e in range(len(dir_list))]




for directory in dir_list:
     
    
    print("Reading " + directory + " ...")

    for file in os.listdir(base_loc + "/" + directory):
        if fnmatch.fnmatch(file, '*_coverage_pc.obj'):
            filepath = base_loc + "/" + directory + "/" + file 
            break

    coordinates = np.loadtxt(filepath,dtype = str)
    print ("Done loading ...")

    coordinates = coordinates[coordinates[:,0] == "v",1:].astype(float)

    print ("Done pruning ...")

    longitudes = np.arctan2(coordinates[:,1],coordinates[:,0]) * 180./np.pi
    latitudes = np.arctan(coordinates[:,2]/np.linalg.norm(coordinates[:,0:2],axis = 1)) * 180./np.pi
    # radii = np.linalg.norm()
    print ("Done computing ...")

    plt.hist2d(longitudes, latitudes, bins=(50, 25),norm=LogNorm(),cmin  = 30)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Coverage, case " + directory.split("_")[1])
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    # plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/conferences/GNSKi_2019/paper/Figures/coverage_" + directory + ".pdf")
    plt.clf()










