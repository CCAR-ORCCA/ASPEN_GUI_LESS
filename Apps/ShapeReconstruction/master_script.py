import os
import json
import numpy as np

import os
import platform
import sys
import itertools
import time
from multiprocessing import Pool
import socket

def generate_all_cases_dictionnary_list(base_dictionnary,all_cases_dictionnary,base_location,sim_name):
       
    keys, values = zip(*all_cases_dictionnary.items())
    dictionnary_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_cases_dictionnary_list = [{**dictionnary_list[e],**base_dictionnary} for e in range(len(dictionnary_list))]

    for e in range(len(dictionnary_list)):
        all_cases_dictionnary_list[e]["INPUT_DIR"] = base_location + "ShapeReconstruction/input/" + sim_name + "_" + str(e)
        all_cases_dictionnary_list[e]["OUTPUT_DIR"] = base_location + "ShapeReconstruction/output/" + sim_name + "_" + str(e)

    return all_cases_dictionnary_list

# Replace the paths after 'base_location' with the existing directory under which the input/ and /output sub-directories
# will be created and populated
if (socket.gethostname() == "fortuna"):
    base_location = "/orc_raid/bebe0705/"
else:
    base_location = "../"






def run_sim(data,quiet = True):

    if not quiet:
        print("\t Case " + data["INPUT_DIR"].split("/")[-1])
        print("\t - Making directory")

    os.system("mkdir " + data["INPUT_DIR"])
    os.system("mkdir " + data["OUTPUT_DIR"])

    if (os.path.isfile(data["OUTPUT_DIR"] + "/fit_shape.obj") is False):

        if not quiet:
            print("\t - Copying input file in build/")

        with open('input_file.json', 'w') as outfile:
            json.dump(data, outfile)

        if not quiet:    
            print("\t - Saving input file in input/ and output/")
        
        with open(data["INPUT_DIR"] + '/input_file.json', 'w') as outfile:
            json.dump(data, outfile)
        with open(data["OUTPUT_DIR"] + '/input_file.json', 'w') as outfile:
            json.dump(data, outfile)

        if not quiet:
            print("\t - Running case " +  data["INPUT_DIR"].split("/")[-1])

        if not quiet:

            os.system("./ShapeReconstruction " + data["OUTPUT_DIR"] + '/input_file.json' + " 2>&1 | tee -a " + data["OUTPUT_DIR"] + "/log.txt" )
        else:
            os.system("./ShapeReconstruction " + data["OUTPUT_DIR"] + "/input_file.json  >> " + data["OUTPUT_DIR"] + "/log.txt 2>&1" )
    
    else:
        if not quiet:

            print("Case " + data["INPUT_DIR"].split("/")[-1] + " has already finished running. skipping case ...")

def start_sims(n_pools = 1):

    base_dictionnary = {
    "CR_TRUTH" : 1.2,
    "USE_BA" : True,
    "USE_ICP" : True,
    "USE_TRUE_RIGID_TRANSFORMS" : False,
    "MIN_TRIANGLE_ANGLE" : 30,
    "MAX_TRIANGLE_SIZE" : 5,
    "HARMONICS_DEGREE" : 10,
    "USE_HARMONICS" : True,
    "SMA" : 1000.,
    "E" : 0.25,
    "I" : 1.4,
    "RAAN"  : 0.2,
    "PERI_OMEGA" : 0.3,
    "M0" : 1.57,
    "SPIN_PERIOD" : 12.,
    "LONGITUDE_SPIN" : 0., 
    "DENSITY" : 1900,
    "MRP_0" : [0,0,0],
    "SURFACE_APPROX_ERROR" : 1,
    "N_ITER_BUNDLE_ADJUSTMENT" : 3,
    "N_ITER_SHAPE_FILTER" : 3,
    "IOD_ITERATIONS" : 100,
    "IOD_RIGID_TRANSFORMS_NUMBER" : 7,
    "IOD_PARTICLES" : 100,
    "USE_TARGET_POI" : False,
    "USE_BEZIER_SHAPE" : True,
    "TF" : 150,
    "NUMBER_OF_EDGES" : 2000,
    "SAVE_TRANSFORMED_SOURCE_PC" : False
    }


    all_cases_dictionnary = {
    "BA_H" : [0,1,2,3],
    "INSTRUMENT_FREQUENCY_SHAPE" : [0.0003,0.0004],
    "LOS_NOISE_SD_BASELINE" : [5e-1,1e0,1.5e0],
    "LATITUDE_SPIN" : [1. * np.pi / 180,0.,10. * np.pi / 180]
    }

    all_data = generate_all_cases_dictionnary_list(base_dictionnary,
        all_cases_dictionnary,base_location,"thesis_fast_slam")

    if (n_pools > 1):
        p = Pool(n_pools)
        p.map(run_sim, all_data)
    else:
        for data in all_data:
            run_sim(data,quiet = False)













