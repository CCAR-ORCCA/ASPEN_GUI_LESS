import os
import json
import numpy as np

import os
import platform
import sys
import itertools
import time


def generate_all_cases_dictionnary_list(base_dictionnary,all_cases_dictionnary,base_location,sim_name):
       
    keys, values = zip(*all_cases_dictionnary.items())
    dictionnary_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_cases_dictionnary_list = [{**dictionnary_list[e],**base_dictionnary} for e in range(len(dictionnary_list))]

    for e in range(len(dictionnary_list)):
        all_cases_dictionnary_list[e]["INPUT_DIR"] = base_location + "ShapeReconstruction/input/" + sim_name + "_" + str(e)
        all_cases_dictionnary_list[e]["OUTPUT_DIR"] = base_location + "ShapeReconstruction/output/" + sim_name + "_" + str(e)

    return all_cases_dictionnary_list

if (platform.system() == 'Linux'):
    base_location = "/orc_raid/bebe0705/"
else:
    base_location = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/"


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
"BA_H" : 4,
"NUMBER_OF_EDGES" : 2000
}

# all_cases_dictionnary = {
# "USE_BEZIER_SHAPE" : [False,True],
# "INSTRUMENT_FREQUENCY_SHAPE" : [0.0004,0.0005,0.0006],
# "NUMBER_OF_EDGES" : [1500,2000,2500,3000]
# }

all_cases_dictionnary = {
"LOS_NOISE_SD_BASELINE" : [5e-1,1e0,1.5e0],
"INSTRUMENT_FREQUENCY_SHAPE" : [0.0003,0.0004,0.0005,0.0006],
"LATITUDE_SPIN" : [1. * np.pi / 180,0.,10. * np.pi / 180]
}


all_data = generate_all_cases_dictionnary_list(base_dictionnary,
    all_cases_dictionnary,base_location,"thesis_fast_slam")



for data in all_data:

    print("\t Case " + data["INPUT_DIR"].split("/")[-1])
    print("\t - Making directory")

    os.system("mkdir " + data["INPUT_DIR"])
    os.system("mkdir " + data["OUTPUT_DIR"])

    if (os.path.isfile(data["OUTPUT_DIR"] + "/fit_shape.obj") is False):

        print("\t - Copying input file in build/")

        with open('input_file.json', 'w') as outfile:
            json.dump(data, outfile)

        print("\t - Saving input file in input/ and output/")
        with open(data["INPUT_DIR"] + '/input_file.json', 'w') as outfile:
            json.dump(data, outfile)
        with open(data["OUTPUT_DIR"] + '/input_file.json', 'w') as outfile:
            json.dump(data, outfile)

        print("\t - Running case " +  data["INPUT_DIR"].split("/")[-1])

        os.system("./ShapeReconstruction 2>&1 | tee -a " + data["OUTPUT_DIR"] + "/log.txt" )
    else:
        print("Case " + data["INPUT_DIR"].split("/")[-1] + " has already finished running. skipping case ...")














