import os
import json
import numpy as np

import os
import platform


if (platform.system() == 'Linux'):
	base_location = "/orc_raid/bebe0705/"
else:
	base_location = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/"


# NAVIGATION_TIMES : number of observation times
# DENSITY : density of small body (kg/m^3)
# HARMONICS_DEGREE : max degree/order of the considered spherical harmonics expansions
# USE_HARMONICS : if True, the true dynamics will use a spherical harmonics gravity model
# USE_HARMONICS_ESTIMATED_DYNAMICS : if True, the estimated dynamics will use a spherical harmonics gravity model computed from the fit shape
# INSTRUMENT_FREQUENCY : instrument frequency in Hz.  1./INSTRUMENT_FREQUENCY defines the time in between two successive observations.
# LOS_NOISE_SD_BASELINE : 1 sigma range measurement noise (m)
# LOS_NOISE_FRACTION_MES_TRUTH : if non zero , increases the standard deviation of the applied noise by an amount proportional to the measured range
# SHAPE_RECONSTRUCTION_OUTPUT_DIR : directory where output of shape reconstruction phase was stored 
# USE_TRUE_STATES : if true, will initialize the filter with the true spacecraft and small body states
# INPUT_DIR : directory where inputs for Navigation are stored
# OUTPUT_DIR : directory where outpurs from Navigation should be stored
# PROCESS_NOISE_SIGMA_VEL : 1 sigma amplitude of spacecraft SNC process noise (m/s^2)
# PROCESS_NOISE_SIGMA_OMEG : 1 sigma amplitude of small-body attitude state SNC process noise (rad/s^2)
# SKIP_FACTOR : between 0 and 1. if == to 1 , will use all pixels to determine position/attitude mes. Values between 0.9 and 1 seem good enough


all_data = [

{
"NAVIGATION_TIMES" : 80,
"DENSITY" : 1900,
"HARMONICS_DEGREE" : 10,
"USE_HARMONICS" : True,
"USE_HARMONICS_ESTIMATED_DYNAMICS" : True,
"INSTRUMENT_FREQUENCY" : 1./3600,
"LOS_NOISE_SD_BASELINE" : 5e-1,
"LOS_NOISE_FRACTION_MES_TRUTH" : 0,
"SHAPE_RECONSTRUCTION_OUTPUT_DIR" : base_location + "ShapeReconstruction/output/test_0/",
"USE_TRUE_STATES": False,
"INPUT_DIR" : base_location + "Navigation/input/test_0",
"OUTPUT_DIR" : base_location + "Navigation/output/test_0",
"PROCESS_NOISE_SIGMA_VEL": 1e-10 ,
"PROCESS_NOISE_SIGMA_OMEG": 1e-10 ,
"SKIP_FACTOR": 0.94
},

{
"NAVIGATION_TIMES" : 80,
"DENSITY" : 1900,
"HARMONICS_DEGREE" : 10,
"USE_HARMONICS" : True,
"USE_HARMONICS_ESTIMATED_DYNAMICS" : True,
"INSTRUMENT_FREQUENCY" : 1./3600,
"LOS_NOISE_SD_BASELINE" : 5e-1,
"LOS_NOISE_FRACTION_MES_TRUTH" : 0,
"SHAPE_RECONSTRUCTION_OUTPUT_DIR" : base_location + "ShapeReconstruction/output/test_1/",
"USE_TRUE_STATES": False,
"INPUT_DIR" : base_location + "Navigation/input/test_1",
"OUTPUT_DIR" : base_location + "Navigation/output/test_1",
"PROCESS_NOISE_SIGMA_VEL": 1e-10 ,
"PROCESS_NOISE_SIGMA_OMEG": 1e-10 ,
"SKIP_FACTOR": 0.94
}


# {
# "NAVIGATION_TIMES" : 80,
# "DENSITY" : 1900,
# "HARMONICS_DEGREE" : 10,
# "USE_HARMONICS" : True,
# "USE_HARMONICS_ESTIMATED_DYNAMICS" : True,
# "INSTRUMENT_FREQUENCY" : 1./3600,
# "LOS_NOISE_SD_BASELINE" : 5e-1,
# "LOS_NOISE_FRACTION_MES_TRUTH" : 0,
# "SHAPE_RECONSTRUCTION_OUTPUT_DIR" : base_location + "ShapeReconstruction/output/test_0/",
# "USE_TRUE_STATES": False,
# "INPUT_DIR" : base_location + "Navigation/input/test_2",
# "OUTPUT_DIR" : base_location + "Navigation/output/test_2",
# "PROCESS_NOISE_SIGMA_VEL": 1e-10 ,
# "PROCESS_NOISE_SIGMA_OMEG": 1e-11 ,
# "SKIP_FACTOR": 0.94
# }


]

for data in all_data:
	print("\t Case " + data["INPUT_DIR"].split("/")[-1])

	print("\t - Making directory")

	os.system("mkdir " + data["INPUT_DIR"])
	os.system("mkdir " + data["OUTPUT_DIR"])

	print("\t - Copying input file in build/")

	with open('input_file.json', 'w') as outfile:
		json.dump(data, outfile)

	print("\t - Saving input file in output/")
	
	with open(data["INPUT_DIR"] + '/input_file.json', 'w') as outfile:
		json.dump(data, outfile)
	
	with open(data["OUTPUT_DIR"] + '/input_file.json', 'w') as outfile:
		json.dump(data, outfile)

	print("\t - Running case " +  data["INPUT_DIR"].split("/")[-1])

	os.system("./Navigation | tee " + data["OUTPUT_DIR"] + "/log.txt" )


