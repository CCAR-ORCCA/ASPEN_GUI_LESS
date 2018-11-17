import os
import json
import numpy as np

import os
import platform



if (platform.system() == 'Linux'):
	base_location = "/orc_raid/bebe0705/"
else:
	base_location = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/"


all_data = [

{
"OBSERVATION_TIMES" : 150,
"IOD_PARTICLES" : 100,
"IOD_ITERATIONS" : 100,
"IOD_RIGID_TRANSFORMS_NUMBER" : 20 ,
"SMA" : 1000,
"E" : 0.25,
"I" : 1.4,
"RAAN" :0.2,
"PERI_OMEGA" : 0.3,
"M0" : 1.57, 
"SPIN_PERIOD" : 12, 
"LONGITUDE_SPIN" : 0., 
"LATITUDE_SPIN" : 0., 
"DENSITY" : 1900,
"HARMONICS_DEGREE" : 10,
"USE_HARMONICS" : True,
"INSTRUMENT_FREQUENCY_SHAPE" : 0.0005,
"MRP_0" : [0,0,0],
"N_ITER_BUNDLE_ADJUSTMENT" : 2,
"N_ITER_SHAPE_FILTER" : 3,
"MIN_TRIANGLE_ANGLE" : 30,
"MAX_TRIANGLE_SIZE" : 5,
"SURFACE_APPROX_ERROR" : 1,
"NUMBER_OF_EDGES" : 2000,
"BA_H" : 4,
"LOS_NOISE_SD_BASELINE" : 5e-1,
"USE_BA" : True,
"USE_ICP" : True,
"USE_TRUE_RIGID_TRANSFORMS" : False,
"INPUT_DIR" : base_location + "ShapeReconstruction/input/test_0",
"OUTPUT_DIR" : base_location + "ShapeReconstruction/output/test_0"
}

]

for data in all_data:
	print("\t Case " + data["INPUT_DIR"].split("/")[-1])
	
	os.system("mkdir " + data["INPUT_DIR"])
	os.system("mkdir " + data["OUTPUT_DIR"])

	print("\t - Making directory")
	print("\t - Copying input file in build/")

	with open('input_file.json', 'w') as outfile:
		json.dump(data, outfile)

	print("\t - Saving input file in input/ and output/")
	with open(data["INPUT_DIR"] + '/input_file.json', 'w') as outfile:
		json.dump(data, outfile)
	with open(data["OUTPUT_DIR"] + '/input_file.json', 'w') as outfile:
		json.dump(data, outfile)

	print("\t - Running case " +  data["INPUT_DIR"].split("/")[-1])

	os.system("./ShapeReconstruction | tee " + data["OUTPUT_DIR"] + "/log.txt" )


