import os
import json
import numpy as np

import os
import platform


if (platform.system() == 'Linux'):
	base_location = "/orc_raid/bebe0705/ShapeReconstruction/"
else:
	base_location = "../"


all_data = [

{
"OBSERVATION_TIMES" : 160,
"NAVIGATION_TIMES" : 500,
"SMA" : 1000,
"E" : 0.25,
"I" : 1.4,
"RAAN" :0.2,
"PERI_OMEGA" : 0.3,
"M0" : 1.57, 
"SPIN_PERIOD" : 12, 
"LONGITUDE_SPIN" : 0.5, 
"LATITUDE_SPIN" : 0.5, 
"DENSITY" : 1900,
"HARMONICS_DEGREE" : 10,
"USE_HARMONICS" : True,
"INSTRUMENT_FREQUENCY_SHAPE" : 0.0005,
"MRP_0" : [0,0,0],
"dir" : base_location + "output/test_0"}
]

for data in all_data:
	print("\t Case " + data["dir"].split("/")[-1])
	
	os.system("mkdir " + data["dir"])
	print("\t - Making directory")

	print("\t - Copying input file in build/")

	with open('input_file.json', 'w') as outfile:
		json.dump(data, outfile)

	print("\t - Saving input file in output/")
	with open(data["dir"] + '/input_file.json', 'w') as outfile:
		json.dump(data, outfile)
	print("\t - Running case " +  data["dir"].split("/")[-1])

	os.system("./ShapeReconstruction")


