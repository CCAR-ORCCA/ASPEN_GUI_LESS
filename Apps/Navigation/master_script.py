import os
import json
import numpy as np

import os
import platform


if (platform.system() == 'Linux'):
	base_location = "/orc_raid/bebe0705/Navigation/"

else:
	base_location = "../"


all_data = [

{
"OBSERVATION_TIMES" : 80,
"DENSITY" : 1900,
"HARMONICS_DEGREE" : 10,
"USE_HARMONICS" : False,
"INSTRUMENT_FREQUENCY" : 0.0005,
"LOS_NOISE_SD_BASELINE" : 5e-1,
"dir" : base_location + "output/test_0",
"input_dir" : base_location + "input/test_0"
}

]

for data in all_data:
	print("\t Case " + data["dir"].split("/")[-1])
	

	print("\t - Making directory")

	os.system("mkdir " + data["dir"])

	print("\t - Copying input file in build/")

	with open('input_file.json', 'w') as outfile:
		json.dump(data, outfile)

	print("\t - Saving input file in output/")
	with open(data["dir"] + '/input_file.json', 'w') as outfile:
		json.dump(data, outfile)
	print("\t - Running case " +  data["dir"].split("/")[-1])

	os.system("./ShapeReconstruction| tee " + data["dir"] + "/log.txt" )


