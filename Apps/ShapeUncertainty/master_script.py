import os
import json
import numpy as np

# THIS SCRIPT MUST BE LOCATED IN THE SAME DIRECTORY AS ./build 
# BUT WILL BE CALLED FROM ./build

all_data = [


# {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 5e-3,
# "CORRELATION_DISTANCE" : 5e-2,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_0"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 7.5e-3,
# "CORRELATION_DISTANCE" : 5e-2,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_1"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 5e-3,
# "CORRELATION_DISTANCE" : 7.5e-2,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_2"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 1e-2,
# "CORRELATION_DISTANCE" : 1e-1,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_3"},


# {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 1e-2,
# "CORRELATION_DISTANCE" : 5e-2,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_4"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 1e-2,
# "CORRELATION_DISTANCE" : 7.5e-2,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_5"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 1e-2,
# "CORRELATION_DISTANCE" : 1e-1,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_6"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/itokawa_8.obj",
# "ERROR_STANDARD_DEV" : 1e-2,
# "CORRELATION_DISTANCE" : 1.25e-1,
# "N_MONTE_CARLO" : 10000,
#  "dir" : "../output/test_7"},


# {"PATH_SHAPE" : "../../../resources/shape_models/67P_lowlowres.obj",
# "ERROR_STANDARD_DEV" : 2.5e-2,
# "CORRELATION_DISTANCE" : 1e-1,
# "N_MONTE_CARLO" : 25000,
#  "dir" : "../output/test_8"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/67P_lowlowres.obj",
# "ERROR_STANDARD_DEV" : 5e-2,
# "CORRELATION_DISTANCE" : 1e-1,
# "N_MONTE_CARLO" : 25000,
#  "dir" : "../output/test_9"},
#  {"PATH_SHAPE" : "../../../resources/shape_models/67P_lowlowres.obj",
# "ERROR_STANDARD_DEV" : 2.5e-2,
# "CORRELATION_DISTANCE" : 1.5e-1,
# "N_MONTE_CARLO" : 25000,
#  "dir" : "../output/test_10"},
 {"PATH_SHAPE" : "../../../resources/shape_models/67P_lowlowres.obj",
"ERROR_STANDARD_DEV" : 7.5e-2,
"CORRELATION_DISTANCE" : 3e-1,
"N_MONTE_CARLO" : 10000,
 "dir" : "../output/test_11"},
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

	os.system("./ShapeUncertainty | tee " + data["dir"] + "/log.txt" )


