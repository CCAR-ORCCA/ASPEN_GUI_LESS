import os
import json
import numpy as np

# all_data = [{"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_0"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_1"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_2"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_3"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_4"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_5"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.75,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_6"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.75,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_7"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.75,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_8"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.75,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_9"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.75,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_10"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.75,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_11"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_12"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_13"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_14"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_15"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_16"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_17"}]


# all_data = [{"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_12"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_13"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_14"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_15"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_16"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.695,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : 0., "dir" : "../output/data_17"}]


# all_data = [{"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 1e12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : np.pi/2, "dir" : "../output/data_18"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 1e12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : np.pi/2, "dir" : "../output/data_19"},
# {"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 1e12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : np.pi/2, "dir" : "../output/data_20"}]



# all_data = [{"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 5,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 1e12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : np.pi/2, "dir" : "../output/data_21"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 7,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 1e12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : np.pi/2, "dir" : "../output/data_22"},
# {"ORBIT_FRACTION" : 0.5,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.25,
# "I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 3., "SPIN_PERIOD" : 1e12, "LONGITUDE_SPIN" : 0., "LATITUDE_SPIN" : np.pi/2, "dir" : "../output/data_23"}]

all_data = [{"ORBIT_FRACTION" : 0.25,"OBSERVATION_TIMES" : 10,"SMA" : 1000,"E" : 0.25,
"I" : 1.4,"RAAN" :0.2,"PERI_OMEGA" : 0.3,"M0" : 1.57, "SPIN_PERIOD" : 12, "LONGITUDE_SPIN" : 0.5, "LATITUDE_SPIN" : 0.5, "dir" : "../output/test_0"}]

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

	os.system("./IOD")


