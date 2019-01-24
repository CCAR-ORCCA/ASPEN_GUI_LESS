import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import json
import np_array_to_latex
plt.switch_backend('Qt5Agg')


base_loc = '/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output'
dir_list = next(os.walk(base_loc))[1]

case_number_list = [int(dir_list[e].split("_")[1]) for e in range(len(dir_list))]
case_number_list.sort()
dir_list = ["case_" + str(case_number_list[e]) for e in range(len(dir_list))]

all_times = []
all_iod_res_icp = []
all_converged_res_icp = []
runtimes = []
cases = []
parameter_matrices_arrays = []
row_headers = ["IOD particles","Number of edges", "Bezier shape","Instrument frequency (mHz)"]
col_headers_arrays = []
col_headers_array = []
array_length = 6
parameter_matrix = np.zeros([4,array_length])

for directory in dir_list:
    filepath = base_loc + "/" + directory + '/log.txt'  
    
    iod_res_icp = []
    converged_res_icp = []
    time = []
    store_converged_res_icp_counter = 0




    print("Reading " + directory + " ...")

    with open(base_loc + "/" + directory + "/input_file.json") as f:
    	data = json.load(f)
    	print("\tImaging frequency: " + str(data["INSTRUMENT_FREQUENCY_SHAPE"]) + " Hz")
    	print("\tNumber of IOD particles: " + str(data["IOD_PARTICLES"]))
    	print("\tBezier shape?: " + str(data["USE_BEZIER_SHAPE"]))
    	print("\tNumber of edges: " + str(data["NUMBER_OF_EDGES"]))


    with open(filepath) as fp:  

        line = fp.readline()
        while line:
            split_line = line.split(" ")

            store_converged_res_icp_counter = store_converged_res_icp_counter - 1

            if (len(split_line)) > 5:
                if split_line[1] == "Residuals" and split_line[2] == "from" and split_line[3] == "iod":
                    iod_res_icp += [float(split_line[5])]


            if (len(split_line)) > 1:
            	if split_line[0] == "Leaving" and split_line[1] == "ICPBase.":
            		store_converged_res_icp_counter = 3

            if (store_converged_res_icp_counter == 0):            	converged_res_icp += [float(split_line[1])]


            if split_line[0] == "###################" and split_line[1] == "Index":
                time += [split_line[-4]]

            if (len(split_line)) > 4:

	            if split_line[0] == "Time" and split_line[4] == "simulation":
	            	runtime = float(split_line[-1])

            line = fp.readline()



    all_times += [np.array(time,dtype = float)]
    all_iod_res_icp += [np.array(iod_res_icp,dtype = float)]
    all_converged_res_icp += [np.array(converged_res_icp,dtype = float)]
    runtimes += [runtime]
    cases += [directory]

    if len(col_headers_array) < array_length:
    	parameter_matrix[0,len(col_headers_array)] =  int(data["IOD_PARTICLES"])
    	parameter_matrix[1,len(col_headers_array)] =  int(data["NUMBER_OF_EDGES"])
    	parameter_matrix[2,len(col_headers_array)] =  int(data["USE_BEZIER_SHAPE"])
    	parameter_matrix[3,len(col_headers_array)] =  1000 * float(data["INSTRUMENT_FREQUENCY_SHAPE"])
    	col_headers_array += [directory.replace("_"," ")]

    if (len(col_headers_array) == array_length):

    	col_headers_arrays += [col_headers_array]
    	parameter_matrices_arrays += [parameter_matrix]

    	np_array_to_latex.np_array_to_latex(parameter_matrix,
    		"/Users/bbercovici/GDrive/CUBoulder/Research/conferences/GNSKi_2019/paper/Figures/inputs_" 
    		+ str(col_headers_array[0]).split(" ")[1] + "_" + str(col_headers_array[-1]).split(" ")[1],
    		row_headers = row_headers,
    		column_headers = col_headers_array)

    	parameter_matrix = np.zeros([4,min(array_length,len(dir_list) - int(str(col_headers_array[-1]).split(" ")[1]) - 1)])
    	col_headers_array = []


linestyles = ['-.','--','-d','-v','-D','+','*']
w
for case in range(len(all_times)):
	plt.scatter(case * np.ones(1),runtimes[case])

plt.title("Total runtime")
plt.xlabel("Case")
plt.ylabel("Runtime (s)")
plt.xticks(range(len(cases)), cases,rotation = 90)
plt.show()

plt.clf()

for case in range(len(all_times)):
	plt.semilogy(all_times[case][1:]/3600,all_iod_res_icp[case],linestyles[case % len(linestyles)],label = cases[case])

plt.title("A-priori registration residuals from IOD solution")
plt.xlabel("Time (hours)")
plt.ylabel("Registration residuals (m)")
plt.legend(loc = "best",ncol = 10)

plt.show()

plt.clf()

for case in range(len(all_times)):
	plt.scatter(case * np.ones(len(all_iod_res_icp[case])),all_iod_res_icp[case])

plt.title("A-priori registration residuals from IOD solution")
plt.xlabel("Case")
plt.ylabel("Registration residuals (m)")
plt.legend(loc = "best")
plt.xticks(range(len(cases)), cases,rotation = 90)
plt.tight_layout()
plt.show()

plt.clf()

for case in range(len(all_times)):
	plt.semilogy(all_times[case][0:len(all_converged_res_icp[case])]/3600,all_converged_res_icp[case],linestyles[case % len(linestyles)], label = cases[case])

plt.legend(loc = "best",ncol = 10)

plt.title("ICP registration residuals")
plt.xlabel("Time (hours)")
plt.ylabel("Registration residuals (m)")
plt.tight_layout()
plt.show()
plt.clf()


for case in range(len(all_times)):
	plt.scatter(case * np.ones(len(all_converged_res_icp[case])),all_converged_res_icp[case])

plt.title("ICP registration residuals")
plt.xlabel("Case")
plt.ylabel("Registration residuals (m)")
plt.xticks(range(len(cases)), cases,rotation = 90)
plt.tight_layout()
plt.show()

plt.clf()





