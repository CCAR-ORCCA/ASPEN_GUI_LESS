import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import np_array_to_latex as nptlatex
from np_array_to_latex import np_array_to_latex
from pprint import pprint
import json
import os

def create_output_table(cov_mrp_model,cov_mrp_mc,cov_cm_model,cov_cm_mc,cov_inertia_model,
	cov_inertia_mc,cov_dims_model,cov_dims_mc,cov_moments_model,cov_moments_mc,prefix,output_dir):

	np_array_to_latex(cov_mrp_model,output_dir + "/" + prefix +"mrp_cov",decimals = 3,type = "e",
		is_symmetric = "upper")
	np_array_to_latex(cov_mrp_mc,output_dir + "/" + prefix +"cov_mrp_mc",decimals = 3,type = "e",
		is_symmetric = "upper")
	np_array_to_latex((cov_mrp_model - cov_mrp_mc) / cov_mrp_mc * 100,output_dir + "/" + prefix +"cov_mrp_dev",decimals = 3,
		is_symmetric = "upper")

	np_array_to_latex(cov_cm_model,output_dir + "/" + prefix +"cm_cov",decimals = 3,
		is_symmetric = "upper",type = "e")
	np_array_to_latex(cov_cm_mc,output_dir + "/" + prefix +"cov_cm_mc",decimals = 3,
		is_symmetric = "upper",type = "e")
	np_array_to_latex((cov_cm_model - cov_cm_mc) / cov_cm_mc * 100,output_dir + "/" + prefix +"cov_cm_dev",decimals = 3,
		is_symmetric = "upper")

	np_array_to_latex(cov_inertia_model,output_dir + "/" + prefix +"inertia_cov",decimals = 3,
		is_symmetric = "upper",type = "e")
	np_array_to_latex(cov_inertia_mc,output_dir + "/" + prefix +"cov_inertia_mc",decimals = 3,
		is_symmetric = "upper",type = "e")
	np_array_to_latex((cov_inertia_model - cov_inertia_mc) / cov_inertia_mc * 100,output_dir + "/" + prefix +"cov_inertia_dev",decimals = 3,
		is_symmetric = "upper")

	np_array_to_latex(cov_dims_model,output_dir + "/" + prefix +"dims_cov",decimals = 3,
		is_symmetric = "upper",type = "e")
	np_array_to_latex(cov_dims_mc,output_dir + "/" + prefix +"cov_dims_mc",decimals = 3,
		is_symmetric = "upper",type = "e")
	np_array_to_latex((cov_dims_model - cov_dims_mc) / cov_dims_mc * 100,output_dir + "/" + prefix +"cov_dims_dev",decimals = 3,
		is_symmetric = "upper")


	np_array_to_latex(cov_moments_model[0:3,0:3],output_dir + "/" + prefix +"moments_cov",decimals = 3,type = "e",
		is_symmetric = "upper")
	np_array_to_latex(cov_moments_mc[0:3,0:3],output_dir + "/" + prefix +"cov_moments_mc",decimals = 3,type = "e",
		is_symmetric = "upper")
	np_array_to_latex((cov_moments_model[0:3,0:3] - cov_moments_mc[0:3,0:3]) / cov_moments_mc[0:3,0:3] * 100,output_dir + "/" + prefix +"cov_moments_dev",decimals = 3,
		is_symmetric = "upper")



def create_input_table(all_results_dirs,savepath):

    column_labels = [ "Case " + str(int(all_results_dirs[i].split("_")[-1]) + 1) for i in range(len(all_results_dirs)) ] 
    row_labels = ["$Noise standard deviation (m) $","Correlation distance (m)","Monte-Carlo samples (-)"]

    res = np.zeros([3,len(all_results_dirs)])

    for i in range(len(all_results_dirs)):
        with open(all_results_dirs[i] + "/input_file.json") as f:
            data = json.load(f)
        res[0,i] = data["ERROR_STANDARD_DEV"]
        res[1,i] = data["CORRELATION_DISTANCE"]
        res[2,i] = int(data["N_MONTE_CARLO"])

    np_array_to_latex(res,savepath,row_headers = row_labels,column_headers = column_labels,
  column = True,type = 'f', decimals = 2, ante_decimals = 6)




def draw_2d_covariance(axis,cov,color):

	cov_2d = np.zeros([2,2])

	if (axis == 0):

		cov_2d[0,0] = cov[1,1]
		cov_2d[1,1] = cov[2,2]
		cov_2d[1,0] = cov[1,2]
		cov_2d[0,1] = cov[2,1]

	elif (axis == 1):

		cov_2d[0,0] = cov[0,0]
		cov_2d[1,1] = cov[2,2]
		cov_2d[1,0] = cov[0,2]
		cov_2d[0,1] = cov[2,0]

	elif (axis == 2):

		cov_2d[0,0] = cov[0,0]
		cov_2d[1,1] = cov[1,1]
		cov_2d[1,0] = cov[0,1]
		cov_2d[0,1] = cov[1,0]

	dims,M = np.linalg.eigh(cov_2d)

	if (np.linalg.det(M)< 0):
		M[:,0] = - M[:,0]
	
	# the axes matrix is of the form 

	# M = [cos t - sin t ]
	#     [ sin t cos t]

	angle = np.arctan2(M[1,0],M[0,0]) * 180./np.pi + 90


	# the horizontal axis is along the largest uncertainty

	width = 2 * 3 * np.sqrt(dims)[1]
	height = 2 * 3 * np.sqrt(dims)[0]

	e = Ellipse((0, 0), width, height, angle,
		facecolor = None,
		edgecolor = color,
		fill = False)
	

	plt.gca().add_artist(e)

def draw_slice(axis,slices,output_dir = None,zoom = False,prefix = ""):

	cut_names = ["Y-Z","X-Z","X-Y"]
	zoom_options = " with zoom ... \n" if zoom else " without zoom ... \n"



	print "- Plotting " + cut_names[axis] + " slice "  + zoom_options


	np.random.seed(0)

	cmap = plt.cm.get_cmap(plt.cm.viridis)

	indices = range(len(slices))[::-1]

	for s in indices:

		path = slices[s]
		lines_file = np.loadtxt(path)
		
		x_max = - float("inf")
		x_min = float("inf")

		y_max = - float("inf")
		y_min = float("inf")

		if s == 0:
			c = "black"
			alpha = 1
		else:
			c = "lightblue"
			alpha = 0.7

		for i in range(lines_file.shape[0]):
			x = [1e3 * lines_file[i][0],1e3 * lines_file[i][2]]
			y = [1e3 * lines_file[i][1],1e3 * lines_file[i][3]]
			x_max = max(max(x),x_max)
			y_max = max(max(y),y_max)
			x_min = min(min(x),x_min)
			y_min = min(min(y),y_min)

			plt.gca().add_line(mpl.lines.Line2D(x, y,color = c,alpha = alpha))


	if axis == 0:
		plt.xlabel("Y (m)")
		plt.ylabel("Z (m)")
	elif axis == 1:
		plt.xlabel("X (m)")
		plt.ylabel("Z (m)")
	elif axis == 2:
		plt.xlabel("X (m)")
		plt.ylabel("Y (m)")

	plt.scatter(0,0,marker = ".",color = "black" )
	
	if zoom is False:
		plt.xlim(1.5 * x_min, 1.5 * x_max)
		plt.ylim(1.5 * y_min, 1.5 * y_max)

	plt.axis("equal")

	if (len(output_dir) > 0):
		if (zoom is True):
			plt.savefig(output_dir + "/" + prefix +"slice_zoom_" + str(axis) + ".pdf", bbox_inches='tight')
		else:
			plt.savefig(output_dir + "/" + prefix +"slice_" + str(axis) + ".pdf", bbox_inches='tight')
	else:
		plt.show()

	plt.cla()
	plt.clf()

def draw_dispersions(name,
	axis,
	parameters,
	cov_mc,
	cov_model,
	labels,
	output_dir = None,
	prefix = ""):

	cut_names = ["Y-Z","X-Z","X-Y"]

	print "- Plotting " + name + " in " + cut_names[axis] + " slice " 

	if axis == 0:
		plt.scatter(parameters[1,: ],parameters[2,: ],marker = ",",color = "lightblue" ,alpha = 0.7)
	elif axis == 1:
		plt.scatter(parameters[0,: ],parameters[2,: ],marker = ",",color = "lightblue" ,alpha = 0.7)
	elif axis == 2:
		plt.scatter(parameters[0,: ],parameters[1,: ],marker = ",",color = "lightblue" ,alpha = 0.7)
		
	draw_2d_covariance(axis,cov_mc,color = "darkblue")
	draw_2d_covariance(axis,cov_model,color = "red")

	if axis == 0:
		plt.xlabel(labels[1])
		plt.ylabel(labels[2])
	elif axis == 1:
		plt.xlabel(labels[0])
		plt.ylabel(labels[2])
	else:
		plt.xlabel(labels[0])
		plt.ylabel(labels[1])

	plt.scatter(0,0,marker = ".",color = "black" )
	
	plt.axis("equal")
	

	if (len(output_dir) > 0):
		plt.savefig(output_dir + "/" + prefix +"" + name + "_" + str(axis) + ".pdf", bbox_inches='tight')
	else:
		plt.show()

	plt.cla()
	plt.clf()



def list_results(mainpath,output_dir):

    all_results_dirs  = [x[0] for x in os.walk(mainpath)][1:]

    all_results_dirs_tag = np.array([int(all_results_dirs[i].split("_")[-1]) for i in range(len(all_results_dirs))])

    sorted_order = np.argsort(all_results_dirs_tag)

    all_results_dirs = [all_results_dirs[sorted_order[i]] for i in range(len(sorted_order))]

    print "\t Found " + str(len(all_results_dirs)) + " result directories\n\n"

    for i in range(len(all_results_dirs)):
        print str(i) + " : " + all_results_dirs[i] + "\n"

        with open(all_results_dirs[i] + "/input_file.json") as f:
            data = json.load(f)

        pprint(data)
        print "\n"

    index_str = raw_input(" Which one should be processed ? Pick a number or enter 'all'\n")
    save_str = raw_input(" Should results be saved? (y/n) ?\n")

    if index_str != "all":
        if save_str is "y":
            plot_all_results(all_results_dirs[int(index_str)],output_dir)
        elif save_str is "n":
            plt.switch_backend('Qt5Agg')
            plot_all_results(all_results_dirs[int(index_str)])
        else:
            raise(TypeError("Unrecognized input: " + str(save_str)))
    else:

        if save_str is "y":
            for i in range(len(all_results_dirs)):
                plot_all_results(all_results_dirs[i],output_dir)

        elif save_str is "n":
            plt.switch_backend('Qt5Agg')
            for i in range(len(all_results_dirs)):
                plot_all_results(all_results_dirs[i])

        else:
            raise(TypeError("Unrecognized input: " + str(save_str)))


def plot_all_results(input_dir,output_dir = ""):

	prefix = input_dir.split("/")[-1] + "_"

	print("Plotting case " + prefix + "\n")

	slices_x = [input_dir + "/slice_x_"+str(i)+ ".txt" for i in range(20)]
	slices_y = [input_dir + "/slice_y_"+str(i)+ ".txt" for i in range(20)]
	slices_z = [input_dir + "/slice_z_"+str(i)+ ".txt" for i in range(20)]

	slices_x = [input_dir + "/slice_x_baseline.txt"] + slices_x
	slices_y = [input_dir + "/slice_y_baseline.txt"] + slices_y
	slices_z = [input_dir + "/slice_z_baseline.txt"] + slices_z

	cov_cm_mc = 1e6 * np.loadtxt(input_dir + "/cov_cm_mc.txt")
	cov_cm_model = 1e6 * np.loadtxt(input_dir + "/cm_cov.txt")
	cm = 1e3 * np.loadtxt(input_dir + "/cm_spread.txt")

	cov_inertia_mc = np.loadtxt(input_dir + "/cov_inertia_mc.txt")
	cov_inertia_model = np.loadtxt(input_dir + "/inertia_cov.txt")
	inertia = np.loadtxt(input_dir + "/inertia_spread.txt")
	inertia = (inertia.T - np.mean(inertia,axis = 1)).T

	cov_moments_mc =  np.loadtxt(input_dir + "/cov_moments_mc.txt")
	cov_moments_model =  np.loadtxt(input_dir + "/moments_cov.txt")
	moments =  np.loadtxt(input_dir + "/moments_spread.txt")
	moments = (moments.T - np.mean(moments,axis = 1)).T

	cov_dims_mc = np.loadtxt(input_dir + "/cov_dims_mc.txt")
	cov_dims_model = np.loadtxt(input_dir + "/dims_cov.txt")
	dims = np.loadtxt(input_dir + "/dims_spread.txt")
	dims = (dims.T - np.mean(dims,axis = 1)).T

	cov_mrp_mc =  np.loadtxt(input_dir + "/cov_mrp_mc.txt")
	cov_mrp_model =  np.loadtxt(input_dir + "/mrp_cov.txt")
	mrp = np.loadtxt(input_dir + "/mrp_spread.txt")
	mrp = (mrp.T - np.mean(mrp,axis = 1)).T

	fig = plt.figure()

	if (len(output_dir)>0):
		create_output_table(cov_mrp_model,cov_mrp_mc,cov_cm_model,cov_cm_mc,cov_inertia_model,
		cov_inertia_mc,cov_dims_model,cov_dims_mc,(1e3) ** 4 *cov_moments_model,(1e3) ** 4 *cov_moments_mc,prefix,output_dir)

		inertia_tensor = np.zeros([3,3])
		inertia_vector = np.loadtxt(input_dir + "/I.txt")
		inertia_tensor[0,0] = inertia_vector[0]
		inertia_tensor[1,1] = inertia_vector[1]
		inertia_tensor[2,2] = inertia_vector[2]
		inertia_tensor[0,1] = inertia_vector[3]
		inertia_tensor[0,2] = inertia_vector[4]
		inertia_tensor[1,2] = inertia_vector[5]


		moments_mean = np.loadtxt(input_dir + "/moments.txt")[0:3]
		np_array_to_latex((1e3) ** 2 * moments_mean,output_dir + "/" + prefix +"moments",decimals = 3,type = "e")

		dims_mean = np.loadtxt(input_dir + "/dims.txt")
		np_array_to_latex(dims_mean,output_dir + "/" + prefix +"dims",decimals = 3,type = "e")




	draw_slice(0,slices_x,output_dir = output_dir,prefix = prefix)
	draw_slice(1,slices_y,output_dir = output_dir,prefix = prefix)
	draw_slice(2,slices_z,output_dir = output_dir,prefix = prefix)


	draw_dispersions("com",0,cm,cov_cm_mc,cov_cm_model,[r"$\delta \mathbf{c}_{m,x}$ ($\mathrm{m}$)",r"$\delta \mathbf{c}_{m,y}$ ($\mathrm{m}$)",r"$\delta \mathbf{c}_{m,z}$ ($\mathrm{m}$)"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("com",1,cm,cov_cm_mc,cov_cm_model,[r"$\delta \mathbf{c}_{m,x}$ ($\mathrm{m}$)",r"$\delta \mathbf{c}_{m,y}$ ($\mathrm{m}$)",r"$\delta \mathbf{c}_{m,z}$ ($\mathrm{m}$)"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("com",2,cm,cov_cm_mc,cov_cm_model,[r"$\delta \mathbf{c}_{m,x}$ ($\mathrm{m}$)",r"$\delta \mathbf{c}_{m,y}$ ($\mathrm{m}$)",r"$\delta \mathbf{c}_{m,z}$ ($\mathrm{m}$)"],output_dir = output_dir,
		prefix = prefix)

	draw_dispersions("dims",0,dims,cov_dims_mc,cov_dims_model,[r"$\delta a$ ($\mathrm{km}$)",r"$\delta b$ ($\mathrm{km}$)",r"$\delta c$ ($\mathrm{km}$)"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("dims",1,dims,cov_dims_mc,cov_dims_model,[r"$\delta a$ ($\mathrm{km}$)",r"$\delta b$ ($\mathrm{km}$)",r"$\delta c$ ($\mathrm{km}$)"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("dims",2,dims,cov_dims_mc,cov_dims_model,[r"$\delta a$ ($\mathrm{km}$)",r"$\delta b$ ($\mathrm{km}$)",r"$\delta c$ ($\mathrm{km}$)"],output_dir = output_dir,
		prefix = prefix)

	draw_dispersions("moments",0,(1e3) ** 2 * moments,(1e3) ** 4 *cov_moments_mc,(1e3) ** 4 *cov_moments_model,[r"$\delta A$ ($\mathrm{m}^2$)",r"$\delta B$ ($\mathrm{m}^2$)",r"$\delta C$ ($\mathrm{m}^2$)"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("moments",1,(1e3) ** 2 * moments,(1e3) ** 4 *cov_moments_mc,(1e3) ** 4 *cov_moments_model,[r"$\delta A$ ($\mathrm{m}^2$)",r"$\delta B$ ($\mathrm{m}^2$)",r"$\delta C$ ($\mathrm{m}^2$)"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("moments",2,(1e3) ** 2 * moments,(1e3) ** 4 *cov_moments_mc,(1e3) ** 4 *cov_moments_model,[r"$\delta A$ ($\mathrm{m}^2$)",r"$\delta B$ ($\mathrm{m}^2$)",r"$\delta C$ ($\mathrm{m}^2$)"],output_dir = output_dir,
		prefix = prefix)
	
	draw_dispersions("mrp",0,mrp,cov_mrp_mc,cov_mrp_model,[r"$\delta \sigma_1$",r"$\delta \sigma_2$",r"$\delta \sigma_3$"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("mrp",1,mrp,cov_mrp_mc,cov_mrp_model,[r"$\delta \sigma_1$",r"$\delta \sigma_2$",r"$\delta \sigma_3$"],output_dir = output_dir,
		prefix = prefix)
	draw_dispersions("mrp",2,mrp,cov_mrp_mc,cov_mrp_model,[r"$\delta \sigma_1$",r"$\delta \sigma_2$",r"$\delta \sigma_3$"],output_dir = output_dir,
		prefix = prefix)

mainpath = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeUncertainty/output"
output_dir = "/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures"

list_results(mainpath = mainpath,output_dir = output_dir)

