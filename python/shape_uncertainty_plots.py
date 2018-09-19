import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from np_array_to_latex import np_array_to_latex

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

def draw_slice(axis,slices,cm,cov_mc,cov_model,cov_numpy = None,save = False,zoom = False):

	cut_names = ["Y-Z","X-Z","X-Y"]
	zoom_options = " with zoom ... \n" if zoom else " without zoom ... \n"



	print "- Plotting " + cut_names[axis] + " slice "  + zoom_options



	np.random.seed(0)

	cmap = plt.cm.get_cmap(plt.cm.viridis)


	indices = range(len(slices))[::-1]

	if zoom is False:
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
	
	for s in range(int(float(cm.shape[1]) / 5)):
		if axis == 0:
			plt.scatter( cm[1,s ], cm[2,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
		elif axis == 1:
			plt.scatter( cm[0,s ],  cm[2,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
			
		elif axis == 2:
			plt.scatter( cm[0,s ], cm[1,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
		
	draw_2d_covariance(axis,cov_mc,color = "lightblue")
	draw_2d_covariance(axis,cov_model,color = "red")
	if cov_numpy is not None:
		draw_2d_covariance(axis,cov_numpy,color = "green")



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

	if (save is True):
		if (zoom is True):
			plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/slice_zoom_" + str(axis) + ".pdf", bbox_inches='tight')
		else:
			plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/slice_" + str(axis) + ".pdf", bbox_inches='tight')
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
	save = False):

	cut_names = ["Y-Z","X-Z","X-Y"]

	print "- Plotting " + name + " in " + cut_names[axis] + " slice " 


	for s in range(int(float(parameters.shape[1]) / 5)):
		if axis == 0:
			plt.scatter(parameters[1,s ],parameters[2,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
		elif axis == 1:
			plt.scatter(parameters[0,s ],parameters[2,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
			
		elif axis == 2:
			plt.scatter(parameters[0,s ],parameters[1,s ],marker = ".",color = "lightblue" ,alpha = 0.7)
		
	draw_2d_covariance(axis,cov_mc,color = "lightblue")
	draw_2d_covariance(axis,cov_model,color = "red")
	

	if axis == 0:
		plt.xlabel(labels[1])
		plt.ylabel(labels[2])
	elif axis == 1:
		plt.xlabel(labels[0])
		plt.ylabel(labels[2])
	elif axis == 2:
		plt.xlabel(labels[0])
		plt.ylabel(labels[1])

	plt.scatter(0,0,marker = ".",color = "black" )
	
	plt.axis("equal")

	if (save is True):
		plt.savefig("/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/" + name + "_" + str(axis) + ".pdf", bbox_inches='tight')
	else:
		plt.show()

	plt.cla()
	plt.clf()


slices_x = [
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_baseline.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_0.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_1.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_2.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_3.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_4.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_5.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_6.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_7.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_8.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_9.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_10.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_11.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_12.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_13.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_14.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_15.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_16.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_17.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_18.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_x_19.txt"
]




slices_y = [
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_baseline.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_0.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_1.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_2.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_3.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_4.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_5.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_6.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_7.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_8.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_9.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_10.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_11.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_12.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_13.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_14.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_15.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_16.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_17.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_18.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_y_19.txt"]





slices_z = [
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_baseline.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_0.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_1.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_2.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_3.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_4.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_5.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_6.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_7.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_8.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_9.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_10.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_11.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_12.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_13.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_14.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_15.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_16.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_17.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_18.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/slice_z_19.txt"
]

cov_cm_mc = 1e6 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/cov_cm_mc.txt")
cov_cm_model = 1e6 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/cm_cov.txt")
cm = 1e3 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/cm_spread.txt")

cov_inertia_mc = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/cov_inertia_mc.txt")
cov_inertia_model = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/inertia_cov.txt")
inertia = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/inertia_spread.txt")
inertia = (inertia.T - np.mean(inertia,axis = 1)).T

cov_moments_mc = (1e3) ** 10 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/cov_moments_mc.txt")
cov_moments_model = (1e3) ** 10 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/moments_cov.txt")
moments = (1e3) ** 5 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/moments_spread.txt")
moments = (moments.T - np.mean(moments,axis = 1)).T

cov_dims_mc = 1e6 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/cov_dims_mc.txt")
cov_dims_model = 1e6 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/dims_cov.txt")
dims = 1e3 * np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/dims_spread.txt")
dims = (dims.T - np.mean(dims,axis = 1)).T


cov_mrp_mc =  np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/cov_mrp_mc.txt")
cov_mrp_model =  np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/mrp_cov.txt")
mrp = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/output/mrp_spread.txt")
mrp = (mrp.T - np.mean(mrp,axis = 1)).T






fig = plt.figure()



np_array_to_latex(cov_mrp_model,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/mrp_cov",decimals = 3,type = "e")
np_array_to_latex(cov_mrp_mc,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_mrp_mc",decimals = 3,type = "e")
np_array_to_latex((cov_mrp_model - cov_mrp_mc) / cov_mrp_mc * 100,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_mrp_dev",decimals = 3)

np_array_to_latex(cov_cm_model,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cm_cov",decimals = 5)
np_array_to_latex(cov_cm_mc,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_cm_mc",decimals = 5)
np_array_to_latex((cov_cm_model - cov_cm_mc) / cov_cm_mc * 100,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_cm_dev",decimals = 3)

np_array_to_latex(cov_inertia_model,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/inertia_cov",decimals = 5)
np_array_to_latex(cov_inertia_mc,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_inertia_mc",decimals = 5)
np_array_to_latex((cov_inertia_model - cov_inertia_mc) / cov_inertia_mc * 100,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_inertia_dev",decimals = 3)

np_array_to_latex(cov_dims_model,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/dims_cov",decimals = 5)
np_array_to_latex(cov_dims_mc,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_dims_mc",decimals = 5)
np_array_to_latex((cov_dims_model - cov_dims_mc) / cov_dims_mc * 100,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_dims_dev",decimals = 3)


np_array_to_latex(cov_moments_model[0:3,0:3],"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/moments_cov",decimals = 3,type = "e")
np_array_to_latex(cov_moments_mc[0:3,0:3],"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_moments_mc",decimals = 3,type = "e")
np_array_to_latex((cov_moments_model[0:3,0:3] - cov_moments_mc[0:3,0:3]) / cov_moments_mc[0:3,0:3] * 100,"/Users/bbercovici/GDrive/CUBoulder/Research/papers/shape_uncertainty/R0/Figures/cov_moments_dev",decimals = 3)

w
draw_dispersions("mrp",0,mrp,cov_mrp_mc,cov_mrp_model,[r"$\delta \sigma_1$",r"$\delta \sigma_2$",r"$\delta \sigma_3$"],save = True)
draw_dispersions("mrp",1,mrp,cov_mrp_mc,cov_mrp_model,[r"$\delta \sigma_1$",r"$\delta \sigma_2$",r"$\delta \sigma_3$"],save = True)
draw_dispersions("mrp",2,mrp,cov_mrp_mc,cov_mrp_model,[r"$\delta \sigma_1$",r"$\delta \sigma_2$",r"$\delta \sigma_3$"],save = True)


draw_dispersions("dims",0,dims,cov_dims_mc,cov_dims_model,[r"$\delta a$ ($\mathrm{m}$)",r"$\delta b$ ($\mathrm{m}$)",r"$\delta c$ ($\mathrm{m}$)"],save = True)
draw_dispersions("dims",1,dims,cov_dims_mc,cov_dims_model,[r"$\delta a$ ($\mathrm{m}$)",r"$\delta b$ ($\mathrm{m}$)",r"$\delta c$ ($\mathrm{m}$)"],save = True)
draw_dispersions("dims",2,dims,cov_dims_mc,cov_dims_model,[r"$\delta a$ ($\mathrm{m}$)",r"$\delta b$ ($\mathrm{m}$)",r"$\delta c$ ($\mathrm{m}$)"],save = True)


draw_dispersions("moments",0,moments,cov_moments_mc,cov_moments_model,[r"$\delta A$ ($\mathrm{m}^5$)",r"$\delta B$ ($\mathrm{m}^5$)",r"$\delta C$ ($\mathrm{m}^5$)"],save = True)
draw_dispersions("moments",1,moments,cov_moments_mc,cov_moments_model,[r"$\delta A$ ($\mathrm{m}^5$)",r"$\delta B$ ($\mathrm{m}^5$)",r"$\delta C$ ($\mathrm{m}^5$)"],save = True)
draw_dispersions("moments",2,moments,cov_moments_mc,cov_moments_model,[r"$\delta A$ ($\mathrm{m}^5$)",r"$\delta B$ ($\mathrm{m}^5$)",r"$\delta C$ ($\mathrm{m}^5$)"],save = True)


draw_slice(0,slices_x,cm,cov_cm_mc,cov_cm_model,save = True,zoom = True)
draw_slice(1,slices_y,cm,cov_cm_mc,cov_cm_model,save = True,zoom = True)
draw_slice(2,slices_z,cm,cov_cm_mc,cov_cm_model,save = True,zoom = True)

draw_slice(0,slices_x,cm,cov_cm_mc,cov_cm_model,save = True,zoom = False)
draw_slice(1,slices_y,cm,cov_cm_mc,cov_cm_model,save = True,zoom = False)
draw_slice(2,slices_z,cm,cov_cm_mc,cov_cm_model,save = True,zoom = False)


