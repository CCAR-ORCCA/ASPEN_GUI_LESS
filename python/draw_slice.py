import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl





def draw_slice(slices,cm = []):


	cmap = plt.cm.get_cmap(plt.cm.viridis)
	fig = plt.figure()
	
	for s in range(len(slices)):

		path = slices[s]
		lines_file = np.loadtxt(path)
		
		x_max = - float("inf")
		x_min = float("inf")

		y_max = - float("inf")
		y_min = float("inf")

		for i in range(lines_file.shape[0]):
			x = [lines_file[i][0],lines_file[i][2]]
			y = [lines_file[i][1],lines_file[i][3]]
			x_max = max(max(x),x_max)
			y_max = max(max(y),y_max)
			x_min = min(min(x),x_min)
			y_min = min(min(y),y_min)

			plt.gca().add_line(mpl.lines.Line2D(x, y,color = cmap(200 *s )))

		if len(cm) == len(slices) : 
			plt.scatter(cm[s][0],cm[s][1],color = cmap(200 *s ))


	plt.gca().set_xlim(1.5 * x_min, 1.5 * x_max)
	plt.gca().set_ylim(1.5 * y_min, 1.5 * y_max)
	plt.axis("equal")
	plt.show()


slices = [
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/build/slice_0.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/build/slice_1.txt",
"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/build/slice_2.txt",
# "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/build/slice_3.txt",
# "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/build/slice_4.txt",
# "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/build/slice_5.txt",
# "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/TestBezier/build/slice_6.txt"
]

draw_slice(slices)





