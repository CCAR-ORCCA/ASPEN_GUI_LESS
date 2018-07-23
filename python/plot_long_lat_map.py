import matplotlib.pyplot as plt
import numpy as np

def plot_maps(step):
	long_lat_map = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/maps/longitude_latitude_" + str(step)+ ".txt")
	long_lat_map = long_lat_map[0:step,:]

	n_bins_longitude = 72 
	n_bins_latitude = 36 

	d_bin_longitude = 360./ n_bins_longitude
	d_bin_latitude = 180./ n_bins_latitude

	bins = np.zeros([n_bins_latitude, n_bins_longitude])
	bins[:] = 0

	for i in range(long_lat_map.shape[0]):
		bin_longitude = int(long_lat_map[i,0] / d_bin_longitude) + n_bins_longitude/2
		bin_latitude = int(long_lat_map[i,1] / d_bin_latitude) + n_bins_latitude/2
		bins[n_bins_latitude - bin_latitude - 1, bin_longitude] += 1


	bins[bins[:] == 0.] = np.nan

	plt.imshow(bins)
	plt.colorbar()
	plt.xlabel("Latitude bin")
	plt.ylabel("Longitude bin")
	plt.title("Flyover map after BA at timestep " + str(step))

	plt.show()

	plt.scatter(long_lat_map[:,0],long_lat_map[:,1],c = range(long_lat_map.shape[0]))
	labels = [str(i) for i in range(long_lat_map.shape[0])]

	for i, label in enumerate(labels):
		plt.gca().annotate(label,(long_lat_map[i,0],long_lat_map[i,1]))

	plt.xlim([-180,180])
	plt.ylim([-90,90])
	plt.show()


def plot_maps_before(step):
	long_lat_map = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/maps/longitude_latitude_before_" + str(step)+ ".txt")
	long_lat_map = long_lat_map[0:step,:]

	n_bins_longitude = 72 
	n_bins_latitude = 36 

	d_bin_longitude = 360./ n_bins_longitude
	d_bin_latitude = 180./ n_bins_latitude

	bins = np.zeros([n_bins_latitude, n_bins_longitude])

	for i in range(long_lat_map.shape[0]):
		bin_longitude = int(long_lat_map[i,0] / d_bin_longitude) + n_bins_longitude/2
		bin_latitude = int(long_lat_map[i,1] / d_bin_latitude) + n_bins_latitude/2
		bins[n_bins_latitude - bin_latitude - 1, bin_longitude] += 1

	bins[bins[:] == 0.] = np.nan
	


	plt.imshow(bins)
	plt.colorbar()
	plt.xlabel("Latitude bin")
	plt.ylabel("Longitude bin")
	plt.title("Flyover map before BA at timestep " + str(step))

	plt.show()

	plt.scatter(long_lat_map[:,0],long_lat_map[:,1],c = range(long_lat_map.shape[0]))
	labels = [str(i) for i in range(long_lat_map.shape[0])]

	for i, label in enumerate(labels):
		plt.gca().annotate(label,(long_lat_map[i,0],long_lat_map[i,1]))

	plt.xlim([-180,180])
	plt.ylim([-90,90])
	plt.show()


def plot_iod_maps(step):
	long_lat_map = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/maps/longitude_latitude_IOD_" + str(step)+ ".txt")
	long_lat_map = long_lat_map[0:step,:]

	true_long_lat_map = np.loadtxt("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/maps/true_longitude_latitude_IOD_" + str(step)+ ".txt")
	true_long_lat_map = true_long_lat_map[0:step,:]

	n_bins_longitude = 72 
	n_bins_latitude = 36 

	d_bin_longitude = 360./ n_bins_longitude
	d_bin_latitude = 180./ n_bins_latitude

	bins = np.zeros([n_bins_latitude, n_bins_longitude])

	for i in range(long_lat_map.shape[0]):
		bin_longitude = int(long_lat_map[i,0] / d_bin_longitude) + n_bins_longitude/2
		bin_latitude = int(long_lat_map[i,1] / d_bin_latitude) + n_bins_latitude/2
		bins[n_bins_latitude - bin_latitude - 1, bin_longitude] += 1

	bins[bins[:] == 0.] = np.nan
	
	plt.imshow(bins)
	plt.colorbar()
	plt.xlabel("Latitude bin")
	plt.ylabel("Longitude bin")
	plt.title("Flyover map according to IOD at " + str(step))

	plt.show()

	plt.scatter(long_lat_map[:,0],long_lat_map[:,1],c = range(long_lat_map.shape[0]))
	plt.scatter(true_long_lat_map[:,0],true_long_lat_map[:,1],c = "green",marker  ='.')

	labels = [str(i) for i in range(long_lat_map.shape[0])]

	for i, label in enumerate(labels):
		plt.gca().annotate(label,(long_lat_map[i,0],long_lat_map[i,1]))

	plt.xlim([-180,180])
	plt.ylim([-90,90])
	plt.show()


# plot_maps_before(180)
# plot_maps(180)
plot_iod_maps(360)

