import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


# def show_graph_with_labels(adjacency_matrix, mylabels):
#     plt.clf()

#     # removing nans
#     adjacency_matrix = np.nan_to_num(adjacency_matrix)
#     rows, cols = np.where(adjacency_matrix != 0)
#     edges = zip(rows.tolist(), cols.tolist())
#     gr = nx.Graph()
#     gr.add_edges_from(edges)
#     nx.draw(gr, node_size=20, labels=mylabels, with_labels=True)

#     plt.show()


# def plot_adjacency_graph(path):
   
#     mydata = np.genfromtxt(path, delimiter=',')
#     plot_res_adjacency_mat(mydata)
  
    # show_graph_with_labels(mydata,{i: str(i) for i in range(mydata.shape[0])})

def plot_res_adjacency_mat(data,factor,interpolation):

    max_error = np.amax(np.diagonal(data,1))

    # For all the Dk point clouds
    for i in tqdm(range(data.shape[0])): 
        # For all the Sk point clouds, necessarily different from the Dk
        for j in range(i + 1):

            if data[i,j] > factor * max_error or data[i,j] < 0:
                data[i,j] = np.nan

            
            data[j,i] = np.nan
    plt.subplot(133)

    plt.imshow(data,cmap='jet_r',interpolation = interpolation)

    plt.colorbar()
    plt.title("PC matching residuals, " + str((factor - 1) * 100) + " % tolerance")
    plt.xlabel("Point cloud label")
    plt.ylabel("Point cloud label")


    # Adding a flag on the smallest and largest residuals
    coords_max = np.unravel_index(np.nanargmax(data), data.shape)
    plt.scatter(coords_max[1],coords_max[0], color='red', s=40)

    coords_min = np.unravel_index(np.nanargmin(data), data.shape)
    plt.scatter(coords_min[1],coords_min[0], color='green', s=40)

    plt.show()
    return data


def plot_overlap_adjacency_mat(data,interpolation):
    
    for i in range(data.shape[0]):
        data[i,i] = np.nan
        for j in range(i + 1,data.shape[0]):
            print data[i,j],data[j,i]
            
            if data[i,j] < 0 or data[j,i] < 0:
                data[j,i] = np.nan
                data[i,j] = np.nan

            data[i,j] = np.nan


    plt.subplot(131)
    plt.tight_layout()

    plt.imshow(data * 100, cmap = 'jet',interpolation = interpolation)
    plt.colorbar()
    plt.title("Accepted point-pairs (%)")
    plt.xlabel("Point cloud label")
    plt.ylabel("Point cloud label")
    data_t = np.copy(data)
    
    
    return data_t



    
def plot_N_pairs_mat(data,interpolation):
    
    for i in range(data.shape[0]):
        for j in range(i + 1,data.shape[0]):
            data[i,j] = np.nan
    
    plt.subplot(132) 

    plt.imshow(data , cmap = 'jet',interpolation = interpolation)
    plt.colorbar()
    plt.title("Max number of point-pairs")
    plt.xlabel("Point cloud label")
    plt.ylabel("Point cloud label")



def plot_point_cloud_quality(data_overlap,data_res):


    # minimum residuals, maximum overlap
    # data_quality = data_overlap / data_res * data_res
    data_quality = np.copy(data_overlap)

    N_pairs_max = float(data_quality.shape[0]) *( data_quality.shape[0] - 1)/ 2

    print "Density: " + str(float((~np.isnan(data_quality)).sum()) /N_pairs_max * 100) + " %" 


    plt.clf()
    plt.imshow(data_quality)
    plt.colorbar()
    plt.title("Point-cloud pairs quality ")
    plt.xlabel("Point cloud label")
    plt.ylabel("Point cloud label")
    plt.show()
    return data_quality


def plot_results(path,factor,interpolation):

    data_res = np.loadtxt(path + "/connectivity_res.txt")
    data_overlap = np.loadtxt(path + "/connectivity_overlap.txt")
    data_N_pairs = np.loadtxt(path + "/connectivity_N_pairs.txt")

    data_overlap = plot_overlap_adjacency_mat(data_overlap,interpolation)
    plot_N_pairs_mat(data_N_pairs,interpolation)
    data_res = plot_res_adjacency_mat(data_res,factor,interpolation)
    data_quality = plot_point_cloud_quality(data_overlap,data_res)
    return data_quality

    

data_quality = plot_results("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/",float('Inf'),interpolation = "none")
# plot_results("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/ShapeReconstruction/output/",10,interpolation = "quadric")




