#ifndef HEADER_FILTER
#define HEADER_FILTER

#include "ShapeModel.hpp"
#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "Interpolator.hpp"
#include "FilterArguments.hpp"

#include <RigidBodyKinematics.hpp>

#include <boost/progress.hpp>
#include <numeric>
#include <sstream>
#include <iomanip>


/**
Filter class hosting:
- the instrument
- the true shape model
- the estimated shape model
- Filtering tools:
# the partial derivatives evaluation
# shape refinement
*/
class Filter {

public:



	/**
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	@param estimated_shape_model Pointer to the estimated shape model
	@param arguments filter parameters
	*/
	Filter(FrameGraph * frame_graph,
	       Lidar * lidar,
	       ShapeModel * true_shape_model,
	       ShapeModel * estimated_shape_model,
	       FilterArguments * arguments);


	/**
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	*/
	Filter(FrameGraph * frame_graph,
	       Lidar * lidar,
	       ShapeModel * true_shape_model) ;

	/**
	@param N_iteration number of iteration of the filter with each batch of information
	@param plot_measurement true if the measurements should be saved
	@param save_shape_model true if the shape model should be saved after each measurement
	*/
	void run(unsigned int N_iteration, bool plot_measurements, bool save_shape_model);


	/**
	Collects 3D point cloud measurements and stores them to an OBJ file
	@param path Path to obj file of the form XXX.obj (ex: test.obj)
	*/
	void get_surface_point_cloud(std::string path);


	/**
	Collects 3D point cloud measurements and stores them to an OBJ file
	using a precomputed orbit and attracting body attitude
	@param orbit_path Path to the orbit file
	@param orbit_time_path Path to the orbit time file
	@param attitude_path Path to the attitude file
	@param attitude_time_path Path to the attitude time file
	@param savepath Path to obj file of the form XXX.obj (ex: test.obj)
	*/
	void get_surface_point_cloud_from_trajectory(
	    std::string orbit_path,
	    std::string orbit_time_path,
	    std::string attitude_path,
	    std::string attitude_time_path,
	    std::string savepath);


	/**
	Collects 3D point cloud measurements and stores them to an OBJ file
	using a precomputed orbit and attracting body attitude
	@param orbit_states Orbit states
	@param orbit_time Orbit time
	@param attitude_states Attitude states
	@param attitude_time Attitude time
	@param savepath Path to obj file of the form XXX.obj (ex: test.obj)
	*/
	void get_surface_point_cloud_from_trajectory(
	    arma::mat * orbit_states,
	    arma::vec * orbit_time,
	    arma::mat * attitude_states,
	    arma::vec * attitude_time,
	    std::string savepath) ;



	/**
	Solves the square linear system info_mat * x = normal_mat
	by means of a cholesky decomposition
	@param info_mat Symetrical information matrix
	@param normal_mat Normal matrix
	@return x Solution
	*/
	arma::vec cholesky(arma::mat & info_mat, arma::mat & normal_mat) const;

protected:

	void correct_shape(unsigned int time_index, bool first_iter, bool last_iter);

	void correct_observed_features(std::vector<Ray * > & good_rays,
	                               std::set<Vertex *> & seen_vertices,
	                               std::set<Facet *> & seen_facets,
	                               arma::mat & N_mat,
	                               std::map<Facet *,
	                               std::vector<unsigned int> > & facet_to_index_of_vertices) ;

	void get_observed_features(std::vector<Ray * > & good_rays,
	                           std::set<Vertex *> & seen_vertices,
	                           std::set<Facet *> & seen_facets,
	                           arma::mat & N_mat,
	                           std::map<Facet *,
	                           std::vector<unsigned int> > & facet_to_index_of_vertices) ;




	std::vector<arma::rowvec> partial_range_partial_coordinates(const arma::vec & P,
	        const arma::vec & u, Facet * facet) ;


	FilterArguments * arguments;
	FrameGraph * frame_graph;
	Lidar * lidar;
	ShapeModel * true_shape_model;
	ShapeModel * estimated_shape_model;




};


#endif