#ifndef HEADER_FILTER
#define HEADER_FILTER

#include "ShapeModel.hpp"
#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "RigidBodyKinematics.hpp"
#include <boost/progress.hpp>


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
	@param t0 Initial time (s)
	@param t1 Final time (s)
	@param omega Angular rate of the instrument about the target (rad/s)
	@param min_normal_observation_angle Minimum angle for a ray to be used (rad)
	@param min_facet_normal_angle_difference Minimum angle separating to normals associated with the same vertex
	@param minimum_ray_per_facet Minimum number of rays per facet to include the facet in the
	@param ridge_coef Non-zero value regularizes the information matrix by introducing a bias
	estimation process
	*/
	Filter(FrameGraph * frame_graph,
	       Lidar * lidar,
	       ShapeModel * true_shape_model,
	       ShapeModel * estimated_shape_model,
	       double t0,
	       double tf,
	       double omega,
	       double min_normal_observation_angle,
	       double min_facet_normal_angle_difference,
	       unsigned int minimum_ray_per_facet,
	       double ridge_coef);

	/**
	@param N_iteration number of iteration of the filter with each batch of information
	@param plot_measurement true if the measurements should be saved
	@param save_shape_model true if the shape model should be saved after each measurement
	*/
	void run(unsigned int N_iteration, bool plot_measurements, bool save_shape_model);

protected:

	void correct_shape();

	void correct_observed_features(std::vector<Ray * > & good_rays,
	                               std::set<Vertex *> & seen_vertices,
	                               std::set<Facet *> & seen_facets,
	                               arma::mat & N_mat,
	                               std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices) ;
	void get_observed_features(std::vector<Ray * > & good_rays,
	                           std::set<Vertex *> & seen_vertices,
	                           std::set<Facet *> & seen_facets,
	                           arma::mat & N_mat,
	                           std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices) ;





	std::vector<arma::rowvec> partial_range_partial_coordinates(const arma::vec & P,
	        const arma::vec & u, Facet * facet) ;




	double t0;
	double tf;
	double min_normal_observation_angle;
	double omega;
	double min_facet_normal_angle_difference;
	unsigned int minimum_ray_per_facet;
	double ridge_coef;

	FrameGraph * frame_graph;
	Lidar * lidar;
	ShapeModel * true_shape_model;
	ShapeModel * estimated_shape_model;



};


#endif