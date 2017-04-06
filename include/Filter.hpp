#ifndef HEADER_FILTER
#define HEADER_FILTER

#include "ShapeModel.hpp"
#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "RigidBodyKinematics.hpp"
#include <boost/progress.hpp>
#include <numeric>
#include <sstream>
#include <iomanip>

/**
Class storing the filter parameters
	*/
class Arguments {
public:

	/**
	Default constructor
	*/
	Arguments() {

	}

	/**
	Constructor
	@param t0 Initial time (s)
	@param t1 Final time (s)
	@param orbit_rate Angular rate of the instrument about the target (rad/s), 313
	@param body_spin_rate Angular rate of the instrument about the target (rad/s), M3
	@param min_normal_observation_angle Minimum angle for a ray to be used (rad)
	@param min_facet_normal_angle_difference Minimum angle separating to normals associated with the same vertex
	@param minimum_ray_per_facet Minimum number of rays per facet to include the facet in the
	@param ridge_coef Non-zero value regularizes the information matrix by introducing a bias
	estimation process
	@param reject_outliers True if facet residuals differing from the mean by more than one sigma should be excluded
	@param split_status True if the shape model is to be split
	@param use_cholesky True is Cholesky decomposition should be used to solve the normal equation
	@param recycle_facets True if facets that are degenerated should be removed
	*/

	Arguments(double t0,
	          double tf,
	          double min_normal_observation_angle,
	          double orbit_rate,
	          double inclination,
	          double body_spin_rate,
	          double min_facet_normal_angle_difference,
	          unsigned int minimum_ray_per_facet,
	          double ridge_coef,
	          bool reject_outliers,
	          bool split_status,
	          bool use_cholesky,
	          bool recycle_facets) {

		this -> t0 = t0;
		this -> tf = tf;
		this -> min_normal_observation_angle = min_normal_observation_angle;
		this -> orbit_rate = orbit_rate;
		this -> inclination = inclination;
		this -> body_spin_rate = body_spin_rate;
		this -> min_facet_normal_angle_difference = min_facet_normal_angle_difference;
		this -> minimum_ray_per_facet = minimum_ray_per_facet;
		this -> ridge_coef = ridge_coef;
		this -> reject_outliers = reject_outliers;
		this -> split_status = split_status;
		this -> use_cholesky = use_cholesky;
		this -> recycle_facets = recycle_facets;
	}

	double get_t0() const {
		return this -> t0;

	}
	double get_tf() const {
		return this -> tf;

	}
	double get_min_normal_observation_angle() const {
		return this -> min_normal_observation_angle;

	}
	double get_orbit_rate() const {
		return this -> orbit_rate;

	}
	double get_body_spin_rate() const {
		return this -> body_spin_rate;
	}

	double get_inclination() const {
		return this -> inclination;
	}

	double get_min_facet_normal_angle_difference() const {
		return this -> min_facet_normal_angle_difference;

	}

	bool get_split_status() const {
		return this -> split_status;

	}

	unsigned int get_minimum_ray_per_facet() const {
		return this -> minimum_ray_per_facet;

	}
	double get_ridge_coef() const {
		return this -> ridge_coef;

	}
	bool get_reject_outliers() const {
		return this -> reject_outliers;

	}

	bool get_use_cholesky() const {
		return this -> use_cholesky;
	}

	bool get_recycle_facets() const {
		return this -> recycle_facets;
	}





protected:
	double t0;
	double tf;
	double min_normal_observation_angle;

	double orbit_rate;
	double inclination;
	double body_spin_rate;

	double min_facet_normal_angle_difference;
	unsigned int minimum_ray_per_facet;
	double ridge_coef;

	bool reject_outliers;
	bool split_status;
	bool use_cholesky;
	bool recycle_facets;

};




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
	       Arguments * arguments);


	/**
	@param N_iteration number of iteration of the filter with each batch of information
	@param plot_measurement true if the measurements should be saved
	@param save_shape_model true if the shape model should be saved after each measurement
	*/
	void run(unsigned int N_iteration, bool plot_measurements, bool save_shape_model);

	/**
	Solves the square linear system info_mat * x = normal_mat
	by means of a cholesky decomposition
	@param info_mat Symetrical information matrix
	@param normal_mat Normal matrix
	@return x Solution
	*/
	arma::vec cholesky(arma::mat & info_mat, arma::mat & normal_mat) const;

protected:

	void correct_shape(unsigned int time_index, bool last_iter);

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


	Arguments * arguments;
	FrameGraph * frame_graph;
	Lidar * lidar;
	ShapeModel * true_shape_model;
	ShapeModel * estimated_shape_model;




};


#endif