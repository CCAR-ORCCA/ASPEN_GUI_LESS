#ifndef HEADER_FILTER
#define HEADER_FILTER

#include "ShapeModel.hpp"
#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "Interpolator.hpp"
#include "FilterArguments.hpp"
#include "PC.hpp"
#include "ICP.hpp"
#include "ICPException.hpp"
#include "Args.hpp"
#include "Wrappers.hpp"
#include "RK.hpp"



#include <RigidBodyKinematics.hpp>

#include <boost/progress.hpp>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <limits>



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
	@param filter_arguments filter parameters
	*/
	Filter(FrameGraph * frame_graph,
	       Lidar * lidar,
	       ShapeModel * true_shape_model,
	       ShapeModel * estimated_shape_model,
	       FilterArguments * filter_arguments);


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
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	@param filter_arguments filter parameters
	*/
	Filter(FrameGraph * frame_graph,
	       Lidar * lidar,
	       ShapeModel * true_shape_model,
	       FilterArguments * filter_arguments) ;

	/**
	Runs the shape reconstruction filter
	@param orbit_path Path to the orbit file
	@param orbit_time_path Path to the orbit time file
	@param attitude_path Path to the attitude file
	@param attitude_time_path Path to the attitude time file
	@param plot_measurement true if the measurements should be saved
	@param save_shape_model true if the shape model should be saved after each measurement
	@param inertial_traj True if provided trajectory is inertial
	*/
	void run_shape_reconstruction(std::string orbit_path,
	                              std::string orbit_time_path,
	                              std::string attitude_path,
	                              std::string attitude_time_path,
	                              bool plot_measurements,
	                              bool save_shape_model,
	                              bool inertial_traj);

	/**
	Runs the attitude estimation filter
	@param orbit_path Path to the orbit file
	@param orbit_time_path Path to the orbit time file
	@param attitude_path Path to the attitude file
	@param attitude_time_path Path to the attitude time file
	@param savepath Path to obj file of the form XXX.obj (ex: test.obj)
	@param inertial_traj True if provided trajectory is inertial
	*/
	void run_attitude_estimation(
	    std::string orbit_path,
	    std::string orbit_time_path,
	    std::string attitude_path,
	    std::string attitude_time_path,
	    bool inertial_traj);


	/**
	Register the source and destination point clouds
	@param index time index
	@param time time
	*/
	void register_pcs(int index, double time);

	/**
	Extracts the spin axis from the rigid transform
	@param dcm DCM
	@param x
	@param points_pairs Source/Destination point pairs
	*/
	void extract_spin_axis(arma::mat & dcm,
	                       arma::vec & x,
	                       std::vector<std::pair<std::shared_ptr<PointNormal>,
	                       std::shared_ptr<PointNormal> > > * point_pairs) ;


	/**

	Runs one shape reconstruction pass (N shape correction updates followed by a 
	facet recycling step)
	@param time_index index
	@param time_index_formatted string denoting the current time
	*/
	void shape_reconstruction_pass(unsigned int time_index, 
		std::string time_index_formatted);

	

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

	/**
	Moves the latest measurements to the corresponding point clouds
	and stores them to file
	@param t time
	*/
	void store_point_clouds(int index);



	/**
	Computes a measurement of the angular velocity
	@param point_pairs Pointer to the paired source and destination point clouds
	*/
	// void measure_omega(std::vector<std::pair<std::shared_ptr<PointNormal>,
	//                    std::shared_ptr<PointNormal> > > * point_pairs);


	/**
	Computes a measurement of the angular velocity
	@param dcm M in f(C) = MC + x
	*/
	void measure_omega(arma::mat & dcm) ;


	/**
	Computes an estimate of angular velocity of the target
	assuming that it is undergoing torque free motion using a CKF.
	@param point_pairs Pointer to the paired source and destination point clouds
	@param time Time
	*/
	void compute_omega_KF(std::vector<std::pair<std::shared_ptr<PointNormal>,
	                      std::shared_ptr<PointNormal> > > * point_pairs,
	                      double time) ;


	/**
	Computes an estimate of the center of mass of the target
	assuming that it is undergoing torque free motion using a CKF.
	@param dcm DCM obtained from the ICP rigid transform
	@param X translation vector obtained from the ICP rigid transform
	*/
	void estimate_cm_KF(arma::mat & dcm, arma::vec & x);

	/**
	Computes a measurement of the direction of the rigid's body spin axis
	@param dcm DCM obtained from the ICP rigid transform
	*/
	void measure_spin_axis(arma::mat & dcm);






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
	                           std::vector<unsigned int> > & facet_to_index_of_vertices,
	                           double & mean,
	                           double & stdev

	                          ) ;



	std::vector<arma::rowvec> partial_range_partial_coordinates(const arma::vec & P,
	        const arma::vec & u, Facet * facet) ;


	FilterArguments * filter_arguments;
	FrameGraph * frame_graph;
	Lidar * lidar;
	ShapeModel * true_shape_model;
	ShapeModel * estimated_shape_model;

	std::shared_ptr<PC> destination_pc = nullptr;
	std::shared_ptr<PC> source_pc = nullptr;
	std::shared_ptr<PC> destination_pc_shape = nullptr;
	std::shared_ptr<PC> source_pc_shape = nullptr;


	std::set<Facet *> seen_facets_destination;


};


#endif