#ifndef HEADER_FILTER
#define HEADER_FILTER

#include "ShapeModel.hpp"
#include "ShapeModelTri.hpp"

#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "Interpolator.hpp"
#include "FilterArguments.hpp"
#include "PC.hpp"
#include "ICP.hpp"
#include "ICPException.hpp"
#include "Args.hpp"
#include "Wrappers.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeFitter.hpp"

#include "CGAL_interface.hpp"
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
		ShapeModelTri * true_shape_model,
		ShapeModelTri * estimated_shape_model,
		FilterArguments * filter_arguments);


	/**
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	*/
	Filter(FrameGraph * frame_graph,
		Lidar * lidar,
		ShapeModelTri * true_shape_model) ;

	/**
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	@param filter_arguments filter parameters
	*/
	Filter(FrameGraph * frame_graph,
		Lidar * lidar,
		ShapeModelTri * true_shape_model,
		FilterArguments * filter_arguments) ;

	
	/**
	Runs the shape reconstruction filter
	@param times vector of measurement times
	@param interpolator pointer to state interpolator used within the filter
	@param save_shape_model true if the true shape model must be saved prior 
	to the run
	*/
	void run_shape_reconstruction(arma::vec &times ,
		Interpolator * interpolator,
		bool save_shape_model);




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



	



protected:

	void correct_shape(unsigned int time_index, bool first_iter, bool last_iter,
		bool plot_measurement);

	void correct_observed_features(std::vector<Ray * > & good_rays,
		std::set<ControlPoint *> & seen_vertices,
		std::set<Facet *> & seen_facets,

		arma::mat & N_mat,
		std::map<Facet *,
		std::vector<unsigned int> > & facet_to_index_of_vertices) ;

	void get_observed_features(std::vector<Ray * > & good_rays,
		std::set<ControlPoint *> & seen_vertices,
		std::set<Facet *> & seen_facets,
		std::set<Facet *> & spurious_facets,
		arma::mat & N_mat,
		std::map<Facet *,
		std::vector<unsigned int> > & facet_to_index_of_vertices
		) ;



	std::vector<arma::rowvec> partial_range_partial_coordinates(const arma::vec & P,
		const arma::vec & u, Facet * facet) ;



	/**
	Computes the new relative states from the (sigma,omega),(r,r') relative states
	@param X_S relative state at present time (12x1)
	@param dcm_LB reference to [LB] dcm at present time (12x1)
	@param dcm_LB_t_D reference to [LB] dcm at past measurement time (12x1)
	@param LN_t_S reference to [LN] dcm at current time
	@param LN_t_D reference to [LN] dcm at past measurement time
	@param mrp_BN reference to the mrp instantiating [BN] at the current time
	@param mrp_BN_t_D reference to the mrp instantiating [BN] at the past measurement time
	@param mrp_LB reference to the mrp instantiating [LB] at the current time
	@param lidar_pos reference to relative position of the spacecraft w/r to the barycentric B frame
	@param lidar_vel reference to relative velocity of the spacecraft w/r to the barycentric B frame
	*/
	void get_new_relative_states(const arma::vec & X_S, arma::mat & dcm_LB, arma::mat & dcm_LB_t_D, arma::mat & LN_t_S, 
	arma::mat & LN_t_D, arma::vec & mrp_BN, arma::vec & mrp_BN_t_D,
	arma::vec & mrp_LB, arma::vec & lidar_pos,arma::vec & lidar_vel );

	/**
	Computes a measurement of the angular velocity
	@param dcm M in f(C) = MC + x
	*/
	void measure_omega(arma::mat & dcm) ;


	/**
	Computes a measurement of the direction of the rigid's body spin axis
	@param dcm DCM obtained from the ICP rigid transform
	*/
	void measure_spin_axis(arma::mat & dcm);

	/**
	Computes the new relative states from the (sigma,omega),(r,r') relative states
	@param X_S relative state at present time (12x1)
	@param time measurement time
	@param M matrix output from the ICP
	@param LN_t_S reference to [LN] dcm at current time
	@param LN_t_D reference to [LN] dcm at past measurement time
	@param mrp_BN reference to the mrp instantiating [BN] at the current time
	*/
	void perform_measurements(const arma::vec & X_S, double time, const arma::mat & M,  arma::mat & LN_t_S, 
	arma::mat & LN_t_D, arma::vec & mrp_BN);


	FilterArguments * filter_arguments;
	FrameGraph * frame_graph;
	Lidar * lidar;
	ShapeModelTri * true_shape_model;
	ShapeModelTri * estimated_shape_model;

	std::shared_ptr<PC> destination_pc = nullptr;
	std::shared_ptr<PC> source_pc = nullptr;
	std::shared_ptr<PC> destination_pc_shape = nullptr;


};


#endif