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

	/**
	Fits the shape using the prescribed point cloud
	@param N_iter maximum number of iterations
	@param J standard deviation of update norm below which convergence is reached
	@param DS DCM aligning the provided point cloud with the shape
	@param X_DS translation vector aligning the provided point cloud with the shape
	*/
	void fit_shape(PC * pc, 
		unsigned int N_iter ,
		double J ,
		const arma::mat & DS , 
		const arma::vec & X_DS );



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
	void measure_omega(const arma::mat & dcm) ;


	/**
	Computes a measurement of the direction of the rigid's body spin axis
	@param dcm DCM obtained from the ICP rigid transform
	*/
	void measure_spin_axis(const arma::mat & dcm);

	/**
	Computes the new relative states from the (sigma,omega),(r,r') relative states
	@param X_S relative state at present time (12x1)
	@param time measurement time
	@param NE_tD_EN_tS_pc matrix output from the ICP. Incremental rotation measure
	@param X_relative_from_pc new measure of the total relative displacement
	@param LN_t_S true [LN] dcm at current time
	@param LN_t_D true [LN] dcm at past measurement time
	@param mrp_BN true mrp instantiating [BN] at the current time
	@param X_relative_true true relatuve motion
	@param from_shape true if M was obtained from registration of shape destination point cloud to source
	@param offset_DCM DCM aligning the tracked body frame B and its estimate E at t0
	@param OL_t0 position of spacecraft in the body frame when measurements start to be accumulated
	@param LN_t0 [LN] DCM at the time observations are starting
	*/
	void perform_measurements_pc(const arma::vec & X_S, 
		double time, 
		const arma::mat & NE_tD_EN_tS_pc,
		const arma::vec & X_relative_from_pc,
		const arma::mat & LN_t_S, 
		const arma::mat & LN_t_D, 
		const arma::vec & mrp_BN,
		const arma::vec & X_relative_true ,
		const arma::mat & offset_DCM,
		const arma::vec & OL_t0,
		const arma::mat & LN_t0);


	/**
	Computes the new relative states from the (sigma,omega),(r,r') relative states
	@param X_S relative state at present time (12x1)
	@param time measurement time
	@param M matrix output from the ICP.
	@param NE_tD_EN_tS_pc matrix output from the pc to pc ICP. Always measures an incremental rotation
	@param X_pc translational output from the pc to pc ICP
	@param LN_t_S reference to [LN] dcm at current time
	@param LN_t_D reference to [LN] dcm at past measurement time
	@param mrp_BN reference to the mrp instantiating [BN] at the current time
	@param X_relative_true true relative motion
	@param offset_DCM DCM aligning the tracked body frame B and its estimate E at t0
	@param OL_t0 position of spacecraft in the body frame when measurements start to be accumulated
	@param LN_t0 [LN] DCM at the time observations are starting
	*/
	void perform_measurements_shape(
		const arma::vec & X_S, 
		double time, 
		const arma::mat & M,
		const arma::mat & NE_tD_EN_tS_pc,
		const arma::vec & X_pc,
		const arma::mat & LN_t_S, 
		const arma::mat & LN_t_D, 
		const arma::vec & mrp_BN,
		const arma::vec & X_relative_true,
		const arma::mat & offset_DCM,
		const arma::vec & OL_t0,
		const arma::mat & LN_t0);


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