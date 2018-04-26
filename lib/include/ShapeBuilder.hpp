#ifndef HEADER_SHAPEBUILDER
#define HEADER_SHAPEBUILDER

#include <armadillo>
#include <memory>
#include <set>
#include <string>

class Lidar;
class PC;

class ShapeModelTri;
class ShapeModelBezier;
class ShapeBuilderArguments;
class FrameGraph;
class PointNormal;


/**
ShapeBuilder class hosting:
- the instrument
- the true shape model
- the estimated shape model
- ShapeBuildering tools:
# the partial derivatives evaluation
# shape refinement
*/
class ShapeBuilder {

public:



	/**
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	*/
	ShapeBuilder(FrameGraph * frame_graph,
		Lidar * lidar,
		ShapeModelTri * true_shape_model) ;

	/**
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	@param filter_arguments filter parameters
	*/
	ShapeBuilder(FrameGraph * frame_graph,
		Lidar * lidar,
		ShapeModelTri * true_shape_model,
		ShapeBuilderArguments * filter_arguments) ;

	
	/**
	Runs the shape reconstruction filter
	@param times vector of measurement times
	@param X reference to state used within the filter
	@param save_shape_model true if the true shape model must be saved prior 
	to the run
	*/
	void run_shape_reconstruction(const arma::vec &times ,
		const std::vector<arma::vec> & X,
		bool save_shape_model);




	std::shared_ptr<ShapeModelBezier> get_estimated_shape_model() const;




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
	void store_point_clouds(int index,const arma::mat & M_pc = arma::eye<arma::mat>(3,3),const arma::mat & X_pc = arma::zeros<arma::vec>(3));

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




	/**
	Computes the new relative states from the (sigma,omega),(r,r') relative states
	@param X_S relative state at present time (12x1)
	@param dcm_LB reference to [LB] dcm at present time (12x1)
	@param mrp_LN reference to the mrp instantiating [LN] at the current time
	@param lidar_pos reference to relative position of the spacecraft w/r to the barycentric B frame
	@param lidar_vel reference to relative velocity of the spacecraft w/r to the barycentric B frame
	*/
	void get_new_states(const arma::vec & X_S, 
	arma::mat & dcm_LB, 
	arma::vec & mrp_LN, 
	arma::vec & lidar_pos,
	arma::vec & lidar_vel);

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
	Concatenates the destination and source point clouds. The latter is merged into the former
	@param M_pc dcm matrix from the ICP registering the source point cloud to the destination point cloud
	@param X_pc translation vector from the ICP registering the source point cloud to the destination point cloud
	*/
	void concatenate_point_clouds(unsigned int index);

	void initialize_shape(unsigned int index);


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


	ShapeBuilderArguments * filter_arguments;
	FrameGraph * frame_graph;
	Lidar * lidar;
	ShapeModelTri * true_shape_model;
	std::shared_ptr<ShapeModelBezier> estimated_shape_model;

	std::shared_ptr<PC> destination_pc = nullptr;
	std::shared_ptr<PC> source_pc = nullptr;
	std::shared_ptr<PC> destination_pc_shape = nullptr;

	std::vector< std::shared_ptr<PointNormal> > concatenated_pc_vector;


	std::vector< std::shared_ptr<PC> > all_registered_pc;


};


#endif