#ifndef HEADER_SHAPEBUILDER
#define HEADER_SHAPEBUILDER

#include <armadillo>
#include <memory>
#include <set>
#include <string>
#include <OrbitConversions.hpp>

class Lidar;

template <class PointType> class ShapeModelTri;
template <class PointType> class ShapeModelBezier;
template <class PointType> class PointCloud;

class ShapeBuilderArguments;
class FrameGraph;
class PointNormal;
class ControlPoint;
struct RigidTransform;

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
		ShapeModelTri<ControlPoint> * true_shape_model) ;

	/**
	Constructor
	@param frame_graph Pointer to the graph storing the reference frames
	@param lidar Pointer to instrument
	@param true_shape_model Pointer to the true shape model
	@param filter_arguments filter parameters
	*/
	ShapeBuilder(FrameGraph * frame_graph,
		Lidar * lidar,
		ShapeModelTri<ControlPoint> * true_shape_model,
		ShapeBuilderArguments * filter_arguments) ;

	
	/**
	Runs the shape reconstruction filter
	@param times vector of measurement times
	@param X reference to state used within the filter
	to the run
	@param dir path to directory where to save outputs
	*/
	void run_shape_reconstruction(const arma::vec &times ,
		const std::vector<arma::vec> & X,
		const std::string dir);



	/**
	Runs the iod filter
	@param times vector of measurement times
	@param X reference to state used within the filter
	@param dir path to directory where to save outputs
	*/
	void run_iod(const arma::vec &times ,
		const std::vector<arma::vec> & X,
		std::string dir);




	std::shared_ptr<ShapeModelBezier< ControlPoint > > get_estimated_shape_model() const;


	arma::vec get_final_measured_attitude() const;
	arma::vec get_final_measured_omega() const;



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
	@param dir path to output directory
	*/
	void store_point_clouds(int index,const std::string dir);



	/**
	Extracts an apriori rigid transform (M_pc_a_priori,X_pc_a_priori) that works
	`best` to prealign the source and destination point clouds at the current time.
	This `best` transform is either 1) the last rigid transform that the ICP/Bundle adjuster returned 
	or 2) a prediction of the rigid transform using the IOD arc yielding an estimate of the spacecraft trajectory.
	The `best` transform is chosen in terms of the point-pairs residuals
	@param[in] M_pc_a_priori rotational component of the a-priori rigid transform
	@param[in] X_pc_a_priori translational component of the a-priori rigid transform
	@param cartesian_state Cartesian state representative of the IOD arc 
	@param times vector of times
	@param time_index current time index
	@param epoch_time_index time index of the time that is the epoch (i.e first time) in the IOC arc
	@param BN_measured measured/bundle adjusted [BN] dcms. Note that at the time this method is called, BN[time_index] is not available yet
	@param M_pcs measured/bundle-adjusted absolute rigid transforms dcms. Note that at the time this method is called, M_pcs[time_index] is not available yet
	@param X_pcs measured/bundle-adjusted absolute rigid transforms translations. Note that at the time this method is called, X_pcs[time_index] is not available yet
	@param mrps_LN spacecraft inertial attitude time-history
	*/
	void get_best_a_priori_rigid_transform(
		arma::mat::fixed<3,3> & M_pc_a_priori,
		arma::vec::fixed<3> & X_pc_a_priori,
		const OC::CartState & cartesian_state,
		const arma::vec & times,
		const int & time_index,
		const int & epoch_time_index,
		const std::vector<arma::mat::fixed<3,3>> & BN_measured,
		const std::map<int,arma::mat::fixed<3,3>> &  M_pcs,
		const std::map<int,arma::vec::fixed<3> > &  X_pcs,
		const std::vector<arma::vec::fixed<3> > & mrps_LN);


	/**
	Returns covariance of estimated state
	@return covariance of estimated state (position,velocity,mrp,angular velocity)
	*/
	arma::mat get_covariance_estimated_state() const;


	/**
	Returns  estimated state
	@return estimated state (position,velocity,mrp,angular velocity)
	*/
	arma::vec::fixed<13> get_estimated_state() const;





protected:

	static void extract_a_priori_transform(
		arma::mat::fixed<3,3> & M, 
		arma::vec::fixed<3> & X,
		const int index,
		const arma::vec::fixed<3> & r_k_hat,
		const arma::vec::fixed<3> & r_km1_hat,
		const std::vector<arma::mat::fixed<3,3> > & BN_measured,
		const std::map<int,arma::mat::fixed<3,3>> &  M_pcs,
		const std::map<int,arma::vec::fixed<3>> &  X_pcs,
		const std::vector<arma::vec::fixed<3> > & mrps_LN);

	arma::vec get_center_collected_pcs(
		int first_pc_index,
		int last_pc_index) const;

	arma::vec get_center_collected_pcs(int first_pc_index,
		int last_pc_index,
		const std::vector<RigidTransform> & absolute_rigid_transforms,
		const std::vector<RigidTransform> & absolute_true_rigid_transforms) const;

	arma::vec::fixed<3> get_center_collected_pcs() const;


	void estimate_coverage(
		std::string dir,PointCloud<PointNormal> * pc  = nullptr) const;


	/**
	Computes the new relative states from the (sigma,omega),(r,r') relative states
	@param X_S relative state at present time (12x1)
	@param dcm_LB reference to [LB] dcm at present time (12x1)
	@param mrp_LN reference to the mrp instantiating [LN] at the current time
	@param lidar_pos reference to relative position of the spacecraft w/r to the barycentric B frame
	@param lidar_vel reference to relative velocity of the spacecraft w/r to the barycentric B frame
	*/
	void get_new_states(const arma::vec & X_S, 
		arma::mat::fixed<3,3> & dcm_LB, 
		arma::vec::fixed<3> & lidar_pos,
		arma::vec::fixed<3> & lidar_vel,
		std::vector<arma::vec::fixed<3>> & mrps_LN,
		std::vector<arma::mat::fixed<3,3>> & BN_true,
		std::vector<arma::mat::fixed<3,3>> & HN_true);



	void save_true_ground_track(const std::vector<arma::mat> & BN_true,
		const std::vector<arma::mat> & HN_true) const;


	void save_estimated_ground_track(
		std::string path,
		const arma::vec & times,
		const int t0 ,
		const int tf, 
		const OC::KepState & est_kep_state,
		const std::vector<arma::mat> BN_measured) const;

	
	/**
	Assembles the rigid transforms needed to evaluate the IOD cost function
	@param rigid_transforms rigid transforms to assemble
	@param obs_indices vector storing indices of times to be used in this IOD run
	@param times vector of times
	@param mrps_LN time history of (true) [LN] in mrp form
	@param X_pcs map of computed absolute rigid transform translations, indexed by timestamp
	@param M_pcs map of computed absolute rigid transform rotations, indexed by timestamp
	*/
	void assemble_rigid_transforms_IOD(std::vector<RigidTransform> & sequential_rigid_transforms,
		std::vector<RigidTransform> & absolute_rigid_transforms,
		const std::vector<int> & obs_indices,
		const arma::vec & times, 
		const std::vector<arma::vec::fixed<3> >  & mrps_LN,
		const std::map<int,arma::vec::fixed<3> > & X_pcs,
		const std::map<int,arma::mat::fixed<3,3> > & M_pcs) const;



	void compute_rigid_transform_covariances(std::vector<arma::mat> & rigid_transforms_covariances,
		const arma::vec & times, 
		const int t0_index,
		const int tf_index,
		const std::vector<arma::vec>  & mrps_LN,
		const std::map<int,arma::vec> &  X_pcs,
		const std::map<int,arma::mat> &  M_pcs) const ;



	/**
	Computes a initial a-priori state minimizing the associated rigid transform norm 
	@param times vector of times
	@param t0_index timestamp of epoch in current IOD run
	@param tf_index timestamp of last considered state in IOD run
	@param mrps_LN time history of (true) [LN] in mrp form
	@param X_pcs map of computed absolute rigid transform translations, indexed by timestamp
	@param M_pcs map of computed absolute rigid transform rotations, indexed by timestamp
	@param R_pcs map of absolute rigid transform covariance matrices 
	*/
	void run_IOD_finder(
		const arma::vec & times,
		const int t0 ,
		const int tf, 
		const std::vector<arma::vec::fixed<3> > & mrps_LN,
		const std::map<int, arma::vec::fixed<3> > & X_pcs,
		const std::map<int, arma::mat::fixed<3,3> > & M_pcs,
		const std::map<int, arma::mat::fixed<6,6> > & R_pcs,
		OC::CartState & cart_state,
		arma::vec & epoch_state,
		arma::vec & final_state,
		arma::mat & epoch_cov,
		arma::mat & final_cov) const;




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



	void save_attitude(std::string prefix,int index,const std::vector<arma::mat::fixed<3,3> > & BN) const;

	static void run_psr(PointCloud<PointNormal> * pc,
		const std::string dir,
		ShapeModelTri<ControlPoint> & psr_shape,
		ShapeBuilderArguments * filter_arguments);


	/**
	Concatenates the destination and source point clouds. The latter is merged into the former
	@param M_pc dcm matrix from the ICP registering the source point cloud to the destination point cloud
	@param X_pc translation vector from the ICP registering the source point cloud to the destination point cloud
	*/
	void concatenate_point_clouds(unsigned int index);

	void initialize_shape(unsigned int cutoff_index);

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


	/**
	Saves true, estimated rigid transforms to file along with their covariances
	*/
	void save_rigid_transforms(std::string dir, 
	const std::map<int,arma::vec::fixed<3> > & X_pcs,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs,
	const std::map<int,arma::vec::fixed<3> > & X_pcs_true,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs_true,
	const std::map<int,arma::mat::fixed<6,6> > & R_pcs);


	ShapeBuilderArguments * filter_arguments;
	FrameGraph * frame_graph;
	Lidar * lidar;
	ShapeModelTri<ControlPoint> * true_shape_model;

	std::shared_ptr<ShapeModelBezier< ControlPoint > > estimated_shape_model;

	std::shared_ptr<PointCloud < PointNormal > > destination_pc = nullptr;
	std::shared_ptr<PointCloud < PointNormal > > source_pc = nullptr;
	std::shared_ptr<PointCloud < PointNormal > > destination_pc_shape = nullptr;
	std::vector< std::shared_ptr<PointNormal> > concatenated_pc_vector;
	std::vector< std::shared_ptr<PointCloud < PointNormal > > > all_registered_pc;

	arma::mat LN_t0;
	arma::mat LB_t0;
	arma::mat::fixed<13,13>  covariance_estimated_state;
	arma::vec::fixed<13> estimated_state;
	OC::KepState true_kep_state_t0;


	arma::vec x_t0;


};


#endif