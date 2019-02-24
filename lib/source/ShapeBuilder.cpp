#include <ShapeBuilder.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelBezier.hpp>

#include <Lidar.hpp>
#include <FrameGraph.hpp>
#include <ShapeBuilderArguments.hpp>

#include <IterativeClosestPointToPlane.hpp>
#include <IterativeClosestPoint.hpp>

#include <BundleAdjuster.hpp>
#include <CustomException.hpp>
#include <ControlPoint.hpp>
#include <ShapeModelImporter.hpp>
#include <ShapeFitterBezier.hpp>
#include <IODFinder.hpp>
#include <CGAL_interface.hpp>

#include <EstimationNormals.hpp>
#include <IODBounds.hpp>
#include <PointCloud.hpp>
#include <PointCloudIO.hpp>

#include <boost/progress.hpp>
#include <chrono>
#include <cmath>
#include <RigidBodyKinematics.hpp>
#include <BatchAttitude.hpp>

#define IOFLAGS_shape_builder 0

ShapeBuilder::ShapeBuilder(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri<ControlPoint> * true_shape_model,
	ShapeBuilderArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> filter_arguments = filter_arguments;
}

ShapeBuilder::ShapeBuilder(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri<ControlPoint> * true_shape_model) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
}

void ShapeBuilder::run_shape_reconstruction(const arma::vec &times ,
	const std::vector<arma::vec> & X,const std::string dir) {

	std::cout << "Running the filter and saving results to " << dir << std::endl;

	arma::vec X_S = arma::zeros<arma::vec>(X[0].n_rows);

	arma::vec::fixed<3> lidar_pos = X_S.rows(0,2);
	arma::vec::fixed<3> lidar_vel = X_S.rows(3,5);

	arma::mat::fixed<3,3> dcm_LB = arma::eye<arma::mat>(3, 3);
	std::vector<RigidTransform> rigid_transforms;
	std::vector<arma::vec::fixed<3> > mrps_LN;
	std::vector<arma::mat::fixed<3,3 > > BN_measured;
	std::vector<arma::mat::fixed<3,3 > > BN_true;
	std::vector<arma::mat::fixed<3,3 > > HN_true;
	std::vector<double> estimated_mu;
	std::map<int,arma::vec::fixed<3> > X_pcs;
	std::map<int,arma::mat::fixed<3,3> > M_pcs;
	std::map<int,arma::vec::fixed<3> > X_pcs_true;
	std::map<int,arma::mat::fixed<3,3> > M_pcs_true;
	std::map<int,arma::mat::fixed<6,6> > R_pcs;

	arma::mat epoch_cov,final_cov;
	arma::vec epoch_state,final_state;

	arma::vec iod_guess;

	int last_iod_epoch_index = 0;
	int cutoff_index = 0;
	OC::CartState iod_state;
	int epoch_time_index = 0;

	arma::mat::fixed<3,3> M_pc = arma::eye<arma::mat>(3,3);
	arma::vec::fixed<3> X_pc = arma::zeros<arma::vec>(3);

	BundleAdjuster ba_test(
		this -> lidar -> get_los_noise_sd_baseline(),
		&this -> all_registered_pc,
		this -> filter_arguments -> get_N_iter_bundle_adjustment() ,
		this -> filter_arguments -> get_ba_h(),
		&this -> LN_t0,
		&this -> x_t0,
		dir,
		&mrps_LN,
		&BN_measured);


	for (int time_index = 0; time_index < times.n_rows; ++time_index) {

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index + 1;
		std::string time_index_formatted = ss.str();

		std::cout << "\n################### Index : " << time_index << " / " <<  times.n_rows - 1  << ", Time : " << times(time_index) << " / " <<  times( times.n_rows - 1) << " ########################" << std::endl;

		X_S = X[time_index];

		this -> get_new_states(X_S,dcm_LB,lidar_pos,lidar_vel,mrps_LN,BN_true,HN_true);
		
		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(X_S.subvec(0,2));
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrps_LN[time_index]);

		// Setting the small body to its new attitude
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(X_S.subvec(6,8));

		// Getting the true observations (noise is added)
		this -> lidar -> send_flash(this -> true_shape_model,true);

		#if IOFLAGS_shape_builder
		this -> lidar -> save(dir + "/pc_" + std::to_string(time_index),true);
		#endif

		// The rigid transform best aligning the two point clouds is found
		// The solution to this first registration will be used to prealign the 
		// shape model and the source point cloud



		this -> store_point_clouds(time_index,dir);

		if (this -> destination_pc != nullptr && this -> source_pc == nullptr){
			this -> all_registered_pc.push_back(this -> destination_pc);

			// It is legit to use the true attitude state at the first timestep
			// as it just defines an offset
			BN_measured.push_back(RBK::mrp_to_dcm(X_S.subvec(6,8)));

			M_pcs[time_index] = arma::eye<arma::mat>(3,3);;
			X_pcs[time_index] = arma::zeros<arma::vec>(3);

			M_pcs_true[time_index] = arma::eye<arma::mat>(3,3);;
			X_pcs_true[time_index] = arma::zeros<arma::vec>(3);

			R_pcs[time_index] = arma::zeros<arma::mat>(6,6);

			true_shape_model -> save(dir + "/true_shape_L0.obj",- this -> LN_t0 * this -> x_t0, this -> LN_t0);
			

		}

		else if (this -> destination_pc != nullptr && this -> source_pc != nullptr){


			if(this -> filter_arguments -> get_use_icp()){

				arma::mat::fixed<3,3> M_pc_a_priori;
				arma::vec::fixed<3> X_pc_a_priori;

				std::cout << "getting best a-priori rigid transform\n";



				for (int j = 0; j < this -> source_pc -> size(); ++j){

					if (this -> source_pc -> get_point(j).get_point_coordinates().n_rows != 3){
						throw(std::runtime_error("Point " + std::to_string(j) + " in source_pc is fishy"));
					}
				}

				for (int j = 0; j < this ->  destination_pc -> size(); ++j){

					if (this -> destination_pc -> get_point(j).get_point_coordinates().n_rows != 3){
						throw(std::runtime_error("Point " + std::to_string(j) + " in destination_pc is fishy"));
					}
				}









				this -> get_best_a_priori_rigid_transform(
					M_pc_a_priori,
					X_pc_a_priori,
					iod_state,
					times,
					time_index,
					epoch_time_index,
					BN_measured,
					M_pcs,
					X_pcs,
					mrps_LN);

				IterativeClosestPointToPlane icp_pc(this -> destination_pc, this -> source_pc);
				
				icp_pc.register_pc(
					this -> filter_arguments -> get_los_noise_sd_baseline(),
					M_pcs.at(time_index - 1),
					M_pc_a_priori,
					X_pc_a_priori
					);

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D == t_0
				M_pc = icp_pc.get_dcm();
				X_pc = icp_pc.get_x();
				R_pcs[time_index] = icp_pc.get_R();
			}

			M_pcs_true[time_index] = this -> LB_t0 * dcm_LB.t();
			X_pcs_true[time_index] = M_pcs_true[time_index] *(- this -> frame_graph -> convert(arma::zeros<arma::vec>(3),"B","L")) - this -> LN_t0 * this -> x_t0;


			std::cout << "Error in ICP rigid transform : \n";

			arma::vec dX_pc = X_pc - X_pcs_true[time_index];
			arma::vec dsigma_pc = RBK::dcm_to_mrp(M_pc * M_pcs_true[time_index].t());

			std::cout << dX_pc(0) << " " << dX_pc(1) << " " << dX_pc(2) << " " << dsigma_pc(0) << " " << dsigma_pc(1) << " "  << dsigma_pc(2) << std::endl;
			/****************************************************************************/
				/********** ONLY FOR DEBUG: MAKES ICP USE TRUE RIGID TRANSFORMS *************/
			
			if (this -> filter_arguments -> get_use_true_rigid_transforms()){
				std::cout << "MAKES ICP USE TRUE RIGID TRANSFORMS\n";
				M_pc = M_pcs_true[time_index];
				X_pc = X_pcs_true[time_index];

			}
				/****************************************************************************/
				/****************************************************************************/

			// The measured BN dcm is saved
			// using the ICP measurement
			// M_pc(k) is [LB](t_0) * [BL](t_k) = [LN](t_0)[NB](t_0) * [BN](t_k) * [NL](t_k);
			BN_measured.push_back( BN_measured.front() * this -> LN_t0.t() *  M_pc * RBK::mrp_to_dcm(mrps_LN[time_index]));
			
			// The source pc is registered, using the rigid transform that 
			// the ICP returned
			this -> source_pc -> transform(M_pc,X_pc);
			this -> all_registered_pc.push_back(this -> source_pc);

			M_pcs[time_index] = M_pc;
			X_pcs[time_index] = X_pc;

			assert(BN_measured.size() == BN_true.size());
			assert(M_pcs.size() == BN_true.size());
			assert(X_pcs.size() == BN_true.size());

			if (this -> filter_arguments -> get_use_ba()){
				
				if(ba_test.update_overlap_graph()){
					std::cout << "Detected loop closure. Running bundle adjustment ...";
					ba_test.run(M_pcs,X_pcs,R_pcs,BN_measured,mrps_LN);
				}
				
			}

			std::cout << "True state at epoch time of index "<< epoch_time_index << " before running IOD: " << X[epoch_time_index].subvec(0,5).t();
			std::cout << "True position at index "<< time_index << " before running IOD: " << X[time_index].subvec(0,5).t() << std::endl;
			
			epoch_time_index = std::max(time_index - this -> filter_arguments -> get_iod_rigid_transforms_number(),0);

			this -> run_IOD_finder(times, 
				epoch_time_index ,
				time_index, 
				mrps_LN,
				X_pcs,
				M_pcs,
				R_pcs,
				iod_state,
				epoch_state,
				final_state,
				epoch_cov,
				final_cov);

			if (epoch_time_index == 0){
				this -> r0_from_kep_arc = epoch_state.subvec(0,2);
				std::cout << "r0_from_kep_arc: " << this -> r0_from_kep_arc.t();
				std::cout << "true r0: " << this -> x_t0.subvec(0,2).t();
			}

			// Should define extrapolated location of target of interest at the next timestep
				// right here

			this -> estimate_coverage(dir +"/"+ std::to_string(time_index) + "_");
			
			if (time_index <= times.n_rows - 2){
				arma::vec::fixed<3> r_extrapolated_next_time = (
					iod_state.convert_to_kep(0).convert_to_cart(times(time_index + 1) - times(epoch_time_index)).get_position_vector());


				std::cout << "\nExtrapolated position at next timestep: " << r_extrapolated_next_time.t();
				std::cout << "True position at next timestep: " << X[time_index +1].subvec(0,2).t();



			// Assumes a fixed rotation axis and angular velocity
			// As well as uniform time sampling
				arma::mat::fixed<3,3> BN_extrapolated_next_time = (BN_measured.back() * (BN_measured.end()[-2]).t()) * BN_measured.back() ; 

				if (this -> target_of_interest_L0_frame.n_rows != 0){
					this -> lidar_to_target_of_interest_N_frame = (- r_extrapolated_next_time 
						+ BN_extrapolated_next_time.t() 
						* ( BN_measured.front() 
							* ( this -> r0_from_kep_arc + RBK::mrp_to_dcm(mrps_LN.front()).t() * this -> target_of_interest_L0_frame)));

					if (arma::dot(arma::normalise(this -> lidar_to_target_of_interest_N_frame),arma::normalise(r_extrapolated_next_time)) > 0){
						this -> lidar_to_target_of_interest_N_frame.reset();
						std::cout << "\tTarget has faded\n";
					}
				}
			}


			if (time_index > this -> filter_arguments -> get_iod_rigid_transforms_number())
				estimated_mu.push_back(iod_state.get_mu());

			// Bundle adjustment is periodically run
			// If an overlap with previous measurements is detected
			// or if the bundle adjustment has not been run for a 
			// certain number of observations
			// Should probably replace this by an adaptive threshold based
			// on a prediction of the alignment error

			// N rigids transforms : (t0 --  t1), (t1 -- t2), ... , (tN-1 -- tN)
			// span N+1 times
			// The bundle adjustment covers two IOD runs so that the end state of the first run can be stiched
			// first run:  (tk --  tk+ 1), (tk + 1 -- tk + 2), ... , (tk + N-1 -- tk + N)
			// second run:  (tk + N --  tk + N + 1), (tk + N + 1 -- tk + N + 2), ... , (tk + 2N-1 -- tk + 2N)

			if (time_index - last_iod_epoch_index == this -> filter_arguments -> get_iod_rigid_transforms_number() && time_index != times.n_rows - 1){

				this -> save_attitude(dir + "/true",time_index,BN_true);

				if (!this -> filter_arguments -> get_use_ba() ){
					this -> save_attitude(dir + "/measured_no_BA",time_index,BN_measured);
					last_iod_epoch_index = time_index;
				}

				else{

					// std::cout << "\n-- Applying BA to successive point clouds\n";

					// this -> save_attitude(dir + "/measured_before_BA",time_index,BN_measured);

					// if (!this -> filter_arguments -> get_use_true_rigid_transforms()){

					// 	ba_test.run(M_pcs,X_pcs,R_pcs,BN_measured,mrps_LN,false);
					// }

					// std::cout << "\n-- Saving attitude...\n";

					// this -> save_attitude(dir + "/measured_after_BA",time_index,BN_measured);

					std::cout << "\n-- Estimating coverage...\n";


					std::cout << "\n-- Running IOD Finder ...\n";

					this -> run_IOD_finder(times, 
						epoch_time_index ,
						time_index, 
						mrps_LN,
						X_pcs,
						M_pcs,
						R_pcs,
						iod_state,
						epoch_state,
						final_state,
						epoch_cov,
						final_cov);



					last_iod_epoch_index = time_index;

				}


			}


			else if (time_index == times.n_rows - 1){


				if (this -> filter_arguments -> get_use_ba()){
					ba_test.set_h(0);
					ba_test.run(M_pcs,X_pcs,R_pcs,BN_measured,mrps_LN);
				}

				std::cout << " -- Saving attitude ...\n";
				this -> save_attitude(dir + "/measured_after_BA",time_index,BN_measured);

				std::cout << "\n-- Running IOD Finder ...\n";

				this -> run_IOD_finder(times,
					epoch_time_index ,
					time_index, 
					mrps_LN,
					X_pcs,
					M_pcs,
					R_pcs,
					iod_state,
					epoch_state,
					final_state,
					epoch_cov,
					final_cov);


				this -> save_rigid_transforms(dir, 
					X_pcs,
					M_pcs,
					X_pcs_true,
					M_pcs_true,
					R_pcs);

				std::cout << " -- Estimating coverage ...\n";
				PointCloud<PointNormal> global_pc;
				this -> estimate_coverage(dir +"/"+ std::to_string(time_index) + "_",&global_pc);

				std::cout << " -- Making PSR a-priori ...\n";
				ShapeModelTri<ControlPoint> psr_shape("",this -> frame_graph);

				ShapeBuilder::run_psr(&global_pc,dir,psr_shape,this -> filter_arguments);
				psr_shape.construct_kd_tree_shape();

				this -> estimated_shape_model = std::make_shared<ShapeModelBezier<ControlPoint>>(ShapeModelBezier<ControlPoint>(psr_shape,"E",this -> frame_graph));
				for (int e = 0; e < this -> estimated_shape_model -> get_NElements(); ++e){
					this -> estimated_shape_model -> get_element(e).set_owning_shape(this -> estimated_shape_model.get());
				}


				std::cout << " -- Fitting PSR a-priori ...\n";

				if (this -> filter_arguments -> get_use_bezier_shape()){
					std::cout << " --- Elevating degree ...\n";

					this -> estimated_shape_model -> elevate_degree();

					std::cout << " -- Populating mass properties ...\n";

					this -> estimated_shape_model -> populate_mass_properties_coefs_deterministics();

					std::cout << " -- Updating mass properties ...\n";

					this -> estimated_shape_model -> update_mass_properties();

					std::cout << " -- Saving both ...\n";

					this -> estimated_shape_model -> save_both(dir + "/elevated_shape");
				}

				std::cout << " -- Calling shape fitter ...\n";
				std::cout << this -> estimated_shape_model -> get_NControlPoints() << std::endl;

				ShapeFitterBezier shape_fitter(&psr_shape,
					this -> estimated_shape_model.get(),&global_pc); 
				shape_fitter.fit_shape_batch(this -> filter_arguments -> get_N_iter_shape_filter(),
					this -> filter_arguments -> get_ridge_coef());
				this -> estimated_shape_model -> update_mass_properties();	

				this -> estimated_shape_model -> save_both(dir + "/fit_shape");


				arma::vec::fixed<3> initial_spacecraft_position = - this -> LN_t0.t() * this -> estimated_shape_model -> get_center_of_mass();

				std::cout << " -- Rotating shape ...\n";


				this -> estimated_shape_model -> rotate(this -> LN_t0.t());

				std::cout << " -- Translating shape ...\n";

				this -> estimated_shape_model -> translate(initial_spacecraft_position);

				// The estimated shape should now be aligned with the true shape model

				// The estimated shape model is bary-centered 
				std::cout << " -- Updating mass properties before N_frame save ...\n";

				this -> estimated_shape_model -> update_mass_properties();	
				this -> estimated_shape_model -> save_both(dir + "/fit_shape_N_frame");

				this -> estimated_shape_model -> rotate(BN_measured.front());
				std::cout << " -- Updating mass properties before B_frame save ...\n";

				this -> estimated_shape_model -> update_mass_properties();	
				this -> estimated_shape_model -> save_both(dir + "/fit_shape_B_frame");

				BatchAttitude batch_attitude(times,M_pcs);

				// Estimating small body state
				arma::vec::fixed<6> a_priori_state;
				a_priori_state.subvec(0,2) = RBK::dcm_to_mrp(BN_measured.front());
				a_priori_state.subvec(3,5) = 4 * arma::inv(RBK::Bmat(RBK::dcm_to_mrp(BN_measured.front()))) * (RBK::dcm_to_mrp(BN_measured[1]) - RBK::dcm_to_mrp(BN_measured.front()))/(times(1) - times(0));

				batch_attitude.set_a_priori_state(a_priori_state);
				batch_attitude.set_inertia_estimate(this -> estimated_shape_model -> get_inertia());

				batch_attitude.run(R_pcs,mrps_LN);

				// The mean and standard deviation of all the measured mu's is extracted
				arma::vec mu_estimated(estimated_mu.size());

				for (int i = 0; i < estimated_mu.size(); ++i){
					mu_estimated(i) = estimated_mu[i];
				}


				this -> estimated_state = arma::zeros<arma::vec>(14);
				this -> estimated_state.subvec(0,5) = final_state.subvec(0,5);
				this -> estimated_state.subvec(6,11) = batch_attitude.get_attitude_state_history().back();
				this -> estimated_state(12) = arma::mean(mu_estimated);
				this -> estimated_state(13) = 1.4;

				this -> covariance_estimated_state = arma::zeros<arma::mat>(14,14);
				this -> covariance_estimated_state.submat(0,0,5,5) = final_cov.submat(0,0,5,5);
				this -> covariance_estimated_state.submat(6,6,11,11) = batch_attitude.get_attitude_state_covariances_history().back();
				this -> covariance_estimated_state(12,12) = std::pow(arma::stddev(mu_estimated),2);
				this -> covariance_estimated_state(13,13) = std::pow(0.1,2);


				return;

			}



		}

	}


}


void ShapeBuilder::run_iod(const arma::vec &times ,
	const std::vector<arma::vec> & X,std::string dir) {

	// std::cout << "Running the iod filter" << std::endl;

	// arma::vec X_S = arma::zeros<arma::vec>(X[0].n_rows);

	// arma::vec lidar_pos = X_S.rows(0,2);
	// arma::vec lidar_vel = X_S.rows(3,5);

	// arma::mat dcm_LB = arma::eye<arma::mat>(3, 3);
	// std::vector<RigidTransform> rigid_transforms;
	// std::vector<arma::vec> mrps_LN;
	// std::vector<arma::mat> BN_measured;
	// std::vector<arma::mat> BN_true;
	// std::vector<arma::mat> HN_true;
	// std::map<int,arma::vec> X_pcs;
	// std::map<int,arma::mat> M_pcs;

	// arma::vec iod_guess;

	// int last_iod_epoch_index = 0;
	// int cutoff_index = 0;


	// arma::mat M_pc = arma::eye<arma::mat>(3,3);
	// arma::vec X_pc = arma::zeros<arma::vec>(3);

	// for (int time_index = 0; time_index < times.n_rows; ++time_index) {

	// 	std::stringstream ss;
	// 	ss << std::setw(6) << std::setfill('0') << time_index + 1;
	// 	std::string time_index_formatted = ss.str();

	// 	std::cout << "\n################### Index : " << time_index << " / " <<  times.n_rows - 1  << ", Time : " << times(time_index) << " / " <<  times( times.n_rows - 1) << " ########################" << std::endl;

	// 	X_S = X[time_index];

	// 	this -> get_new_states(X_S,dcm_LB,lidar_pos,lidar_vel,mrps_LN,BN_true,HN_true);

	// 	// Setting the Lidar frame to its new state
	// 	this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(X_S.subvec(0,2));
	// 	this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrps_LN[time_index]);

	// 	// Setting the small body to its new attitude
	// 	this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(X_S.subvec(6,8));

	// 	// Getting the true observations (noise is added)
	// 	this -> lidar -> send_flash(this -> true_shape_model,true);

	// 	// The rigid transform best aligning the two point clouds is found
	// 	// The solution to this first registration will be used to prealign the 
	// 	// shape model and the source point cloud

	// 	this -> store_point_clouds(time_index,dir);
	// 	if (this -> destination_pc != nullptr && this -> source_pc == nullptr){
	// 		this -> all_registered_pc.push_back(this -> destination_pc);
	// 	}

	// 	else if (this -> destination_pc != nullptr && this -> source_pc != nullptr){

	// 		M_pcs[time_index] = arma::eye<arma::mat>(3,3);;
	// 		X_pcs[time_index] = arma::zeros<arma::vec>(3);


	// 		// The point-cloud to point-cloud ICP is used for point cloud registration
	// 		// This ICP can fail. If so, the update is still applied and will be fixed 
	// 		// in the bundle adjustment

	// 		try{

	// 			IterativeClosestPointToPlane icp_pc(this -> destination_pc, this -> source_pc);
	// 			icp_pc.set_save_rigid_transform(this -> LN_t0.t(),this -> x_t0);
	// 			icp_pc.register_pc(M_pc,X_pc);

	// 		// These two align the consecutive point clouds 
	// 		// in the instrument frame at t_D == t_0
	// 			M_pc = icp_pc.get_dcm();
	// 			X_pc = icp_pc.get_x();	
	// 		}
	// 		catch(ICPException & e){
	// 			std::cout << e.what() << std::endl;
	// 		}
	// 		catch(ICPNoPairsException & e){
	// 			std::cout << e.what() << std::endl;
	// 		}
	// 		catch(std::runtime_error & e){
	// 			std::cout << e.what() << std::endl;
	// 		}

	// 		arma::mat M_pc_true = this -> LB_t0 * dcm_LB.t();
	// 		arma::vec pos_in_L = - this -> frame_graph -> convert(arma::zeros<arma::vec>(3),"B","L");
	// 		arma::vec X_pc_true = M_pc_true * pos_in_L - this -> LN_t0 * this -> x_t0;

	// 		RBK::dcm_to_mrp(M_pc_true).save(dir + "/sigma_tilde_true_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);
	// 		X_pc_true.save(dir + "/X_tilde_true_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);

	// 		RBK::dcm_to_mrp(M_pc).save(dir + "/sigma_tilde_before_ba_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);
	// 		X_pc.save(dir + "/X_tilde_before_ba_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);


	// 			/****************************************************************************/
	// 			/********** ONLY FOR DEBUG: MAKES ICP USE TRUE RIGID TRANSFORMS *************/
	// 		if (!this -> filter_arguments -> get_use_icp()){
	// 			std::cout << "\t Using true rigid transform\n";
	// 			M_pc = M_pc_true;
	// 			X_pc = X_pc_true;
	// 		}
	// 			/****************************************************************************/
	// 			/****************************************************************************/

	// 		// The source pc is registered, using the rigid transform that 
	// 		// the ICP returned
	// 		PointCloudIO<PointNormal>::save_to_obj(*this -> source_pc,
	// 			dir + "/source_" + std::to_string(time_index) + ".obj",
	// 			this -> LN_t0.t(), 
	// 			this -> x_t0);


	// 		this -> source_pc -> transform(M_pc,X_pc);
	// 		this -> all_registered_pc.push_back(this -> source_pc);

	// 		#if IOFLAGS_run_iod			


	// 		PointCloudIO<PointNormal>::save_to_obj(*this -> source_pc,
	// 			dir + "/source_" + std::to_string(time_index) + ".obj",
	// 			this -> LN_t0.t(), 
	// 			this -> x_t0);

	// 		#endif

	// 		M_pcs[time_index] = M_pc;
	// 		X_pcs[time_index] = X_pc;

	// 	}

	// }

	// int final_index;
	// if (!this -> filter_arguments -> get_use_ba()){
	// 	final_index = this -> filter_arguments -> get_iod_rigid_transforms_number() - 1;

	// }


	// else{

	// 	std::cout << " -- Applying BA to successive point clouds\n";
	// 	std::vector<std::shared_ptr<PC > > pc_to_ba;

	// 	throw(std::runtime_error("not implemented yet"));

	// 	// BundleAdjuster bundle_adjuster(0, 
	// 	// 	this -> filter_arguments -> get_iod_rigid_transforms_number() - 1,
	// 	// 	M_pcs,
	// 	// 	X_pcs,
	// 	// 	BN_measured,
	// 	// 	&this -> all_registered_pc,
	// 	// 	this -> filter_arguments -> get_N_iter_bundle_adjustment(),
	// 	// 	this -> LN_t0,
	// 	// 	this -> x_t0,
	// 	// 	mrps_LN,
	// 	// 	true,
	// 	// 	previous_closure_index,
	// 	// 	0);

	// 	std::cout << " -- Running IOD after correction\n";
	// 	// final_index = previous_closure_index;

	// }

	
	// int mc_iter = this -> filter_arguments -> get_iod_mc_iter();

	// arma::mat results(7,mc_iter);
	// arma::mat crude_guesses(6,mc_iter);
	// arma::mat all_covs(7 * mc_iter,7);
	// arma::vec all_rps(mc_iter);
	// arma::vec all_rps_sds(mc_iter);


	// boost::progress_display progress(mc_iter);
	
	// #pragma omp parallel for
	// for (int i = 0; i < mc_iter; ++i){


	// 	std::map<int,arma::vec> X_pcs_noisy;
	// 	std::map<int,arma::mat> M_pcs_noisy;

	// 	arma::vec state,crude_guess;
	// 	arma::mat cov;	

	// 	this -> run_IOD_finder(state,
	// 		cov,
	// 		crude_guess,
	// 		times,
	// 		last_iod_epoch_index ,
	// 		this -> filter_arguments -> get_iod_rigid_transforms_number() - 1, 
	// 		mrps_LN,
	// 		X_pcs,
	// 		M_pcs);

	// 	results.submat(0,i,5,i) = state.subvec(0,5);
	// 	results.submat(6,i,6,i) = state(6);

	// 	OC::CartState cart_state(state.subvec(0,5),state(6));
	// 	OC::KepState kep_state = cart_state.convert_to_kep(0);

	// 	crude_guesses.col(i) = crude_guess;
	// 	all_rps(i) = kep_state.get_a() * (1 - kep_state.get_eccentricity());
	// 	arma::rowvec::fixed<7> drpdstate = IODFinder::partial_rp_partial_state(state);
	// 	all_rps_sds(i) = std::sqrt(arma::dot(drpdstate.t(),cov * drpdstate.t() ));


	// 	all_covs.rows(7 * i, 7 * i + 6) = cov;

	// 	++progress;

	// }

	// results.save(dir + "/results.txt",arma::raw_ascii);
	// all_rps.save(dir + "/all_rps.txt",arma::raw_ascii);
	// all_rps_sds.save(dir + "/all_rps_sds.txt",arma::raw_ascii);
	// crude_guesses.save(dir + "/crude_guesses.txt",arma::raw_ascii);
	// all_covs.save(dir + "/all_covs.txt",arma::raw_ascii);
	// arma::vec results_mean = arma::mean(results,1);
	// arma::mat::fixed<7,7> cov_mc = arma::zeros<arma::mat>(7,7);

	// for (unsigned int i = 0; i < results.n_cols; ++i){
	// 	cov_mc +=  (results.col(i) - results_mean) * (results.col(i) - results_mean).t();
	// }

	// cov_mc *= 1./(results.n_cols-1);
	// cov_mc.save(dir + "/cov_mc.txt",arma::raw_ascii);

}


std::shared_ptr<ShapeModelBezier< ControlPoint > > ShapeBuilder::get_estimated_shape_model() const{
	return this -> estimated_shape_model;
}




void ShapeBuilder::save_attitude(std::string prefix,int index,const std::vector<arma::mat::fixed<3,3> > & BN) const{


	arma::mat mrp_BN = arma::mat( 3,BN.size());
	for (int i = 0; i < mrp_BN.n_cols; ++i){
		mrp_BN.col(i) = RBK::dcm_to_mrp(BN.at(i));

	}

	mrp_BN.save( prefix + "_BN_" + std::to_string(index) + ".txt",arma::raw_ascii);

}


void ShapeBuilder::run_IOD_finder(const arma::vec & times,
	const int t0 ,
	const int tf, 
	const std::vector<arma::vec::fixed<3> > & mrps_LN,
	const std::map<int,arma::vec::fixed<3> > & X_pcs,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs,
	const std::map<int, arma::mat::fixed<6,6> > & R_pcs,
	OC::CartState & cart_state,
	arma::vec & epoch_state,
	arma::vec & final_state,
	arma::mat & epoch_cov,
	arma::mat & final_cov) const{

	// Forming the absolute/rigid transforms
	std::vector<RigidTransform> sequential_rigid_transforms;
	std::vector<RigidTransform> absolute_rigid_transforms;

	std::cout << "\n\tAssembling rigid transforms\n";

	std::vector<int> obs_indices;

	for (int i = 0; i < tf + 1; ++i){
		obs_indices.push_back(i);
	}

	this -> assemble_rigid_transforms_IOD(
		sequential_rigid_transforms,
		absolute_rigid_transforms,
		obs_indices,
		times,
		mrps_LN,
		X_pcs,
		M_pcs);

	// Crude IO
	std::cout << "\n\tGetting center of collected point clouds\n";

	// Get the center of the collected pcs
	// and use this as a crude guess for the very first position vector
	// This vector must obviously be expressed in the N frame
	arma::vec::fixed<3> r_start_crude = - this -> LN_t0.t() * this -> get_center_collected_pcs();


	std::cout << "r_0 before mapping forward: " << r_start_crude.t() << std::endl;

	// This position is mapped forward in time to the first start time in the IOD arc
	// This start time is t0
	for (int t = 0; t < t0; ++t){
		r_start_crude = sequential_rigid_transforms[t].M .t() * (r_start_crude + sequential_rigid_transforms[t].X);
	}

	// Applying the k-th rigid transform (index starting at 0) 
	// to the k-th position vector maps it to the (k+1)th position vector

	// Should the positions along the arc be initialized from the rigid transforms
	// or from the keplerian propagation of r_start_crude? 
	// Well, I do not have a velocity crude guess at t_start, 
	// so I have no choice but to use the rigid transforms




	std::vector<arma::vec::fixed<3> > IOD_arc_positions;
	IOD_arc_positions.push_back(r_start_crude);

	
	for (int t = t0; t < tf; ++t){
		r_start_crude = sequential_rigid_transforms[t].M .t() * (r_start_crude + sequential_rigid_transforms[t].X);
		IOD_arc_positions.push_back(r_start_crude);
	}

	// A first PSO run refines the velocity and standard gravitational parameter at the start time of 
	// the observation arc
	// This arc is comprised of rigid transforms indexed between t0 and tf inclusive

	std::vector<RigidTransform> sequential_rigid_transforms_arc;
	std::vector<RigidTransform> absolute_rigid_transforms_arc;
	std::vector<arma::vec::fixed<3> > mrps_LN_arc;

	for (auto rt : sequential_rigid_transforms){
		if (rt.index_start >= t0){
			sequential_rigid_transforms_arc.push_back(rt);
		}
	}

	for (auto rt : absolute_rigid_transforms){
		if (rt.index_end >= t0){
			absolute_rigid_transforms_arc.push_back(rt);
		}
	}

	for (int i = 0; i < mrps_LN.size(); ++i){
		if (i >= t0){
			mrps_LN_arc.push_back(mrps_LN[i]);
		}
	}

	IODFinder iod_finder(
		&sequential_rigid_transforms_arc, 
		&absolute_rigid_transforms_arc, 
		mrps_LN_arc,
		this -> filter_arguments -> get_iod_iterations(),
		this -> filter_arguments -> get_iod_particles());


	arma::vec guess = {
		IOD_arc_positions.front()(0),
		IOD_arc_positions.front()(1),
		IOD_arc_positions.front()(2),
		0,
		0,
		0,
		2
	};

	std::cout << "Initial guess on the IOD arc: " << guess.t() << std::endl;

	arma::vec lower_bounds = {
		guess(0) - 100,
		guess(1) - 100,
		guess(2) - 100,
		-1e-1,
		-1e-1,
		-1e-1,
		1
	};

	arma::vec upper_bounds = {
		guess(0) + 100,
		guess(1) + 100,
		guess(2) + 100,
		1e-1,
		1e-1,
		1e-1,
		3
	};
	

	iod_finder.run_pso(lower_bounds, upper_bounds,0,guess);
	epoch_state = iod_finder.get_result();
	final_state = arma::zeros<arma::vec>(epoch_state.n_rows);

	
	iod_finder.run_batch(epoch_state,final_state,epoch_cov,final_cov,R_pcs);
	cart_state.set_state(epoch_state.subvec(0,5));
	cart_state.set_mu(epoch_state(6));

}



void ShapeBuilder::assemble_rigid_transforms_IOD(
	std::vector<RigidTransform> & sequential_rigid_transforms,
	std::vector<RigidTransform> & absolute_rigid_transforms,
	const std::vector<int> & obs_indices,
	const arma::vec & times, 
	const std::vector<arma::vec::fixed<3> >  & mrps_LN,
	const std::map<int,arma::vec::fixed<3> > & X_pcs,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs) const{


	for (auto k : obs_indices){

		if (k != 0){

	// Adding the rigid transform. M_p_k and X_p_k represent the incremental rigid transform 
	// from t_k to t_(k-1)
			arma::mat M_p_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * M_pcs.at(k - 1).t() * M_pcs.at(k) * RBK::mrp_to_dcm(mrps_LN[k]);
			arma::vec X_p_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * M_pcs.at(k - 1).t() * (X_pcs.at(k) - X_pcs.at(k - 1));

			RigidTransform rigid_transform;
			rigid_transform.M = M_p_k;
			rigid_transform.X = X_p_k;
			rigid_transform.t_start = times(k-1);
			rigid_transform.t_end = times(k);
			rigid_transform.index_start = k - 1;
			rigid_transform.index_end = k;


			sequential_rigid_transforms.push_back(rigid_transform);

			RigidTransform rt;

			rt.t_end = times(k);
			rt.t_start = times(0);

			rt.index_end = k;
			rt.index_start = 0;	

			rt.X = X_pcs.at(k);
			rt.M = M_pcs.at(k);

			absolute_rigid_transforms.push_back(rt);

		}
		else{

			RigidTransform rt;
			rt.t_start = times(0);
			rt.t_end= times(0);

			rt.index_start = 0;
			rt.index_end= 0;

			rt.X = X_pcs.at(0);
			rt.M = M_pcs.at(0);

			absolute_rigid_transforms.push_back(rt);

		}



	}
}



void ShapeBuilder::store_point_clouds(int index,const std::string dir) {


	// No point cloud has been collected yet
	if (this -> destination_pc == nullptr) {

		PointCloud<PointNormal > pc(this -> lidar -> get_focal_plane());
		this -> destination_pc = std::make_shared<PointCloud<PointNormal>>(pc);
		this -> destination_pc -> build_kdtree (false);
		arma::vec::fixed<3> los = {1,0,0};

		EstimationNormals<PointNormal,PointNormal> estimate_normals(*this -> destination_pc,*this -> destination_pc);
		estimate_normals.set_los_dir(los);
		estimate_normals.estimate(6);

		#if IOFLAGS_shape_builder
		PointCloudIO<PointNormal>::save_to_obj(*this -> destination_pc, dir + "/destination_" + std::to_string(index) + ".obj",
			this -> LN_t0.t(),this -> x_t0);
		#endif

	}

	else {

		// Only one destination point cloud has been collected
		if (this -> source_pc == nullptr) {

			PointCloud<PointNormal > pc(this -> lidar -> get_focal_plane());
			this -> source_pc = std::make_shared<PointCloud<PointNormal>>(pc);
			this -> source_pc -> build_kdtree (false);

			arma::vec::fixed<3> los = {1,0,0};

			EstimationNormals<PointNormal,PointNormal> estimate_normals(*this -> source_pc,*this -> source_pc);
			estimate_normals.set_los_dir(los);
			estimate_normals.estimate(6);


			#if IOFLAGS_shape_builder
			PointCloudIO<PointNormal>::save_to_obj(*this -> source_pc, dir + "/source_" + std::to_string(index) + ".obj",
				this -> LN_t0.t(),this -> x_t0);
			PointCloudIO<PointNormal>::save_to_obj(*this -> destination_pc, dir + "/destination_" + std::to_string(index) + ".obj",
				this -> LN_t0.t(),this -> x_t0);
			#endif


		}

		// Two point clouds have been collected : "nominal case")
		else {

			// The source and destination point clouds are combined into the new source point cloud

			this -> destination_pc = this -> source_pc;



			PointCloud<PointNormal > pc(this -> lidar -> get_focal_plane());
			this -> source_pc = std::make_shared<PointCloud<PointNormal>>(pc);
			this -> source_pc -> build_kdtree (false);

			arma::vec::fixed<3> los = {1,0,0};

			EstimationNormals<PointNormal,PointNormal> estimate_normals(*this -> source_pc,*this -> source_pc);
			estimate_normals.set_los_dir(los);
			estimate_normals.estimate(6);


			#if IOFLAGS_shape_builder
			PointCloudIO<PointNormal>::save_to_obj(*this -> source_pc, dir + "/source_" + std::to_string(index) + ".obj",
				this -> LN_t0.t(),this -> x_t0);
			#endif

		}
	}
}


void ShapeBuilder::get_new_states(
	const arma::vec & X_S, 
	arma::mat::fixed<3,3> & dcm_LB, 
	arma::vec::fixed<3> & lidar_pos,
	arma::vec::fixed<3> & lidar_vel,
	std::vector<arma::vec::fixed<3>> & mrps_LN,
	std::vector<arma::mat::fixed<3,3> > & BN_true,
	std::vector<arma::mat::fixed<3,3> > & HN_true){

	// Getting the new small body inertial attitude
	// and spacecraft relative position expressed in the small body centered inertia frame
	lidar_pos = X_S.rows(0, 2);
	lidar_vel = X_S.rows(3, 5);

	// The [LN] DCM is assembled. Note that e_r does not exactly have to point towards the target
	// barycenter
	arma::vec e_r ;

	std::cout << "\t Possible choice for los in N frame: \n";
	std::cout << "\t\tFrom poi: " << arma::normalise(this -> lidar_to_target_of_interest_N_frame).t();
	std::cout << "\t\tNadir: " << - arma::normalise(lidar_pos).t();

	if (this -> filter_arguments -> get_use_target_poi() && mrps_LN.size() > 10){
		e_r = arma::normalise(this -> lidar_to_target_of_interest_N_frame);
		std::cout << "Pointing at POI\n";
	}
	else{
		e_r = - arma::normalise(lidar_pos);
		std::cout << "Pointing at nadir\n";

	}
	arma::vec e_h = arma::normalise(arma::cross(e_r,-lidar_vel));
	arma::vec e_t = arma::cross(e_h,e_r);

	arma::mat dcm_LN(3,3);
	arma::mat dcm_HN(3,3);

	dcm_LN.row(0) = e_r.t();
	dcm_LN.row(1) = e_t.t();
	dcm_LN.row(2) = e_h.t();

	dcm_HN.row(0) = -e_r.t();
	dcm_HN.row(1) = -e_t.t();
	dcm_HN.row(2) = e_h.t();

	arma::vec mrp_LN = RBK::dcm_to_mrp(dcm_LN);
	dcm_LB = dcm_LN * RBK::mrp_to_dcm(X_S.rows(6, 8)).t();

	HN_true.push_back(dcm_HN);
	BN_true.push_back(RBK::mrp_to_dcm(X_S.rows(6, 8)));
	mrps_LN.push_back(mrp_LN);



	if (this -> LN_t0.n_rows == 0){
		this -> LN_t0 = dcm_LN;
		this -> x_t0 = lidar_pos;
		this -> LB_t0 = dcm_LB;
		OC::CartState true_cart_state_t0(X_S.rows(0,5),this -> true_shape_model -> get_volume() * 1900 * arma::datum::G);
		this -> true_kep_state_t0 = true_cart_state_t0.convert_to_kep(0);
	}

}



arma::vec ShapeBuilder::get_center_collected_pcs(
	int first_pc_index,
	int last_pc_index,
	const std::vector<RigidTransform> & absolute_rigid_transforms,
	const std::vector<RigidTransform> & absolute_true_rigid_transforms) const{

	arma::vec center = {0,0,0};
	int N = last_pc_index - first_pc_index + 1;
	arma::vec pc_center;
	for (int i = first_pc_index; i < last_pc_index + 1; ++i){		

		pc_center = EstimationFeature<PointNormal,PointNormal>::compute_center(*this -> all_registered_pc[i]);

		center += 1./N * (absolute_rigid_transforms.at(i).M * (
			absolute_true_rigid_transforms.at(i).M.t() * (pc_center - absolute_true_rigid_transforms.at(i).X) ) + absolute_rigid_transforms.at(i).X); 
	}
	return center;
}


arma::vec ShapeBuilder::get_center_collected_pcs(
	int first_pc_index,
	int last_pc_index) const{

	arma::vec center = {0,0,0};
	int N = last_pc_index - first_pc_index + 1;
	arma::vec pc_center;

	for (int i = first_pc_index; i < last_pc_index + 1; ++i){	

		pc_center = EstimationFeature<PointNormal,PointNormal>::compute_center(*this -> all_registered_pc[i]);

		center += 1./N * pc_center ; 
	}
	return center;
}


arma::vec::fixed<3> ShapeBuilder::get_center_collected_pcs() const{

	arma::vec::fixed<3> center = {0,0,0};
	int N = this -> all_registered_pc.size();
	int N_p = 0;
	for (int i = 0; i < N; ++i){	
		center += this -> all_registered_pc[i] -> size() * EstimationFeature<PointNormal,PointNormal>::compute_center(*this -> all_registered_pc[i]) ; 
		N_p += this -> all_registered_pc[i] -> size();
	}
	return center / N_p;
}




void ShapeBuilder::estimate_coverage(std::string dir,PointCloud<PointNormal> * pc){

	std::cout << "\n-- Fetching points ...\n";

	// A PC is formed with all the registered point clouds
	PointCloud<PointNormal> global_pc;
	std::vector<int> last_pc_indices;

	for (int i = 0; i < this -> all_registered_pc.size(); ++i){
		for (int j = 0; j <  this -> all_registered_pc[i] -> size(); ++j){
			if (j == int(this -> all_registered_pc[i] -> size()) - 1){
				last_pc_indices.push_back(global_pc.size());
			}
			global_pc.push_back( this -> all_registered_pc[i] -> get_point(j));

		}
	}

	std::cout << "\n-- Number of points in global pc: " << global_pc.size() << std::endl;
	std::cout << "\n-- Building KD-tree ..." << std::endl;


	auto start = std::chrono::system_clock::now();

	// The KD tree of this pc is built
	global_pc.build_kdtree(false);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << "\n-- Done building KD-tree in " << elapsed_seconds.count( ) << " seconds" << std::endl;

	// The normals are NOT re-estimated. they come from the concatenated point clouds

	start = std::chrono::system_clock::now();

	arma::uvec S(global_pc.size());
	
	std::cout << "\n-- Computing coverage ..."<< std::endl;

	#pragma omp parallel for 
	for (int i = 0; i < global_pc.size(); ++i){

		// The closest neighbors are extracted
		// std::map<double, int > closest_points = global_pc.get_closest_N_points(global_pc.get_point_coordinates(i),3);
		std::vector<int> closest_points = global_pc.get_nearest_neighbors_radius(global_pc.get_point_coordinates(i),1.);

		S(i) = closest_points.size();
	}

	end = std::chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "\n-- Done computing coverage in " << elapsed_seconds.count( ) << " seconds" << std::endl;
	std::cout << "\n-- Finding unsatisfying points ..." << std::endl;

	arma::uvec unsatisfying_points = arma::find(S <= 3);
	
	// std::cout << "\n-- Getting POI index ..." << std::endl;

	// int POI_index = last_pc_indices[S.rows(last_pc_indices.front(),last_pc_indices.back()).index_min()];
	
	// // std::cout << "\n-- Setting target of interest from POI #..." << POI_index << std::endl;
	// // TODO: there's probably a NAN in POI_index at some point or something of the like
	// // this -> target_of_interest_L0_frame = global_pc.get_point_coordinates(POI_index);

	std::cout << "Uniformity score: " << double(S.size() - unsatisfying_points.size())/S.size() * 100 << " %\n";
	// std::cout << "Point of interest coordinates in L0 frame: " << this -> target_of_interest_L0_frame.t();

	
	if (pc != nullptr){
		*pc = global_pc;
		PointCloudIO<PointNormal>::save_to_obj(global_pc,dir + "coverage_pc.obj",this -> LN_t0.t(), this -> x_t0);
		PointCloudIO<PointNormal>::save_to_obj(global_pc,dir + "coverage_pc_as_is.obj");

	}

}


void ShapeBuilder::run_psr(PointCloud<PointNormal> * pc,
	const std::string dir,
	ShapeModelTri<ControlPoint> & psr_shape,
	ShapeBuilderArguments * filter_arguments){

	std::string pc_cgal_path = dir + "/pc_cgal.txt";
	std::string shape_cgal_path = dir + "/shape_cgal.obj";

	double percentage_point_kept = 1;

	PointCloud<PointNormal> pc_downsampled;

	std::vector<int> indices;
	for (int i = 0; i < pc -> size(); ++i){
		indices.push_back(i);
	}

	std::cout << "\tShuffling\n";
	std::random_shuffle ( indices.begin(), indices.end() );

	int points_kept = static_cast<int>(percentage_point_kept/100. * pc -> size() );

	for (int i = 0; i < points_kept; ++i){
		pc_downsampled.push_back(pc -> get_point(indices[i]));
	}

	std::cout << "\tSaving to txt\n";

	PointCloudIO<PointNormal>::save_to_txt(pc_downsampled, pc_cgal_path);

	// sm_angle = 30.0; // Min triangle angle in degrees.
	// sm_radius = 30; // Max triangle size w.r.t. point set average spacing.
	// sm_distance = 0.5; // Surface Approximation error w.r.t. point set average spacing.

	CGALINTERFACE::CGAL_interface(pc_cgal_path.c_str(), 
		shape_cgal_path.c_str(),
		filter_arguments -> get_number_of_edges(),
		filter_arguments -> get_min_triangle_angle(),
		filter_arguments -> get_max_triangle_size(),
		filter_arguments -> get_surface_approx_error());

	ShapeModelImporter::load_obj_shape_model(shape_cgal_path, 1, true,psr_shape);

}


void ShapeBuilder::extract_a_priori_transform(
	arma::mat::fixed<3,3> & M, 
	arma::vec::fixed<3> & X,
	const int index,
	const arma::vec::fixed<3> & r_k_hat,
	const arma::vec::fixed<3> & r_km1_hat,
	const std::vector<arma::mat::fixed<3,3>> & BN_measured,
	const std::map<int,arma::mat::fixed<3,3>> &  M_pcs,
	const std::map<int,arma::vec::fixed<3> > &  X_pcs,
	const std::vector<arma::vec::fixed<3> > & mrps_LN){

	// in all that follows, _k refers to the current time 

	assert(mrps_LN.size() == BN_measured.size() + 1 );// should already have collected the spacecraft attitude at this time
	assert(mrps_LN.size() == index + 1);

	// The a-priori rotation is extracted first
	arma::mat::fixed<3,3> BN_km1 = BN_measured.back();
	arma::mat::fixed<3,3> BN_km2 = *std::prev(BN_measured.end(), 2);

	// Assumes a fixed rotation axis and angular velocity
	arma::mat::fixed<3,3> BN_k_hat = (BN_km1 * BN_km2.t()) * BN_km1 ; 
	
	// not sure this spectral decomposition is needed

	arma::cx_vec eigval;
	arma::cx_mat eigvec;
	
	arma::eig_gen( eigval, eigvec, BN_k_hat );

	// This is a suitable a-priori rigid transform dcm
	M = RBK::mrp_to_dcm(mrps_LN.front()) * BN_measured.front().t() * BN_k_hat * RBK::mrp_to_dcm(mrps_LN.back()).t();

	// 

	arma::mat::fixed<3,3> Mp_k_hat = (
		RBK::mrp_to_dcm(mrps_LN[index - 1]).t() 
		* M_pcs.at(index - 1).t() 
		* M 
		* RBK::mrp_to_dcm(mrps_LN.back()));




	// This is a suitable a-priori rigid transform X
	X = M_pcs.at(index - 1) * RBK::mrp_to_dcm(mrps_LN[index - 1]) * (Mp_k_hat * r_k_hat - r_km1_hat) + X_pcs.at(index - 1);


}


void ShapeBuilder::get_best_a_priori_rigid_transform(
	arma::mat::fixed<3,3> & M_pc_a_priori,
	arma::vec::fixed<3> &  X_pc_a_priori,
	const OC::CartState & cartesian_state,
	const arma::vec & times,
	const int & time_index,
	const int & epoch_time_index,
	const std::vector<arma::mat::fixed<3,3>> & BN_measured,
	const std::map<int,arma::mat::fixed<3,3>> &  M_pcs,
	const std::map<int,arma::vec::fixed<3> > &  X_pcs,
	const std::vector<arma::vec::fixed<3> > & mrps_LN){


	OC::KepState kep_state_epoch = cartesian_state.convert_to_kep(0);

	OC::CartState cart_state_tkm1 = kep_state_epoch.convert_to_cart(times(time_index - 1) - times(epoch_time_index));
	OC::CartState cart_state_tk = kep_state_epoch.convert_to_cart(times(time_index) - times(epoch_time_index));

	arma::vec::fixed<3> r_km1_hat = cart_state_tkm1.get_position_vector();
	arma::vec::fixed<3> r_k_hat = cart_state_tk.get_position_vector();

	arma::mat::fixed<3,3> M_pc_iod = arma::eye<arma::mat>(3,3);
	arma::vec::fixed<3> X_pc_iod = arma::zeros<arma::vec>(3);
	if (!r_k_hat.has_nan() && !r_km1_hat.has_nan()){
		ShapeBuilder::extract_a_priori_transform(M_pc_iod,
			X_pc_iod,
			time_index,
			r_k_hat,
			r_km1_hat,
			BN_measured,
			M_pcs,
			X_pcs,
			mrps_LN);
	}

	// The quality of the pre-alignment is assessed by looking at which of the two predictions
	// (M_pc_iod,X_pc_iod) or (M_pc,X_pc) yields the best pairs

	// Previous rigid transform
	IterativeClosestPointToPlane icp_pc_prealign(this -> destination_pc, this -> source_pc);
	icp_pc_prealign.compute_pairs(4,M_pcs.at(time_index - 1),X_pcs.at(time_index - 1));
	
	double res_previous_rt = icp_pc_prealign.compute_residuals(M_pcs.at(time_index - 1),X_pcs.at(time_index - 1));
	int N_pairs_previous_rt = icp_pc_prealign.get_point_pairs().size();
	
	std::cout << "\t Residuals from previous rt: " << res_previous_rt << " from " << icp_pc_prealign.get_point_pairs().size()<< " pairs" << std::endl;
	std::cout << "\t Rigid transforms from previous rt: " << RBK::dcm_to_mrp(M_pcs.at(time_index - 1)).t() << " , " << X_pcs.at(time_index - 1).t() << std::endl << std::endl;

	// IOD rigid transform
	int N_pairs_iod = 0;
	double res_previous_iod = std::numeric_limits<double>::infinity();
	
	try{
		icp_pc_prealign.compute_pairs(4,M_pc_iod,X_pc_iod);
		N_pairs_iod = icp_pc_prealign.get_point_pairs().size();
		std::cout <<"\t N_pairs_iod: " << N_pairs_iod << std::endl;
		res_previous_iod = icp_pc_prealign.compute_residuals(M_pc_iod,X_pc_iod);

		std::cout << "\t Residuals from iod rt: " << res_previous_iod << " from "<< icp_pc_prealign.get_point_pairs().size()  << " pairs" << std::endl << std::endl;
		std::cout << "\t Rigid transforms from iod rt: " << RBK::dcm_to_mrp(M_pc_iod).t() << " , " << X_pc_iod.t() << std::endl << std::endl;
	}
	catch(ICPNoPairsException & e){
		e.what();
	}

	if (res_previous_rt < res_previous_iod || N_pairs_iod == 0){
		std::cout << "\t Choosing previous rt a-priori\n";

		icp_pc_prealign.clear_point_pairs();
		icp_pc_prealign.set_minimum_h(4);
		icp_pc_prealign.register_pc(this -> filter_arguments -> get_los_noise_sd_baseline(),
			M_pcs.at(std::max(0,time_index - 2)),
			M_pcs.at(time_index - 1),
			X_pcs.at(time_index - 1));

		M_pc_a_priori = icp_pc_prealign.get_dcm();
		X_pc_a_priori = icp_pc_prealign.get_x();
	}
	else{
		std::cout << "\t Choosing iod rt a-priori\n";

		M_pc_a_priori = M_pc_iod;
		X_pc_a_priori = X_pc_iod;
	}


}




void ShapeBuilder::save_rigid_transforms(std::string dir, 
	const std::map<int,arma::vec::fixed<3> > & X_pcs,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs,
	const std::map<int,arma::vec::fixed<3> > & X_pcs_true,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs_true,
	const std::map<int,arma::mat::fixed<6,6> > & R_pcs){


	assert(X_pcs.size() == this -> all_registered_pc.size() );
	assert(X_pcs.size() == M_pcs.size());
	assert(M_pcs.size() == X_pcs_true.size());
	assert(X_pcs_true.size() == M_pcs_true.size());
	assert(M_pcs_true.size() == R_pcs.size());
	assert(R_pcs.size() == X_pcs.size());


	arma::mat X_pcs_arma(static_cast<int>(X_pcs.size() - 1) ,3);
	arma::mat mrp_pcs_arma(static_cast<int>(X_pcs.size() - 1) ,3);
	arma::mat X_error_arma(static_cast<int>(X_pcs.size() - 1) ,3);

	arma::mat X_pcs_true_arma(static_cast<int>(X_pcs.size() - 1) ,3);
	arma::mat mrp_pcs_true_arma(static_cast<int>(X_pcs.size() - 1) ,3);
	arma::mat mrp_error_arma(static_cast<int>(X_pcs.size() - 1) ,3);
	arma::mat R_pcs_arma(static_cast<int>(X_pcs.size() - 1) ,36);


	for (int i = 0; i < static_cast<int>(X_pcs.size()) - 1; ++i){
		X_pcs_arma.row(i) = X_pcs.at(i + 1).t();
		mrp_pcs_arma.row(i) = RBK::dcm_to_mrp(M_pcs.at(i + 1)).t();

		X_pcs_true_arma.row(i) = X_pcs_true.at(i + 1).t();
		mrp_pcs_true_arma.row(i) = RBK::dcm_to_mrp(M_pcs_true.at(i + 1)).t();

		R_pcs_arma.row(i) = arma::vectorise(R_pcs.at(i+1)).t();

		X_error_arma.row(i) = X_pcs_arma.row(i) - X_pcs_true_arma.row(i);
		mrp_error_arma.row(i) = RBK::dcm_to_mrp(M_pcs.at(i + 1) * M_pcs_true.at(i + 1).t()).t();

	}

	X_pcs_arma.save(dir + "/X_pcs_arma.txt",arma::raw_ascii);
	mrp_pcs_arma.save(dir + "/M_pcs_arma.txt",arma::raw_ascii);
	X_pcs_true_arma.save(dir + "/X_pcs_true_arma.txt",arma::raw_ascii);
	mrp_pcs_true_arma.save(dir + "/M_pcs_true_arma.txt",arma::raw_ascii);

	X_error_arma.save(dir + "/X_error_arma.txt",arma::raw_ascii);
	mrp_error_arma.save(dir + "/mrp_error_arma.txt",arma::raw_ascii);

	R_pcs_arma.save(dir + "/R_pcs_arma.txt",arma::raw_ascii);

}

arma::mat ShapeBuilder::get_covariance_estimated_state() const{
	return this -> covariance_estimated_state;
}


arma::vec ShapeBuilder::get_estimated_state() const{
	return this -> estimated_state;
}



