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
// #include <IOFlags.hpp>
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

#define IOFLAGS_shape_builder 1

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

	std::cout << "Running the filter" << std::endl;

	arma::vec X_S = arma::zeros<arma::vec>(X[0].n_rows);

	arma::vec::fixed<3> lidar_pos = X_S.rows(0,2);
	arma::vec::fixed<3> lidar_vel = X_S.rows(3,5);

	arma::mat::fixed<3,3> dcm_LB = arma::eye<arma::mat>(3, 3);
	std::vector<RigidTransform> rigid_transforms;
	std::vector<arma::vec::fixed<3> > mrps_LN;
	std::vector<arma::mat::fixed<3,3 > > BN_measured;
	std::vector<arma::mat::fixed<3,3 > > BN_true;
	std::vector<arma::mat::fixed<3,3 > > HN_true;
	std::map<int,arma::vec::fixed<3> > X_pcs;
	std::map<int,arma::mat::fixed<3,3> > M_pcs;
	std::map<int,arma::mat::fixed<6,6> > R_pcs;

	arma::vec iod_guess;

	int last_ba_call_index = 0;
	int cutoff_index = 0;
	int epoch_time_index = 0;
	OC::CartState iod_state;

	arma::mat::fixed<3,3> M_pc = arma::eye<arma::mat>(3,3);
	arma::vec::fixed<3> X_pc = arma::zeros<arma::vec>(3);

	BundleAdjuster ba_test(&this -> all_registered_pc,this -> filter_arguments -> get_N_iter_bundle_adjustment() ,
		5,
		&this -> LN_t0,
		&this -> x_t0,
		dir);


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
			
		}

		else if (this -> destination_pc != nullptr && this -> source_pc != nullptr){


			if(this -> filter_arguments -> get_use_icp()){

				arma::mat::fixed<3,3> M_pc_a_priori;
				arma::vec::fixed<3> X_pc_a_priori;

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
				icp_pc.register_pc(M_pc_a_priori,X_pc_a_priori);

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D == t_0
				M_pc = icp_pc.get_dcm();
				X_pc = icp_pc.get_x();
				R_pcs[time_index] = icp_pc.get_R();
			}
				/****************************************************************************/
				/********** ONLY FOR DEBUG: MAKES ICP USE TRUE RIGID TRANSFORMS *************/
			if (this -> filter_arguments -> get_use_true_rigid_transforms()){
				std::cout << "MAKES ICP USE TRUE RIGID TRANSFORMS\n";
				M_pc = this -> LB_t0 * dcm_LB.t();

				arma::vec pos_in_L = - this -> frame_graph -> convert(arma::zeros<arma::vec>(3),"B","L");
				X_pc = M_pc * pos_in_L - this -> LN_t0 * this -> x_t0;
				R_pcs[time_index] = arma::eye<arma::mat>(6,6);

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

			ba_test.update_overlap_graph();
			ba_test.run(M_pcs,X_pcs,BN_measured,mrps_LN,false);
			
			this -> run_IOD_finder(times, epoch_time_index ,time_index, mrps_LN,X_pcs,M_pcs,R_pcs,iod_state);

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

			if (time_index - last_ba_call_index == this -> filter_arguments -> get_iod_rigid_transforms_number()){

				this -> save_attitude(dir + "/true",time_index,BN_true);

				if (!this -> filter_arguments -> get_use_ba() ){
					this -> save_attitude(dir + "/measured_no_BA",time_index,BN_measured);
					last_ba_call_index = time_index;
				}

				else{

					std::cout << "\n-- Applying BA to successive point clouds\n";

					this -> save_attitude(dir + "/measured_before_BA",time_index,BN_measured);

					if (!this -> filter_arguments -> get_use_true_rigid_transforms()){
						
						ba_test.run(M_pcs,X_pcs,BN_measured,mrps_LN,true);
					}

					std::cout << "\n-- Saving attitude...\n";

					this -> save_attitude(dir + "/measured_after_BA",time_index,BN_measured);

					std::cout << "\n-- Estimating coverage...\n";

					this -> estimate_coverage(dir +"/"+ std::to_string(time_index) + "_");

					std::cout << "\n-- Moving on...\n";

					std::cout << "True position : " << X_S.subvec(0,2).t();
					std::cout << "True velocity : " << X_S.subvec(3,5).t();
					std::cout << "True position at t0 : " << (this -> x_t0).t() << std::endl;

					this -> run_IOD_finder(times, epoch_time_index ,time_index, mrps_LN,X_pcs,M_pcs,R_pcs,iod_state);

					last_ba_call_index = time_index;
				}
			}


			else if (this -> filter_arguments -> get_use_ba() && time_index == times.n_rows - 1){

				std::cout << " -- Applying BA to whole point cloud batch\n";
				this -> save_attitude(dir + "/measured_before_BA",time_index,BN_measured);

				ba_test.run(
					M_pcs,
					X_pcs,
					BN_measured,
					mrps_LN,
					true);

				std::cout << " -- Saving attitude...\n";
				this -> save_attitude(dir + "/measured_after_BA",time_index,BN_measured);

				std::cout << " -- Estimating coverage...\n";
				PointCloud<PointNormal> global_pc;
				this -> estimate_coverage(dir +"/"+ std::to_string(time_index) + "_",&global_pc);

				std::cout << " -- Making PSR a-priori...\n";
				ShapeModelTri<ControlPoint> psr_shape("",this -> frame_graph);
				ShapeBuilder::run_psr(&global_pc,dir,psr_shape,this -> filter_arguments);
				psr_shape.construct_kd_tree_shape();

				std::cout << " -- Fitting PSR a-priori...\n";
				ShapeModelBezier<ControlPoint> bezier_shape(psr_shape,"E",this -> frame_graph);
				bezier_shape.elevate_degree();
				bezier_shape.save_to_obj(dir + "/elevated_shape.obj");
				ShapeFitterBezier shape_fitter(&psr_shape,&bezier_shape,&global_pc); 
				shape_fitter.fit_shape_batch(this -> filter_arguments -> get_N_iter_shape_filter(),
					this -> filter_arguments -> get_ridge_coef());

				bezier_shape.save_to_obj(dir + "/fit_shape.obj");

				throw(std::runtime_error("not implemented yet"));
			}

			#if IOFLAGS_shape_builder
			PointCloudIO<PointNormal>::save_to_obj(*this -> source_pc,
				"../output/pc/source_" + std::to_string(time_index) + ".obj",
				this -> LN_t0.t(), 
				this -> x_t0);
			#endif

			if (time_index == times.n_rows - 1 || !this -> filter_arguments -> get_use_icp()){
				std::cout << "- Initializing shape model" << std::endl;

				this -> initialize_shape(cutoff_index);
				this -> estimated_shape_model -> save(dir + "/fit_source_" + std::to_string(time_index)+ ".b");

				arma::vec center_of_mass = this -> estimated_shape_model -> get_center_of_mass();

			// The estimated shape model is bary-centered 
				this -> estimated_shape_model -> shift_to_barycenter();
				this -> estimated_shape_model -> update_mass_properties();			
				this -> estimated_shape_model -> shift_to_barycenter();
				this -> estimated_shape_model -> update_mass_properties();

			// After being bary-centered, its inertial attitude is set
			// Its coordinates are still expressed in the L0 frame
			// I want them in the B frame 
			// So need to apply the following transform:
			// [BN](0)[NL](0)

				arma::mat dcm = BN_measured.front() * this -> LN_t0.t();
				this -> estimated_shape_model -> rotate(dcm);
				this -> estimated_shape_model -> update_mass_properties();

			// The measured states are saved
			// as they will be provided to the navigation filter
			// as a-priori


			// The final position is obtained from inverting the rigid transforms,
			// using the computed position of the center of mass in the stitching frame

				double dt = times(times.n_rows - 1) - times(times.n_rows - 2);

				arma::vec final_pos = RBK::mrp_to_dcm(mrps_LN.back()).t() * M_pcs[M_pcs.size() - 1].t() * (X_pcs[X_pcs.size() - 1]- center_of_mass);
				arma::vec final_pos_before = RBK::mrp_to_dcm(mrps_LN[mrps_LN.size() - 2]).t() * M_pcs[M_pcs.size() - 2].t() * (X_pcs[X_pcs.size() - 2] - center_of_mass);

				arma::vec final_vel = (final_pos - final_pos_before) / dt;


			// the final angular velocity is obtained by finite differencing
			// of the successive BAed (or not!) attitude measurements

				arma::vec sigma_final = RBK::dcm_to_mrp(BN_measured.back());
				arma::vec sigma_final_before = RBK::dcm_to_mrp(BN_measured[BN_measured.size() -2]);

				arma::vec dmrp = sigma_final - sigma_final_before;

			// dmrp/dt == 1/4 Bmat(sigma_before) * omega
				arma::vec omega_final = 4./dt * arma::inv(RBK::Bmat(sigma_final_before)) * (sigma_final - sigma_final_before);

				this -> filter_arguments -> set_position_final(final_pos);
				this -> filter_arguments -> set_velocity_final(final_vel);

				this -> filter_arguments -> set_mrp_EN_final(sigma_final);
				this -> filter_arguments -> set_omega_EN_final(omega_final);


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

	// int last_ba_call_index = 0;
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
	// 		last_ba_call_index ,
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
	OC::CartState & cart_state) const{


	// Forming the absolute/rigid transforms
	std::vector<RigidTransform> sequential_rigid_transforms;
	std::vector<RigidTransform> absolute_rigid_transforms;

	std::cout << "\n\tAssembling rigid transforms\n";

	std::vector<int> obs_indices;

	for (int i = t0; i < tf + 1; ++i){
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

	arma::mat crude_positions(3,IOD_arc_positions.size());
	for (int i = 0; i < IOD_arc_positions.size(); ++i){
		crude_positions.col(i) = IOD_arc_positions[i];
	}

	crude_positions.save("crude_positions.txt",arma::raw_ascii);

	// A first PSO run refines the velocity and standard gravitational parameter at the start time of 
	// the observation arc

	IODFinder iod_finder(&sequential_rigid_transforms, &absolute_rigid_transforms, mrps_LN,
		this -> filter_arguments -> get_iod_iterations(),this -> filter_arguments -> get_iod_particles());


	arma::vec guess = {
		IOD_arc_positions.front()(0),
		IOD_arc_positions.front()(1),
		IOD_arc_positions.front()(2),
		0,
		0,
		0,
		2
	};

	arma::vec lower_bounds = {
		guess(0) - 10,
		guess(1) - 10,
		guess(2) - 10,
		-1e-1,
		-1e-1,
		-1e-1,
		1
	};

	arma::vec upper_bounds = {
		guess(0) + 10,
		guess(1) + 10,
		guess(2) + 10,
		1e-1,
		1e-1,
		1e-1,
		3
	};
	

	iod_finder.run_pso(lower_bounds, upper_bounds,1,guess);
	arma::vec state = iod_finder.get_result();
	arma::mat cov;
	iod_finder.run_batch(state,cov,R_pcs);

	cart_state.set_state(state.subvec(0,5));
	cart_state.set_mu(state(6));
}






void ShapeBuilder::assemble_rigid_transforms_IOD(
	std::vector<RigidTransform> & sequential_rigid_transforms,
	std::vector<RigidTransform> & absolute_rigid_transforms,
	std::vector<RigidTransform> & absolute_true_rigid_transforms,
	const arma::vec & times, 
	const int t0_index,
	const int tf_index,
	const std::vector<arma::vec::fixed<3> >  & mrps_LN,
	const std::map<int,arma::vec::fixed<3> > & X_pcs,
	const std::map<int,arma::mat::fixed<3,3> > & M_pcs) const{

	throw("not implemented");

	// RigidTransform rt;
	// rt.t_k = times(0);
	// rt.X = X_pcs.at(0);
	// rt.M = M_pcs.at(0);

	// absolute_rigid_transforms.push_back(rt);
	// absolute_true_rigid_transforms.push_back(rt);


	// std::map<int,arma::vec> X_pcs_noisy;
	// std::map<int,arma::mat> M_pcs_noisy;


	// X_pcs_noisy[0] = arma::zeros<arma::vec>(3);
	// M_pcs_noisy[0] = arma::eye<arma::mat>(3,3);

	// for (int k = 1; k < X_pcs.size(); ++k){
	// 	X_pcs_noisy[k] = X_pcs.at(k) + this -> filter_arguments -> get_rigid_transform_noise_sd("X") * arma::randn<arma::vec>(3);
	// 	M_pcs_noisy[k] = M_pcs.at(k) * RBK::mrp_to_dcm(this -> filter_arguments -> get_rigid_transform_noise_sd("sigma") * arma::randn<arma::vec>(3));
	// }

	// for (int k = t0_index ; k <=  tf_index; ++ k){

	// 	if (k != 0){

	// // Adding the rigid transform. M_p_k and X_p_k represent the incremental rigid transform 
	// // from t_k to t_(k-1)
	// 		arma::mat M_p_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * M_pcs_noisy.at(k - 1).t() * M_pcs_noisy.at(k) * RBK::mrp_to_dcm(mrps_LN[k]);
	// 		arma::vec X_p_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * M_pcs_noisy.at(k - 1).t() * (X_pcs_noisy.at(k) - X_pcs_noisy.at(k - 1));



	// 		RigidTransform rigid_transform;
	// 		rigid_transform.M = M_p_k;
	// 		rigid_transform.X = X_p_k;
	// 		rigid_transform.t_k = times(k - t0_index);
	// 		sequential_rigid_transforms.push_back(rigid_transform);

	// 		RigidTransform rt;
	// 		rt.t_k = times(k - t0_index);
	// 		rt.X = X_pcs_noisy[k];
	// 		rt.M = M_pcs_noisy[k];

	// 		absolute_rigid_transforms.push_back(rt);

	// 		RigidTransform rt_true;
	// 		rt_true.t_k = times(k - t0_index);
	// 		rt_true.X = X_pcs.at(k);
	// 		rt_true.M = M_pcs.at(k);

	// 		absolute_true_rigid_transforms.push_back(rt_true);


	// 	}
	// }
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
			PointCloudIO<PointNormal>::save_to_obj(*this -> destination_pc, dir + "/destination_" + std::to_string(index - 1) + ".obj",
				this -> LN_t0.t(),this -> x_t0);
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
	arma::vec e_r = - arma::normalise(lidar_pos);
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




void ShapeBuilder::initialize_shape(unsigned int cutoff_index){

	throw(std::runtime_error("not implemented yet"));

	// std::string pc_path = "../output/pc/source_transformed_poisson.cgal";
	// std::string pc_path_obj = "../output/pc/source_transformed_poisson.obj";
	// std::string a_priori_path = "../output/shape_model/apriori.obj";
	// std::string pc_aligned_path_obj = "../output/pc/source_aligned_poisson.obj";
	// std::shared_ptr<PC> destination_pc_concatenated;


	// if (this -> filter_arguments -> get_use_icp()){

	// // The point clouds are bundle-adjusted
	// 	std::vector<std::shared_ptr< PC>> kept_pcs;

	// 	if (this -> filter_arguments -> get_use_ba()){

	// 		// Only the point clouds that looped with the first one are kept

	// 		std::cout << " - Keeping all pcs until # " << cutoff_index + 1<< " over a total of " << this -> all_registered_pc.size() << std::endl;

	// 		for(int pc = 0; pc <= cutoff_index; ++pc){
	// 			std::cout << "keeping pc " << pc << " / " << cutoff_index << std::endl;
	// 			kept_pcs.push_back(this -> all_registered_pc.at(pc));
	// 		}


	// 	}
	// 	else{
	// 		kept_pcs = this -> all_registered_pc;
	// 	}

	// 	std::cout << "-- Constructing point cloud...\n";
	// 	std::shared_ptr<PC> pc_before_ba = std::make_shared<PC>(PC(kept_pcs,this -> filter_arguments -> get_points_retained()));

	// 	pc_before_ba -> save("../output/pc/source_transformed_before_ba.obj",this -> LN_t0.t(),this -> x_t0);



	// 	destination_pc_concatenated = std::make_shared<PC>(PC(kept_pcs,this -> filter_arguments -> get_points_retained()));

	// 	destination_pc_concatenated -> save(
	// 		pc_path, 
	// 		arma::eye<arma::mat>(3,3), 
	// 		arma::zeros<arma::vec>(3), 
	// 		true,
	// 		false);



	// 	destination_pc_concatenated -> save(
	// 		pc_path_obj, 
	// 		arma::eye<arma::mat>(3,3), 
	// 		arma::zeros<arma::vec>(3), 
	// 		false,
	// 		true);



	// 	// The concatenated point cloud is saved after being transformed so as to "overlap" with the true shape. It
	// 	// should perfectly overlap without noise and bundle-adjustment/ICP errors


	// 	destination_pc_concatenated -> save(pc_aligned_path_obj, this -> LN_t0.t(),this -> x_t0);


	// }

	// else{

	// 	arma::mat points,normals;

	// 	this -> true_shape_model -> random_sampling(this -> filter_arguments -> get_points_retained(),points,normals);

	// 	destination_pc_concatenated = std::make_shared<PC>(PC(points,normals));

	// 	destination_pc_concatenated -> save(
	// 		pc_path, 
	// 		arma::eye<arma::mat>(3,3), 
	// 		arma::zeros<arma::vec>(3), 
	// 		true,
	// 		false);

	// 	destination_pc_concatenated -> save(
	// 		pc_path_obj, 
	// 		arma::eye<arma::mat>(3,3), 
	// 		arma::zeros<arma::vec>(3), 
	// 		false,
	// 		true);


	// }


	// std::cout << "-- Running PSR...\n";
	// CGALINTERFACE::CGAL_interface(pc_path.c_str(),a_priori_path.c_str(),this -> filter_arguments -> get_N_edges());



	// ShapeModelTri a_priori_obj("", nullptr);
	// ShapeModelImporter::load_obj_shape_model(a_priori_path, 1, true,a_priori_obj);




	// std::shared_ptr<ShapeModelBezier< ControlPoint > > a_priori_bezier = std::make_shared<ShapeModelBezier>(ShapeModelBezier(&a_priori_obj,"E", this -> frame_graph));

	// // the shape is elevated to the prescribed degree
	// unsigned int starting_degree = a_priori_bezier -> get_degree();
	// for (unsigned int i = starting_degree; i < this -> filter_arguments -> get_shape_degree(); ++i){
	// 	a_priori_bezier -> elevate_degree();
	// }

	// a_priori_bezier -> initialize_index_table();
	// a_priori_bezier -> save_both("../output/shape_model/a_priori_bezier");

	// ShapeFitterBezier shape_fitter(a_priori_bezier.get(),destination_pc_concatenated.get());

	// shape_fitter.fit_shape_batch(this -> filter_arguments -> get_N_iter_shape_filter(),this -> filter_arguments -> get_ridge_coef());

	// a_priori_bezier -> save_both("../output/shape_model/fit_a_priori");

	// // The estimated shape model is finally initialized
	// this -> estimated_shape_model = a_priori_bezier;
	// this -> estimated_shape_model -> update_mass_properties();

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
	for (int i = 0; i < N; ++i){	
		center += 1./N * EstimationFeature<PointNormal,PointNormal>::compute_center(*this -> all_registered_pc[i]) ; 
	}
	return center;
}




void ShapeBuilder::estimate_coverage(std::string dir,PointCloud<PointNormal> * pc) const{

	std::cout << "\n-- Fetching points ...\n";

	// A PC is formed with all the registered point clouds
	PointCloud<PointNormal> global_pc;

	for (int i = 0; i < this -> all_registered_pc.size(); ++i){
		for (int j = 0; j <  this -> all_registered_pc[i] -> size(); ++j){
			global_pc.push_back( this -> all_registered_pc[i] -> get_point(j));
		}
	}

	std::cout << "\n-- Number of points in global pc: " << global_pc.size() << std::endl;

	// The KD tree of this pc is built
	global_pc.build_kdtree(false);


	// The normals are NOT re-estimated. they come from the concatenated point clouds


	// For each point in this global pc, its N nearest neighbors are searched
	// The diameter of the search area defines the ... area Si associated with its normal
	// So for each point p_i of normal n_i, to which a neighborhood area Si is assigned
	// we define an approximated coverage criterion as
	// 
	// 			 ||sum(Si * n_i)||^2
	// eta = 1 - ----------------
	// 				(sum(Si))^2
	// 
	// which is a consistent criterion in the event of a perfect, uniform sampling
	// along with the exact surface normals. when eta ~ 1, coverage is complete

	arma::vec S(global_pc.size());
	S.fill(0);
	double sum_S = 0;
	double max_S = -1;
	double Sx = 0;
	double Sy = 0;
	double Sz = 0;


	#pragma omp parallel for reduction(+:sum_S,Sx,Sy,Sz), reduction(max: max_S)
	for (int i = 0; i < global_pc.size(); ++i){

		// The closest neighbors are extracted
		std::map<double, int > closest_points = global_pc.get_closest_N_points(global_pc.get_point_coordinates(i),6);
		double surface = std::pow((--closest_points.end()) -> first,2) * arma::datum::pi;

		// S is defined as pi * r_mean ^ 2

		S(i) = std::max(surface,arma::datum::pi * std::pow(3 ,2));

		// the surface normal sum is incremented
		Sx  += S(i) * global_pc.get_normal_coordinates(i)(0);
		Sy  += S(i) * global_pc.get_normal_coordinates(i)(1);
		Sz  += S(i) * global_pc.get_normal_coordinates(i)(2);
		
		sum_S += S(i);
		max_S = std::max(max_S,S(i));
	}

	// The coverage criterion is evaluated
	std::cout << "\n-- Max sampling radius : " << std::sqrt(max_S) << std::endl;
	std::cout << "\n-- Missing surface (%) : " << 100 * std::sqrt(Sx * Sx + Sy * Sy + Sz * Sz) / sum_S;


	PointCloudIO<PointNormal>::save_to_obj(global_pc,dir + "coverage_pc.obj",this -> LN_t0.t(), this -> x_t0);
	PointCloudIO<PointNormal>::save_to_obj(global_pc,dir + "coverage_pc_as_is.obj");


	if (pc != nullptr){
		*pc = global_pc;
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


	// icp_pc_prealign.set_minimum_h(4);
	// 	icp_pc_prealign.register_pc(M_pcs.at(time_index - 1),X_pcs.at(time_index - 1));

	// 	M_pc_a_priori = icp_pc_prealign.get_dcm();
	// 	X_pc_a_priori = icp_pc_prealign.get_x();
	// 	std::cout << "\t Choosing previous rt a-priori\n";

	OC::KepState kep_state_epoch = cartesian_state.convert_to_kep(0);


	OC::CartState cart_state_tk = kep_state_epoch.convert_to_cart(times(time_index) - times(epoch_time_index));
	OC::CartState cart_state_tkm1 = kep_state_epoch.convert_to_cart(times(time_index - 1) - times(epoch_time_index));


	arma::vec::fixed<3> r_k_hat = cart_state_tk.get_position_vector();
	arma::vec::fixed<3> r_km1_hat = cart_state_tkm1.get_position_vector();


	arma::mat::fixed<3,3> M_pc_iod;
	arma::vec::fixed<3> X_pc_iod;

	ShapeBuilder::extract_a_priori_transform(M_pc_iod,
		X_pc_iod,
		time_index,
		r_k_hat,
		r_km1_hat,
		BN_measured,
		M_pcs,
		X_pcs,
		mrps_LN);

	// The quality of the pre-alignment is assessed by looking at which of the two predictions
	// (M_pc_iod,X_pc_iod) or (M_pc,X_pc) yields the best pairs

	// Previous rigid transform
	IterativeClosestPoint icp_pc_prealign(this -> destination_pc, this -> source_pc);
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
		res_previous_iod = icp_pc_prealign.compute_residuals(M_pc_iod,X_pc_iod);
		N_pairs_iod = icp_pc_prealign.get_point_pairs().size();

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
		icp_pc_prealign.register_pc(M_pcs.at(time_index - 1),X_pcs.at(time_index - 1));

		M_pc_a_priori = icp_pc_prealign.get_dcm();
		X_pc_a_priori = icp_pc_prealign.get_x();
	}
	else{
		std::cout << "\t Choosing iod rt a-priori\n";

		M_pc_a_priori = M_pc_iod;
		X_pc_a_priori = X_pc_iod;
	}


}





