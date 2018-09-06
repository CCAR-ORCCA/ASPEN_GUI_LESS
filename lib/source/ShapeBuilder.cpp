#include "ShapeBuilder.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelBezier.hpp"

#include "Lidar.hpp"
#include "FrameGraph.hpp"
#include "ShapeBuilderArguments.hpp"
#include "PC.hpp"
#include "ICP.hpp"
#include "BundleAdjuster.hpp"
#include "CustomException.hpp"
#include "ControlPoint.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeFitterBezier.hpp"
#include "IOFlags.hpp"
#include "IODFinder.hpp"
#include <CGAL_interface.hpp>
#include <RigidBodyKinematics.hpp>
#include "IODBounds.hpp"
#include <boost/progress.hpp>
#include <chrono>
#include <cmath>

ShapeBuilder::ShapeBuilder(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model,
	ShapeBuilderArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> filter_arguments = filter_arguments;
}

ShapeBuilder::ShapeBuilder(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
}

void ShapeBuilder::run_shape_reconstruction(const arma::vec &times ,
	const std::vector<arma::vec> & X,
	bool save_shape_model) {


	std::cout << "Running the filter" << std::endl;

	arma::vec X_S = arma::zeros<arma::vec>(X[0].n_rows);
	arma::mat longitude_latitude = arma::zeros<arma::mat>( times.n_rows, 2);
	arma::mat true_longitude_latitude = arma::zeros<arma::mat>( times.n_rows, 2);


	arma::vec lidar_pos = X_S.rows(0,2);
	arma::vec lidar_vel = X_S.rows(3,5);

	arma::mat dcm_LB = arma::eye<arma::mat>(3, 3);
	std::vector<RigidTransform> rigid_transforms;
	std::vector<arma::vec> mrps_LN;
	std::vector<arma::mat> BN_measured;
	std::vector<arma::mat> BN_true;
	std::vector<arma::mat> HN_true;
	std::map<int,arma::vec> X_pcs;
	std::map<int,arma::mat> M_pcs;

	arma::vec iod_guess;


	int last_ba_call_index = 0;
	int last_IOD_epoch_index = 0;
	int cutoff_index = 0;
	int previous_closure_index = 0;


	arma::mat M_pc = arma::eye<arma::mat>(3,3);
	arma::vec X_pc = arma::zeros<arma::vec>(3);

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

		this -> store_point_clouds(time_index);

		if (this -> destination_pc != nullptr && this -> source_pc == nullptr){
			this -> all_registered_pc.push_back(this -> destination_pc);
			longitude_latitude.row(time_index) = arma::zeros<arma::rowvec>(2);

			#if IOFLAGS_shape_builder
			this -> destination_pc -> save("../output/pc/source_" + std::to_string(0) + ".obj",this -> LN_t0.t(),this -> x_t0);
			#endif

			// It is legit to use the true attitude state at the first timestep
			// as it just defines an offset
			BN_measured.push_back(RBK::mrp_to_dcm(X_S.subvec(6,8)));

			M_pcs[time_index] = arma::eye<arma::mat>(3,3);;
			X_pcs[time_index] = arma::zeros<arma::vec>(3);
			
		}

		if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {

			// The point-cloud to point-cloud ICP is used for point cloud registration
			// This ICP can fail. If so, the update is still applied and will be fixed 
			// in the bundle adjustment
			double longitude,latitude;
			
			ICP icp_pc(this -> destination_pc, this -> source_pc, M_pc, X_pc);

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D == t_0
			M_pc = icp_pc.get_M();
			X_pc = icp_pc.get_X();

				/****************************************************************************/
				/********** ONLY FOR DEBUG: MAKES ICP USE TRUE RIGID TRANSFORMS *************/
			if (!this -> filter_arguments-> get_use_ba()){

				M_pc = this -> LB_t0 * dcm_LB.t();

				arma::vec pos_in_L = - this -> frame_graph -> convert(arma::zeros<arma::vec>(3),"B","L");
				X_pc = M_pc * pos_in_L - this -> LN_t0 * this -> x_t0;
			}
				/****************************************************************************/
				/****************************************************************************/


			// The measured BN dcm is saved
			// using the ICP measurement
			// M_pc(k) is [LB](t_0) * [BL](t_k) = [LN](t_0)[NB](t_0) * [BN](t_k) * [NL](t_k);
			BN_measured.push_back( BN_measured.front() * this -> LN_t0.t() *  M_pc * RBK::mrp_to_dcm(mrps_LN[time_index]));

			// 		// The spacecraft longitude/latitude is computed from the estimated keplerian state
			// 	for (int i = last_IOD_epoch_index; i <= time_index; ++i){

			// 			/******************
			// 			** ESTIMATED STATE*
			// 			*******************/
			// 		double dt = times(i) - times(last_IOD_epoch_index);
			// 		arma::vec u_H = {1,0,0};

			// 		double f = OC::State::f_from_M(est_kep_state.get_M0() + est_kep_state.get_n() * dt,est_kep_state.get_eccentricity());
			// 		arma::mat DCM_HN = RBK::M3(est_kep_state.get_omega() + f) * RBK::M1(est_kep_state.get_inclination()) * RBK::M3(est_kep_state.get_Omega());
			// 		arma::vec u_B = BN_measured[i] * DCM_HN.t() * u_H;

			// 		longitude = 180. / arma::datum::pi * std::atan2(u_B(1),u_B(0));
			// 		latitude = 180. / arma::datum::pi * std::atan(u_B(2)/arma::norm(u_B.subvec(0,1)));
			// 		arma::rowvec long_lat = {longitude,latitude};
			// 		longitude_latitude.row(i) = long_lat;


			// 		this -> fly_over_map.add_label(i,longitude,latitude);


			// 			/*******************
			// 			**** TRUE STATE ****
			// 			********************/
			// 		double true_f = OC::State::f_from_M(this -> true_kep_state_t0.get_M0() + this -> true_kep_state_t0.get_n() * dt,this -> true_kep_state_t0.get_eccentricity());
			// 		arma::mat DCM_HN_true = RBK::M3(this -> true_kep_state_t0.get_omega() + true_f) * RBK::M1(this -> true_kep_state_t0.get_inclination()) * RBK::M3(this -> true_kep_state_t0.get_Omega());

			// 		arma::vec u_B_true = BN_true[i] * DCM_HN_true.t() * u_H;

			// 		double true_longitude = 180. / arma::datum::pi * std::atan2(u_B_true(1),u_B_true(0));
			// 		double true_latitude = 180. / arma::datum::pi * std::atan(u_B_true(2)/arma::norm(u_B_true.subvec(0,1)));

			// 		arma::rowvec true_long_lat = {true_longitude,true_latitude};

			// 		true_longitude_latitude.row(i) = true_long_lat;


			// 	}

			// 	OC::CartState true_cart_state_t0(X_S.rows(0,5),this -> true_shape_model -> get_volume() * 1900 * arma::datum::G);
			// 	this -> true_kep_state_t0 = true_cart_state_t0.convert_to_kep(0);

			// 	// A guess particle is formed by assuming that the keplerian state will be unchanged.
			// 	// The M0 is adjusted to the time of the next epoch
			// 	iod_guess = est_particle;
			// 	iod_guess(5) += est_kep_state.get_n() * (times(time_index) - times(last_IOD_epoch_index));


			// 	longitude_latitude.save("../output/maps/longitude_latitude_IOD_" +std::to_string(time_index) +  ".txt",arma::raw_ascii);
			// 	true_longitude_latitude.save("../output/maps/true_longitude_latitude_IOD_" +std::to_string(time_index) +  ".txt",arma::raw_ascii);

			// 	// The proper containers and indices are reset
			// 	last_IOD_epoch_index = time_index;
			// 	rigid_transforms.clear();

			// }

			// The source pc is registered, using the rigid transform that 
			// the ICP returned
			this -> source_pc -> transform(M_pc,X_pc);
			this -> all_registered_pc.push_back(this -> source_pc);

			M_pcs[time_index] = M_pc;
			X_pcs[time_index] = X_pc;

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
			
			if (this -> filter_arguments -> get_use_ba() && 
				time_index - last_ba_call_index == this -> filter_arguments -> get_iod_rigid_transforms_number()){
				
				// this -> save_true_ground_track(BN_true,HN_true);
				// std::cout << " -- Running IOD before correction\n";


				// OC::KepState estimated_state = this -> run_IOD_finder(times,
				// 	last_ba_call_index ,
				// 	time_index, 
				// 	mrps_LN,
				// 	X_pcs,
				// 	M_pcs);


				// this -> save_estimated_ground_track(
				// 	"estimated_lat_long_before.txt",
				// 	times,
				// 	last_ba_call_index ,
				// 	time_index, 
				// 	estimated_state,
				// 	BN_measured);


				std::cout << " -- Applying BA to successive point clouds\n";
			std::vector<std::shared_ptr<PC > > pc_to_ba;


			this -> save_attitude("measured_before_BA",time_index,BN_measured);
			this -> save_attitude("true",time_index,BN_true);

			BundleAdjuster bundle_adjuster(0, 
				time_index,
				M_pcs,
				X_pcs,
				BN_measured,
				&this -> all_registered_pc,
				this -> filter_arguments -> get_N_iter_bundle_adjustment(),
				this -> LN_t0,
				this -> x_t0,
				mrps_LN,
				false,
				previous_closure_index);

			this -> save_attitude("measured_after_BA",time_index,BN_measured);

			std::cout << " -- Running IOD after correction\n";

				// estimated_state =  this -> run_IOD_finder(times,
				// 	last_ba_call_index ,
				// 	time_index, 
				// 	mrps_LN,
				// 	X_pcs,
				// 	M_pcs);



			last_ba_call_index = time_index;
		}

		else if (this -> filter_arguments -> get_use_ba() && time_index == times.n_rows - 1){

			std::cout << " -- Applying BA to whole point cloud batch\n";
			std::vector<std::shared_ptr<PC > > pc_to_ba;
			int ground_index = 0;
			
			BundleAdjuster bundle_adjuster(0, 
				time_index,
				M_pcs,
				X_pcs,
				BN_measured,
				&this -> all_registered_pc,
				this -> filter_arguments -> get_N_iter_bundle_adjustment(),
				this -> LN_t0,
				this -> x_t0,
				mrps_LN,
				true,
				ground_index);

			cutoff_index = bundle_adjuster.get_cutoff_index();

			std::cout << " -- Running IOD after correction\n";

		}


			#if IOFLAGS_shape_builder
		this -> source_pc -> save("../output/pc/source_" + std::to_string(time_index) + ".obj",this -> LN_t0.t(),this -> x_t0);
			#endif



		if (time_index == times.n_rows - 1 || !this -> filter_arguments -> get_use_icp()){
			std::cout << "- Initializing shape model" << std::endl;

			this -> initialize_shape(cutoff_index,longitude_latitude);
			this -> estimated_shape_model -> save("../output/shape_model/fit_source_" + std::to_string(time_index)+ ".b");
			

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
	const std::vector<arma::vec> & X) {


	std::cout << "Running the iod filter" << std::endl;

	arma::vec X_S = arma::zeros<arma::vec>(X[0].n_rows);

	arma::vec lidar_pos = X_S.rows(0,2);
	arma::vec lidar_vel = X_S.rows(3,5);

	arma::mat dcm_LB = arma::eye<arma::mat>(3, 3);
	std::vector<RigidTransform> rigid_transforms;
	std::vector<arma::vec> mrps_LN;
	std::vector<arma::mat> BN_measured;
	std::vector<arma::mat> BN_true;
	std::vector<arma::mat> HN_true;
	std::map<int,arma::vec> X_pcs;
	std::map<int,arma::mat> M_pcs;

	arma::vec iod_guess;

	int last_ba_call_index = 0;
	int last_IOD_epoch_index = 0;
	int cutoff_index = 0;
	int previous_closure_index = 0;


	arma::mat M_pc = arma::eye<arma::mat>(3,3);
	arma::vec X_pc = arma::zeros<arma::vec>(3);

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

		this -> store_point_clouds(time_index);

		if (this -> destination_pc != nullptr && this -> source_pc == nullptr){
			this -> all_registered_pc.push_back(this -> destination_pc);

			#if IOFLAGS_run_iod
			this -> destination_pc -> save("../output/pc/source_" + std::to_string(0) + ".obj",this -> LN_t0.t(),this -> x_t0);
			#endif

			M_pcs[time_index] = arma::eye<arma::mat>(3,3);;
			X_pcs[time_index] = arma::zeros<arma::vec>(3);
			
		}


		if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {

			// The point-cloud to point-cloud ICP is used for point cloud registration
			// This ICP can fail. If so, the update is still applied and will be fixed 
			// in the bundle adjustment
			
			try{
				ICP icp_pc(this -> destination_pc, this -> source_pc, M_pc, X_pc,false,this -> LN_t0.t(),this -> x_t0);

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D == t_0
				M_pc = icp_pc.get_M();
				X_pc = icp_pc.get_X();	
			}
			catch(ICPException & e){
				std::cout << e.what() << std::endl;
			}
			catch(ICPNoPairsException & e){
				std::cout << e.what() << std::endl;
			}

			arma::mat M_pc_true = this -> LB_t0 * dcm_LB.t();
			arma::vec pos_in_L = - this -> frame_graph -> convert(arma::zeros<arma::vec>(3),"B","L");
			arma::vec X_pc_true = M_pc_true * pos_in_L - this -> LN_t0 * this -> x_t0;

			RBK::dcm_to_mrp(M_pc_true).save("../output/transforms/sigma_tilde_true_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);
			X_pc_true.save("../output/transforms/X_tilde_true_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);
			
			RBK::dcm_to_mrp(M_pc).save("../output/transforms/sigma_tilde_before_ba_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);
			X_pc.save("../output/transforms/X_tilde_before_ba_" + std::to_string(time_index ) + ".txt",arma::raw_ascii);
			

				/****************************************************************************/
				/********** ONLY FOR DEBUG: MAKES ICP USE TRUE RIGID TRANSFORMS *************/
			if (!this -> filter_arguments -> get_use_icp()){
				std::cout << "\t Using true rigid transform\n";
				M_pc = M_pc_true;
				X_pc = X_pc_true;
			}
				/****************************************************************************/
				/****************************************************************************/

			// The source pc is registered, using the rigid transform that 
			// the ICP returned
			this -> source_pc -> transform(M_pc,X_pc);
			this -> all_registered_pc.push_back(this -> source_pc);

			#if IOFLAGS_run_iod
			this -> source_pc -> save("../output/pc/source_" + std::to_string(time_index) + ".obj",this -> LN_t0.t(),this -> x_t0);
			#endif

			M_pcs[time_index] = M_pc;
			X_pcs[time_index] = X_pc;

		}

	}

	int final_index;
	if (!this -> filter_arguments -> get_use_ba()){
		final_index = this -> filter_arguments -> get_iod_rigid_transforms_number() - 1;
		
	}


	else{

		std::cout << " -- Applying BA to successive point clouds\n";
		std::vector<std::shared_ptr<PC > > pc_to_ba;


		BundleAdjuster bundle_adjuster(0, 
			this -> filter_arguments -> get_iod_rigid_transforms_number() - 1,
			M_pcs,
			X_pcs,
			BN_measured,
			&this -> all_registered_pc,
			this -> filter_arguments -> get_N_iter_bundle_adjustment(),
			this -> LN_t0,
			this -> x_t0,
			mrps_LN,
			true,
			previous_closure_index,
			0);

		std::cout << " -- Running IOD after correction\n";
		final_index = previous_closure_index;

	}

	
	int mc_iter = this -> filter_arguments -> get_iod_mc_iter();

	arma::mat results(7,mc_iter);
	arma::mat crude_guesses(6,mc_iter);
	arma::mat all_covs(7 * mc_iter,7);

	boost::progress_display progress(mc_iter);
	
	#pragma omp parallel for
	for (int i = 0; i < mc_iter; ++i){


		std::map<int,arma::vec> X_pcs_noisy;
		std::map<int,arma::mat> M_pcs_noisy;

		arma::vec state,crude_guess;
		arma::mat cov;	

		this -> run_IOD_finder(state,
			cov,
			crude_guess,
			times,
			last_ba_call_index ,
			this -> filter_arguments -> get_iod_rigid_transforms_number() - 1, 
			mrps_LN,
			X_pcs,
			M_pcs);

		results.submat(0,i,5,i) = state.subvec(0,5);
		results.submat(6,i,6,i) = state(6);


		crude_guesses.col(i) = crude_guess;


		all_covs.rows(7 * i, 7 * i + 6) = cov;
		
		++progress;

	}

	results.save("../output/results.txt",arma::raw_ascii);
	crude_guesses.save("../output/crude_guesses.txt",arma::raw_ascii);
	all_covs.save("../output/all_covs.txt",arma::raw_ascii);
	arma::vec results_mean = arma::mean(results,1);
	arma::mat::fixed<7,7> cov_mc = arma::zeros<arma::mat>(7,7);

	for (unsigned int i = 0; i < results.n_cols; ++i){
		cov_mc +=  (results.col(i) - results_mean) * (results.col(i) - results_mean).t();
	}

	cov_mc *= 1./(results.n_cols-1);
	cov_mc.save("../output/cov_mc.txt",arma::raw_ascii);

}


std::shared_ptr<ShapeModelBezier> ShapeBuilder::get_estimated_shape_model() const{
	return this -> estimated_shape_model;
}




void ShapeBuilder::save_attitude(std::string prefix,int index,const std::vector<arma::mat> & BN) const{


	arma::mat mrp_BN = arma::mat( 3,BN.size());
	for (int i = 0; i < mrp_BN.n_cols; ++i){
		mrp_BN.col(i) = RBK::dcm_to_mrp(BN.at(i));

	}

	mrp_BN.save("../output/filter/BN_" + prefix + "_" + std::to_string(index) + ".txt",arma::raw_ascii);

}


void ShapeBuilder::run_IOD_finder(arma::vec & state,
	arma::mat & cov,
	arma::vec & crude_guess,
	const arma::vec & times,
	const int t0 ,
	const int tf, 
	const std::vector<arma::vec> & mrps_LN,
	const std::map<int,arma::vec> & X_pcs,
	const std::map<int,arma::mat> M_pcs) const{


	// The IOD Finder is ran before running bundle adjustment
	std::vector<RigidTransform> sequential_rigid_transforms;
	std::vector<RigidTransform> absolute_rigid_transforms;
	std::vector<RigidTransform> absolute_true_rigid_transforms;

	ShapeBuilder::assemble_rigid_transforms_IOD(sequential_rigid_transforms,
		absolute_rigid_transforms,
		absolute_true_rigid_transforms,
		times,
		t0,
		tf,
		mrps_LN,
		X_pcs,
		M_pcs);


	IODFinder iod_finder(&sequential_rigid_transforms, 
		&absolute_rigid_transforms,
		mrps_LN,
		this -> filter_arguments -> get_rigid_transform_noise_sd("X"),
		this -> filter_arguments -> get_rigid_transform_noise_sd("sigma"),
		this -> filter_arguments -> get_iod_iterations(), 
		this -> filter_arguments -> get_iod_particles());


	arma::vec true_particle(7);
	true_particle.subvec(0,5) = this -> true_kep_state_t0.get_state();
	true_particle(6) = this -> true_kep_state_t0.get_mu();


	// Crude IO

	// get the center of the collected pcs
	arma::vec center = this -> get_center_collected_pcs(t0,tf,absolute_rigid_transforms,
		absolute_true_rigid_transforms);
	
	arma::vec r0_crude = - this -> LN_t0.t() * center;
	arma::vec r1_crude = sequential_rigid_transforms[0].M .t() * (r0_crude + sequential_rigid_transforms[0].X);
	arma::vec r2_crude = sequential_rigid_transforms[1].M .t() * (r1_crude + sequential_rigid_transforms[1].X);

	// 2nd order interpolation
	arma::mat A = arma::zeros<arma::mat>(9,9) ;
	A.submat(0,0,2,2) = arma::eye<arma::mat>(3,3);
	A.submat(3,0,5,2) = arma::eye<arma::mat>(3,3);
	A.submat(6,0,8,2) = arma::eye<arma::mat>(3,3);

	A.submat(3,3,5,5) = sequential_rigid_transforms[0].t_k * arma::eye<arma::mat>(3,3);
	A.submat(3,6,5,8) = std::pow(sequential_rigid_transforms[0].t_k,2) * arma::eye<arma::mat>(3,3);

	A.submat(6,3,8,5) = sequential_rigid_transforms[1].t_k * arma::eye<arma::mat>(3,3);
	A.submat(6,6,8,8) = std::pow(sequential_rigid_transforms[1].t_k,2) * arma::eye<arma::mat>(3,3);

	arma::vec R = arma::vec(9);
	R.subvec(0,2) = r0_crude;
	R.subvec(3,5) = r1_crude;
	R.subvec(6,8) = r2_crude;

	arma::vec coefs = arma::solve(A,R);

	arma::vec v1_crude = coefs.subvec(3,5) + 2 * sequential_rigid_transforms[1].t_k * coefs.subvec(6,8);

	crude_guess = arma::zeros<arma::vec>(6);
	crude_guess.subvec(0,2) = r0_crude;
	crude_guess.subvec(3,5) = v1_crude;


	arma::vec dr =  (r1_crude - r0_crude);
	double dt = (sequential_rigid_transforms[1].t_k - sequential_rigid_transforms[0].t_k);
	arma::vec v0_crude = dr / dt;


	arma::vec l_bounds = {
		r0_crude(0) - 100,
		r0_crude(1) - 100,
		r0_crude(2) - 100,
		v0_crude(0) - 100 / dt,
		v0_crude(1) - 100 / dt,
		v0_crude(2) - 100 / dt,
		MU_MIN
	};

	arma::vec u_bounds = {
		r0_crude(0) + 100,
		r0_crude(1) + 100,
		r0_crude(2) + 100,
		v0_crude(0) + 100 / dt,
		v0_crude(1) + 100 / dt,
		v0_crude(2) + 100 / dt,
		MU_MAX
	};


	arma::vec guess_particle = {
		r0_crude(0),r0_crude(1),r0_crude(2),
		v0_crude(0),v0_crude(1),v0_crude(2),
		NAN
	};

	iod_finder.run(l_bounds,u_bounds,"cartesian",0,guess_particle);



	state = iod_finder.get_result();

	iod_finder.run_batch(state,cov);


}





void ShapeBuilder::assemble_rigid_transforms_IOD(std::vector<RigidTransform> & sequential_rigid_transforms,
	std::vector<RigidTransform> & absolute_rigid_transforms,
	std::vector<RigidTransform> & absolute_true_rigid_transforms,
	const arma::vec & times, 
	const int t0_index,
	const int tf_index,
	const std::vector<arma::vec>  & mrps_LN,
	const std::map<int,arma::vec> & X_pcs,
	const std::map<int,arma::mat> & M_pcs) const{

	RigidTransform rt;
	rt.t_k = times(0);
	rt.X = X_pcs.at(0);
	rt.M = M_pcs.at(0);

	absolute_rigid_transforms.push_back(rt);
	absolute_true_rigid_transforms.push_back(rt);


	std::map<int,arma::vec> X_pcs_noisy;
	std::map<int,arma::mat> M_pcs_noisy;


	X_pcs_noisy[0] = arma::zeros<arma::vec>(3);
	M_pcs_noisy[0] = arma::eye<arma::mat>(3,3);

	for (int k = 1; k < X_pcs.size(); ++k){
		X_pcs_noisy[k] = X_pcs.at(k) + this -> filter_arguments -> get_rigid_transform_noise_sd("X") * arma::randn<arma::vec>(3);
		M_pcs_noisy[k] = M_pcs.at(k) * RBK::mrp_to_dcm(this -> filter_arguments -> get_rigid_transform_noise_sd("sigma") * arma::randn<arma::vec>(3));
	}

	for (int k = t0_index ; k <=  tf_index; ++ k){

		if (k != 0){

	// Adding the rigid transform. M_p_k and X_p_k represent the incremental rigid transform 
	// from t_k to t_(k-1)
			arma::mat M_p_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * M_pcs_noisy.at(k - 1).t() * M_pcs_noisy.at(k) * RBK::mrp_to_dcm(mrps_LN[k]);
			arma::vec X_p_k = RBK::mrp_to_dcm(mrps_LN[k - 1]).t() * M_pcs_noisy.at(k - 1).t() * (X_pcs_noisy.at(k) - X_pcs_noisy.at(k - 1));



			RigidTransform rigid_transform;
			rigid_transform.M = M_p_k;
			rigid_transform.X = X_p_k;
			rigid_transform.t_k = times(k - t0_index);
			sequential_rigid_transforms.push_back(rigid_transform);

			RigidTransform rt;
			rt.t_k = times(k - t0_index);
			rt.X = X_pcs_noisy[k];
			rt.M = M_pcs_noisy[k];

			absolute_rigid_transforms.push_back(rt);

			RigidTransform rt_true;
			rt_true.t_k = times(k - t0_index);
			rt_true.X = X_pcs.at(k);
			rt_true.M = M_pcs.at(k);

			absolute_true_rigid_transforms.push_back(rt_true);


		}
	}
}



void ShapeBuilder::store_point_clouds(int index,const arma::mat & M_pc,const arma::mat & X_pc) {


	// No point cloud has been collected yet
	if (this -> destination_pc == nullptr) {

		this -> destination_pc = std::make_shared<PC>(PC(this -> lidar -> get_focal_plane(),index));

	}

	else {

		// Only one source point cloud has been collected
		if (this -> source_pc == nullptr) {

			this -> source_pc = std::make_shared<PC>(PC(this -> lidar -> get_focal_plane(),index));


		}

		// Two point clouds have been collected : "nominal case")
		else {

			// The source and destination point clouds are combined into the new source point cloud
			this -> destination_pc = this -> source_pc;

			this -> source_pc = std::make_shared<PC>(PC(this -> lidar -> get_focal_plane(),index));

		}
	}
}


void ShapeBuilder::get_new_states(
	const arma::vec & X_S, 
	arma::mat & dcm_LB, 
	arma::vec & lidar_pos,
	arma::vec & lidar_vel,
	std::vector<arma::vec> & mrps_LN,
	std::vector<arma::mat> & BN_true,
	std::vector<arma::mat> & HN_true){

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



void ShapeBuilder::save_true_ground_track(const std::vector<arma::mat> & BN_true,
	const std::vector<arma::mat> & HN_true) const{


	arma::vec u_H = {1,0,0};

	arma::mat true_longitude_latitude(BN_true.size(),2);

	for (int i = 0; i < BN_true.size(); ++i){

		arma::vec u_B_true = BN_true[i] * HN_true[i].t() * u_H;

		double true_longitude = 180. / arma::datum::pi * std::atan2(u_B_true(1),u_B_true(0));
		double true_latitude = 180. / arma::datum::pi * std::atan(u_B_true(2)/arma::norm(u_B_true.subvec(0,1)));

		arma::rowvec true_long_lat = {true_longitude,true_latitude};

		true_longitude_latitude.row(i) = true_long_lat;
	}

	true_longitude_latitude.save("true_lat_long.txt",arma::raw_ascii);


}


void ShapeBuilder::save_estimated_ground_track(
	std::string path,
	const arma::vec & times,
	const int t0 ,
	const int tf, 
	const OC::KepState & est_kep_state,
	const std::vector<arma::mat> BN_measured) const{


	arma::mat longitude_latitude(tf - t0 + 1,2);
	arma::vec u_H = {1,0,0};


	for (int i = t0; i <= tf; ++i){


		double dt = times(i) - times(t0);

		double f = OC::State::f_from_M(est_kep_state.get_M0() + est_kep_state.get_n() * dt,est_kep_state.get_eccentricity());
		arma::mat DCM_HN = RBK::M3(est_kep_state.get_omega() + f) * RBK::M1(est_kep_state.get_inclination()) * RBK::M3(est_kep_state.get_Omega());
		arma::vec u_B = BN_measured.at(i) * DCM_HN.t() * u_H;

		double longitude = 180. / arma::datum::pi * std::atan2(u_B(1),u_B(0));
		double latitude = 180. / arma::datum::pi * std::atan(u_B(2)/arma::norm(u_B.subvec(0,1)));
		arma::rowvec long_lat = {longitude,latitude};
		longitude_latitude.row(i) = long_lat;
	}

	longitude_latitude.save(path,arma::raw_ascii);

}

void ShapeBuilder::initialize_shape(unsigned int cutoff_index,arma::mat & longitude_latitude){


	std::string pc_path = "../output/pc/source_transformed_poisson.cgal";
	std::string pc_path_obj = "../output/pc/source_transformed_poisson.obj";
	std::string a_priori_path = "../output/shape_model/apriori.obj";
	std::string pc_aligned_path_obj = "../output/pc/source_aligned_poisson.obj";
	std::shared_ptr<PC> destination_pc_concatenated;
	longitude_latitude.save("../output/maps/longitude_latitude_final.txt",arma::raw_ascii);


	if (this -> filter_arguments -> get_use_icp()){

	// The point clouds are bundle-adjusted
		std::vector<std::shared_ptr< PC>> kept_pcs;

		if (this -> filter_arguments -> get_use_ba()){

			// Only the point clouds that looped with the first one are kept

			std::cout << " - Keeping all pcs until # " << cutoff_index + 1<< " over a total of " << this -> all_registered_pc.size() << std::endl;

			for(int pc = 0; pc <= cutoff_index; ++pc){
				std::cout << "keeping pc " << pc << " / " << cutoff_index << std::endl;
				kept_pcs.push_back(this -> all_registered_pc.at(pc));
			}


		}
		else{
			kept_pcs = this -> all_registered_pc;
		}

		std::cout << "-- Constructing point cloud...\n";
		std::shared_ptr<PC> pc_before_ba = std::make_shared<PC>(PC(kept_pcs,this -> filter_arguments -> get_points_retained()));

		pc_before_ba -> save("../output/pc/source_transformed_before_ba.obj",this -> LN_t0.t(),this -> x_t0);



		destination_pc_concatenated = std::make_shared<PC>(PC(kept_pcs,this -> filter_arguments -> get_points_retained()));

		destination_pc_concatenated -> save(
			pc_path, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			true,
			false);



		destination_pc_concatenated -> save(
			pc_path_obj, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			false,
			true);



		// The concatenated point cloud is saved after being transformed so as to "overlap" with the true shape. It
		// should perfectly overlap without noise and bundle-adjustment/ICP errors


		destination_pc_concatenated -> save(pc_aligned_path_obj, this -> LN_t0.t(),this -> x_t0);


	}

	else{

		arma::mat points,normals;

		this -> true_shape_model -> random_sampling(this -> filter_arguments -> get_points_retained(),points,normals);

		destination_pc_concatenated = std::make_shared<PC>(PC(points,normals));

		destination_pc_concatenated -> save(
			pc_path, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			true,
			false);

		destination_pc_concatenated -> save(
			pc_path_obj, 
			arma::eye<arma::mat>(3,3), 
			arma::zeros<arma::vec>(3), 
			false,
			true);


	}




	std::cout << "-- Running PSR...\n";
	CGALINTERFACE::CGAL_interface(pc_path.c_str(),a_priori_path.c_str(),this -> filter_arguments -> get_N_edges());


	ShapeModelImporter shape_io_guess(a_priori_path, 1, true);

	ShapeModelTri a_priori_obj("", nullptr);

	shape_io_guess.load_obj_shape_model(&a_priori_obj);

	std::shared_ptr<ShapeModelBezier> a_priori_bezier = std::make_shared<ShapeModelBezier>(ShapeModelBezier(&a_priori_obj,"E", this -> frame_graph));

	// the shape is elevated to the prescribed degree
	unsigned int starting_degree = a_priori_bezier -> get_degree();
	for (unsigned int i = starting_degree; i < this -> filter_arguments -> get_shape_degree(); ++i){
		a_priori_bezier -> elevate_degree();
	}

	a_priori_bezier -> initialize_index_table();
	a_priori_bezier -> save_both("../output/shape_model/a_priori_bezier");

	ShapeFitterBezier shape_fitter(a_priori_bezier.get(),destination_pc_concatenated.get());

	shape_fitter.fit_shape_batch(this -> filter_arguments -> get_N_iter_shape_filter(),this -> filter_arguments -> get_ridge_coef());

	a_priori_bezier -> save_both("../output/shape_model/fit_a_priori");

	// The estimated shape model is finally initialized
	this -> estimated_shape_model = a_priori_bezier;
	this -> estimated_shape_model -> update_mass_properties();

}



arma::vec ShapeBuilder::get_center_collected_pcs(
	int first_pc_index,
	int last_pc_index,
	const std::vector<RigidTransform> & absolute_rigid_transforms,
	const std::vector<RigidTransform> & absolute_true_rigid_transforms) const{

	arma::vec center = {0,0,0};
	int N = last_pc_index - first_pc_index + 1;
	arma::vec pc_center;
	for (int i = 0; i < N; ++i){	

		pc_center = this -> all_registered_pc[i]-> get_center();

		center += 1./N * (absolute_rigid_transforms.at(i).M * (
			absolute_true_rigid_transforms.at(i).M.t() * (pc_center - absolute_true_rigid_transforms.at(i).X) ) + absolute_rigid_transforms.at(i).X); 
	}
	return center;
}





