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


#include <boost/progress.hpp>
#include <chrono>


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

	arma::vec lidar_pos = X_S.rows(0,2);
	arma::vec lidar_vel = X_S.rows(3,5);

	arma::mat dcm_LB = arma::eye<arma::mat>(3, 3);
	arma::vec mrp_LN(3);
	std::vector<RigidTransform> rigid_transforms;
	std::vector<arma::vec> mrps_LN;
	int last_ba_call_index = 0;
	

	arma::mat M_pc = arma::eye<arma::mat>(3,3);
	arma::vec X_pc = arma::zeros<arma::vec>(3);

	for (int time_index = 0; time_index < times.n_rows; ++time_index) {

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index + 1;
		std::string time_index_formatted = ss.str();

		std::cout << "\n################### Index : " << time_index << " / " <<  times.n_rows - 1  << ", Time : " << times(time_index) << " / " <<  times( times.n_rows - 1) << " ########################" << std::endl;

		X_S = X[time_index];

		this -> get_new_states(X_S,dcm_LB,mrp_LN,lidar_pos,lidar_vel );
		mrps_LN.push_back(mrp_LN);

		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(X_S.subvec(0,2));
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

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
			
			this -> fly_over_map.add_label(time_index,0,0);

			#if IOFLAGS_shape_builder
			this -> destination_pc -> save("../output/pc/source_" + std::to_string(0) + ".obj",this -> LN_t0.t(),this -> x_t0);
			#endif
		}



		if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {


			// The point-cloud to point-cloud ICP is used for point cloud registration
			// This ICP can fail. If so, the update is still applied and will be fixed 
			// in the bundle adjustment
			bool icp_converged = false;
			double longitude,latitude;
			arma::mat M_p_k_old = M_pc;
			arma::vec X_p_k_old = X_pc;

			try{
				ICP icp_pc(this -> destination_pc, this -> source_pc, M_pc, X_pc);

			// These two align the consecutive point clouds 
			// in the instrument frame at t_D == t_0
				M_pc = icp_pc.get_M();
				X_pc = icp_pc.get_X();

				/****************************************************************************/
				// ONLY FOR DEBUG: MAKES ICP USE TRUE RIGID TRANSFORMS
				if (!this -> filter_arguments-> get_use_ba()){

					M_pc = this -> LB_t0 * dcm_LB.t();

					arma::vec pos_in_L = - this -> frame_graph -> convert(arma::zeros<arma::vec>(3),"B","L");
					X_pc = M_pc * pos_in_L - this -> LN_t0 * this -> x_t0;

					arma::mat M_transformed = this -> LN_t0.t() * M_pc * RBK::mrp_to_dcm(mrp_LN);
					arma::vec X_transformed = this -> LN_t0.t() * X_pc;


					// std::cout << - M_transformed * lidar_pos + this -> x_t0 + X_transformed << std::endl;

					// throw;

				}
				/****************************************************************************/




				// Adding the rigid transform



				arma::mat M_p_k = RBK::mrp_to_dcm(mrps_LN[time_index - 1]).t() * M_p_k_old.t() * M_pc * RBK::mrp_to_dcm(mrps_LN[time_index]);
				arma::vec X_p_k = RBK::mrp_to_dcm(mrps_LN[time_index - 1]).t() * M_p_k_old.t() * (X_pc - X_p_k_old);


				// std::cout << - M_p_k * lidar_pos + this -> x_t0 + X_p_k  << std::endl;


				RigidTransform rigid_transform;
				rigid_transform.M_k = M_p_k;
				rigid_transform.X_k = X_p_k;
				rigid_transform.t_k = times(time_index);
				rigid_transforms.push_back(rigid_transform);

				if (rigid_transforms.size() == 10){

					

					IODFinder iod_finder(&rigid_transforms, 
						500, 
						200,
						true);

					arma::vec true_particle(7);
					true_particle.subvec(0,5) = this -> true_kep_state_t0.get_state();
					true_particle(6) = this -> true_kep_state_t0.get_mu();

					double mu_min = 0.5 * this -> true_kep_state_t0.get_mu();
					double mu_max = 1.5 * this -> true_kep_state_t0.get_mu();

					// arma::vec lower_bounds = {750,0,0,0,0,0,mu_min};
					// arma::vec upper_bounds = {1250,0.9999,arma::datum::pi,2 * arma::datum::pi,2 * arma::datum::pi,2 * arma::datum::pi,mu_max};


					arma::vec lower_bounds = true_particle - 0.1 * arma::ones<arma::vec>(true_particle.n_rows);
					arma::vec upper_bounds = true_particle + 0.1 * arma::ones<arma::vec>(true_particle.n_rows);


					iod_finder.run(lower_bounds,upper_bounds);
					OC::KepState estimated_state = iod_finder.get_result();
					

					std::cout << " Evaluating the cost function at the true state: " << IODFinder::cost_function(true_particle,&rigid_transforms) << std::endl;

					std::cout << " True keplerian state at epoch: \n" << this -> true_kep_state_t0.get_state() << " with mu :" << this -> true_kep_state_t0.get_mu() << std::endl;
					std::cout << " Estimated keplerian state at epoch: \n" << estimated_state.get_state() << " with mu :" << estimated_state.get_mu() << std::endl;

					throw;

				}

				arma::vec u_L = {1,0,0};
				arma::vec u_B = M_pc * u_L;

				longitude = 180. / arma::datum::pi * std::atan2(u_B(1),u_B(0));
				latitude = 180. / arma::datum::pi * std::atan(u_B(2)/arma::norm(u_B.subvec(0,1)));

				arma::rowvec long_lat = {longitude,latitude};

				longitude_latitude.row(time_index) = long_lat;
				this -> fly_over_map.add_label(time_index,longitude,latitude);
				icp_converged = true;

			}

			catch(ICPException & e){
				std::cout << e.what() << std::endl;
			}

			// The source pc is registered, using the rigid transform that 
			// the ICP returned
			this -> source_pc -> transform(M_pc,X_pc);
			this -> all_registered_pc.push_back(this -> source_pc);


			// Bundle adjustment is periodically run
			// If an overlap with previous measurements is detected
			// or if the bundle adjustment has not been run for a 
			// certain number of observations
			// Should probably replace this by an adaptive threshold based
			// on a prediction of the alignment error

			if (this -> filter_arguments-> get_use_ba() && time_index - last_ba_call_index == 30){

				last_ba_call_index = time_index;


				std::cout << " -- Applying BA to successive point clouds\n";
				std::vector<std::shared_ptr<PC > > pc_to_ba;

				int ground_pc_ba_index = this -> all_registered_pc.size() - 30 - 1;

				for (unsigned int pc = ground_pc_ba_index; pc <= ground_pc_ba_index + 30; ++pc){
					pc_to_ba.push_back(this -> all_registered_pc[pc]);
				}
				longitude_latitude.save("../output/maps/longitude_latitude_before_" +std::to_string(time_index) +  ".txt",arma::raw_ascii);

				BundleAdjuster bundle_adjuster(&pc_to_ba,
					this -> filter_arguments -> get_N_iter_bundle_adjustment(),
					&this -> fly_over_map,
					longitude_latitude,

					this -> LN_t0,
					this -> x_t0,
					false,
					false);
				longitude_latitude.save("../output/maps/longitude_latitude_" +std::to_string(time_index) +  ".txt",arma::raw_ascii);


			}

			if (icp_converged){
				if (this -> filter_arguments -> get_use_ba() && this -> fly_over_map.has_flyovers(longitude,latitude) && time_index > last_ba_call_index + 15){

					std::cout << " -- Flyover detected\n";
					last_ba_call_index = time_index;
					longitude_latitude.save("../output/maps/longitude_latitude_before_" +std::to_string(time_index) +  ".txt",arma::raw_ascii);

					BundleAdjuster bundle_adjuster(&this -> all_registered_pc,
						this -> filter_arguments -> get_N_iter_bundle_adjustment(),
						&this -> fly_over_map,
						longitude_latitude,

						this -> LN_t0,
						this -> x_t0,
						true,
						false);
					longitude_latitude.save("../output/maps/longitude_latitude_" +std::to_string(time_index) +  ".txt",arma::raw_ascii);

				}
			}





			#if IOFLAGS_shape_builder
			this -> source_pc -> save("../output/pc/source_" + std::to_string(time_index) + ".obj",this -> LN_t0.t(),this -> x_t0);
			#endif



			if (time_index == times.n_rows - 1 || !this -> filter_arguments -> get_use_icp()){
				std::cout << "- Initializing shape model" << std::endl;

				this -> initialize_shape(time_index,longitude_latitude);

				this -> estimated_shape_model -> save("../output/shape_model/fit_source_" + std::to_string(time_index)+ ".b");
				return;

			}

		}

	}


}


std::shared_ptr<ShapeModelBezier> ShapeBuilder::get_estimated_shape_model() const{
	return this -> estimated_shape_model;
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



void ShapeBuilder::perform_measurements_shape(const arma::vec & X_S, 
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
	const arma::mat & LN_t0){


	this -> filter_arguments -> append_mrp_mes(RBK::dcm_to_mrp( M ));


	this -> filter_arguments -> append_time(time);
	this -> filter_arguments -> append_omega_true(X_S.rows(9, 11));
	this -> filter_arguments -> append_mrp_true(mrp_BN);

	this -> filter_arguments -> append_relative_pos_mes(arma::zeros<arma::vec>(3));
	this -> filter_arguments -> append_relative_pos_true(arma::zeros<arma::vec>(3));

}



void ShapeBuilder::get_new_states(
	const arma::vec & X_S, 
	arma::mat & dcm_LB, 
	arma::vec & mrp_LN, 
	arma::vec & lidar_pos,
	arma::vec & lidar_vel){

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
	dcm_LN.row(0) = e_r.t();
	dcm_LN.row(1) = e_t.t();
	dcm_LN.row(2) = e_h.t();

	mrp_LN = RBK::dcm_to_mrp(dcm_LN);
	dcm_LB = dcm_LN * RBK::mrp_to_dcm(X_S.rows(6, 8)).t();

	if (this -> LN_t0.n_rows == 0){
		this -> LN_t0 = dcm_LN;
		this -> x_t0 = lidar_pos;
		this -> LB_t0 = dcm_LB;

		OC::CartState true_cart_state_t0(X_S.rows(0,5),this -> true_shape_model -> get_volume() * 1900 * arma::datum::G);
		this -> true_kep_state_t0 = true_cart_state_t0.convert_to_kep(0);

	}

}

void ShapeBuilder::initialize_shape(unsigned int time_index,arma::mat & longitude_latitude){


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
			std::vector<int> bin = this -> fly_over_map.get_bin(36,18);
			int max  = bin[0];
			for (auto iter = bin.begin(); iter != bin.end(); ++iter){
				if (*iter > max){
					max = *iter;
				}
			}
			std::cout << " - Keeping all pcs until # " << max << " over a total of " << this -> all_registered_pc.size() << std::endl;

			for(int pc = 0; pc <= max; ++pc){
				kept_pcs.push_back(this -> all_registered_pc.at(pc));
			}
			for (int pc = max + 1; pc < this -> all_registered_pc.size(); ++pc){
				this -> fly_over_map.remove_label(std::stoi(this -> all_registered_pc. at(pc) -> get_label()));
			}

			BundleAdjuster bundle_adjuster(&kept_pcs,
				this -> filter_arguments -> get_N_iter_bundle_adjustment(),
				&this -> fly_over_map,
				longitude_latitude,
				this -> LN_t0,
				this -> x_t0,
				true,true);
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


		BundleAdjuster bundle_adjuster(&kept_pcs,
			0,
			&this -> fly_over_map,
			longitude_latitude,
			this -> LN_t0,
			this -> x_t0,
			true,true);


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
	CGALINTERFACE::CGAL_interface(pc_path,a_priori_path,this -> filter_arguments -> get_N_edges());


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


}





