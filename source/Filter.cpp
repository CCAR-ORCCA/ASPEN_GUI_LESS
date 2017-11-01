#include "Filter.hpp"
#include <chrono>


Filter::Filter(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model,
	ShapeModelTri * estimated_shape_model,
	FilterArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> filter_arguments = filter_arguments;
}

Filter::Filter(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model,
	FilterArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> filter_arguments = filter_arguments;

}

Filter::Filter(FrameGraph * frame_graph,
	Lidar * lidar,
	ShapeModelTri * true_shape_model) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;

}



void Filter::run_shape_reconstruction(arma::vec &times ,
	Interpolator * interpolator,
	bool save_shape_model) {


	std::cout << "Running the filter" << std::endl;

	arma::vec X_S = interpolator -> interpolate(times(0), true);

	arma::vec lidar_pos = X_S.rows(6,8);
	arma::vec lidar_vel = X_S.rows(9,11);

	arma::vec e_r;
	arma::vec e_t;
	arma::vec e_h;

	arma::mat dcm_LB = arma::eye<arma::mat>(3, 3);
	arma::mat dcm_LB_t_D = arma::eye<arma::mat>(3, 3);

	arma::vec mrp_LB = {0,0,0};
	arma::vec mrp_BN = X_S.rows(0,2);
	arma::vec mrp_BN_t_D = X_S.rows(0,2);

	arma::mat LN_t_S = arma::eye<arma::mat>(3, 3);
	arma::mat LN_t_D = arma::eye<arma::mat>(3, 3);

	arma::vec volume_dif = arma::vec(times.size());
	arma::vec surface_dif = arma::vec(times.size());

	bool start_filter = false;
	unsigned int pc_size = 0;

	if (save_shape_model == true) {
		this -> true_shape_model -> save("../output/shape_model/true_shape_model.obj");
	}

	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index + 1;
		std::string time_index_formatted = ss.str();

		std::cout << "\n################### Index : " << time_index << " / " << times.n_rows - 1  << ", Time : " << times(time_index) << " / " <<  times(times.n_rows - 1) << " ########################" << std::endl;

		X_S = interpolator -> interpolate(times(time_index), true);

		this -> get_new_relative_states(X_S,dcm_LB,dcm_LB_t_D,LN_t_S, 
			LN_t_D,mrp_BN,mrp_BN_t_D,mrp_LB,lidar_pos,lidar_vel );

		
		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LB);

		// Setting the small body to its inertial attitude. This should not affect the 
		// measurements at all
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_BN);

		// Getting the true observations (noise is added)
		this -> lidar -> send_flash(this -> true_shape_model);

		if (start_filter == false){
			unsigned int hits =  this -> lidar -> get_number_of_hits();
			if (pc_size < hits){
				pc_size = hits;
				std::cout << "Got " + std::to_string(pc_size) + " hits. Waiting for better geometry\n";
			}
			else{
				start_filter = true;
				std::cout << "Starting filter with " + std::to_string(pc_size) + " hits\n";

			}
		}


		// Point cloud registration and attitude estimation occurs first
		if ((this -> destination_pc == nullptr && start_filter == true ) || this -> destination_pc != nullptr)
			this -> store_point_clouds(time_index);

		
		if(this -> source_pc == nullptr && this -> destination_pc != nullptr){

			arma::mat I = arma::eye<arma::mat>(3,3);
			this -> perform_measurements(X_S, times(time_index),I,I,I, mrp_BN);

			this -> destination_pc -> save("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/pc/destination_transformed_poisson.cgal", arma::eye<arma::mat>(3, 3), 
				arma::zeros<arma::vec>(3), true, false);
			this -> destination_pc -> save("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/pc/destination_transformed_poisson.obj", arma::eye<arma::mat>(3, 3), 
				arma::zeros<arma::vec>(3), false, true);

				// A poisson surface reconstruction is ran over the point cloud
				// to obtained a partially covering, well behaved, apriori shape model
			CGALINTERFACE::CGAL_interface("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/pc/destination_transformed_poisson.cgal",
				"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/shape_model/apriori.obj");


			// The estimated shape model is finally constructed using
			// the convex hull
			ShapeModelImporter shape_io_estimated(
				"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/shape_model/apriori.obj",
				1, true);


			shape_io_estimated.load_shape_model(this -> estimated_shape_model);
			this -> estimated_shape_model -> construct_kd_tree_facet(false);

			ShapeFitter shape_fitter(this -> estimated_shape_model,this -> destination_pc.get());
			shape_fitter.fit_shape();

		}


		// The rigid transform best aligning the two point clouds is found
		else if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {

			ICP icp(this -> destination_pc, this -> source_pc);

			arma::mat M = icp.get_M();
			arma::mat X = icp.get_X();

			this -> perform_measurements(X_S, times(time_index),M,LN_t_S, 
				LN_t_D, mrp_BN);

			this -> source_pc -> save("../output/pc/source_" + std::to_string(time_index) + ".obj");
			this -> destination_pc -> save("../output/pc/destination_" + std::to_string(time_index) + ".obj");
			this -> source_pc -> save("../output/pc/source_transformed_" + std::to_string(time_index) + ".obj", M, X);


		}

		if (this -> filter_arguments -> get_number_of_measurements() > 0){
			// The attitude of the estimated shape model
			// is set using the latest mrp measurement
			arma::vec mrp_EN = this -> filter_arguments -> get_latest_mrp_mes();

			this -> frame_graph -> get_frame(
				this -> estimated_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(
				mrp_EN);
			}

		}

	}


	void Filter::measure_spin_axis(arma::mat & dcm) {

		std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);
		this -> filter_arguments -> append_spin_axis_mes(prv.second);

	}





	void Filter::measure_omega(arma::mat & dcm) {
		std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);
		this -> filter_arguments -> append_omega_mes(this -> lidar -> get_frequency() * prv.first * this -> filter_arguments -> get_latest_spin_axis_mes());
	}


	void Filter::store_point_clouds(int index) {


	// No point cloud has been collected yet
		if (this -> destination_pc == nullptr) {

			this -> destination_pc = std::make_shared<PC>(PC(
				this -> lidar -> get_focal_plane(),
				this -> frame_graph));
		}

		else {
		// Only one source point cloud has been collected
			if (this -> source_pc == nullptr) {


				this -> source_pc = std::make_shared<PC>(PC(
					this -> lidar -> get_focal_plane(),
					this -> frame_graph));

			}

		// Two point clouds have been collected : "nominal case")
			else {

			// If false, then the center of mass is still being figured out
			// the source and destination point cloud are being exchanged
				if (this -> filter_arguments -> get_estimate_shape() == false ||
					this -> filter_arguments -> get_has_transitioned_to_shape() == false) {

				// The source and destination point clouds are combined into the new source point cloud
					this -> destination_pc = this -> source_pc;

				this -> source_pc = std::make_shared<PC>(PC(
					this -> lidar -> get_focal_plane(),
					this -> frame_graph));
			}

			// Else, the center of mass is figured out and the
			// shape model is used to produce the "destination" point cloud
			else {

			// this -> destination_pc = this -> source_pc;
			// this -> destination_pc_shape = std::make_shared<PC>(PC(this -> estimated_shape_model));
			// this -> source_pc = std::make_shared<PC>(PC(
			// 	this -> lidar -> get_focal_plane(),
			// 	this -> frame_graph));

			}
		}
	}
}










arma::vec Filter::cholesky(arma::mat & info_mat, arma::mat & normal_mat) const {
	arma::vec x = arma::vec(normal_mat.n_rows);

	// // The cholesky decomposition of the information matrix is computed
	// arma::mat R = arma::chol(info_mat);

	// // R.t() * z = N is solved first
	// arma::vec z = arma::vec(normal_mat.n_rows);
	// z(0) = normal_mat(0) / R.row(0)(0);

	// for (unsigned int i = 1; i < normal_mat.n_rows; ++i) {

	// 	double partial_sum = 0;
	// 	for (unsigned int j = 0; j < i ; ++j) {
	// 		partial_sum += R.row(j)(i) * z(j);
	// 	}

	// 	z(i) = (normal_mat(i) - partial_sum) / R.row(i)(i);
	// }


	// // R * x = z is now solved
	// x(x.n_rows - 1) = z(z.n_rows - 1) / R.row(z.n_rows - 1)(z.n_rows - 1);

	// for (int i = x.n_rows - 2; i > -1; --i) {

	// 	double partial_sum = 0;
	// 	for (unsigned int j = i + 1; j < z.n_rows ; ++j) {
	// 		partial_sum += R.row(i)(j) * x(j);
	// 	}

	// 	x(i) = (z(i) - partial_sum) / R.row(i)(i);
	// }

	return x;

}





std::vector<arma::rowvec> Filter::partial_range_partial_coordinates(const arma::vec & P, const arma::vec & u, Facet * facet) {

	std::vector<arma::rowvec> partials;

	// // It is required to "de-normalized" the normal
	// // vector so as to have a consistent
	// // partial derivative
	// arma::vec n = 2 * facet -> get_area() * (*facet -> get_facet_normal());

	// std::vector<std::shared_ptr<ControlPoint > > * vertices = facet -> get_vertices();

	// arma::vec * V0 =  vertices -> at(0) -> get_coordinates();
	// arma::vec * V1 =  vertices -> at(1) -> get_coordinates();
	// arma::vec * V2 =  vertices -> at(2) -> get_coordinates();



	// arma::rowvec drhodV0 = (n.t()) / arma::dot(u, n) + (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	// 	- n * u.t() / arma::dot(u, n)) * RBK::tilde(*V2 - *V1);


	// arma::rowvec drhodV1 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	// 	- n * u.t() / arma::dot(u, n)) * RBK::tilde(*V0 - *V2);


	// arma::rowvec drhodV2 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	// 	- n * u.t() / arma::dot(u, n)) * RBK::tilde(*V1 - *V0);

	// partials.push_back(drhodV0);
	// partials.push_back(drhodV1);
	// partials.push_back(drhodV2);

	return partials;

}

void Filter::perform_measurements(const arma::vec & X_S, double time, const arma::mat & M,  arma::mat & LN_t_S, 
	arma::mat & LN_t_D, arma::vec & mrp_BN){

	// Attitude is measured. The DCM extracted from the ICP 
			// corresponds to M = [LN](t_D)[NB](t_D)[BN](t_S)[NL](t_S)
			// We want [NB](t_D)[BN](t_S)
			// So we need to get
			// M = [NL](t_D)M[LN](t_S)

	arma::mat M_p = LN_t_D.t() * M * LN_t_S;

	// Measurements are stored
	arma::vec mrp_mes_pc;

	if (this -> filter_arguments -> get_number_of_measurements() == 0){

		assert(arma::trace(LN_t_D) == 3);
		this -> measure_spin_axis(LN_t_D);
		this -> measure_omega(LN_t_D);

		this -> filter_arguments -> append_time(time);
		this -> filter_arguments -> append_omega_true(X_S.rows(3, 5));
		
		// No need to remove the initial offset. It is added to the mrp measurement 
		this -> filter_arguments -> append_mrp_true(mrp_BN);
		this -> filter_arguments -> append_mrp_mes(mrp_BN);

	}
	else{

		mrp_mes_pc = RBK::dcm_to_mrp(RBK::mrp_to_dcm(this -> filter_arguments -> get_latest_mrp_mes())  * M_p );

		this -> filter_arguments -> append_mrp_mes(mrp_mes_pc);

		this -> measure_spin_axis(M_p);
		this -> measure_omega(M_p);

		this -> filter_arguments -> append_time(time);
		this -> filter_arguments -> append_omega_true(X_S.rows(3, 5));
		this -> filter_arguments -> append_mrp_true(mrp_BN);

	}

}



void Filter::get_new_relative_states(const arma::vec & X_S, arma::mat & dcm_LB, arma::mat & dcm_LB_t_D, arma::mat & LN_t_S, 
	arma::mat & LN_t_D, arma::vec & mrp_BN, arma::vec & mrp_BN_t_D,
	arma::vec & mrp_LB, arma::vec & lidar_pos,arma::vec & lidar_vel ){

	// Swapping new and old attitude
	dcm_LB_t_D = dcm_LB;
	mrp_BN_t_D = mrp_BN;


	// Getting the new small body inertial attitude
	// and spacecraft relative position
	mrp_BN = X_S.rows(0,2);
	lidar_pos = X_S.rows(6, 8);
	lidar_vel = X_S.rows(9, 11);


	// The [LB] DCM is assembled. Note that e_r does not exactly have to point towards the target
	// barycenter
	arma::vec e_r = - arma::normalise(lidar_pos);
	arma::vec e_h = arma::normalise(arma::cross(e_r,-lidar_vel));
	arma::vec e_t = arma::cross(e_h,e_r);

	dcm_LB.row(0) = e_r.t();
	dcm_LB.row(1) = e_t.t();
	dcm_LB.row(2) = e_h.t();

	mrp_LB = RBK::dcm_to_mrp(dcm_LB);


	// The [LN] DCM at the present time (t_S) and at the past observation time (t_D) is built
	LN_t_S = dcm_LB * RBK::mrp_to_dcm(mrp_BN);
	LN_t_D = dcm_LB_t_D * RBK::mrp_to_dcm(mrp_BN_t_D);
}


