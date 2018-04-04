#include "Observations.hpp"
#include "ShapeModelBezier.hpp"

// Observations: need range measurements between
// spacecraft and bezier surface
// Will probably need to remove const before args
// if args stores pointer to ref frame

arma::vec Observations::obs_lidar_range_true(double t,
	const arma::vec & x, 
	const Args & args){

	// Position of spacecraft relative to small body in inertial frame
	arma::vec lidar_pos = x.rows(0,2);
	arma::vec lidar_vel = x.rows(3,5);

	// Attitude of spacecraft relative to inertial
	arma::vec e_r = - arma::normalise(lidar_pos);
	arma::vec e_h = arma::normalise(arma::cross(e_r,-lidar_vel));
	arma::vec e_t = arma::cross(e_h,e_r);

	arma::mat dcm_LN(3,3);
	dcm_LN.row(0) = e_r.t();
	dcm_LN.row(1) = e_t.t();
	dcm_LN.row(2) = e_h.t();
	arma::vec mrp_LN = RBK::dcm_to_mrp(dcm_LN);

	// Setting the Lidar frame to its new state
	Lidar *  lidar = args.get_lidar();
	FrameGraph *  frame_graph = args.get_frame_graph();


	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);


	// Setting the small body to its inertial attitude. 
	frame_graph -> get_frame(args.get_true_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(x.rows(6,8));

	// Getting the true observations (noise is NOT added, 
	// it will be added elsewhere in the filter)
	

	lidar -> send_flash(args.get_true_shape_model(),false);


	// The range measurements are extracted from the lidar and stored in an armadillo vector
	auto focal_plane = lidar -> get_focal_plane();
	
	arma::vec ranges = arma::vec(focal_plane -> size());
	lidar -> save("../output/lidar/pc_true_" + std::to_string(t),true);


	

	for (unsigned int i = 0; i < ranges.n_rows; ++i){
		ranges(i) = focal_plane -> at(i) -> get_true_range();
	}

	return ranges;

}


arma::vec Observations::obs_lidar_range_computed(
	double t,
	const arma::vec & x, 
	const Args & args){

	// Position of spacecraft relative to small body in inertial frame
	arma::vec lidar_pos = x.rows(0,2);
	arma::vec lidar_vel = x.rows(3,5);

	// Attitude of spacecraft relative to inertial
	arma::vec e_r = - arma::normalise(lidar_pos);
	arma::vec e_h = arma::normalise(arma::cross(e_r,-lidar_vel));
	arma::vec e_t = arma::cross(e_h,e_r);

	arma::mat dcm_LN(3,3);
	dcm_LN.row(0) = e_r.t();
	dcm_LN.row(1) = e_t.t();
	dcm_LN.row(2) = e_h.t();
	arma::vec mrp_LN = RBK::dcm_to_mrp(dcm_LN);

	// Setting the Lidar frame to its new state
	auto lidar = args.get_lidar();
	auto frame_graph = args.get_frame_graph();

	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

	// Setting the small body to its inertial attitude. This should not affect the 
	// measurements at all
	frame_graph -> get_frame(args.get_estimated_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(x.rows(6,8));


	// Getting the true observations (noise is added)
	lidar -> send_flash(args.get_estimated_shape_model(),false);

	// The range measurements are extracted from the lidar and stored in an armadillo vector
	auto focal_plane = lidar -> get_focal_plane();
	
	arma::vec ranges = arma::vec(focal_plane -> size());
	lidar -> save("../output/lidar/pc_bezier_" + std::to_string(t),true);
	for (unsigned int i = 0; i < ranges.n_rows; ++i){
		ranges(i) = focal_plane -> at(i) -> get_true_range();
	}

	return ranges;

}

arma::mat Observations::obs_lidar_range_jac(double t,const arma::vec & x, const Args & args){

	Lidar * lidar = args.get_lidar();
	auto focal_plane = lidar -> get_focal_plane();
	arma::mat H = arma::zeros<arma::mat>(focal_plane -> size(),3);

	auto P_cm = static_cast<ShapeModelBezier * >(args.get_estimated_shape_model()) -> get_cm_cov();

	args.get_sigma_consider_vector_ptr() -> clear();
	args.get_biases_consider_vector_ptr() -> clear();
	args.get_sigmas_range_vector_ptr() -> clear();
	FrameGraph *  frame_graph = args.get_frame_graph();


	for (unsigned int i = 0; i < focal_plane -> size(); ++i){

		if (focal_plane -> at(i) -> get_hit_element() != nullptr){

			arma::vec u = *focal_plane -> at(i) -> get_direction_target_frame();
			arma::vec n;

			Bezier * bezier = static_cast<Bezier *>(focal_plane -> at(i) -> get_hit_element());

			if (bezier == nullptr){
				n = frame_graph -> convert(focal_plane -> at(i) -> get_hit_element() -> get_normal(),
					args.get_estimated_shape_model() -> get_ref_frame_name(),"N");
			}	
			else{

				double u_t, v_t;

				focal_plane -> at(i) -> get_impact_coords( u_t, v_t);
				
				n = frame_graph -> convert(bezier -> get_normal(u_t,v_t),
					args.get_estimated_shape_model() -> get_ref_frame_name(),"N");

				auto P = bezier -> covariance_surface_point(u_t,v_t,u);

				double sigma_range = std::sqrt(arma::dot(u,P * u));
				double sigma_cm = std::sqrt(arma::dot(u,P_cm * u));

				args.get_sigma_consider_vector_ptr() -> push_back(sigma_cm);
				args.get_biases_consider_vector_ptr() -> push_back(bezier -> get_range_bias(u_t,v_t,u));
				args.get_sigmas_range_vector_ptr() -> push_back(sigma_range);


			}

			H.row(i) = - n.t() / arma::dot(n,u);
		}
		else{
			args.get_sigma_consider_vector_ptr() -> push_back(-1);
			args.get_biases_consider_vector_ptr() -> push_back(-1);
			args.get_sigmas_range_vector_ptr() -> push_back(-1);
		}

	}

	return H;

}




arma::vec Observations::obs_pos_ekf_computed(double t,const arma::vec & x,const Args & args){
	return x.rows(0,2);

}


arma::vec Observations::obs_pos_ekf_lidar(double t,const arma::vec & x,const Args & args){

	auto lidar = args.get_lidar();
	auto focal_plane = lidar -> get_focal_plane();
	unsigned int N_mes = focal_plane -> size();
	
	// args should hold
	// mrp_BN_estimated : estimated small body attitude;
	// mrp_LN_true : true spacecraft attitude;
	// mrp_BN_true : estimated small body attitude;
	// x_true: true spacecraft relative position


	BatchFilter filter(args);
	
	filter.set_observations_fun(
		Observations::obs_lidar_range_computed,
		Observations::obs_lidar_range_jac,
		Observations::obs_lidar_range_true);

	std::vector<double> times;
	times.push_back(t);

	arma::vec x_bar_bar = x.rows(0,2);
	int iter = filter.run(40,*args. get_true_pos(),x_bar_bar,times,std::pow(args.get_sd_noise(),2) * arma::ones<arma::mat>(1,1),arma::zeros<arma::mat>(1,1));


	// The covariance in the position is extracted here
	arma::mat P_hat = filter.get_estimated_covariance_history().front();


	// Nasty hack to get around const Args & args
	(*args.get_lidar_position_covariance_ptr()) = P_hat;

	return filter.get_estimated_state_history().front();
}

arma::mat Observations::obs_pos_ekf_computed_jac(double t,const arma::vec & x,const Args & args){

	arma::mat H = arma::zeros<arma::mat>(3,x.n_rows);
	H.submat(0,0,2,2) = arma::eye<arma::mat>(3,3);
	return H;

}




