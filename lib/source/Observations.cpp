#include "Observations.hpp"

// Observations: need range measurements between
// spacecraft and bezier surface
// Will probably need to remove const before args
// if args stores pointer to ref frame

arma::vec Observations::obs_lidar_range_true(double t,
	const arma::vec & x, 
	const Args & args){

	arma::vec mrp_BN_true = *args.get_mrp_BN_true();
	arma::vec mrp_LN_true = *args.get_mrp_LN_true();


	// Position of spacecraft relative to small body
	arma::vec lidar_pos = x.rows(0,2);


	arma::vec mrp_LB = RBK::dcm_to_mrp(RBK::mrp_to_dcm(mrp_LN_true) *  RBK::mrp_to_dcm(-mrp_BN_true));

	// Setting the Lidar frame to its new state
	Lidar *  lidar = args.get_lidar();
	FrameGraph *  frame_graph = args.get_frame_graph();


	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LB);


	// Setting the small body to its inertial attitude. This should not affect the 
	// measurements at all
	frame_graph -> get_frame(args.get_true_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_BN_true);

	// Getting the true observations (noise is NOT added, 
	// it will be added elsewhere in the filter)
	

	lidar -> send_flash(args.get_true_shape_model(),false);


	// The range measurements are extracted from the lidar and stored in an armadillo vector
	auto focal_plane = lidar -> get_focal_plane();
	
	arma::vec ranges = arma::vec(focal_plane -> size());
	lidar -> save("../output/lidar/pc_true",true);


	

	for (unsigned int i = 0; i < ranges.n_rows; ++i){
		ranges(i) = focal_plane -> at(i) -> get_true_range();
	}

	return ranges;

}


arma::vec Observations::obs_lidar_range_computed(
	double t,
	const arma::vec & x, 
	const Args & args){

	arma::vec mrp_BN_estimated = *args.get_mrp_BN_estimated();
	arma::vec mrp_LN_true = *args.get_mrp_LN_true();

	// Position of spacecraft relative to small body (estimated)

	arma::vec lidar_pos = x.rows(0,2);

	arma::vec mrp_LB = RBK::dcm_to_mrp(RBK::mrp_to_dcm(mrp_LN_true) *  RBK::mrp_to_dcm(-mrp_BN_estimated));

	// Setting the Lidar frame to its new state
	auto lidar = args.get_lidar();
	auto frame_graph = args.get_frame_graph();

	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LB);

	// Setting the small body to its inertial attitude. This should not affect the 
	// measurements at all
	frame_graph -> get_frame(args.get_estimated_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_BN_estimated);


	// Getting the true observations (noise is added)
	lidar -> send_flash(args.get_estimated_shape_model(),false);

	// The range measurements are extracted from the lidar and stored in an armadillo vector
	auto focal_plane = lidar -> get_focal_plane();
	
	arma::vec ranges = arma::vec(focal_plane -> size());
	lidar -> save("../output/lidar/pc_bezier",true);
	for (unsigned int i = 0; i < ranges.n_rows; ++i){
		ranges(i) = focal_plane -> at(i) -> get_true_range();
	}

	

	return ranges;

}

arma::mat Observations::obs_lidar_range_jac(double t,const arma::vec & x, const Args & args){

	Lidar * lidar = args.get_lidar();
	auto focal_plane = lidar -> get_focal_plane();
	arma::mat H = arma::zeros<arma::mat>(focal_plane -> size(),3);
	double alpha = 1;

	for (unsigned int i = 0; i < focal_plane -> size(); ++i){
		if (focal_plane -> at(i) -> get_hit_element() != nullptr){

			arma::vec u = *focal_plane -> at(i) -> get_direction_target_frame();
			arma::vec n;

			Bezier * bezier = static_cast<Bezier *>(focal_plane -> at(i) -> get_hit_element());

			if (bezier == nullptr){
				n = focal_plane -> at(i) -> get_hit_element() -> get_normal();
			}	
			else{

				double u_t, v_t;

				focal_plane -> at(i) -> get_impact_coords( u_t, v_t);
				
				n = bezier -> get_normal(u_t,v_t);
				arma::mat P = bezier -> covariance_surface_point(u_t,v_t,u);
				
				alpha = 1./ std::sqrt(arma::dot(u,P * u ));

			}

			H.row(i) = - n.t() / arma::dot(n,u) * alpha;
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
	int iter = filter.run(10,*args. get_true_pos(),x_bar_bar,times,arma::ones<arma::mat>(1,1),arma::zeros<arma::mat>(1,1));


	// The covariance in the position is extracted here
	arma::mat P_hat = filter.get_estimated_covariance_history().front();

	std::cout << "Position measurement covariance: " << std::endl;
	std::cout  << P_hat << std::endl;

	// Nasty hack to get around const Args & args
	(*args.get_lidar_position_covariance_ptr()) = P_hat;

	return filter.get_estimated_state_history().front();
}

arma::mat Observations::obs_pos_ekf_computed_jac(double t,const arma::vec & x,const Args & args){

	arma::mat H = arma::zeros<arma::mat>(3,x.n_rows);
	H.submat(0,0,2,2) = arma::eye<arma::mat>(3,3);
	return H;

}




