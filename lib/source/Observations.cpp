#include "Observations.hpp"
#include "ShapeModelBezier.hpp"
#include "IOFlags.hpp"
#include <Ray.hpp>

// Observations: need range measurements between
// spacecraft and bezier surface
// Will probably need to remove const before args
// if args stores pointer to ref frame

arma::vec Observations::obs_lidar_range_true(double t,
	const arma::vec & x, 
	const Args & args){

	// Position of spacecraft relative to small body in inertial frame
	arma::vec::fixed<3> lidar_pos = x.rows(0,2);
	arma::vec::fixed<3> lidar_vel = args.get_true_vel();

	// Attitude of spacecraft relative to inertial
	arma::vec::fixed<3> e_r = - arma::normalise(lidar_pos);
	arma::vec::fixed<3> e_h = arma::normalise(arma::cross(e_r,-lidar_vel));
	arma::vec::fixed<3> e_t = arma::cross(e_h,e_r);

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
	frame_graph -> get_frame(args.get_true_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(args.get_true_mrp_BN());

	// Getting the true observations (noise is NOT added, 
	// it will be added elsewhere in the filter)

	lidar -> send_flash(args.get_true_shape_model(),false,args.get_skip_factor());


	// The range measurements are extracted from the lidar and stored in an armadillo vector
	auto focal_plane = lidar -> get_focal_plane();
	
	arma::vec ranges = arma::vec(focal_plane -> size());

	#if IOFLAGS_observations
	lidar -> save("../output/lidar/pc_true_" + std::to_string(t),true);
	#endif

	for (unsigned int i = 0; i < ranges.n_rows; ++i){
		ranges(i) = focal_plane -> at(i) -> get_true_range();
	}

	return ranges;

}


arma::vec Observations::obs_lidar_range_computed(
	double t,
	const arma::vec & x, 
	const Args & args){


	std::cout << "-- in obs_lidar_range_computed\n";

	// Position of spacecraft relative to small body in inertial frame
	arma::vec::fixed<3> lidar_pos = x.rows(0,2);

	// Attitude of spacecraft relative to inertial
	arma::vec::fixed<3> e_r = - arma::normalise(args.get_true_pos());
	arma::vec::fixed<3> e_h = arma::normalise(arma::cross(e_r,-args.get_true_vel()));
	arma::vec::fixed<3> e_t = arma::cross(e_h,e_r);

	arma::mat::fixed<3,3> dcm_LN;
	dcm_LN.row(0) = e_r.t();
	dcm_LN.row(1) = e_t.t();
	dcm_LN.row(2) = e_h.t();
	arma::vec::fixed<3> mrp_LN = RBK::dcm_to_mrp(dcm_LN);

	// Setting the Lidar frame to its new state
	std::cout << "-- setting lidar to new state\n";

	auto lidar = args.get_lidar();
	auto frame_graph = args.get_frame_graph();

	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
	frame_graph -> get_frame(lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

	// Setting the small body to its inertial attitude. 
	std::cout << "-- setting the estimated small body to its inertial attitude\n";

	frame_graph -> get_frame(args.get_estimated_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(x.rows(3,5));
	std::cout << "-- setting the true small body to its inertial attitude\n";

	frame_graph -> get_frame(args.get_true_shape_model() -> get_ref_frame_name()) -> set_mrp_from_parent(args.get_true_mrp_BN());


	// Getting the true observations (noise is added)
	std::cout << "-- ray tracing\n";
	lidar -> send_flash(args.get_estimated_shape_model(),false,args.get_skip_factor());
	std::cout << "-- done ray tracing\n";

	// The range measurements are extracted from the lidar and stored in an armadillo vector
	auto focal_plane = lidar -> get_focal_plane();
	
	arma::vec ranges = arma::vec(focal_plane -> size());
	
	#if IOFLAGS_observations
	lidar -> save("../output/lidar/pc_bezier_" + std::to_string(t),true);
	#endif

	for (unsigned int i = 0; i < ranges.n_rows; ++i){
		ranges(i) = focal_plane -> at(i) -> get_true_range();
	}

	return ranges;

}

arma::mat Observations::obs_lidar_range_jac_pos(double t,const arma::vec & x, const Args & args){



	std::cout << "\tin obs_lidar_range_jac_pos\n";

	Lidar * lidar = args.get_lidar();
	auto focal_plane = lidar -> get_focal_plane();
	arma::mat H = arma::zeros<arma::mat>(focal_plane -> size(),3);

	args.get_sigma_consider_vector_ptr() -> clear();

	FrameGraph *  frame_graph = args.get_frame_graph();

	std::cout << "\tBrowsing focal plane\n";

	for (unsigned int i = 0; i < focal_plane -> size(); ++i){

		if (focal_plane -> at(i) -> get_hit_element() != -1){

			arma::vec::fixed<3> u = focal_plane -> at(i) -> get_direction_target_frame();
			arma::vec::fixed<3> n;

			const Bezier & bezier = args.get_estimated_shape_model() -> get_element(focal_plane -> at(i) -> get_hit_element());

			double u_t, v_t;

			focal_plane -> at(i) -> get_impact_coords( u_t, v_t);

			n = bezier.get_normal_coordinates(u_t,v_t);

			auto P = bezier.covariance_surface_point(u_t,v_t,u);

			double sigma_range = std::sqrt(arma::dot(u,P * u));

			args.get_sigma_consider_vector_ptr() -> push_back(sigma_range);

			H.row(i) = - frame_graph -> convert(n,args.get_estimated_shape_model() -> get_ref_frame_name(),"N").t() / arma::dot(n,u);
		}
		

	}

	return H;

}



arma::mat Observations::obs_lidar_range_jac_pos_mrp(double t,const arma::vec & x, const Args & args){

	std::cout << "\tin obs_lidar_range_jac_pos_mrp\n";

	Lidar * lidar = args.get_lidar();
	auto focal_plane = lidar -> get_focal_plane();
	arma::mat H = arma::zeros<arma::mat>(focal_plane -> size(),6);
	arma::vec::fixed<3> u,n,n_inertial;
	arma::mat P(3,3);

	args.get_sigma_consider_vector_ptr() -> clear();

	FrameGraph *  frame_graph = args.get_frame_graph();
	std::cout << "\tBrowsing focal plane\n";

	for (unsigned int i = 0; i < focal_plane -> size(); ++i){

		if (focal_plane -> at(i) -> get_hit_element() != -1){

			u = focal_plane -> at(i) -> get_direction_target_frame();

			const Bezier & bezier = args.get_estimated_shape_model() -> get_element(focal_plane -> at(i) -> get_hit_element());
			
			double u_t, v_t;

			focal_plane -> at(i) -> get_impact_coords( u_t, v_t);


			n = bezier.get_normal_coordinates(u_t,v_t);

			P = bezier.covariance_surface_point(u_t,v_t,u);

			double sigma_range = std::sqrt(arma::dot(u,P * u));

			args.get_sigma_consider_vector_ptr() -> push_back(sigma_range);
			
			

			n_inertial = frame_graph -> convert(n,args.get_estimated_shape_model() -> get_ref_frame_name(),"N");
			
			// Partial of range measurements with respect to position
			H.submat(i,0,i,2) = - n_inertial.t() / arma::dot(n,u);

			// Partial of range measurements with respect to attitude
			H.submat(i,3,i,5) = - 4 * n.t() / arma::dot(n,u) * RBK::tilde(focal_plane -> at(i) -> get_impact_point_target_frame());


		}
		else{
			args.get_sigma_consider_vector_ptr() -> push_back(-1);
		}

	}

	return H;

}




arma::vec Observations::obs_pos_ekf_computed(double t,const arma::vec & x,const Args & args){
	return x.rows(0,2);

}


arma::vec Observations::obs_pos_mrp_ekf_computed(double t,const arma::vec & x,const Args & args){

	arma::vec Ybar(6);
	Ybar.rows(0,2) = x.rows(0,2);//position
	Ybar.rows(3,5) = x.rows(6,8);//mrp_BN

	return Ybar;

}


arma::vec Observations::obs_pos_ekf_lidar(double t,const arma::vec & x,const Args & args){

	
	// Setting the Lidar frame to its new state	
	BatchFilter filter(args);
	
	filter.set_observations_funs(
		Observations::obs_lidar_range_computed,
		Observations::obs_lidar_range_jac_pos,
		Observations::obs_lidar_range_true);

	std::vector<double> times;
	times.push_back(t);

	arma::vec y_bar_bar = x.rows(0,2);

	int iter = filter.run(40,
		args. get_true_pos(),
		y_bar_bar,times,
		std::pow(args.get_sd_noise(),2) * arma::ones<arma::mat>(1,1),
		arma::zeros<arma::mat>(1,1));


	// Nasty hack to get around const Args & args
	(*args.get_batch_output_covariance_ptr()) = filter.get_estimated_covariance_history().front();

	return filter.get_estimated_state_history().front();
}


arma::vec Observations::obs_pos_mrp_ekf_lidar(double t,const arma::vec & x,const Args & args){

	// Setting the Lidar frame to its new state	
	BatchFilter filter(args);
	
	filter.set_observations_funs(
		Observations::obs_lidar_range_computed,
		Observations::obs_lidar_range_jac_pos_mrp,
		Observations::obs_lidar_range_true);

	std::vector<double> times;
	times.push_back(t);

	arma::vec y_bar_bar(6);

	y_bar_bar.rows(0,2) = x.rows(0,2);
	y_bar_bar.rows(3,5) = x.rows(6,8);

	arma::vec X0_truth(6);
	X0_truth.subvec(0,2) = args. get_true_pos();
	X0_truth.subvec(3,5) = args. get_true_mrp_BN();

	int iter = filter.run(40,
		X0_truth,
		y_bar_bar,
		times,
		std::pow(args.get_sd_noise(),2) * arma::ones<arma::mat>(1,1),
		arma::zeros<arma::mat>(1,1));

	// Nasty hack to get around const Args & args
	(*args.get_batch_output_covariance_ptr()) = filter.get_estimated_covariance_history().front();

	return filter.get_estimated_state_history().front();
}


arma::mat Observations::obs_pos_ekf_computed_jac(double t,const arma::vec & x,const Args & args){

	arma::mat H = arma::zeros<arma::mat>(3,x.n_rows);
	H.submat(0,0,2,2) = arma::eye<arma::mat>(3,3);
	return H;

}


arma::mat Observations::obs_pos_mrp_ekf_computed_jac(double t,const arma::vec & x,const Args & args){

	arma::mat H = arma::zeros<arma::mat>(6,x.n_rows);
	H.submat(0,0,2,2) = arma::eye<arma::mat>(3,3);
	H.submat(3,6,5,8) = arma::eye<arma::mat>(3,3);

	return H;

}




