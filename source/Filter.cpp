#include "Filter.hpp"
#include <chrono>


Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model,
               ShapeModel * estimated_shape_model,
               FilterArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> filter_arguments = filter_arguments;
}

Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model,
               FilterArguments * filter_arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> filter_arguments = filter_arguments;

}

Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;

}



void Filter::get_surface_point_cloud_from_trajectory(
    arma::mat * orbit_states,
    arma::vec * orbit_time,
    arma::mat * attitude_states,
    arma::vec * attitude_time,
    std::string savepath) {

	std::cout << "Collecting surface data" << std::endl;

	// Lidar position
	arma::vec lidar_pos_inertial = arma::vec(3);
	arma::vec lidar_pos = arma::vec(3);
	arma::vec lidar_vel = arma::vec(3);

	arma::vec u;
	arma::vec v;
	arma::vec w;

	arma::mat dcm_TN = arma::zeros<arma::mat>(3, 3);
	arma::mat dcm_LT = arma::zeros<arma::mat>(3, 3);

	arma::vec mrp_LN = arma::vec(3);
	arma::vec mrp_TN = arma::vec(3);
	arma::vec mrp_LT = arma::vec(3);

	// An interpolator of the attitude state is created
	Interpolator interpolator_attitude(attitude_time, attitude_states);
	Interpolator interpolator_orbit(orbit_time, orbit_states);

	arma::vec interpolated_orbit(6);
	arma::vec interpolated_attitude(6);



	int N = (int)((orbit_time -> at(orbit_time -> n_rows - 1 ) - orbit_time -> at(0) ) * this -> lidar -> get_frequency());
	arma::vec times = 1. / this -> lidar -> get_frequency() * arma::regspace(0, N);


	for (unsigned int time_index = 0; time_index < times.n_rows; ++time_index) {

		std::cout << "\n################### Time : " << times(time_index) << " / " <<  times(times.n_rows - 1) << " ########################" << std::endl;

		interpolated_orbit = interpolator_orbit.interpolate(times(time_index), false);
		interpolated_attitude = interpolator_attitude.interpolate(times(time_index), true);

		// L frame position and velocity (in T frame)

		lidar_pos = interpolated_orbit.rows(0, 2);
		lidar_vel = interpolated_orbit.rows(3, 5);

		// LT DCM
		u = - arma::normalise(lidar_pos);
		v = arma::normalise(arma::cross(lidar_pos, lidar_vel));
		w = arma::cross(u, v);

		dcm_LT = arma::join_rows(u, arma::join_rows(v, w)).t();

		// TN DCM
		mrp_TN = interpolated_attitude.rows(0, 2);
		dcm_TN = RBK::mrp_to_dcm(mrp_TN);

		// LN DCM
		mrp_LN = RBK::dcm_to_mrp(dcm_LT * dcm_TN);
		lidar_pos_inertial = dcm_TN.t() * lidar_pos;


		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos_inertial);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		// Setting the true and estimated frame to their new state
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_TN);

		// Getting the true observations (noise free)
		this -> lidar -> send_flash(this -> true_shape_model, false, true);

	}

	this -> lidar -> save_surface_measurements(savepath);

}



void Filter::get_surface_point_cloud_from_trajectory(
    std::string orbit_path,
    std::string orbit_time_path,
    std::string attitude_path,
    std::string attitude_time_path,
    std::string savepath) {

	std::cout << "Collecting surface data" << std::endl;

	// Matrix of orbit states
	arma::mat orbit_states;
	orbit_states.load(orbit_path);

	// Vector of orbit times
	arma::vec orbit_time;
	orbit_time.load(orbit_time_path);

	// Matrix of orbit states
	arma::mat attitude_states;
	attitude_states.load(attitude_path);

	// Vector of orbit times
	arma::vec attitude_time;
	attitude_time.load(attitude_time_path);

	// Lidar position
	arma::vec lidar_pos_inertial = arma::vec(3);
	arma::vec lidar_pos = arma::vec(3);
	arma::vec lidar_vel = arma::vec(3);

	arma::vec u;
	arma::vec v;
	arma::vec w;

	arma::mat dcm_TN = arma::zeros<arma::mat>(3, 3);
	arma::mat dcm_LT = arma::zeros<arma::mat>(3, 3);

	arma::vec mrp_LN = arma::vec(3);
	arma::vec mrp_TN = arma::vec(3);
	arma::vec mrp_LT = arma::vec(3);

	// An interpolator of the attitude state is created
	Interpolator interpolator_attitude(&attitude_time, &attitude_states);
	Interpolator interpolator_orbit(&orbit_time, &orbit_states);


	arma::vec interpolated_orbit(6);
	arma::vec interpolated_attitude(6);


	int N = (int)((orbit_time(orbit_time.n_rows - 1 ) - orbit_time(0) ) * this -> lidar -> get_frequency());

	arma::vec times = 1. / this -> lidar -> get_frequency() * arma::regspace(0, N);


	for (unsigned int time_index = 0; time_index < times.n_rows; ++time_index) {


		std::cout << "\n################### Time : " << times(time_index) << " / " <<  times(times.n_rows - 1) << " ########################" << std::endl;

		interpolated_orbit = interpolator_orbit.interpolate(times(time_index), false);
		interpolated_attitude = interpolator_attitude.interpolate(times(time_index), true);

		// L frame position and velocity (in T frame)

		lidar_pos = interpolated_orbit.rows(0, 2);
		lidar_vel = interpolated_orbit.rows(3, 5);


		// LT DCM
		u = - arma::normalise(lidar_pos);
		v = arma::normalise(arma::cross(lidar_pos, lidar_vel));
		w = arma::cross(u, v);

		dcm_LT = arma::join_rows(u, arma::join_rows(v, w)).t();

		// TN DCM
		mrp_TN = interpolated_attitude.rows(0, 2);
		dcm_TN = RBK::mrp_to_dcm(mrp_TN);

		// LN DCM
		mrp_LN = RBK::dcm_to_mrp(dcm_LT * dcm_TN);

		lidar_pos_inertial = dcm_TN.t() * lidar_pos;


		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos_inertial);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		// Setting the true and estimated frame to their new state
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_TN);

		// Getting the true observations (noise free)
		this -> lidar -> send_flash(this -> true_shape_model, false, true);

	}

	this -> lidar -> save_surface_measurements(savepath);

}


void Filter::run_shape_reconstruction(std::string orbit_path,
                                      std::string orbit_time_path,
                                      std::string attitude_path,
                                      std::string attitude_time_path,
                                      bool plot_measurements,
                                      bool save_shape_model,
                                      bool inertial_traj) {



	std::cout << "Running the filter" << std::endl;

	// Matrix of orbit states
	arma::mat orbit_states;
	orbit_states.load(orbit_path);

	// Vector of orbit times
	arma::vec orbit_time;
	orbit_time.load(orbit_time_path);

	// Matrix of orbit states
	arma::mat attitude_states;
	attitude_states.load(attitude_path);

	// Vector of orbit times
	arma::vec attitude_time;
	attitude_time.load(attitude_time_path);

	// Lidar position
	arma::vec lidar_pos_inertial = arma::vec(3);
	arma::vec lidar_pos = arma::vec(3);
	arma::vec lidar_vel = arma::vec(3);

	arma::vec u;
	arma::vec v;
	arma::vec w;

	arma::mat dcm_TN = arma::zeros<arma::mat>(3, 3);
	arma::mat dcm_LT = arma::zeros<arma::mat>(3, 3);

	arma::vec mrp_LN = arma::vec(3);
	arma::vec mrp_TN = arma::vec(3);
	arma::vec mrp_LT = arma::vec(3);

	// An interpolator of the attitude state is created
	Interpolator interpolator_attitude(&attitude_time, &attitude_states);
	Interpolator interpolator_orbit(&orbit_time, &orbit_states);

	arma::vec interpolated_orbit(6);
	arma::vec interpolated_attitude(6);

	int N = (int)((orbit_time(orbit_time.n_rows - 1 ) - orbit_time(0) ) * this -> lidar -> get_frequency());

	arma::vec times = 1. / this -> lidar -> get_frequency() * arma::regspace(0, N);


	arma::vec volume_dif = arma::vec(times.size());
	arma::vec surface_dif = arma::vec(times.size());


	if (save_shape_model == true) {
		this -> estimated_shape_model -> save("../output/shape_model/shape_model_000000.obj");
		this -> true_shape_model -> save("../output/shape_model/true_shape_model.obj");
	}


	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {


		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index + 1;
		std::string time_index_formatted = ss.str();

		std::cout << "\n################### Time : " << times(time_index) << " / " <<  times(times.n_rows - 1) << " ########################" << std::endl;

		interpolated_attitude = interpolator_attitude.interpolate(times(time_index), true);
		interpolated_orbit = interpolator_orbit.interpolate(times(time_index), false);

		if (inertial_traj == true) {
			arma::mat dcm = RBK::mrp_to_dcm(interpolated_attitude.rows(0, 2));
			interpolated_orbit.rows(0, 2) = dcm * interpolated_orbit.rows(0, 2);
			interpolated_orbit.rows(3, 5) = dcm * interpolated_orbit.rows(3, 5) - arma::cross(interpolated_attitude.rows(3, 5), interpolated_orbit.rows(0, 2));
		}

		// L frame position and velocity (in T frame)
		lidar_pos = interpolated_orbit.rows(0, 2);
		lidar_vel = interpolated_orbit.rows(3, 5);


		// LT DCM
		u = - arma::normalise(lidar_pos);
		v = arma::normalise(arma::cross(lidar_pos, lidar_vel));
		w = arma::cross(u, v);

		dcm_LT = arma::join_rows(u, arma::join_rows(v, w)).t();

		// TN DCM
		mrp_TN = interpolated_attitude.rows(0, 2);
		dcm_TN = RBK::mrp_to_dcm(mrp_TN);


		// LN DCM
		mrp_LN = RBK::dcm_to_mrp(dcm_LT * dcm_TN);
		lidar_pos_inertial = dcm_TN.t() * lidar_pos + *this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> get_origin_from_parent();


		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos_inertial);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		// Setting the true frame to its new state
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_TN);




		// Getting the true observations (noise is added)
		this -> lidar -> send_flash(this -> true_shape_model, false, false);

		/*
		Point cloud registration and attitude estimation occurs first
		*/
		this -> store_point_clouds(time_index);

		// The rigid transform best aligning the two point clouds is found
		if (this -> destination_pc != nullptr && this -> source_pc != nullptr) {

			// The two point clouds are registered
			this -> register_pcs(time_index, times(time_index));

			this -> filter_arguments -> append_omega_true(dcm_TN.t() * interpolated_attitude.rows(3, 5));
			this -> filter_arguments -> append_mrp_true(RBK::dcm_to_mrp(dcm_TN));

			// The latest estimate of the center of mass is saved
			arma::vec latest_estimate = this -> filter_arguments ->  get_latest_cm_hat();
			// std::ofstream shape_file;
			// shape_file.open("cm_" + std::to_string(time_index) + ".obj");
			// shape_file << "v " << latest_estimate(0) << " " << latest_estimate(1) << " " << latest_estimate(2) << std::endl;
			// shape_file.close();
		}



		// The attitude of the estimated shape model
		// is set using the latest mrp measurement
		arma::vec mrp_EN = this -> filter_arguments -> get_latest_mrp_mes();
		this -> frame_graph -> get_frame(
		    this -> estimated_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(
		        mrp_EN);


		/**
		Shape estimation occurs second, if the center of mass has been properly determined
		*/

		if (this -> filter_arguments -> get_estimate_shape() == true) {


			if (plot_measurements == true) {
				this -> lidar -> send_flash(this -> estimated_shape_model, true, false);

				// The prefit residuals are computed
				this -> lidar -> compute_residuals();
				this -> lidar -> plot_ranges("../output/measurements/computed_prefit_" + time_index_formatted, 1);

				this -> lidar -> plot_ranges("../output/measurements/residuals_prefit_" + time_index_formatted, 2);
				this -> lidar -> plot_ranges("../output/measurements/true_" + time_index_formatted, 0);
			}



			for (unsigned int pass = 0 ; pass < this -> filter_arguments -> get_number_of_shape_passes() ; ++pass) {

				unsigned int size_before = this -> estimated_shape_model -> get_NFacets();
				this -> shape_reconstruction_pass(time_index,
				                                  time_index_formatted);

				if (this -> estimated_shape_model -> get_NFacets() == size_before) {
					std::cout << "Shape unchanged after " << pass + 1 << " passes. " << std::endl;
					if (this -> filter_arguments -> get_has_transitioned_to_shape() == false) {
						this -> filter_arguments -> set_has_transitioned_to_shape(true);
						std::ofstream shape_file;
						shape_file.open("../output/attitude/switch_time.txt");
						shape_file << times(time_index) << std::endl;
					}

					break;
				}

			}


			// The postfit residuals are stored
			if (plot_measurements == true) {

				this -> estimated_shape_model -> construct_kd_tree();

				this -> lidar -> send_flash(this -> estimated_shape_model, true, false);

				// The postfit residuals are computed
				this -> lidar -> compute_residuals();

				this -> lidar -> plot_ranges("../output/measurements/residuals_postfit_" + time_index_formatted, 2);
				this -> lidar -> plot_ranges("../output/measurements/computed_postfit_" + time_index_formatted, 1);
			}

			if (save_shape_model == true) {
				this -> estimated_shape_model -> save("../output/shape_model/shape_model_" + time_index_formatted + ".obj");
			}




		}


		// The volume difference between the estimated shape and the true shape is stored
		volume_dif(time_index) = std::abs(this -> estimated_shape_model -> get_volume() -  this -> true_shape_model -> get_volume()) / this -> true_shape_model -> get_volume();
		surface_dif(time_index) = std::abs(this -> estimated_shape_model -> get_surface_area() -  this -> true_shape_model -> get_surface_area()) / this -> true_shape_model -> get_surface_area();






	}

	volume_dif.save("../output/volume_dif.txt", arma::raw_ascii);
	surface_dif.save("../output/surface_dif.txt", arma::raw_ascii);

}

void Filter::shape_reconstruction_pass(unsigned int time_index,
                                       std::string time_index_formatted) {



// The filter is iterated N times
	for (unsigned int iteration = 1; iteration <= this -> filter_arguments -> get_N_iterations() ; ++iteration) {

		// The kd-tree of the estimated shape is rebuilt
		this -> estimated_shape_model -> construct_kd_tree();

		// A flash is sent at the estimated shape model
		this -> lidar -> send_flash(this -> estimated_shape_model, true, false);

		// The postfit esiduals are computed
		this -> lidar -> compute_residuals();

		if (iteration ==  1) {
			this -> correct_shape(time_index, true, false);
		}
		// Last iteration
		else if (iteration ==  this -> filter_arguments -> get_N_iterations()) {

			this -> correct_shape(time_index, false, true);
		}
		else {
			this -> correct_shape(time_index, false, false);
		}

	}


}







void Filter::register_pcs(int index, double time) {



	// If the center of mass is still being figured out
	if (this -> filter_arguments -> get_estimate_shape() == false ||
	        this -> filter_arguments -> get_has_transitioned_to_shape() == false) {






		ICP icp(this -> destination_pc, this -> source_pc);

		arma::mat dcm = icp.get_DCM();
		arma::vec X = icp.get_X();
		arma::mat R = icp.get_R();


		this -> source_pc -> save("../output/pc/source_" + std::to_string(index) + ".obj");
		this -> destination_pc -> save("../output/pc/destination_" + std::to_string(index) + ".obj");
		this -> source_pc -> save("../output/pc/source_transformed_" + std::to_string(index) + ".obj", dcm, X);


		// Spin axis is measured
		this -> measure_spin_axis(dcm);

		// Center of mass location is estimated
		this -> estimate_cm_KF(dcm, X);

		// Angular velocity is measured
		this -> measure_omega(dcm);

		// Attitude is measured
		arma::vec mrp_mes_pc = RBK::dcm_to_mrp(RBK::mrp_to_dcm(this -> filter_arguments -> get_latest_mrp_mes())  * dcm );

		this -> filter_arguments -> append_mrp_mes(mrp_mes_pc);
		this -> filter_arguments -> append_time(time);

	}

	// If the center of mass is "Figured out" , the filter switches to
	// shape estimation and determines the attitude based on the current shape estimate
	else {

		// Center of mass location
		arma::vec cm_bar = this -> filter_arguments -> get_latest_cm_hat();
		// The a-priori orientation between the model-generated point cloud and the
		// collected point cloud is obtained from the last measure of the orientation
		// The mrp measuring the orientation of the body frame is extracted
		arma::vec mrp_mes_past = this -> filter_arguments -> get_latest_mrp_mes();

		// It is propagated forward in time assuming a constant angular velocity
		Args args;
		args.set_constant_omega(this -> filter_arguments -> get_latest_omega_mes());
		RK45 rk_sigma(mrp_mes_past,
		              this -> filter_arguments -> get_latest_time(),
		              time,
		              1e-1,
		              &args,
		              false);


		rk_sigma.run(&sigma_dot_wrapper,
		             nullptr,
		             &event_function_mrp,
		             false);


		arma::mat dcm_bar = RBK::mrp_to_dcm(rk_sigma.get_X() -> col(rk_sigma.get_X() -> n_cols - 1));
		arma::vec X_bar = - (dcm_bar - arma::eye<arma::mat>(3, 3)) * cm_bar;

		/*
		The results of this ICP factor in the rotation from
		the previous timestep to the current time
		*/

		arma::mat dcm_shape;
		arma::vec X_shape;
		double J_res_shape = std::numeric_limits<double>::infinity();
		bool icp_shape_converged = true;

		try {
			ICP icp_shape(this -> destination_pc_shape, this -> source_pc, dcm_bar, X_bar);

			dcm_shape = icp_shape.get_DCM();
			X_shape = icp_shape.get_X();
			J_res_shape = icp_shape.get_J_res();
		}
		catch (const ICPException & error ) {
			std::cerr << "Registration using the shape failed" << std::endl;
			std::cerr << error.what() << std::endl;
			J_res_shape = std::numeric_limits<double>::infinity();
			icp_shape_converged = false;
		}
		catch (const std::runtime_error & error) {
			std::cerr << "Registration using the shape failed" << std::endl;
			std::cerr << error.what() << std::endl;
			icp_shape_converged = false;
		}

		// An ICP solution is also obtained from the actual source and destination
		// point clouds

		arma::mat dcm;
		arma::vec X;


		try {
			ICP icp(this -> destination_pc, this -> source_pc);

			dcm = icp.get_DCM();
			X = icp.get_X();
		}
		catch (const ICPException & error ) {
			std::cerr << "For consecutive registration" << std::endl;
			std::cerr << error.what() << std::endl;

			throw (std::runtime_error(""));
		}
		catch (const std::runtime_error & error) {
			std::cerr << "For consecutive registration" << std::endl;
			std::cerr << error.what() << std::endl;
			throw (std::runtime_error(""));
		}

		if (icp_shape_converged) {
			this -> source_pc -> save("../output/pc/source_shape_" + std::to_string(index) + ".obj");
			this -> destination_pc_shape -> save("../output/pc/destination_shape_" + std::to_string(index) + ".obj");
			this -> source_pc -> save("../output/pc/source_transformed_shape_" + std::to_string(index) + ".obj", dcm_shape	, X_shape	);
		}


		std::cout << " RMS residuals from the shape-based ICP: " << J_res_shape << " m" << std::endl;



		if (this -> filter_arguments -> get_maximum_J_rms_shape() > J_res_shape) {
			std::cout << "USING SHAPE" << std::endl;




			arma::mat incremental_dcm = dcm_shape * RBK::mrp_to_dcm(mrp_mes_past).t();

			// The center of mass is still estimated
			this -> estimate_cm_KF(dcm_shape, X_shape);

			// Spin axis is measured
			this -> measure_spin_axis(incremental_dcm);

			// Angular velocity is measured
			this -> measure_omega(incremental_dcm);

			// Attitude is measured
			arma::vec mrp_mes_shape_N = RBK::dcm_to_mrp(dcm_shape);

			this -> filter_arguments -> append_mrp_mes(mrp_mes_shape_N);

			this -> filter_arguments -> append_time(time);

		}

		// Using the two consecutive point clouds to obtain the attitude solution
		else {



			this -> source_pc -> save("../output/pc/source_" + std::to_string(index) + ".obj");
			this -> destination_pc -> save("../output/pc/destination_" + std::to_string(index) + ".obj");
			this -> source_pc -> save("../output/pc/source_transformed_" + std::to_string(index) + ".obj", dcm, X);


			// Spin axis is measured
			this -> measure_spin_axis(dcm);

			// Center of mass location is estimated
			this -> estimate_cm_KF(dcm, X);

			// Angular velocity is measured
			this -> measure_omega(dcm);

			// Attitude is measured
			arma::vec mrp_mes_pc = RBK::dcm_to_mrp(RBK::mrp_to_dcm(this -> filter_arguments -> get_latest_mrp_mes())  * dcm );

			this -> filter_arguments -> append_mrp_mes(mrp_mes_pc);
			this -> filter_arguments -> append_time(time);

		}


	}





}




void Filter::measure_spin_axis(arma::mat & dcm) {


	std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);

	this -> filter_arguments -> append_spin_axis_mes(prv.second);


}


void Filter::estimate_cm_KF(arma::mat & dcm, arma::vec & x) {

	// The center of mass is also extracted
	arma::mat v(3, 3);
	v.col(0) = this -> filter_arguments -> get_latest_spin_axis_mes();
	v.col(1) = arma::normalise(arma::cross(v.col(0), arma::randu(3)));
	v.col(2) = arma::normalise(arma::cross(v.col(0), v.col(1)));


	arma::mat H = (dcm - arma::eye<arma::mat>(3, 3)) * v.cols(1, 2);
	arma::vec lambdas = arma::solve(H.t() * H, - H.t() * x);
	arma::vec cm_obs = v.cols(1, 2) * lambdas;


	// The measurement of the center of mass is fused with the past ones
	double a = 1e8;
	double b = 1e-2;
	double c = 1e-2;

	arma::mat R = a * v.col(0) * v.col(0).t() +  b * v.col(1) * v.col(1).t() +  c * v.col(2) * v.col(2).t();


	// Measurement update for the center of mass
	arma::vec cm_bar = this -> filter_arguments -> get_latest_cm_hat();
	arma::mat P_cm_bar = this -> filter_arguments -> get_latest_P_cm_hat() + this -> filter_arguments -> get_Q_cm();

	arma::mat K = P_cm_bar * arma::inv(P_cm_bar + R);
	arma::vec cm_hat = cm_bar + K * (cm_obs - cm_bar);
	arma::mat P_cm_hat = (arma::eye<arma::mat>(3, 3) - K ) * P_cm_bar * (arma::eye<arma::mat>(3, 3) - K ).t() + K * R * K.t();


	this -> filter_arguments -> append_cm_hat( cm_hat);
	this -> filter_arguments -> append_P_cm_hat(P_cm_hat);


	// A boolean is switched when the norm of the center-of-mass update is less than a given threshold

	if (arma::norm(K * (cm_obs - cm_bar)) < this -> filter_arguments -> get_shape_estimation_cm_trigger_thresh()) {
		this -> filter_arguments -> set_estimate_shape(true);
	}





}



void Filter::measure_omega(arma::mat & dcm) {

	std::pair<double, arma::vec > prv = RBK::dcm_to_prv(dcm);

	this -> filter_arguments -> append_omega_mes(this -> lidar -> get_frequency() * prv.first * this -> filter_arguments -> get_latest_spin_axis_mes());

}


void Filter::store_point_clouds(int index) {

	arma::vec u = {1, 0, 0};

	// No point cloud has been collected yet
	if (this -> destination_pc == nullptr) {

		this -> destination_pc = std::make_shared<PC>(PC(u,
		                         this -> lidar -> get_focal_plane(),
		                         this -> frame_graph));
	}

	else {
		// Only one source point cloud has been collected
		if (this -> source_pc == nullptr) {


			this -> source_pc = std::make_shared<PC>(PC(u,
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

				this -> source_pc = std::make_shared<PC>(PC(u,
				                    this -> lidar -> get_focal_plane(),
				                    this -> frame_graph));
			}

			// Else, the center of mass is figured out and the
			// shape model is used to produce the "destination" point cloud
			else {

				this -> destination_pc = this -> source_pc;
				this -> destination_pc_shape = std::make_shared<PC>(PC(this -> estimated_shape_model));
				this -> source_pc = std::make_shared<PC>(PC(u,
				                    this -> lidar -> get_focal_plane(),
				                    this -> frame_graph));

			}
		}
	}
}








void Filter::correct_shape(unsigned int time_index, bool first_iter, bool last_iter) {

	std::vector<Ray * > good_rays;
	std::set<Vertex * > seen_vertices;
	std::set<Facet * > seen_facets;
	std::set<Facet * > spurious_facets;

	arma::mat N_mat;
	std::map<Facet *, std::vector<unsigned int> > facet_to_index_of_vertices;
	std::map<Facet *, arma::uvec> facet_to_N_mat_cols;

	double mean = 0;
	double stdev = 0;

	this -> get_observed_features(good_rays,
	                              seen_vertices,
	                              seen_facets,
	                              spurious_facets,
	                              N_mat,
	                              facet_to_index_of_vertices,
	                              mean,
	                              stdev
	                             );

	std::cout << " Number of good rays : " << good_rays.size() << std::endl;
	std::cout << " Facets in view : " << seen_facets.size() << std::endl;

	this -> correct_observed_features(good_rays,
	                                  seen_vertices,
	                                  seen_facets,
	                                  N_mat,
	                                  facet_to_index_of_vertices);

	if (first_iter == true) {
		std::map<Facet * , std::vector<double> > facets_to_residuals;

		for (unsigned int ray_index = 0; ray_index < good_rays.size(); ++ray_index) {
			facets_to_residuals[good_rays[ray_index] -> get_computed_hit_facet()].push_back(good_rays[ray_index] -> get_range_residual());

		}


		this -> lidar -> save_range_residuals_per_facet("../output/measurements/facets_residuals_prefit" + std::to_string(time_index),
		        facets_to_residuals);

		this -> lidar -> plot_range_residuals_per_facet("../output/measurements/facets_residuals_prefit" + std::to_string(time_index));







	}

	else if (last_iter == true) {


		std::map<Facet * , std::vector<double> > facets_to_residuals;
		std::map<Facet * , double >   facets_to_residuals_sd;

		// Rays to hit facets and residuals
		for (unsigned int ray_index = 0; ray_index < good_rays.size(); ++ray_index) {
			facets_to_residuals[good_rays[ray_index] -> get_computed_hit_facet()].push_back(good_rays[ray_index] -> get_range_residual());
		}

		// Hit facets raw residuals to hit facets to mean residuals
		double max_area = -1;
		Facet * facet_to_split = facets_to_residuals.begin() -> first;

		for (auto it = facets_to_residuals.begin(); it != facets_to_residuals.end(); ++ it) {
			arma::vec facet_residuals_arma = arma::vec(it -> second.size());
			for (unsigned int res_index = 0; res_index <  it -> second.size(); ++ res_index) {
				facet_residuals_arma(res_index) = it -> second[res_index];
			}
			facets_to_residuals_sd[it -> first] = arma::stddev(facet_residuals_arma);

			if (it-> first -> get_area() > max_area) {
				max_area = it -> first -> get_area();
				facet_to_split = it -> first;
			}
		}

		this -> lidar -> save_range_residuals_per_facet("../output/measurements/facets_residuals_postfit_" + std::to_string(time_index), facets_to_residuals);
		this -> lidar -> plot_range_residuals_per_facet("../output/measurements/facets_residuals_postfit_" + std::to_string(time_index));



		// The facets that were seen from behind are spuriously oriented and are thus removed from the shape model
		std::cout << "Removing spurious_facets" << std::endl;
		while (spurious_facets.size() > 0) {
			std::cout << spurious_facets.size() << std::endl;
			this -> estimated_shape_model -> merge_shrunk_facet(
			    *spurious_facets.begin(),
			    &seen_facets,
			    &spurious_facets);

		}

		std::cout << "Done removing spurious_facets" << std::endl;

		// Splitting the facet with the largest surface area if it has a larger standard deviation that the one that was specified
		if (facets_to_residuals_sd[facet_to_split] > this -> filter_arguments -> get_convergence_facet_residuals()) {
			std::cout << ( "Splitting facet: " + std::to_string(facets_to_residuals_sd[facet_to_split])
			               + " > " + std::to_string(this -> filter_arguments -> get_convergence_facet_residuals())
			               + " m of standard dev ") << std::endl;
			this -> estimated_shape_model -> split_facet(facet_to_split, seen_facets);
		}








		std::cout << "Entering shape quality enforcement" << std::endl;
		// The shape model is treated to remove shrunk facets
		if (this -> filter_arguments -> get_merge_shrunk_facets() == true) {
			this -> estimated_shape_model -> enforce_mesh_quality(
			    this -> filter_arguments -> get_min_facet_angle(),
			    this -> filter_arguments -> get_min_edge_angle(),
			    this -> filter_arguments -> get_max_recycled_facets(),
			    seen_facets);
		}
		std::cout << "Leaving shape quality enforcement" << std::endl;



		

	}

}

void Filter::correct_observed_features(std::vector<Ray * > & good_rays,
                                       std::set<Vertex *> & seen_vertices,
                                       std::set<Facet *> & seen_facets,
                                       arma::mat & N_mat,
                                       std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices) {

	unsigned int gamma = N_mat.n_cols;

	// The information and normal matrices are declared
	arma::mat info_mat = arma::zeros<arma::mat>(gamma, gamma);
	arma::vec normal_mat = arma::zeros<arma::vec>(gamma);
	arma::rowvec H_tilde = arma::zeros<arma::rowvec>( 3 * seen_vertices . size());



	for (unsigned int ray_index = 0; ray_index < good_rays.size(); ++ray_index) {

		Ray * ray = good_rays . at(ray_index);

		// The origin and direction of the ray are converted from the
		// lidar frame to the estimated shape model frame
		arma::vec u = *(ray -> get_direction());
		arma::vec P = *(ray -> get_origin());

		u = this -> frame_graph -> convert(
		        u,
		        this -> lidar -> get_ref_frame_name(),
		        this -> estimated_shape_model -> get_ref_frame_name(),
		        true);

		P = this -> frame_graph -> convert(
		        P,
		        this -> lidar -> get_ref_frame_name(),
		        this -> estimated_shape_model -> get_ref_frame_name(),
		        false);

		// The partial derivatives are computed
		std::vector<arma::rowvec> partials = this -> partial_range_partial_coordinates(P,
		                                     u,
		                                     ray -> get_computed_hit_facet());

		// The indices of the facet present in the facet
		// impacted by this ray are found
		// Here, the indices are purely local
		// to the dimension of seen_vertices
		unsigned int v0_index = facet_to_index_of_vertices[ ray -> get_computed_hit_facet()][0];
		unsigned int v1_index = facet_to_index_of_vertices[ ray -> get_computed_hit_facet()][1];
		unsigned int v2_index = facet_to_index_of_vertices[ ray -> get_computed_hit_facet()][2];

		// The corresponding partitions of H_tilde are set
		H_tilde.cols(3 * v0_index, 3 * v0_index + 2) = partials[0];
		H_tilde.cols(3 * v1_index, 3 * v1_index + 2) = partials[1];
		H_tilde.cols(3 * v2_index, 3 * v2_index + 2) = partials[2];

		// The information matrix and the normal matrices are augmented
		info_mat += (H_tilde * N_mat).t() * (H_tilde * N_mat);
		normal_mat += (H_tilde * N_mat).t() * ray -> get_range_residual();

		// Htilde is reset
		H_tilde = 0 * H_tilde;

	}

	info_mat = info_mat + this -> filter_arguments -> get_ridge_coef() * arma::eye<arma::mat>(info_mat.n_rows, info_mat.n_cols);


	arma::vec alpha;

	// The deviation in the coordinates of the vertices that were seen is computed
	if (this -> filter_arguments -> get_use_cholesky() == true) {
		alpha = this -> cholesky(info_mat, normal_mat);
	}

	else {
		alpha = arma::solve(info_mat, normal_mat);
	}



	arma::vec dV = N_mat * alpha;

	// std::cout << "Info mat conditionning : " << arma::cond(info_mat) << std::endl;
	// std::cout << N_mat << std::endl;
	// std::cout << alpha << std::endl;
	// std::cout << dV << std::endl;

	// The location of the vertices is updated
	for (unsigned int vertex_index = 0; vertex_index < seen_vertices.size(); ++vertex_index) {

		Vertex * vertex = *std::next(seen_vertices.begin(), vertex_index);
		*vertex-> get_coordinates() = *vertex -> get_coordinates() +
		                              dV.rows(3 * vertex_index , 3 * vertex_index + 2 );

	}

	// The mass properties of the shape model are recomputed
	// (center of mass, volume, surface area)
	this -> estimated_shape_model -> update_mass_properties();

	// The facets of the shape model that have been seen are updated
	this -> estimated_shape_model -> update_facets(false);

}

void Filter::get_observed_features(std::vector<Ray * > & good_rays,
                                   std::set<Vertex *> & seen_vertices,
                                   std::set<Facet *> & seen_facets,
                                   std::set<Facet *> & spurious_facets,

                                   arma::mat & N_mat,
                                   std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices,
                                   double & mean,
                                   double & stdev) {




	std::map<Facet *, std::vector<Ray * > > facet_to_rays;
	std::map<Facet *, unsigned int  > hit_count;

	for (unsigned int row_index = 0; row_index < this -> lidar -> get_row_count(); ++row_index) {
		for (unsigned int col_index = 0; col_index < this -> lidar  -> get_col_count(); ++col_index) {

			Ray * ray = this -> lidar -> get_ray(row_index, col_index);

			// If either the true target or the a-priori
			// shape were missed, this measurement is
			// unusable
			if (ray -> get_computed_hit_facet() == nullptr
			        || ray -> get_true_hit_facet() == nullptr ) {
				continue;
			}

			else {

				// Grazing rays are excluded
				arma::vec u = *(ray -> get_direction());

				u = this -> frame_graph -> convert(u,
				                                   this -> lidar -> get_ref_frame_name(),
				                                   this -> estimated_shape_model -> get_ref_frame_name(),
				                                   true);




				if (std::abs(
				            arma::dot(u,
				                      *ray -> get_computed_hit_facet() -> get_facet_normal())) < std::cos(this -> filter_arguments -> get_max_ray_incidence())) {

					// This is a grazing ray (high incidence) that should be excluded
					continue;
				}

				// This should never happen
				if (arma::dot(u, *ray -> get_computed_hit_facet() -> get_facet_normal()) > 0) {
					// std::cout << ray -> get_computed_hit_facet() -> get_facet_normal() -> t() << std::endl;
					// std::cout << u.t() << std::endl;

					// std::cout << arma::dot(u, *ray -> get_computed_hit_facet() -> get_facet_normal()) << std::endl;

					// throw (std::runtime_error("Saw through the shape"));
					std::cout << "Warning: saw through the shape" << std::endl;
					spurious_facets.insert(ray -> get_computed_hit_facet());
				}
			}

			facet_to_rays[ray -> get_computed_hit_facet()].push_back(ray);

		}
	}


	// std::cout << "Facet hit count of the " << facet_to_rays.size() << " facets seen before removing outliers" << std::endl;
	// for (auto pair : facet_to_rays) {
	// 	std::cout << pair.second.size() << std::endl;
	// }



	if (this -> filter_arguments -> get_reject_outliers() == true) {
		// The distribution of the residuals for each facet are computed
		// here, so as to exclude residuals that are "obvious" outliers


		unsigned int size = 0;

		for (auto & pair : facet_to_rays) {


			// The mean is computed
			for (unsigned int ray_index = 0; ray_index < pair.second.size(); ++ray_index) {
				mean += pair.second[ray_index] -> get_range_residual();
				++size;
			}


		}

		mean = mean / size;


		for (auto & pair : facet_to_rays) {

			// The standard deviation is computed
			for (unsigned int ray_index = 0; ray_index < pair.second.size(); ++ray_index) {
				stdev += (pair.second[ray_index] -> get_range_residual() - mean) * (pair.second[ray_index] -> get_range_residual() - mean);

			}


		}

		stdev = std::sqrt(stdev / size);


		// Now that the mean and the standard deviation of the facet
		// residuals has been computed, the outliers can be efficiently excluded

		for (auto & pair : facet_to_rays) {


			std::vector<Ray * > rays_to_keep;

			for (unsigned int ray_index = 0; ray_index < pair.second.size(); ++ray_index) {

				if (std::abs(pair.second[ray_index] -> get_range_residual() - mean) < 2 * stdev) {
					rays_to_keep.push_back(pair.second[ray_index]);
				}

			}

			pair.second = rays_to_keep;

		}

	}



	// std::cout << "Facet hit count of the " << facet_to_rays.size() << " facets seen before removing under-observed facets" << std::endl;
	// for (auto pair : facet_to_rays) {
	// 	std::cout << pair.second.size() << std::endl;
	// }

	for (auto facet_pair : facet_to_rays) {

		if (facet_pair.second.size() >= this -> filter_arguments -> get_minimum_ray_per_facet()) {

			seen_facets.insert(facet_pair.first);
			seen_vertices.insert(facet_pair.first -> get_vertices() -> at(0) . get());
			seen_vertices.insert(facet_pair.first -> get_vertices() -> at(1) . get());
			seen_vertices.insert(facet_pair.first -> get_vertices() -> at(2) . get());



			for (unsigned int ray_index = 0; ray_index < facet_pair.second.size();
			        ++ray_index) {
				good_rays.push_back(facet_pair.second[ray_index]);
				hit_count[facet_pair.first] += 1;

			}

		}

	}


	// std::cout << "Facet hit count of the " << hit_count.size() << " facets seen after removing under-observed facets" << std::endl;
	// for (auto pair : hit_count) {
	// 	std::cout << pair.second << std::endl;
	// }


	// This will help counting how many facets each vertex belongs to
	// std::map<unsigned int, std::vector<unsigned int> > vertex_to_owning_facets;
	std::map<unsigned int, std::vector<Facet *> > vertex_to_owning_facets;

	for (unsigned int seen_facet_index = 0;
	        seen_facet_index < seen_facets . size();
	        ++seen_facet_index) {

		Facet * facet = *std::next(seen_facets . begin(), seen_facet_index);

		unsigned int v0_index = std::distance(
		                            seen_vertices . begin(),
		                            seen_vertices . find(
		                                facet -> get_vertices() -> at(0).get())
		                        );

		unsigned int v1_index = std::distance(
		                            seen_vertices . begin(),
		                            seen_vertices . find(
		                                facet -> get_vertices() -> at(1).get())
		                        );


		unsigned int v2_index = std::distance(
		                            seen_vertices . begin(),
		                            seen_vertices . find(
		                                facet -> get_vertices() -> at(2).get())
		                        );

		vertex_to_owning_facets[v0_index].push_back(facet);
		vertex_to_owning_facets[v1_index].push_back(facet);
		vertex_to_owning_facets[v2_index].push_back(facet);

		facet_to_index_of_vertices[facet].push_back(v0_index);
		facet_to_index_of_vertices[facet].push_back(v1_index);
		facet_to_index_of_vertices[facet].push_back(v2_index);

	}


	// Can only solve for displacements
	// along independent directions
	std::map<unsigned int, std::vector<arma::vec *> > vertex_to_normal;

	for (unsigned int v_index = 0; v_index < seen_vertices.size(); ++ v_index) {

		// Easy: that vertex can only move along one direction
		if (vertex_to_owning_facets[v_index].size() == 1) {
			vertex_to_normal[v_index].push_back(vertex_to_owning_facets[v_index][0] -> get_facet_normal());
		}

		// Easy: that vertex is owned by two facets and can move up to
		//  2 independent directions
		else if (vertex_to_owning_facets[v_index].size() == 2) {
			arma::vec * n1 = vertex_to_owning_facets[v_index][0] -> get_facet_normal();
			arma::vec * n2 = vertex_to_owning_facets[v_index][1] -> get_facet_normal();

			if (arma::norm(arma::cross(*n1, *n2)) > std::sin(this -> filter_arguments -> get_min_facet_normal_angle_difference())) {
				vertex_to_normal[v_index].push_back(n1);
				vertex_to_normal[v_index].push_back(n2);
			}
			else {
				vertex_to_normal[v_index].push_back(n2);
			}

		}

		// This vertex is owned by three facets or more. Have to determine
		// a minimum set of normal to those facets spanning R3.
		else {

			arma::vec * n1;
			arma::vec * n2;
			arma::vec * n3;


			// First of all, two non-colinear facet normals are found
			// The first normal is used as a reference
			// It will be used no matter what
			n1 = vertex_to_owning_facets[v_index][0] -> get_facet_normal();
			vertex_to_normal[v_index].push_back(n1);

			unsigned int n2_index = 0;

			for (unsigned int facet_index = 1; facet_index < vertex_to_owning_facets[v_index].size();
			        ++facet_index) {

				n2 = vertex_to_owning_facets[v_index][facet_index] -> get_facet_normal();

				// If true, we just found another independent normal.
				// We need to select up to one more
				if (arma::norm(arma::cross(*n1, *n2)) > std::sin(this -> filter_arguments -> get_min_facet_normal_angle_difference())) {
					vertex_to_normal[v_index].push_back(n2);
					n2_index = facet_index;
					break;
				}

			}

			// If all the facet normals were not colinear, then at least two were selected and we can
			// try to look for a third one
			if (n2_index != 0) {

				for (unsigned int facet_index = 1; facet_index < vertex_to_owning_facets[v_index].size();
				        ++facet_index) {

					if (facet_index != n2_index) {

						n3 = vertex_to_owning_facets[v_index][facet_index] -> get_facet_normal();
						// If true, we found our third normal
						if (std::abs(arma::dot(arma::cross(*n1, *n2), *n3)) > std::pow(
						            std::sin(this -> filter_arguments -> get_min_facet_normal_angle_difference()), 2)) {
							vertex_to_normal[v_index].push_back(n3);
							break;
						}

					}
				}
			}

		}

	}


	unsigned int gamma = 0;
	for (unsigned int seen_vertex_index = 0; seen_vertex_index < seen_vertices.size(); ++ seen_vertex_index) {
		gamma += vertex_to_normal[seen_vertex_index].size();
	}

	// The matrix mapping the normal displacement of each facet to that of the vertices
	// if constructed
	N_mat = arma::zeros<arma::mat>(3 * seen_vertices.size(), gamma);

	unsigned int col_index = 0;
	for (unsigned int v_index = 0; v_index < seen_vertices.size(); ++ v_index) {

		for (unsigned int normal_index_local = 0; normal_index_local < vertex_to_normal[v_index].size(); ++normal_index_local) {

			N_mat.rows(3 * v_index, 3 * v_index + 2).col(col_index) = *vertex_to_normal[v_index][normal_index_local];
			++col_index;

		}
	}





}



arma::vec Filter::cholesky(arma::mat & info_mat, arma::mat & normal_mat) const {

	// The cholesky decomposition of the information matrix is computed
	arma::mat R = arma::chol(info_mat);

	// R.t() * z = N is solved first
	arma::vec z = arma::vec(normal_mat.n_rows);
	z(0) = normal_mat(0) / R.row(0)(0);

	for (unsigned int i = 1; i < normal_mat.n_rows; ++i) {

		double partial_sum = 0;
		for (unsigned int j = 0; j < i ; ++j) {
			partial_sum += R.row(j)(i) * z(j);
		}

		z(i) = (normal_mat(i) - partial_sum) / R.row(i)(i);
	}


	// R * x = z is now solved
	arma::vec x = arma::vec(normal_mat.n_rows);
	x(x.n_rows - 1) = z(z.n_rows - 1) / R.row(z.n_rows - 1)(z.n_rows - 1);

	for (int i = x.n_rows - 2; i > -1; --i) {

		double partial_sum = 0;
		for (unsigned int j = i + 1; j < z.n_rows ; ++j) {
			partial_sum += R.row(i)(j) * x(j);
		}

		x(i) = (z(i) - partial_sum) / R.row(i)(i);
	}

	return x;

}





std::vector<arma::rowvec> Filter::partial_range_partial_coordinates(const arma::vec & P, const arma::vec & u, Facet * facet) {

	std::vector<arma::rowvec> partials;

	// It is required to "de-normalized" the normal
	// vector so as to have a consistent
	// partial derivative
	arma::vec n = 2 * facet -> get_area() * (*facet -> get_facet_normal());

	std::vector<std::shared_ptr<Vertex > > * vertices = facet -> get_vertices();

	arma::vec * V0 =  vertices -> at(0) -> get_coordinates();
	arma::vec * V1 =  vertices -> at(1) -> get_coordinates();
	arma::vec * V2 =  vertices -> at(2) -> get_coordinates();



	arma::rowvec drhodV0 = (n.t()) / arma::dot(u, n) + (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * RBK::tilde(*V2 - *V1);


	arma::rowvec drhodV1 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * RBK::tilde(*V0 - *V2);


	arma::rowvec drhodV2 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * RBK::tilde(*V1 - *V0);

	partials.push_back(drhodV0);
	partials.push_back(drhodV1);
	partials.push_back(drhodV2);

	return partials;

}

