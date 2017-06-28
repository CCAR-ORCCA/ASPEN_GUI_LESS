#include "Filter.hpp"

Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model,
               ShapeModel * estimated_shape_model,
               FilterArguments * arguments) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;
	this -> estimated_shape_model = estimated_shape_model;
	this -> arguments = arguments;
}

Filter::Filter(FrameGraph * frame_graph,
               Lidar * lidar,
               ShapeModel * true_shape_model) {

	this -> frame_graph = frame_graph;
	this -> lidar = lidar;
	this -> true_shape_model = true_shape_model;

}


void Filter::get_surface_point_cloud(std::string path) {

	std::cout << "Collecting surface data" << std::endl;

	// The vector of times is created first
	// It corresponds to the observation times
	std::vector<double> times;

	times.push_back(this -> arguments -> get_t0());
	double t = times[0];

	while (t < this -> arguments -> get_tf()) {
		t = t + 1. / this -> lidar -> get_frequency();
		times.push_back(t);
	}

	// Memory allocation for the lidar position
	arma::vec lidar_pos_0 = *(this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> get_origin_from_parent());
	arma::vec lidar_pos = arma::vec(3);
	arma::vec lidar_pos_rel = arma::vec(3);

	arma::vec u;
	arma::vec v;
	arma::vec w;

	arma::mat dcm_LN = arma::zeros<arma::mat>(3, 3);
	arma::mat dcm_TN = arma::zeros<arma::mat>(3, 3);

	arma::vec mrp_LN = arma::vec(3);
	arma::vec mrp_TN = arma::vec(3);
	arma::vec mrp_LT = arma::vec(3);

	// Properly orienting the lidar to the target

	u = arma::normalise( - lidar_pos_0);

	v = arma::randu(3);
	v = arma::normalise(v - arma::dot(u, v) * u);
	w = arma::cross(u, v);


	dcm_LN.row(0) = u.t();
	dcm_LN.row(1) = v.t();
	dcm_LN.row(2) = w.t();
	mrp_LN = dcm_to_mrp(dcm_LN);

	this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);


	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {

		// The lidar is on a circular trajectory and is manually steered to the
		// asteroid

		std::cout << "\n################### Time : " << time_index << " ########################" << std::endl;

		arma::mat dcm = (M3(this -> arguments -> get_orbit_rate() * time_index) * M1(this -> arguments -> get_inclination()) * M3(0)).t() ;
		lidar_pos = dcm * lidar_pos_0;


		std::cout << "Lidar pos, inertial" << std::endl;
		std::cout << lidar_pos.t() << std::endl;

		dcm_LN = M3(arma::datum::pi) * dcm.t();

		mrp_LN = dcm_to_mrp(dcm_LN);

		dcm_TN = M3(this -> arguments-> get_body_spin_rate() * time_index).t();
		mrp_TN = dcm_to_mrp(dcm_TN);
		mrp_LT = dcm_to_mrp(dcm_LN * dcm_TN.t());

		std::cout << "Lidar pos, body-fixed frame" << std::endl;
		std::cout << (dcm_TN * lidar_pos).t() << std::endl;


		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		// Setting the true and estimated frame to their new state
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_TN);

		// Getting the true observations (noise free)
		this -> lidar -> send_flash(this -> true_shape_model, false, true);


	}

	this -> lidar -> save_surface_measurements(path);

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
		dcm_TN = mrp_to_dcm(mrp_TN);

		// LN DCM
		mrp_LN = dcm_to_mrp(dcm_LT * dcm_TN);
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
		dcm_TN = mrp_to_dcm(mrp_TN);

		// LN DCM
		mrp_LN = dcm_to_mrp(dcm_LT * dcm_TN);

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






void Filter::run(unsigned int N_iteration, bool plot_measurements, bool save_shape_model) {

	std::cout << "Running the filter" << std::endl;

	// The vector of times is created first
	// It corresponds to the observation times
	std::vector<double> times;

	times.push_back(this -> arguments -> get_t0());
	double t = times[0];

	while (t < this -> arguments -> get_tf()) {
		t = t + 1. / this -> lidar -> get_frequency();
		times.push_back(t);
	}

	// The latitude and longitude are saved to a text file later on
	arma::mat long_lat = arma::mat(times.size(), 2);
	arma::mat long_lat_rel = arma::mat(times.size(), 2);

	// Memory allocation for the lidar position
	arma::vec lidar_pos_0 = *(this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> get_origin_from_parent());
	arma::vec lidar_pos = arma::vec(3);
	arma::vec lidar_pos_rel = arma::vec(3);

	arma::vec u;
	arma::vec v;
	arma::vec w;

	arma::mat dcm_LN = arma::zeros<arma::mat>(3, 3);
	arma::mat dcm_TN = arma::zeros<arma::mat>(3, 3);

	arma::vec mrp_LN = arma::vec(3);
	arma::vec mrp_TN = arma::vec(3);
	arma::vec mrp_LT = arma::vec(3);

	arma::vec volume_dif = arma::vec(times.size());
	arma::vec surface_dif = arma::vec(times.size());

	// Properly orienting the lidar to the target

	u = arma::normalise( - lidar_pos_0);

	v = arma::randu(3);
	v = arma::normalise(v - arma::dot(u, v) * u);
	w = arma::cross(u, v);


	dcm_LN.row(0) = u.t();
	dcm_LN.row(1) = v.t();
	dcm_LN.row(2) = w.t();
	mrp_LN = dcm_to_mrp(dcm_LN);

	this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);



	if (save_shape_model == true) {
		this -> estimated_shape_model -> save("../output/shape_model/shape_model_apriori.obj");
		this -> true_shape_model -> save("../output/shape_model/true_shape_model.obj");
	}


	for (unsigned int time_index = 0; time_index < times.size(); ++time_index) {

		// The lidar is on a circular trajectory and is manually steered to the
		// asteroid

		std::cout << "\n################### Time : " << time_index << " ########################" << std::endl;

		arma::mat dcm = (M3(this -> arguments -> get_orbit_rate() * time_index) * M1(this -> arguments -> get_inclination()) * M3(0)).t() ;
		lidar_pos = dcm * lidar_pos_0;


		std::cout << "Lidar pos, inertial" << std::endl;
		std::cout << lidar_pos.t() << std::endl;

		dcm_LN = M3(arma::datum::pi) * dcm.t();

		mrp_LN = dcm_to_mrp(dcm_LN);

		dcm_TN = M3(this -> arguments-> get_body_spin_rate() * time_index).t();
		mrp_TN = dcm_to_mrp(dcm_TN);
		mrp_LT = dcm_to_mrp(dcm_LN * dcm_TN.t());

		std::cout << "Lidar pos, body-fixed frame" << std::endl;
		std::cout << (dcm_TN * lidar_pos).t() << std::endl;

		// The angles are obtained from the mrp so as to be in the correct range
		arma::vec angles = {std::atan2(lidar_pos(1), lidar_pos(0)), std::asin(lidar_pos(2) / arma::norm(lidar_pos))};

		long_lat.row(time_index) = arma::rowvec(
		{	angles(0), angles(1)});

		lidar_pos_rel = dcm_TN * lidar_pos;

		arma::vec angles_rel = {std::atan2(lidar_pos_rel(1), lidar_pos_rel(0)), std::asin(lidar_pos_rel(2) / arma::norm(lidar_pos_rel))};
		long_lat_rel.row(time_index) = arma::rowvec(
		{	angles_rel(0), angles_rel(1)});


		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		// Setting the true and estimated frame to their new state
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_TN);
		this -> frame_graph -> get_frame(this -> estimated_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_TN);


		// Getting the true observations (noise free)
		this -> lidar -> send_flash(this -> true_shape_model, false);


		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index;
		std::string time_index_formatted = ss.str();

		if (plot_measurements == true) {
			this -> lidar -> send_flash(this -> estimated_shape_model, true);
			this -> lidar -> plot_ranges("../output/measurements/residuals_prefit_" + time_index_formatted, 2);
			this -> lidar -> plot_ranges("../output/measurements/computed_prefit_" + time_index_formatted, 1);
			this -> lidar -> plot_ranges("../output/measurements/true_" + time_index_formatted, 0);

		}

		// The filter is iterated N times
		for (unsigned int iteration = 0; iteration < N_iteration ; ++iteration) {
			this -> lidar -> send_flash(this -> estimated_shape_model, true);

			if (iteration ==  0) {
				this -> correct_shape(time_index, true, false);
			}
			else if (iteration ==  N_iteration - 1) {
				this -> correct_shape(time_index, false, true);
			}
			else {
				this -> correct_shape(time_index, false, false);
			}

		}

		// The postfit residuals are stored
		if (plot_measurements == true) {
			this -> lidar -> send_flash(this -> estimated_shape_model, true);
			this -> lidar -> plot_ranges("../output/measurements/residuals_postfit_" + time_index_formatted, 2);
			this -> lidar -> plot_ranges("../output/measurements/computed_postfit_" + time_index_formatted, 1);
		}

		// Should make a decision about splitting some facets here
		if (save_shape_model == true) {
			this -> estimated_shape_model -> save("../output/shape_model/shape_model_" + time_index_formatted + ".obj");
		}

		// The volume difference between the estimated shape and the true shape is stored
		volume_dif(time_index) = std::abs(this -> estimated_shape_model -> get_volume() -  this -> true_shape_model -> get_volume()) / this -> true_shape_model -> get_volume();
		surface_dif(time_index) = std::abs(this -> estimated_shape_model -> get_surface_area() -  this -> true_shape_model -> get_surface_area()) / this -> true_shape_model -> get_surface_area();

	}

	long_lat.save("../output/long_lat.txt", arma::raw_ascii);
	long_lat_rel.save("../output/long_lat_rel.txt", arma::raw_ascii);
	volume_dif.save("../output/volume_dif.txt", arma::raw_ascii);
	surface_dif.save("../output/surface_dif.txt", arma::raw_ascii);

}



void Filter::run_new(
    std::string orbit_path,
    std::string orbit_time_path,
    std::string attitude_path,
    std::string attitude_time_path) {

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

	int N = (int)((orbit_time(orbit_time . n_rows - 1 ) - orbit_time (0) ) * this -> lidar -> get_frequency());
	arma::vec times = 1. / this -> lidar -> get_frequency() * arma::regspace(0, N);

	for (unsigned int time_index = 0; time_index < times.n_rows; ++time_index) {

		std::stringstream ss;
		ss << std::setw(6) << std::setfill('0') << time_index;
		std::string time_index_formatted = ss.str();

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
		dcm_TN = mrp_to_dcm(mrp_TN);

		// LN DCM
		mrp_LN = dcm_to_mrp(dcm_LT * dcm_TN);
		lidar_pos_inertial = dcm_TN.t() * lidar_pos;


		// Setting the Lidar frame to its new state
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_origin_from_parent(lidar_pos_inertial);
		this -> frame_graph -> get_frame(this -> lidar -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_LN);

		// Setting the true and estimated frame to their new state
		this -> frame_graph -> get_frame(this -> true_shape_model -> get_ref_frame_name()) -> set_mrp_from_parent(mrp_TN);

		// Getting the true observations (noise free)
		this -> lidar -> send_flash(this -> true_shape_model, false, true);
		this -> lidar -> plot_ranges("../output/measurements/true_" + time_index_formatted, 0);

	}

}


void Filter::correct_shape(unsigned int time_index, bool first_iter, bool last_iter) {

	std::vector<Ray * > good_rays;
	std::set<Vertex *> seen_vertices;
	std::set<Facet *> seen_facets;
	arma::mat N_mat;
	std::map<Facet *, std::vector<unsigned int> > facet_to_index_of_vertices;
	std::map<Facet *, arma::uvec> facet_to_N_mat_cols;


	this -> get_observed_features(good_rays,
	                              seen_vertices,
	                              seen_facets,
	                              N_mat,
	                              facet_to_index_of_vertices);

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

		this -> lidar -> save_range_residuals_per_facet("../output/measurements/facets_residuals_prefit_" + std::to_string(time_index), facets_to_residuals);
		this -> lidar -> plot_range_residuals_per_facet("../output/measurements/facets_residuals_prefit_" + std::to_string(time_index));

	}

	else if (last_iter == true) {
		std::map<Facet * , std::vector<double> > facets_to_residuals;

		double max_res = -1;
		Facet * facet_to_split = nullptr;

		for (unsigned int ray_index = 0; ray_index < good_rays.size(); ++ray_index) {
			facets_to_residuals[good_rays[ray_index] -> get_computed_hit_facet()].push_back(good_rays[ray_index] -> get_range_residual());
			if (std::abs(good_rays[ray_index] -> get_range_residual()) > max_res) {
				max_res = std::abs(good_rays[ray_index] -> get_range_residual());
				facet_to_split = good_rays[ray_index] -> get_computed_hit_facet();
			}
		}

		this -> lidar -> save_range_residuals_per_facet("../output/measurements/facets_residuals_postfit_" + std::to_string(time_index), facets_to_residuals);
		this -> lidar -> plot_range_residuals_per_facet("../output/measurements/facets_residuals_postfit_" + std::to_string(time_index));

		if (facet_to_split != nullptr && this -> arguments -> get_split_status() == true) {
			if (facet_to_split -> get_split_count() < this -> arguments -> get_max_split_count())
				this -> estimated_shape_model -> split_facet(facet_to_split);
		}

		if (this -> arguments -> get_recycle_shrunk_facets() == true) {
			this -> estimated_shape_model -> enforce_mesh_quality(this -> arguments -> get_min_facet_angle(),
			        this -> arguments -> get_min_edge_angle());
		}
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

	info_mat = info_mat + this -> arguments -> get_ridge_coef() * arma::eye<arma::mat>(info_mat.n_rows, info_mat.n_cols);


	arma::vec alpha;

	// The deviation in the coordinates of the vertices that were seen is computed
	if (this -> arguments -> get_use_cholesky() == true) {
		alpha = this -> cholesky(info_mat, normal_mat);
	}

	else {
		alpha = arma::solve(info_mat, normal_mat);
	}




	arma::vec dV = N_mat * alpha;
	std::cout << "Info mat conditionning : " << arma::cond(info_mat) << std::endl;
	std::cout << N_mat << std::endl;
	std::cout << alpha << std::endl;
	std::cout << dV << std::endl;

	// The location of the vertices is updated
	for (unsigned int vertex_index = 0; vertex_index < seen_vertices.size(); ++vertex_index) {

		Vertex * vertex = *std::next(seen_vertices.begin(), vertex_index);
		*vertex-> get_coordinates() = *vertex -> get_coordinates() +
		                              dV.rows(3 * vertex_index , 3 * vertex_index + 2 );

	}

	// The mass properties of the shape model are recomputed
	// (center of mass, volume, surface area)
	this -> estimated_shape_model -> update_mass_properties();

	// The shape model is shifted using the new location of the center of mass
	// this -> estimated_shape_model -> shift(-(*this -> estimated_shape_model -> get_center_of_mass()));

	// The facets of the shape model that have been seen are updated
	// There is some overhead because their center will be recomputed
	// after having been shifted, but it's not too bad since
	// seen_facets is only a fraction of the estimated shape model facets
	this -> estimated_shape_model -> update_facets();

	// std::cout << "Volume:" << this -> estimated_shape_model -> get_volume() << std::endl;
}

void Filter::get_observed_features(std::vector<Ray * > & good_rays,
                                   std::set<Vertex *> & seen_vertices,
                                   std::set<Facet *> & seen_facets,
                                   arma::mat & N_mat,
                                   std::map<Facet *, std::vector<unsigned int> > & facet_to_index_of_vertices) {




	std::map<Facet *, std::vector<Ray * > > facet_to_rays;
	std::map<Facet *, unsigned int  > hit_count;

	for (unsigned int row_index = 0; row_index < this -> lidar -> get_row_count(); ++row_index) {
		for (unsigned int col_index = 0; col_index < this -> lidar  -> get_col_count(); ++col_index) {

			Ray * ray = this -> lidar -> get_ray(row_index, col_index);

			// If either the true target or the a-priori
			// shape were missed, this measurement is
			// unusable
			if (ray -> get_computed_hit_facet() == nullptr
			        || ray -> get_true_hit_facet() == nullptr) {
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
				                      *ray -> get_computed_hit_facet() -> get_facet_normal())) < std::sin(this -> arguments -> get_min_normal_observation_angle())) {
					// This is a grazing ray that should be excluded
					continue;
				}

				facet_to_rays[ray -> get_computed_hit_facet()].push_back(ray);

			}
		}
	}
	std::cout << "Facet hit count of the " << facet_to_rays.size() << " facets seen before removing outliers" << std::endl;
	for (auto pair : facet_to_rays) {
		std::cout << pair.second.size() << std::endl;
	}



	if (this -> arguments -> get_reject_outliers() == true) {
		// The distribution of the residuals for each facet are computed
		// here, so as to exclude residuals that are "obvious" outliers

		double mean = 0;
		double stdev = 0;
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

		std::cout << "mean: " << mean << std::endl;
		std::cout << "sd: " << stdev << std::endl;


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



	std::cout << "Facet hit count of the " << facet_to_rays.size() << " facets seen before removing under-observed facets" << std::endl;
	for (auto pair : facet_to_rays) {
		std::cout << pair.second.size() << std::endl;
	}

	for (auto facet_pair : facet_to_rays) {

		if (facet_pair.second.size() >= this -> arguments -> get_minimum_ray_per_facet()) {

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


	std::cout << "Facet hit count of the " << hit_count.size() << " facets seen after removing under-observed facets" << std::endl;
	for (auto pair : hit_count) {
		std::cout << pair.second << std::endl;
	}


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

		// Easy: that vertex is owned by two facets and can mode to
		// up to 2 independent directions
		else if (vertex_to_owning_facets[v_index].size() == 2) {
			arma::vec * n1 = vertex_to_owning_facets[v_index][0] -> get_facet_normal();
			arma::vec * n2 = vertex_to_owning_facets[v_index][1] -> get_facet_normal();

			if (arma::norm(arma::cross(*n1, *n2)) > std::sin(this -> arguments -> get_min_facet_normal_angle_difference())) {
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
				if (arma::norm(arma::cross(*n1, *n2)) > std::sin(this -> arguments -> get_min_facet_normal_angle_difference())) {
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
						if (std::abs(arma::dot(arma::cross(*n1, *n2), *n3)) > std::sin(this -> arguments -> get_min_facet_normal_angle_difference())) {
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
	                       - n * u.t() / arma::dot(u, n)) * tilde(*V2 - *V1);


	arma::rowvec drhodV1 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * tilde(*V0 - *V2);


	arma::rowvec drhodV2 = (*V0 - P).t() / arma::dot(u, n) * (arma::eye<arma::mat>(3, 3)
	                       - n * u.t() / arma::dot(u, n)) * tilde(*V1 - *V0);

	partials.push_back(drhodV0);
	partials.push_back(drhodV1);
	partials.push_back(drhodV2);

	return partials;

}

