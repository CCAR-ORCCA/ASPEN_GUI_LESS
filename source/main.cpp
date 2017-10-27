// #include "Lidar.hpp"
#include "ShapeModelTri.hpp"
#include "ShapeModelImporter.hpp"
// #include "Filter.hpp"
#include "RK.hpp"
#include "Wrappers.hpp"
#include "Interpolator.hpp"
#include "Constants.hpp"
#include "DynamicAnalyses.hpp"
#include <limits>


int main() {


// Ref frame graph
	FrameGraph frame_graph;
	frame_graph.add_frame("N");
	frame_graph.add_frame("T");

	// Shape model formed with triangles
	ShapeModelTri true_shape_model("T", &frame_graph);

#ifdef __APPLE__
	ShapeModelImporter shape_io_truth(
	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_150_scaled.obj", 1, false);
#elif __linux__
	ShapeModelImporter shape_io_truth(
	    "/home/ben/Documents/ASPEN_GUI_LESS/resources/shape_models/itokawa_150_scaled.obj", 1 , false);
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif


	shape_io_truth.load_shape_model(&true_shape_model);
	true_shape_model.construct_kd_tree(false);

// 1) Propagate small body attitude
	arma::vec attitude_0(6);

	double omega = 2 * arma::datum::pi / (12 * 3600);

	arma::vec angular_vel = { 0.1 * omega,omega,  0.1 * omega};

	attitude_0.rows(0, 2) = arma::zeros<arma::vec>(3);
	attitude_0.rows(3, 5) = angular_vel;



// Specifiying filter_arguments such as density, pointer to frame graph and
// attracting shape model
	Args args;
	DynamicAnalyses dyn_analyses(&true_shape_model);

	args.set_frame_graph(&frame_graph);
	args.set_shape_model(&true_shape_model);
	args.set_is_attitude_bool(false);
	args.set_dyn_analyses(&dyn_analyses);

	RK45 rk_attitude(attitude_0,
	                 T0,
	                 TF,
	                 TF-T0,
	                 &args,
	                 "attitude",
	                 1e-10);

	rk_attitude.run(&attitude_dxdt_wrapper,
	                &event_function_mrp_omega,
	                false,
	                "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/");

// // 2) Propagate spacecraft attitude about small body
// // using computed small body attitude
// 	Interpolator interpolator(rk_attitude . get_T(), rk_attitude . get_X());
// 	args.set_interpolator(&interpolator);
// 	args.set_is_attitude_bool(false);

// 	args.set_density(DENSITY);
// 	args.set_mass(DENSITY * true_shape_model . get_volume());



// // Initial condition of the orbiting spacecraft
// 	arma::vec initial_pos = {3000 , 0, 0};
// 	double v = std::sqrt(arma::datum::G * args.get_mass() / arma::norm(initial_pos));

// 	arma::vec initial_vel_inertial = {0, 0, v};


// 	arma::vec orbit_0(6);

// 	orbit_0.rows(0, 2) = initial_pos;
// 	orbit_0.rows(3, 5) = initial_vel_inertial;

// 	RK45 rk_orbit( orbit_0,
// 	               t0,
// 	               tf,
// 	               dt,
// 	               &args,
// 	               check_energy_conservation,
// 	               "orbit_inertial"
// 	             );

// 	rk_orbit.run(&point_mass_dxdt_wrapper,
// 	             nullptr,
// 	             nullptr,
// 	             false,
// 	             "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/");

// // The attitude of the asteroid is also interpolated
// 	arma::mat interpolated_attitude = arma::mat(6, rk_orbit.get_T() -> n_rows);

// 	for (unsigned int i = 0; i < interpolated_attitude.n_cols; ++i) {
// 		interpolated_attitude.col(i) = interpolator.interpolate( rk_orbit.get_T() -> at(i), true);
// 	}
// 	interpolated_attitude.save("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/interpolated_attitude.txt", arma::raw_ascii);

// // Lidar
// 	Lidar lidar(&frame_graph,
// 	            "L",
// 	            ROW_FOV,
// 	            COL_FOV ,
// 	            ROW_RESOLUTION,
// 	            COL_RESOLUTION,
// 	            FOCAL_LENGTH,
// 	            INSTRUMENT_FREQUENCY,
// 	            LOS_NOISE_3SD_BASELINE,
// 	            LOS_NOISE_FRACTION_MES_TRUTH);



// // Filter filter_arguments
// 	FilterArguments shape_filter_args;

// 	shape_filter_args.set_max_ray_incidence(60 * arma::datum::pi / 180.);


// 	shape_filter_args.set_min_facet_normal_angle_difference(45 * arma::datum::pi / 180.);


// 	shape_filter_args.set_split_facets(true);
// 	shape_filter_args.set_use_cholesky(false);

// 	shape_filter_args.set_min_edge_angle(0 * arma::datum::pi / 180);// Minimum edge angle indicating degeneracy
// 	shape_filter_args.set_min_facet_angle(20 * arma::datum::pi / 180);// Minimum facet angle indicating degeneracy

// // Minimum number of rays per facet to update the estimated shape
// 	shape_filter_args.set_min_ray_per_facet(3);

// // Iterations
// 	shape_filter_args.set_N_iterations(5);
// 	shape_filter_args.set_number_of_shape_passe(30);

// // Facets recycling
// 	shape_filter_args.set_merge_shrunk_facets(true);
// 	shape_filter_args.set_max_recycled_facets(5);

// 	shape_filter_args.set_convergence_facet_residuals( 5 * LOS_NOISE_3SD_BASELINE);

// 	arma::vec cm_bar_0 = {1e3, -1e2, -1e3};


// 	shape_filter_args.set_P_cm_0(1e6 * arma::eye<arma::mat>(3, 3));
// 	shape_filter_args.set_cm_bar_0(cm_bar_0);
// 	shape_filter_args.set_Q_cm(0e-3 * arma::eye<arma::mat>(3, 3));

// 	shape_filter_args.set_P_omega_0(1e-3 * arma::eye<arma::mat>(3, 3));
// 	shape_filter_args.set_omega_bar_0(arma::zeros<arma::vec>(3));
// 	shape_filter_args.set_Q_omega(0e-4 * arma::eye<arma::mat>(3, 3));

// 	shape_filter_args.set_estimate_shape(false);
// 	shape_filter_args.set_shape_estimation_cm_trigger_thresh(1);
// 	// shape_filter_args.set_shape_estimation_cm_trigger_thresh(0);



// 	Filter shape_filter(&frame_graph,
// 	                    &lidar,
// 	                    &true_shape_model,
// 	                    &estimated_shape_model,
// 	                    &shape_filter_args);


// 	shape_filter.run_shape_reconstruction(
// 	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/X_RK45_orbit_inertial.txt",
// 	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/T_RK45_orbit_inertial.txt",
// 	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/X_RK45_attitude.txt",
// 	    "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/T_RK45_attitude.txt",
// 	    false,
// 	    true,
// 	    true);


// 	shape_filter_args.save_estimate_time_history();
// 	std::ofstream shape_file;
// 	shape_file.open("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/attitude/true_cm.obj");
// 	shape_file << "v " << 0 << " " << 0 << " " << 0 << std::endl;



	return 0;
}












