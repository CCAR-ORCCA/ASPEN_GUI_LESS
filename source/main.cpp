#include "Lidar.hpp"
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

	// Spherical harmonics coefficients
	arma::mat Cnm;
	arma::mat Snm;


#ifdef __APPLE__
	ShapeModelImporter shape_io_truth(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_150_scaled.obj", 1, false);
	Cnm.load("../gravity/itokawa_150_Cnm_n10_r175.txt", arma::raw_ascii);
	Snm.load("../gravity/itokawa_150_Snm_n10_r175.txt", arma::raw_ascii);

#elif __linux__
	ShapeModelImporter shape_io_truth(
		"/home/ben/Documents/ASPEN_GUI_LESS/resources/shape_models/itokawa_150_scaled.obj", 1 , false);
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif


	shape_io_truth.load_shape_model(&true_shape_model);
	true_shape_model.construct_kd_tree(false);

	DynamicAnalyses dyn_analyses(&true_shape_model);


	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_shape_model(&true_shape_model);
	args.set_is_attitude_bool(false);
	args.set_dyn_analyses(&dyn_analyses);
	args.set_Cnm(&Cnm);
	args.set_Snm(&Snm);
	args.set_degree(10);
	args.set_ref_radius(175);
	args.set_mu(arma::datum::G * true_shape_model . get_volume() * 1900);

	// Initial state
	arma::vec X0 = arma::zeros<arma::vec>(12);

	double omega = 2 * arma::datum::pi / (12 * 3600);

	arma::vec omega_0 = {0,0,omega};
	X0.rows(3,5) = omega_0; // Omega_BN(0)

	arma::vec pos_0 = {3000,0,0};
	X0.rows(6,8) = pos_0; // r_LN(0) in body frame

	arma::vec vel_0_inertial = {0,0.0,0.02};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);

	X0.rows(9,11) = vel_0_body; // r'_LN(0) in body frame

	RK45 rk_coupled(X0,
		T0,
		TF,
		TF-T0,
		&args,
		"attitude_orbit_n10",
		1e-11);

	rk_coupled.run(&joint_sb_spacecraft_body_frame_dyn,
		&event_function_mrp_omega,
		true,
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/");

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












