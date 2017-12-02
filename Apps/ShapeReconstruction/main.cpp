#include <Lidar.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelImporter.hpp>
#include <Filter.hpp>
#include <RK.hpp>
#include <Wrappers.hpp>
#include <Interpolator.hpp>
#include <Constants.hpp>
#include <DynamicAnalyses.hpp>
#include <PC.hpp>
#include <ShapeFitterTri.hpp>

#include <limits>
#include <chrono>
#include <boost/progress.hpp>

int main() {

// Ref frame graph
	FrameGraph frame_graph;
	frame_graph.add_frame("B");
	frame_graph.add_frame("L");
	frame_graph.add_frame("N");
	frame_graph.add_frame("E");

	frame_graph.add_transform("B", "L");
	frame_graph.add_transform("N", "B");
	frame_graph.add_transform("N", "E");

	// Shape model formed with triangles
	ShapeModelTri true_shape_model("B", &frame_graph);
	ShapeModelTri estimated_shape_model("E", &frame_graph);

	// Spherical harmonics coefficients
	arma::mat Cnm;
	arma::mat Snm;


#ifdef __APPLE__
	ShapeModelImporter shape_io_truth(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_128.obj", 1000, false);
	Cnm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Cnm_n10_r175.txt", arma::raw_ascii);
	Snm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Snm_n10_r175.txt", arma::raw_ascii);

#elif __linux__
	ShapeModelImporter shape_io_truth(
		"/home/ben/Documents/ASPEN_GUI_LESS/resources/shape_models/itokawa_150_scaled.obj", 1 , false);
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

	shape_io_truth.load_shape_model(&true_shape_model);
	true_shape_model.construct_kd_tree_shape(false);

	
	DynamicAnalyses dyn_analyses(&true_shape_model);


	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_shape_model(&true_shape_model);
	args.set_is_attitude_bool(false);
	args.set_dyn_analyses(&dyn_analyses);
	args.set_Cnm(&Cnm);
	args.set_Snm(&Snm);
	args.set_degree(5);
	args.set_ref_radius(175);
	args.set_mu(arma::datum::G * true_shape_model . get_volume() * 1900);

	// Initial state
	arma::vec X0 = arma::zeros<arma::vec>(12);

	double omega = 2 * arma::datum::pi / (12 * 3600);

	arma::vec omega_0 = {0,0,omega};
	X0.rows(3,5) = omega_0; // Omega_BN(0)

	arma::vec pos_0 = {1000,0,0};
	X0.rows(6,8) = pos_0; // r_LN(0) in body frame


	// Velocity determined from sma
	double a = 1000;

	double v = sqrt(args.get_mu() * (2 / arma::norm(pos_0) - 1./ a));

	arma::vec vel_0_inertial = {0,0.0,v};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);

	X0.rows(9,11) = vel_0_body; // r'_LN(0) in body frame

	RK45 rk_coupled(X0,
		T0,
		TF,
		TF-T0,
		&args,
		"attitude_orbit_n5",
		1e-11);

	rk_coupled.run(&joint_sb_spacecraft_body_frame_dyn,
		&event_function_mrp_omega,
		true,
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/output/integrators/");


	Interpolator state_interpolator(rk_coupled.get_T(),rk_coupled.get_X());

// Lidar
	Lidar lidar(&frame_graph,
		"L",
		ROW_FOV,
		COL_FOV ,
		ROW_RESOLUTION,
		COL_RESOLUTION,
		FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY,
		LOS_NOISE_3SD_BASELINE,
		LOS_NOISE_FRACTION_MES_TRUTH);


	arma::vec times = arma::regspace<arma::vec>(T0,  1./INSTRUMENT_FREQUENCY,  TF); 
	arma::mat X_interp(12,times.n_rows);


// Filter filter_arguments
	FilterArguments shape_filter_args;

	shape_filter_args.set_estimate_shape(false);

	Filter shape_filter(&frame_graph,
		&lidar,
		&true_shape_model,
		&estimated_shape_model,
		&shape_filter_args);

	auto start = std::chrono::system_clock::now();
	shape_filter.run_shape_reconstruction(times,&state_interpolator,true);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	shape_filter_args.save_results();

	return 0;
}












