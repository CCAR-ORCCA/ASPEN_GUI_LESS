#include "NavigationFilter.hpp"
#include "Observations.hpp"
#include "Dynamics.hpp"
#include "ShapeModelImporter.hpp"
#include "ShapeModelBezier.hpp"



// Various constants that set up the visibility emulator scenario

// Lidar settings
#define ROW_RESOLUTION 64
#define COL_RESOLUTION 64
// #define ROW_FOV 20
// #define COL_FOV 20
#define ROW_FOV 20

#define COL_FOV 20


// Instrument operating frequency
#define INSTRUMENT_FREQUENCY 0.000145 // one flash every 2 hours

// Noise
#define FOCAL_LENGTH 1e1
#define LOS_NOISE_3SD_BASELINE 3e-2
#define LOS_NOISE_FRACTION_MES_TRUTH 0e-5

// Times (s)
#define T0 0
#define TF 864000// 10 days


int main() {


	arma::arma_rng::set_seed(0);

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
	// ShapeModelTri estimated_shape_model("E", &frame_graph);
	ShapeModelBezier estimated_shape_model("E", &frame_graph);

	
	ShapeModelImporter shape_io_truth(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_64_scaled_aligned.obj", 1, false);
	
	ShapeModelImporter shape_io_guess("../input/fit_shape_aligned.b", 1, false);

	shape_io_truth.load_obj_shape_model(&true_shape_model);
	true_shape_model.construct_kd_tree_shape();

	// DEBUG: TRUE SHAPE MODEL == ESTIMATED SHAPE MODEL

	// shape_io_truth.load_obj_shape_model(&estimated_shape_model);
	shape_io_guess.load_bezier_shape_model(&estimated_shape_model);


	estimated_shape_model.construct_kd_tree_shape();
	true_shape_model.save("../output/shape_model/B.obj");
	estimated_shape_model.save_to_obj("../output/shape_model/E.obj");


	// Itokawa angular velocity
	double omega = 2 * arma::datum::pi / (12 * 3600);

	// Integrator extra arguments
	Args args;
	// DynamicAnalyses dyn_analyses(&true_shape_model);
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);
	args.set_estimated_shape_model(&estimated_shape_model);

	// args.set_dyn_analyses(&dyn_analyses);
	args.set_mu(arma::datum::G * true_shape_model . get_volume() * 1900);
	args.set_mass(true_shape_model . get_volume() * 1900);
	args.set_sd_noise(LOS_NOISE_3SD_BASELINE / 3);
	

	// Spacecraft initial state
	// Initial spacecraft state
	arma::vec X0_true_spacecraft = arma::zeros<arma::vec>(6);

	arma::vec pos_0 = {1000,0,0};
	X0_true_spacecraft.rows(0,2) = pos_0; // r_LN(0) in body frame

	// Velocity determined from sma
	double a = arma::norm(pos_0);
	double v = sqrt(args.get_mu() * (2 / arma::norm(pos_0) - 1./ a));
	arma::vec omega_0 = {0,0,omega};
	arma::vec vel_0_inertial = {0,0,v};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);
	X0_true_spacecraft.rows(3,5) = vel_0_body; // r'_LN(0) in body frame

	// DEBUG: Asteroid estimated state == spacecraft initial state
	arma::vec X0_true_small_body = {0,0,0,0,0,omega};
	arma::vec X0_estimated_small_body = {0,0,0,0,0,omega};

	// Initial spacecraft position estimate
	arma::vec P0_spacecraft_vec = {100,100,100,1e-6,1e-6,1e-6};
	arma::mat P0_spacecraft_mat = arma::diagmat(P0_spacecraft_vec);

	arma::vec X0_estimated_spacecraft = X0_true_spacecraft + arma::sqrt(P0_spacecraft_mat) * arma::randn<arma::vec>(6);

	arma::vec X0_true_augmented = arma::zeros<arma::vec>(12);
	X0_true_augmented.subvec(0,5) = X0_true_spacecraft;
	X0_true_augmented.subvec(6,11) = X0_true_small_body;

	arma::vec X0_estimated_augmented = arma::zeros<arma::vec>(12);
	X0_estimated_augmented.subvec(0,5) = X0_estimated_spacecraft;
	X0_estimated_augmented.subvec(6,11) = X0_estimated_small_body;

	// Lidar
	Lidar lidar(&frame_graph,"L",ROW_FOV,COL_FOV ,ROW_RESOLUTION,COL_RESOLUTION,FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY,LOS_NOISE_3SD_BASELINE,LOS_NOISE_FRACTION_MES_TRUTH);

	args.set_lidar(&lidar);

	arma::vec times = arma::regspace<arma::vec>(T0,  1./INSTRUMENT_FREQUENCY,  TF); 
	
	// Times
	std::vector<double> T_obs;
	for (unsigned int i = 0; i < times.n_rows; ++i){
		T_obs.push_back( times(i));
	}

	// A-priori covariance on spacecraft state and asteroid state.
	// Since the asteroid state is not estimated, it is frozen
	arma::vec P0_diag = {0.001,0.001,0.001,0.001,0.001,0.001,1e-20,1e-20,1e-20,1e-20,1e-20,1e-20};

	P0_diag.subvec(0,5) = P0_spacecraft_vec;

	arma::mat P0 = arma::diagmat(P0_diag);

	NavigationFilter filter(args);
	filter.set_observations_fun(
		Observations::obs_pos_ekf_computed,
		Observations::obs_pos_ekf_computed_jac,
		Observations::obs_pos_ekf_lidar);	

	filter.set_estimate_dynamics_fun(
		Dynamics::point_mass_attitude_dxdt_body_frame,
		Dynamics::point_mass_jac_attitude_dxdt_body_frame,
		Dynamics::point_mass_attitude_dxdt_body_frame);


	filter.set_initial_information_matrix(arma::inv(P0));
	filter.set_gamma_fun(Dynamics::gamma_OD_augmented);

	arma::mat Q = std::pow(1e-12,2) * arma::eye<arma::mat>(3,3);


	arma::mat R = arma::zeros<arma::mat>(1,1);

	auto start = std::chrono::system_clock::now();


	int iter = filter.run(1,X0_true_augmented,X0_estimated_augmented,T_obs,R,Q);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << " Done running filter " << elapsed_seconds.count() << " s\n";


	filter.write_estimated_state("../output/filter/X_hat.txt");
	filter.write_true_state("../output/filter/X_true.txt");
	filter.write_T_obs(T_obs,"../output/filter/T_obs.txt");
	filter.write_estimated_covariance("../output/filter/covariances.txt");






















	return 0;
}

