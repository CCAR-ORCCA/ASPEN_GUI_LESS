#include "NavigationFilter.hpp"
#include "Observations.hpp"
#include "Dynamics.hpp"
#include "ShapeModelImporter.hpp"



// Various constants that set up the visibility emulator scenario

// Lidar settings
#define ROW_RESOLUTION 128
#define COL_RESOLUTION 128
#define ROW_FOV 20
#define COL_FOV 20

// Instrument operating frequency
#define INSTRUMENT_FREQUENCY 0.001 // one flash every hour 

// Noise
#define FOCAL_LENGTH 1e1
#define LOS_NOISE_3SD_BASELINE 5e-2
#define LOS_NOISE_FRACTION_MES_TRUTH 0.

// Times (s)
#define T0 0
#define TF 80000// 7 days


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

	ShapeModelImporter shape_io_truth(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/itokawa_64.obj", 1000, false);
	Cnm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Cnm_n10_r175.txt", arma::raw_ascii);
	Snm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Snm_n10_r175.txt", arma::raw_ascii);

	shape_io_truth.load_obj_shape_model(&true_shape_model);
	true_shape_model.construct_kd_tree_shape(false);

	DynamicAnalyses dyn_analyses(&true_shape_model);

	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);
	args.set_is_attitude_bool(false);
	args.set_dyn_analyses(&dyn_analyses);
	args.set_Cnm(&Cnm);
	args.set_Snm(&Snm);
	args.set_degree(5);
	args.set_ref_radius(175);
	args.set_mu(arma::datum::G * true_shape_model . get_volume() * 1900);

	// Initial state
	arma::vec X0_true = arma::zeros<arma::vec>(12);
	double omega = 2 * arma::datum::pi / (12 * 3600);
	arma::vec omega_0 = {0,0,omega};
	X0_true.rows(3,5) = omega_0; // Omega_BN(0)

	arma::vec pos_0 = {2000,0,0};
	X0_true.rows(6,8) = pos_0; // r_LN(0) in body frame

	// Velocity determined from sma
	double a = arma::norm(pos_0);
	double v = sqrt(args.get_mu() * (2 / arma::norm(pos_0) - 1./ a));
	arma::vec vel_0_inertial = {0,0,v};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);
	X0_true.rows(9,11) = vel_0_body; // r'_LN(0) in body frame

	// Lidar
	Lidar lidar(&frame_graph,"L",ROW_FOV,COL_FOV ,ROW_RESOLUTION,COL_RESOLUTION,FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY,LOS_NOISE_3SD_BASELINE,LOS_NOISE_FRACTION_MES_TRUTH);

	arma::vec times = arma::regspace<arma::vec>(T0,  1./INSTRUMENT_FREQUENCY,  TF); 
	std::vector<double> T_obs;
	for (unsigned int i = 0; i < times.n_rows; ++i){
		T_obs.push_back( times(i));
	}

	arma::vec P0_diag = {0.001,0.001,0.001,0.001,0.001,0.001};
	arma::mat P0 = arma::diagmat(P0_diag);

	NavigationFilter filter(args);
	filter.set_observations_fun(Observations::obs_long_lat,
		Observations::obs_jac_long_lat);	
	filter.set_estimate_dynamics_fun(Dynamics::point_mass_dxdt_odeint,
		Dynamics::point_mass_jac_odeint);
	filter.set_initial_information_matrix(arma::inv(P0));

	filter.set_gamma_fun(Dynamics::gamma_OD);

	arma::mat Q = std::pow(1e-6,2) * arma::eye<arma::mat>(3,3);

	arma::vec X0_true = {0,0,1.1,1,0,0.01};
	arma::vec X_bar_0 = {0.01,0.01,1.15,1,0.01,0.02};

	arma::mat R = std::pow(1./3600 * arma::datum::pi / 180,2) * arma::eye<arma::mat>(2,2);

	int iter = filter.run(1,X0_true,X_bar_0,T_obs,R,Q,true);
	
	filter.write_estimated_state("./X_hat.txt");
	filter.write_true_obs("./Y_true.txt");
	filter.write_true_state("./X_true.txt");
	filter.write_T_obs(times,"./T_obs.txt");
	filter.write_residuals("./residuals.txt",R);
	filter.write_estimated_covariance("./covariances.txt");






















	return 0;
}

