#include <Lidar.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelImporter.hpp>
#include <ShapeBuilder.hpp>
#include <Wrappers.hpp>
#include <DynamicAnalyses.hpp>
#include <Dynamics.hpp>
#include <Observer.hpp>
#include <boost/numeric/odeint.hpp>
#include "System.hpp"
#include "Observer.hpp"
#include "PC.hpp"

#include <ShapeFitterTri.hpp>
#include <ShapeFitterBezier.hpp>

#include <limits>
#include <chrono>
#include <boost/progress.hpp>


// Various constants that set up the visibility emulator scenario

// Lidar settings
#define ROW_RESOLUTION 128
#define COL_RESOLUTION 128
#define ROW_FOV 20
#define COL_FOV 20

// Instrument operating frequency
#define INSTRUMENT_FREQUENCY 0.0016

// Noise
#define FOCAL_LENGTH 1e1
#define LOS_NOISE_3SD_BASELINE 1e0
#define LOS_NOISE_FRACTION_MES_TRUTH 0.

// Times (s)
#define T0 0
#define TF 30000// 7 days


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

	// Spherical harmonics coefficients
	arma::mat Cnm;
	arma::mat Snm;

#ifdef __APPLE__
	ShapeModelImporter shape_io_truth(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/bennu_scaled.obj", 1, true);
	Cnm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Cnm_n10_r175.txt", arma::raw_ascii);
	Snm.load("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/gravity/itokawa_150_Snm_n10_r175.txt", arma::raw_ascii);

#elif __linux__
	ShapeModelImporter shape_io_truth(
		"/home/ben/Documents/ASPEN_GUI_LESS/resources/shape_models/itokawa_150_scaled.obj", 1 , false);
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

	shape_io_truth.load_obj_shape_model(&true_shape_model);

	
	true_shape_model.construct_kd_tree_shape(false);

	// ShapeModelBezier true_bezier(&true_shape_model,"",&frame_graph);
	
	// arma::mat R = arma::eye<arma::mat>(3,3);
	
	// arma::mat points = true_bezier.random_sampling(30000,R);
	// points.save("/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/paperGNSki/data/sampled_pc_itokawa_scaled_aligned.txt",arma::raw_ascii);
	// PC::save(points,"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/Apps/paperGNSki/data/sampled_pc_itokawa_scaled_aligned.obj");

	DynamicAnalyses dyn_analyses(&true_shape_model);

	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);
	// args.set_dyn_analyses(&dyn_analyses);
	// args.set_Cnm(&Cnm);
	// args.set_Snm(&Snm);
	// args.set_degree(5);
	// args.set_ref_radius(175);
	// args.set_mu(arma::datum::G * true_shape_model . get_volume() * 1900);

	// Initial state
	arma::vec X0_augmented = arma::zeros<arma::vec>(12);

	double omega = 2 * arma::datum::pi / (12 * 3600);

	arma::vec omega_0 = {0,0,omega};
	X0_augmented.rows(9,11) = omega_0; // Omega_BN(0)

	arma::vec pos_0 = {1000,0,0};
	X0_augmented.rows(0,2) = pos_0; // r_LN(0) in body frame

	// Velocity determined from sma
	double a = 1000;

	double v = sqrt(args.get_mu() * (2 / arma::norm(pos_0) - 1./ a));

	arma::vec vel_0_inertial = {0.1 * v,0,0.9 * v};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);

	X0_augmented.rows(3,5) = vel_0_body; // r'_LN(0) in body frame


	arma::vec times = arma::regspace<arma::vec>(T0,  1./INSTRUMENT_FREQUENCY,  TF); 
	std::vector<double> T_obs;
	for (unsigned int i =0; i < times.n_rows; ++i){
		T_obs.push_back(times(i));
	}

	// Containers
	std::vector<arma::vec> X_augmented;
	auto N_true = X0_augmented.n_rows;


	// Set active inertia here
	args.set_active_inertia(true_shape_model.get_inertia());

	System dynamics(args,
		N_true,
		Dynamics::point_mass_attitude_dxdt_body_frame );

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

	auto tbegin = T_obs.begin();
	auto tend = T_obs.end();

	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_augmented, tbegin, tend,1e-3,
		Observer::push_back_augmented_state(X_augmented));

	

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




// ShapeBuilder filter_arguments
	ShapeBuilderArguments shape_filter_args;

	shape_filter_args.set_estimate_shape(false);

	ShapeBuilder shape_filter(&frame_graph,
		&lidar,
		&true_shape_model,
		&shape_filter_args);


	shape_filter.run_shape_reconstruction(times,X_augmented,true);

	shape_filter_args.save_results();

	return 0;
}












