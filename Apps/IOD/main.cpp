#include <Lidar.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelImporter.hpp>
#include <ShapeBuilder.hpp>
#include <Observations.hpp>
#include <Dynamics.hpp>
#include <Observer.hpp>
#include <System.hpp>
#include <ShapeBuilderArguments.hpp>
#include <PC.hpp>
#include "IODFinder.hpp"
#include "StatePropagator.hpp"
#include <NavigationFilter.hpp>
#include <SBGATSphericalHarmo.hpp>

#include <chrono>
#include <boost/progress.hpp>
#include <boost/numeric/odeint.hpp>
#include <vtkOBJReader.h>

// Various constants that set up the scenario
#define TARGET_SHAPE "itokawa_64_scaled_aligned" // Target shape

// Lidar settings
#define ROW_RESOLUTION 64 // Goldeneye
#define COL_RESOLUTION 64 // Goldeneye
#define ROW_FOV 20 // ?
#define COL_FOV 20 // ?

// Instrument specs
#define FOCAL_LENGTH 1e1 // meters
#define INSTRUMENT_FREQUENCY_SHAPE 0.001 // frequency at which point clouds are collected for the shape reconstruction phase

// Noise
#define LOS_NOISE_SD_BASELINE 50e-2
#define LOS_NOISE_FRACTION_MES_TRUTH 0.

// Times
#define T0 0
#define OBSERVATION_TIMES 15 // shape reconstruction steps

// Shape fitting parameters
#define N_ITER_BUNDLE_ADJUSTMENT 6 // Number of iterations in bundle adjustment

// IOD parameters
#define IOD_RIGID_TRANSFORMS_NUMBER 15 // Number of rigid transforms to be used in each IOD run
#define IOD_PARTICLES 500 // Number of particles (10000 seems a minimum)
#define IOD_ITERATIONS 50 // Number of iterations
#define IOD_MC_ITER 600

// Target properties
#define SPIN_RATE 12. // Spin rate (hours)
#define DENSITY 1900 // Density (kg/m^3)
#define USE_HARMONICS false // if true, will use the spherical harmonics expansion of the target's gravity field
#define HARMONICS_DEGREE 10 // degree of the spherical harmonics expansion

// Rigid transform artificial noise
#define RIGID_TRANSFORM_X_SD 0.0001
#define RIGID_TRANSFORM_SIGMA_SD 0.000

#define USE_BA false // Whether or not the bundle adjustment should be used
#define USE_ICP false // Whether or not the ICP should be used (if not, uses true rigid transforms)

// Initial state
#define SMA 1000
#define E 0.25
#define I 1.4
#define RAAN 0.2
#define PERI_OMEGA 0.3
#define M0 0.1


#define LABEL ""


///////////////////////////////////////////

int main() {

	arma::arma_rng::set_seed(0);


// Ref frame graph
	FrameGraph frame_graph;
	frame_graph.add_frame("B");
	frame_graph.add_frame("L");
	frame_graph.add_frame("N");
	frame_graph.add_frame("E");

	frame_graph.add_transform("N", "B");
	frame_graph.add_transform("N", "E");
	frame_graph.add_transform("N", "L");

	// Shape model formed with triangles
	ShapeModelTri true_shape_model("B", &frame_graph);

	std::string path_to_shape;

#ifdef __APPLE__
	path_to_shape = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/"+ std::string(TARGET_SHAPE) + ".obj";
#elif __linux__
	path_to_shape = "../../../resources/shape_models/" + std::string(TARGET_SHAPE) +".obj";
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

	ShapeModelImporter shape_io_truth(path_to_shape, 1 , true);
	shape_io_truth.load_obj_shape_model(&true_shape_model);
	true_shape_model.construct_kd_tree_shape();


// Lidar
	Lidar lidar(&frame_graph,
		"L",
		ROW_FOV,
		COL_FOV ,
		ROW_RESOLUTION,
		COL_RESOLUTION,
		FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY_SHAPE,
		LOS_NOISE_SD_BASELINE,
		LOS_NOISE_FRACTION_MES_TRUTH);

	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);
	args.set_mu(arma::datum::G * true_shape_model . get_volume() * DENSITY);
	args.set_mass(true_shape_model . get_volume() * DENSITY);
	args.set_lidar(&lidar);
	args.set_sd_noise(LOS_NOISE_SD_BASELINE);
	args.set_sd_noise_prop(LOS_NOISE_FRACTION_MES_TRUTH);
	args.set_true_inertia(true_shape_model.get_inertia());

	/******************************************************/
	/********* Computation of spherical harmonics *********/
	/**************** about orbited shape *****************/
	/******************************************************/
	# if USE_HARMONICS
	vtkSmartPointer<vtkOBJReader> reader = vtkSmartPointer<vtkOBJReader>::New();
	reader -> SetFileName(path_to_shape.c_str());
	reader -> Update(); 

	vtkSmartPointer<SBGATSphericalHarmo> spherical_harmonics = vtkSmartPointer<SBGATSphericalHarmo>::New();
	spherical_harmonics -> SetInputConnection(reader -> GetOutputPort());
	spherical_harmonics -> SetDensity(DENSITY);
	spherical_harmonics -> SetScaleMeters();
	spherical_harmonics -> SetReferenceRadius(true_shape_model.get_circumscribing_radius());
	spherical_harmonics -> IsNormalized(); // can be skipped as normalized coefficients is the default parameter
	spherical_harmonics -> SetDegree(HARMONICS_DEGREE);
	spherical_harmonics -> Update();

	// The spherical harmonics are saved to a file
	spherical_harmonics -> SaveToJson("../output/harmo_" + std::string(TARGET_SHAPE) + ".json");
	args.set_sbgat_harmonics(spherical_harmonics);
	#endif
	
	/******************************************************/
	/******************************************************/
	/******************************************************/


	/******************************************************/
	/******************************************************/
	/***************( True ) Initial state ****************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	// Initial state
	arma::vec X0_augmented = arma::zeros<arma::vec>(12);

	// MRP BN 
	arma::vec mrp_0 = {0.,0.,0.};

	// Angular velocity in body frame
	double omega = 2 * arma::datum::pi / (SPIN_RATE * 3600);
	arma::vec omega_0 = {0e-2 * omega,0e-2 * omega,omega};

	// - sma : semi-major axis [L]
	// 	- e : eccentricity [-]
	// 	- i : inclination in [0,pi] [rad]
	// 	- Omega : right-ascension of ascending node in [0,2 pi] [rad] 
	// 	- omega : longitude of perigee [0,2 pi] [rad] 
	// 	- M0 : mean anomaly at epoch [rad]

	arma::vec kep_state_vec = {SMA,E,I,RAAN,PERI_OMEGA,M0};
	OC::KepState kep_state(kep_state_vec,args.get_mu());
	OC::CartState cart_state = kep_state.convert_to_cart(0);

	X0_augmented.rows(0,2) = cart_state.get_position_vector();
	X0_augmented.rows(3,5) = cart_state.get_velocity_vector();

	X0_augmented.rows(6,8) = mrp_0;
	X0_augmented.rows(9,11) = omega_0;

	double true_mu = arma::datum::G * true_shape_model . get_volume() * DENSITY;

	/******************************************************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	/******************************************************/
	/******************************************************/
	/*********** Propagation of (true) state **************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	std::vector<double> T_obs;
	std::vector<arma::vec> X_augmented;
	
	#if USE_HARMONICS
	StatePropagator::propagateOrbit(T_obs,X_augmented, 
		T0, 1./INSTRUMENT_FREQUENCY_SHAPE,OBSERVATION_TIMES, 
		X0_augmented,
		Dynamics::harmonics_attitude_dxdt_inertial,args,
		"../output/traj/","obs_harmonics_" + std::string(LABEL));
	StatePropagator::propagateOrbit(T0, 2 * arma::datum::pi * std::sqrt(std::pow(SMA,3) / true_mu), 10. , X0_augmented,
		Dynamics::harmonics_attitude_dxdt_inertial,args,
		"../output/traj/","full_orbit_harmonics_" + std::string(LABEL));
	#else 
	StatePropagator::propagateOrbit(T_obs,X_augmented, 
		T0, 1./INSTRUMENT_FREQUENCY_SHAPE,OBSERVATION_TIMES, 
		X0_augmented,
		Dynamics::point_mass_attitude_dxdt_inertial,args,
		"../output/traj/","obs_point_mass_" + std::string(LABEL));

	StatePropagator::propagateOrbit(T0, 2 * arma::datum::pi * std::sqrt(std::pow(SMA,3) / true_mu), 10. , X0_augmented,
		Dynamics::point_mass_attitude_dxdt_inertial,args,
		"../output/traj/","full_orbit_point_mass_" + std::string(LABEL));
	#endif 


	arma::vec times(T_obs.size()); 
	for (int i = 0; i < T_obs.size(); ++i){
		times(i) = T_obs[i];
	}



	/******************************************************/
	/******************************************************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	ShapeBuilderArguments shape_filter_args;
	shape_filter_args.set_los_noise_sd_baseline(LOS_NOISE_SD_BASELINE);
	shape_filter_args.set_N_iter_bundle_adjustment(N_ITER_BUNDLE_ADJUSTMENT);
	shape_filter_args.set_use_ba(USE_BA);
	shape_filter_args.set_iod_rigid_transforms_number(IOD_RIGID_TRANSFORMS_NUMBER);
	shape_filter_args.set_iod_particles(IOD_PARTICLES);
	shape_filter_args.set_iod_iterations(IOD_ITERATIONS);
	shape_filter_args.set_use_icp(USE_ICP);
	shape_filter_args.set_rigid_transform_noise_sd("X",RIGID_TRANSFORM_X_SD);
	shape_filter_args.set_rigid_transform_noise_sd("sigma",RIGID_TRANSFORM_SIGMA_SD);
	shape_filter_args.set_iod_mc_iter(IOD_MC_ITER);
	ShapeBuilder shape_filter(&frame_graph,&lidar,&true_shape_model,&shape_filter_args);

	std::cout << X_augmented.front() << std::endl;

	std::cout << "True mu: " << true_mu << std::endl;
	std::cout << "True period: " << 2 * arma::datum::pi * std::sqrt(std::pow(SMA,3) / true_mu) / (3600) << " hours" << std::endl;

	shape_filter.run_iod(times,X_augmented);


	return 0;
}











