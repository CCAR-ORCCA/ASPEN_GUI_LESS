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
#include <StatePropagator.hpp>

#include <NavigationFilter.hpp>
#include <SBGATSphericalHarmo.hpp>

#include <chrono>
#include <boost/progress.hpp>
#include <boost/numeric/odeint.hpp>
#include <vtkOBJReader.h>

#include "json.hpp"


// Various constants that set up the visibility emulator scenario

// Lidar settings
#define ROW_RESOLUTION 128 // Goldeneye
#define COL_RESOLUTION 128 // Goldeneye
#define ROW_FOV 20 // ?
#define COL_FOV 20 // ?

// Instrument specs
#define FOCAL_LENGTH 1e1 // meters
#define INSTRUMENT_FREQUENCY_NAV 0.000145 // frequency at which point clouds are collected during the navigation phase
#define SKIP_FACTOR 0.94 // between 0 and 1 . Determines the focal plane fraction that will be kept during the navigation phase (as a fraction of ROW_RESOLUTION)

// Noise

#define LOS_NOISE_FRACTION_MES_TRUTH 0.

// Process noise 
#define PROCESS_NOISE_SIGMA_VEL 2e-8 // velocity
#define PROCESS_NOISE_SIGMA_OMEG 1e-10 // angular velocity

// Shape fitting parameters
#define POINTS_RETAINED 2000000 // Number of points to be retained in the shape fitting
#define RIDGE_COEF 0e-5 // Ridge coef (regularization of normal equations)
#define N_EDGES 4000 // Number of edges in a-priori
#define SHAPE_DEGREE 2 // Shape degree
#define N_ITER_SHAPE_FILTER 5 // Filter iterations
#define TARGET_SHAPE "itokawa_64_scaled_aligned" // Target shape
#define N_ITER_BUNDLE_ADJUSTMENT 3 // Number of iterations in bundle adjustment

// IOD parameters

// Navigation parameters
#define USE_PHAT_IN_BATCH false // If true, the state covariance is used to provide an a-priori to the batch
#define N_ITER_MES_UPDATE 10 // Number of iterations in the navigation filter measurement update
#define USE_CONSISTENCY_TEST false // If true, will exit IEKF if consistency test is satisfied


///////////////////////////////////////////

int main() {

	// Loading case parameters from file
	arma::arma_rng::set_seed(0);

	// Loading input data from json file
	std::ifstream i("input_file.json");
	nlohmann::json input_data;
	i >> input_data;
	std::string SHAPE_RECONSTRUCTION_OUTPUT_DIR = input_data["SHAPE_RECONSTRUCTION_OUTPUT_DIR"];
	std::string dir = input_data["dir"];
	
	std::ifstream j(SHAPE_RECONSTRUCTION_OUTPUT_DIR);
	nlohmann::json shape_reconstruction_output_data;
	j >> shape_reconstruction_output_data;

	// Fetching input data 
	double DENSITY = input_data["DENSITY"];
	double INSTRUMENT_FREQUENCY = input_data["INSTRUMENT_FREQUENCY"];
	double LOS_NOISE_SD_BASELINE = input_data["LOS_NOISE_SD_BASELINE"];
	bool USE_HARMONICS = input_data["USE_HARMONICS"];
	int NAVIGATION_TIMES = input_data["NAVIGATION_TIMES"]; 
	int HARMONICS_DEGREE = input_data["HARMONICS_DEGREE"];	

	// std::vector<std::vector<double>> SHAPE_COVARIANCES = shape_reconstruction_output_data["ESTIMATED_SHAPE_COVARIANCES"];
	std::cout << "Loading shape covariances...\n";
	nlohmann::json SHAPE_COVARIANCES = shape_reconstruction_output_data["ESTIMATED_SHAPE_COVARIANCES"];
	std::cout << "Loading estimated shape path...\n";

	std::string ESTIMATED_SHAPE_PATH = shape_reconstruction_output_data["ESTIMATED_SHAPE_PATH"];
	
	std::cout << "Loading estimated shape spherical harmonics path...\n";

	std::string ESTIMATED_SPHERICAL_HARMONICS = shape_reconstruction_output_data["ESTIMATED_SPHERICAL_HARMONICS"];
	throw;
	


	double tf = (NAVIGATION_TIMES - 1) * 1./INSTRUMENT_FREQUENCY;


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
	ShapeModelTri<ControlPoint> true_shape_model("B", &frame_graph);
	ShapeModelBezier<ControlPoint> estimated_shape_model("E", &frame_graph);

	std::string path_to_true_shape,path_to_estimated_shape;

#ifdef __APPLE__
	path_to_true_shape = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/"+ std::string(TARGET_SHAPE) + ".obj";

#elif __linux__
	path_to_true_shape = "../../../resources/shape_models/" + std::string(TARGET_SHAPE) +".obj";
	path_to_estimated_shape = input_data["ESTIMATED_SHAPE_PATH"];
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

	ShapeModelImporter::load_obj_shape_model(path_to_true_shape, 
		1, true,true_shape_model);

	true_shape_model.construct_kd_tree_shape();

	ShapeModelImporter::load_bezier_shape_model(path_to_estimated_shape, 
		1, true,estimated_shape_model);

	estimated_shape_model.construct_kd_tree_shape();


	// Loading shape covariance data
	for (int e = 0; e < estimated_shape_model.get_NElements(); ++e){

		std::vector<double> shape_covariance_param;
		for (auto p : SHAPE_COVARIANCES[e]){
			shape_covariance_param.push_back(p);
		}
		Bezier & element = estimated_shape_model.get_element(e);

		element.set_patch_covariance(shape_covariance_param);

	}




throw;
// Lidar
	Lidar lidar(&frame_graph,
		"L",
		ROW_FOV,
		COL_FOV ,
		ROW_RESOLUTION,
		COL_RESOLUTION,
		FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY,
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
	args.set_use_P_hat_in_batch(USE_PHAT_IN_BATCH);
	args.set_N_iter_mes_update(N_ITER_MES_UPDATE);
	args.set_use_consistency_test(USE_CONSISTENCY_TEST);
	args.set_skip_factor(SKIP_FACTOR);
	args.set_true_inertia(true_shape_model.get_inertia());


	/******************************************************/
	/********* Computation of spherical harmonics *********/
	/**************** about orbited shape *****************/
	/******************************************************/
	if(USE_HARMONICS){

		vtkSmartPointer<vtkOBJReader> reader = vtkSmartPointer<vtkOBJReader>::New();
		reader -> SetFileName(path_to_true_shape.c_str());
		reader -> Update(); 

		vtkSmartPointer<SBGATSphericalHarmo> spherical_harmonics = vtkSmartPointer<SBGATSphericalHarmo>::New();
		spherical_harmonics -> SetInputConnection(reader -> GetOutputPort());
		spherical_harmonics -> SetDensity(DENSITY);
		spherical_harmonics -> SetScaleMeters();
		spherical_harmonics -> SetReferenceRadius(true_shape_model.get_circumscribing_radius());
	// can be skipped as normalized coefficients is the default parameter
		spherical_harmonics -> IsNormalized(); 
		spherical_harmonics -> SetDegree(HARMONICS_DEGREE);
		spherical_harmonics -> Update();

	// The spherical harmonics are saved to a file
		spherical_harmonics -> SaveToJson("../output/harmo_" + std::string(TARGET_SHAPE) + ".json");
		args.set_sbgat_harmonics(spherical_harmonics);
	}
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

	arma::vec::fixed<12> X0_true = {
		input_data["X0_TRUE_SPACECRAFT"][0],
		input_data["X0_TRUE_SPACECRAFT"][1],
		input_data["X0_TRUE_SPACECRAFT"][2],
		input_data["X0_TRUE_SPACECRAFT"][3],
		input_data["X0_TRUE_SPACECRAFT"][4],
		input_data["X0_TRUE_SPACECRAFT"][5],
		input_data["X0_TRUE_SMALL_BODY"][0],
		input_data["X0_TRUE_SMALL_BODY"][1],
		input_data["X0_TRUE_SMALL_BODY"][2],
		input_data["X0_TRUE_SMALL_BODY"][3],
		input_data["X0_TRUE_SMALL_BODY"][4],
		input_data["X0_TRUE_SMALL_BODY"][5]
	};

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
	std::vector<arma::vec> X_true;

	if(USE_HARMONICS){
		StatePropagator::propagateOrbit(T_obs,X_true, 
			0, 1./INSTRUMENT_FREQUENCY,NAVIGATION_TIMES, 
			X0_true,
			Dynamics::harmonics_attitude_dxdt_inertial,args,
			dir + "/","obs_harmonics");
		StatePropagator::propagateOrbit(0, tf, 10. , X0_true,
			Dynamics::harmonics_attitude_dxdt_inertial,args,
			dir + "/","full_orbit_harmonics");
	}
	else{
		StatePropagator::propagateOrbit(T_obs,X_true, 
			0, 1./INSTRUMENT_FREQUENCY,NAVIGATION_TIMES, 
			X0_true,
			Dynamics::point_mass_attitude_dxdt_inertial,args,
			dir + "/","obs_point_mass");

		StatePropagator::propagateOrbit(0, tf, 10. , X0_true,
			Dynamics::point_mass_attitude_dxdt_inertial,args,
			dir + "/","full_orbit_point_mass");
	}

	arma::vec times(T_obs.size()); 
	for (int i = 0; i < T_obs.size(); ++i){
		times(i) = T_obs[i];
	}

	/******************************************************/
	/******************************************************/
	/******************************************************/
	/******************************************************/
	/******************************************************/


/******************************************************/
	/******************************************************/
	/*************** Estimated Initial state ****************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	// Initial state

	arma::vec::fixed<12> X0_estimated = {
		input_data["X0_ESTIMATED_SPACECRAFT"][0],
		input_data["X0_ESTIMATED_SPACECRAFT"][1],
		input_data["X0_ESTIMATED_SPACECRAFT"][2],
		input_data["X0_ESTIMATED_SPACECRAFT"][3],
		input_data["X0_ESTIMATED_SPACECRAFT"][4],
		input_data["X0_ESTIMATED_SPACECRAFT"][5],
		input_data["X0_ESTIMATED_SMALL_BODY"][0],
		input_data["X0_ESTIMATED_SMALL_BODY"][1],
		input_data["X0_ESTIMATED_SMALL_BODY"][2],
		input_data["X0_ESTIMATED_SMALL_BODY"][3],
		input_data["X0_ESTIMATED_SMALL_BODY"][4],
		input_data["X0_ESTIMATED_SMALL_BODY"][5]
	};

// A-priori covariance on spacecraft state and asteroid state.
	
	arma::mat::fixed<12,12> P0;





	std::cout << "True State: " << std::endl;
	std::cout << X0_true.t() << std::endl;

	std::cout << "Initial Estimated state: " << std::endl;
	std::cout << X0_estimated.t() << std::endl;

	std::cout << "Initial Error: " << std::endl;
	std::cout << (X0_true-X0_estimated).t() << std::endl;

	arma::vec nav_times(NAVIGATION_TIMES);

	// Times
	std::vector<double> nav_times_vec;
	for (unsigned int i = 0; i < NAVIGATION_TIMES; ++i){
		nav_times(i) = double(i)/INSTRUMENT_FREQUENCY_NAV;
		nav_times_vec.push_back( nav_times(i));
	}

	NavigationFilter filter(args);
	arma::mat Q = Dynamics::create_Q(PROCESS_NOISE_SIGMA_VEL,
		PROCESS_NOISE_SIGMA_OMEG);
	filter.set_gamma_fun(Dynamics::gamma_OD);

	filter.set_observations_fun(
		Observations::obs_pos_mrp_ekf_computed,
		Observations::obs_pos_mrp_ekf_computed_jac,
		Observations::obs_pos_mrp_ekf_lidar);	


	# if USE_HARMONICS
	filter.set_estimate_dynamics_fun(
		Dynamics::estimated_point_mass_attitude_dxdt_inertial,
		Dynamics::estimated_point_mass_jac_attitude_dxdt_inertial,
		Dynamics::harmonics_attitude_dxdt_inertial);
	#else 
	filter.set_estimate_dynamics_fun(
		Dynamics::estimated_point_mass_attitude_dxdt_inertial,
		Dynamics::estimated_point_mass_jac_attitude_dxdt_inertial,
		Dynamics::point_mass_attitude_dxdt_inertial);
	#endif 


	filter.set_initial_information_matrix(arma::inv(P0));

	auto start = std::chrono::system_clock::now();

	int iter = filter.run(1,
		X0_true,
		X0_estimated,
		nav_times_vec,
		arma::ones<arma::mat>(1,1),
		Q);

	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << " Done running filter " << elapsed_seconds.count() << " s\n";


	filter.write_estimated_state(dir + "/X_hat.txt");
	filter.write_true_state(dir + "/X_true.txt");
	filter.write_T_obs(nav_times_vec,dir + "/nav_times.txt");
	filter.write_estimated_covariance(dir + "/covariances.txt");

	return 0;
}
