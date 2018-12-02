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


// Shape fitting parameters
#define TARGET_SHAPE "itokawa_64_scaled_aligned" // Target shape

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
	std::string OUTPUT_DIR = input_data["OUTPUT_DIR"];

	
	std::ifstream j(SHAPE_RECONSTRUCTION_OUTPUT_DIR + "output_file_from_shape_reconstruction.json");
	nlohmann::json shape_reconstruction_output_data;
	j >> shape_reconstruction_output_data;

	// Fetching input data 
	double DENSITY = input_data["DENSITY"];
	double INSTRUMENT_FREQUENCY_NAV = input_data["INSTRUMENT_FREQUENCY_NAV"];
	double LOS_NOISE_SD_BASELINE = input_data["LOS_NOISE_SD_BASELINE"];
	double LOS_NOISE_FRACTION_MES_TRUTH = input_data["LOS_NOISE_FRACTION_MES_TRUTH"];
	double PROCESS_NOISE_SIGMA_VEL = input_data["PROCESS_NOISE_SIGMA_VEL"];
	double PROCESS_NOISE_SIGMA_OMEG = input_data["PROCESS_NOISE_SIGMA_OMEG"];
	double SKIP_FACTOR = input_data["SKIP_FACTOR"];

	bool USE_HARMONICS = input_data["USE_HARMONICS"];
	bool USE_HARMONICS_ESTIMATED_DYNAMICS = input_data["USE_HARMONICS_ESTIMATED_DYNAMICS"];

	int NAVIGATION_TIMES = input_data["NAVIGATION_TIMES"]; 
	int HARMONICS_DEGREE = input_data["HARMONICS_DEGREE"];	

	std::string ESTIMATED_SHAPE_PATH = shape_reconstruction_output_data["ESTIMATED_SHAPE_PATH"];
	std::string ESTIMATED_SPHERICAL_HARMONICS = shape_reconstruction_output_data["ESTIMATED_SPHERICAL_HARMONICS"];
	nlohmann::json SHAPE_COVARIANCES = shape_reconstruction_output_data["ESTIMATED_SHAPE_COVARIANCES"];
	double CR_TRUTH = shape_reconstruction_output_data["CR_TRUTH"];
	double DISTANCE_FROM_SUN_AU = shape_reconstruction_output_data["DISTANCE_FROM_SUN_AU"];
	double tf = (NAVIGATION_TIMES - 1) * 1./INSTRUMENT_FREQUENCY_NAV;


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
	ShapeModelBezier<ControlPoint> estimated_shape_model_to_elevate("", &frame_graph);


	std::string path_to_true_shape,path_to_estimated_shape;

#ifdef __APPLE__
	path_to_true_shape = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/"+ std::string(TARGET_SHAPE) + ".obj";

#elif __linux__
	path_to_true_shape = "../../../resources/shape_models/" + std::string(TARGET_SHAPE) +".obj";
	path_to_estimated_shape = shape_reconstruction_output_data["ESTIMATED_SHAPE_PATH"];
	path_to_estimated_shape += ".b";
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

	ShapeModelImporter::load_obj_shape_model(path_to_true_shape, 
		1, true,true_shape_model);

	true_shape_model.construct_kd_tree_shape();

	ShapeModelImporter::load_bezier_shape_model(path_to_estimated_shape, 
		1, true,estimated_shape_model);

	ShapeModelImporter::load_bezier_shape_model(path_to_estimated_shape, 
		1, true,estimated_shape_model_to_elevate);

	estimated_shape_model.construct_kd_tree_shape();

		// std::vector<std::vector<double>> SHAPE_COVARIANCES = shape_reconstruction_output_data["ESTIMATED_SHAPE_COVARIANCES"];
	
	// Loading shape covariance data
	for (int e = 0; e < estimated_shape_model.get_NElements(); ++e){

		std::vector<double> shape_covariance_param;
		for (auto p : SHAPE_COVARIANCES[e]){
			shape_covariance_param.push_back(p);
		}
		Bezier & element = estimated_shape_model.get_element(e);

		element.set_patch_covariance(shape_covariance_param);

	}




// Lidar
	Lidar lidar(&frame_graph,
		"L",
		ROW_FOV,
		COL_FOV ,
		ROW_RESOLUTION,
		COL_RESOLUTION,
		FOCAL_LENGTH,
		INSTRUMENT_FREQUENCY_NAV,
		LOS_NOISE_SD_BASELINE,
		LOS_NOISE_FRACTION_MES_TRUTH);

	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);
	args.set_mu_truth(arma::datum::G * true_shape_model . get_volume() * DENSITY);
	args.set_mass_truth(true_shape_model . get_volume() * DENSITY);
	args.set_lidar(&lidar);
	args.set_sd_noise(LOS_NOISE_SD_BASELINE);
	args.set_sd_noise_prop(LOS_NOISE_FRACTION_MES_TRUTH);
	args.set_use_P_hat_in_batch(USE_PHAT_IN_BATCH);
	args.set_N_iter_mes_update(N_ITER_MES_UPDATE);
	args.set_use_consistency_test(USE_CONSISTENCY_TEST);
	args.set_skip_factor(SKIP_FACTOR);
	args.set_inertia_truth(true_shape_model.get_inertia());
	args.set_estimated_shape_model(&estimated_shape_model);
	
	args.set_inertia_estimate(estimated_shape_model.get_inertia());
	args.set_output_dir(OUTPUT_DIR);
	args.set_distance_from_sun_AU(DISTANCE_FROM_SUN_AU);

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
		spherical_harmonics -> SaveToJson(OUTPUT_DIR + "/harmonics_truth.json");
		args.set_sbgat_harmonics_truth(spherical_harmonics);
	}
	/******************************************************/
	/******************************************************/
	/******************************************************/


	if (USE_HARMONICS_ESTIMATED_DYNAMICS){
		estimated_shape_model_to_elevate.elevate_degree();
		estimated_shape_model_to_elevate.elevate_degree();
		estimated_shape_model_to_elevate.elevate_degree();
		estimated_shape_model_to_elevate.elevate_degree();
		estimated_shape_model_to_elevate.save_both(OUTPUT_DIR + "/elevated_estimated_shape_for_harmonics");


		std::string path_to_estimated_elevated_shape = OUTPUT_DIR + "/elevated_estimated_shape_for_harmonics.obj";

		vtkSmartPointer<vtkOBJReader> reader = vtkSmartPointer<vtkOBJReader>::New();
		reader -> SetFileName(path_to_estimated_elevated_shape.c_str());
		reader -> Update(); 

		vtkSmartPointer<SBGATSphericalHarmo> spherical_harmonics = vtkSmartPointer<SBGATSphericalHarmo>::New();
		spherical_harmonics -> SetInputConnection(reader -> GetOutputPort());
		spherical_harmonics -> SetDensity(1);// the density is set to 1 so that the 
		// filter can just multiply the computed acceleration by the estimated density
		spherical_harmonics -> SetScaleMeters();
		spherical_harmonics -> SetReferenceRadius(estimated_shape_model.get_circumscribing_radius());

	// can be skipped as normalized coefficients is the default parameter
		spherical_harmonics -> IsNormalized(); 
		spherical_harmonics -> SetDegree(HARMONICS_DEGREE);
		spherical_harmonics -> Update();

	// The spherical harmonics are saved to a file
		spherical_harmonics -> SaveToJson(OUTPUT_DIR + "/harmonics_estimate.json");
		args.set_sbgat_harmonics_estimate(spherical_harmonics);

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

	arma::vec X0_true = {
		shape_reconstruction_output_data["X0_TRUE_SPACECRAFT"][0],
		shape_reconstruction_output_data["X0_TRUE_SPACECRAFT"][1],
		shape_reconstruction_output_data["X0_TRUE_SPACECRAFT"][2],
		shape_reconstruction_output_data["X0_TRUE_SPACECRAFT"][3],
		shape_reconstruction_output_data["X0_TRUE_SPACECRAFT"][4],
		shape_reconstruction_output_data["X0_TRUE_SPACECRAFT"][5],
		shape_reconstruction_output_data["X0_TRUE_SMALL_BODY"][0],
		shape_reconstruction_output_data["X0_TRUE_SMALL_BODY"][1],
		shape_reconstruction_output_data["X0_TRUE_SMALL_BODY"][2],
		shape_reconstruction_output_data["X0_TRUE_SMALL_BODY"][3],
		shape_reconstruction_output_data["X0_TRUE_SMALL_BODY"][4],
		shape_reconstruction_output_data["X0_TRUE_SMALL_BODY"][5],
		arma::datum::G * true_shape_model . get_volume() * DENSITY,
		CR_TRUTH
	};
	std::cout << "True State: " << std::endl;
	std::cout << X0_true.t() << std::endl;

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

	arma::vec::fixed<14> X0_estimated;
	arma::mat::fixed<14,14> P0 = arma::zeros<arma::mat>(14,14);

	if (input_data["USE_TRUE_STATES"]){

		std::cout << "USING TRUE STATES TO INITIALIZE THE FILTER\n";
		X0_estimated = X0_true;

		P0(0,0) = 1e1;
		P0(1,1) = 1e1;
		P0(2,2) = 1e1;

		P0(3,3) = 1e-1;
		P0(4,4) = 1e-1;
		P0(5,5) = 1e-1;

		P0(6,6) = 1e-4;
		P0(7,7) = 1e-4;
		P0(8,8) = 1e-4;

		P0(9,9) = 1e-10;
		P0(10,10) = 1e-10;
		P0(11,11) = 1e-10;
		P0(12,12) = 10;
		P0(13,13) = 0.1;


	}

	else{
		std::cout << "USING ESTIMATED STATES TO INITIALIZE THE FILTER\n";

		X0_estimated = {
			shape_reconstruction_output_data["X0_ESTIMATED_SPACECRAFT"][0],
			shape_reconstruction_output_data["X0_ESTIMATED_SPACECRAFT"][1],
			shape_reconstruction_output_data["X0_ESTIMATED_SPACECRAFT"][2],
			shape_reconstruction_output_data["X0_ESTIMATED_SPACECRAFT"][3],
			shape_reconstruction_output_data["X0_ESTIMATED_SPACECRAFT"][4],
			shape_reconstruction_output_data["X0_ESTIMATED_SPACECRAFT"][5],
			shape_reconstruction_output_data["X0_ESTIMATED_SMALL_BODY"][0],
			shape_reconstruction_output_data["X0_ESTIMATED_SMALL_BODY"][1],
			shape_reconstruction_output_data["X0_ESTIMATED_SMALL_BODY"][2],
			shape_reconstruction_output_data["X0_ESTIMATED_SMALL_BODY"][3],
			shape_reconstruction_output_data["X0_ESTIMATED_SMALL_BODY"][4],
			shape_reconstruction_output_data["X0_ESTIMATED_SMALL_BODY"][5],
			shape_reconstruction_output_data["ESTIMATED_SMALL_BODY_MU"],
			shape_reconstruction_output_data["ESTIMATED_SMALL_BODY_CR"]
		};

		if(!P0.load(SHAPE_RECONSTRUCTION_OUTPUT_DIR + "/covariance_estimated_state.txt",arma::raw_ascii)){
			throw(std::runtime_error("Error loading P0 from " + SHAPE_RECONSTRUCTION_OUTPUT_DIR + "/covariance_estimated_state.txt"));
		}
	}




	std::cout << "Removing correlations in position/velocity\n";
	P0.submat(0,5,0,5) = arma::diagmat(P0.submat(0,5,0,5).diag());






// A-priori covariance on spacecraft state and asteroid state.
	

	
	std::cout << "Initial Estimated state: " << std::endl;
	std::cout << X0_estimated.t() << std::endl;


	std::cout << "Initial Estimated state covariance: " << std::endl;
	std::cout << P0 << std::endl;

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
	
	arma::mat Q = Dynamics::create_Q(PROCESS_NOISE_SIGMA_VEL,PROCESS_NOISE_SIGMA_OMEG);
	
	filter.set_gamma_fun(Dynamics::gamma_OD);

	filter.set_observations_funs(
		Observations::obs_pos_mrp_ekf_computed,
		Observations::obs_pos_mrp_ekf_computed_jac,
		Observations::obs_pos_mrp_ekf_lidar);	

	/****************************************/
	/****************************************/
	/******* TRUE STATE DYNAMICS *******/
	/****************************************/
	/****************************************/
	/****************************************/

	SystemDynamics dynamics_system_truth(args);

	dynamics_system_truth.add_next_state("r",3,false);
	dynamics_system_truth.add_next_state("r_dot",3,false);
	dynamics_system_truth.add_next_state("sigma_BN",3,true);
	dynamics_system_truth.add_next_state("omega_BN",3,false);
	dynamics_system_truth.add_next_state("mu",1,false);
	dynamics_system_truth.add_next_state("CR_TRUTH",1,false);

	dynamics_system_truth.add_dynamics("r",Dynamics::velocity,{"r_dot"});
	
	if (USE_HARMONICS){
		dynamics_system_truth.add_dynamics("r_dot",Dynamics::spherical_harmonics_acceleration_truth,{"r","sigma_BN"});
	}
	else{
		dynamics_system_truth.add_dynamics("r_dot",Dynamics::point_mass_acceleration,{"r","mu"});
	}
	
	dynamics_system_truth.add_dynamics("r_dot",Dynamics::SRP_cannonball,{"CR_TRUTH"});
	
	dynamics_system_truth.add_dynamics("sigma_BN",Dynamics::dmrp_dt,{"sigma_BN","omega_BN"});
	dynamics_system_truth.add_dynamics("omega_BN",Dynamics::domega_dt_truth,{"sigma_BN","omega_BN"});


	filter.set_true_dynamics_system(dynamics_system_truth);

	/****************************************/
	/*************** END OF *****************/
	/********** TRUE STATE DYNAMICS *********/
	/****************************************/
	/****************************************/
	/****************************************/

	/******************************************************/
	/******************************************************/
	/*********** Propagation of (true) state **************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	std::vector<double> T_obs;
	std::vector<arma::vec> X_true;

	std::cout << "Propagating true state\n";
	
	StatePropagator::propagate(T_obs,X_true, 
		0, 1./INSTRUMENT_FREQUENCY_NAV,NAVIGATION_TIMES, 
		X0_true,
		dynamics_system_truth,
		args,
		OUTPUT_DIR + "/","true_orbit");

	StatePropagator::propagate(0, tf, 10. , X0_true,
		dynamics_system_truth,
		args,
		OUTPUT_DIR + "/","true_orbit_dense");
	
	arma::vec times(T_obs.size()); 
	for (int i = 0; i < T_obs.size(); ++i){
		times(i) = T_obs[i];
	}


	/******************************************************/
	/******************************************************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	/****************************************/
	/****************************************/
	/******* ESTIMATED STATE DYNAMICS *******/
	/****************************************/
	/****************************************/
	/****************************************/

	// Estimated state dynamics

	SystemDynamics dynamics_system_estimate(args);

	dynamics_system_estimate.add_next_state("r",3,false);
	dynamics_system_estimate.add_next_state("r_dot",3,false);
	dynamics_system_estimate.add_next_state("sigma_BN",3,true);
	dynamics_system_estimate.add_next_state("omega_BN",3,false);
	dynamics_system_estimate.add_next_state("mu",1,false);
	dynamics_system_estimate.add_next_state("CR_ESTIMATE",1,false);

	dynamics_system_estimate.add_dynamics("r",Dynamics::velocity,{"r_dot"});
	
	if (USE_HARMONICS_ESTIMATED_DYNAMICS){
		dynamics_system_estimate.add_dynamics("r_dot",Dynamics::spherical_harmonics_acceleration_estimate,{"r","sigma_BN","mu"});
	}
	else{
		dynamics_system_estimate.add_dynamics("r_dot",Dynamics::point_mass_acceleration,{"r","mu"});
	}
	dynamics_system_estimate.add_dynamics("r_dot",Dynamics::SRP_cannonball,{"CR_ESTIMATE"});
	dynamics_system_estimate.add_dynamics("sigma_BN",Dynamics::dmrp_dt,{"sigma_BN","omega_BN"});
	dynamics_system_estimate.add_dynamics("omega_BN",Dynamics::domega_dt_estimate,{"sigma_BN","omega_BN"});


	
	dynamics_system_estimate.add_jacobian("r","r_dot",Dynamics::identity_33,{"r_dot"});
	
	if (USE_HARMONICS_ESTIMATED_DYNAMICS){
		dynamics_system_estimate.add_jacobian("r_dot","r",Dynamics::spherical_harmonics_gravity_gradient_matrix_estimate,{"r","sigma_BN","mu"});
		dynamics_system_estimate.add_jacobian("r_dot","mu",Dynamics::spherical_harmonics_acceleration_estimate_unit_mu,{"r","sigma_BN"});
	}
	else{
		dynamics_system_estimate.add_jacobian("r_dot","r",Dynamics::point_mass_gravity_gradient_matrix,{"r","mu"});
		dynamics_system_estimate.add_jacobian("r_dot","mu",Dynamics::point_mass_acceleration_unit_mu,{"r"});
	}

	dynamics_system_estimate.add_jacobian("r_dot","CR_ESTIMATE",Dynamics::SRP_cannonball_unit_C,{"CR_ESTIMATE"});

	dynamics_system_estimate.add_jacobian("sigma_BN","sigma_BN",Dynamics::partial_mrp_dot_partial_mrp,{"sigma_BN","omega_BN"});
	dynamics_system_estimate.add_jacobian("sigma_BN","omega_BN",Dynamics::partial_mrp_dot_partial_omega,{"sigma_BN"});
	dynamics_system_estimate.add_jacobian("omega_BN","omega_BN",Dynamics::partial_omega_dot_partial_omega_estimate,{"sigma_BN","omega_BN"});
	
	filter.set_estimated_dynamics_system(dynamics_system_estimate);


	/****************************************/
	/*************** END OF *****************/
	/******* ESTIMATED STATE DYNAMICS *******/
	/****************************************/
	/****************************************/
	/****************************************/


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


	filter.write_estimated_state(OUTPUT_DIR + "/X_hat.txt");
	filter.write_true_state(OUTPUT_DIR + "/X_true.txt");
	filter.write_T_obs(nav_times_vec,OUTPUT_DIR + "/nav_times.txt");
	filter.write_estimated_covariance(OUTPUT_DIR + "/covariances.txt");

	return 0;
}
