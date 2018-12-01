#include <Lidar.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelBezier.hpp>
#include <ShapeModelImporter.hpp>
#include <ShapeBuilder.hpp>
#include <Observations.hpp>
#include <Dynamics.hpp>
#include <Observer.hpp>
#include <SystemDynamics.hpp>
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
#define TARGET_SHAPE "itokawa_64_scaled_aligned" // Target shape

// IOD parameters

// Navigation parameters
#define USE_PHAT_IN_BATCH false // If true, the state covariance is used to provide an a-priori to the batch
#define N_ITER_MES_UPDATE 10 // Number of iterations in the navigation filter measurement update
#define USE_CONSISTENCY_TEST false // If true, will exit IEKF if consistency test is satisfied

// A-priori covariance
#define T0 0

///////////////////////////////////////////

int main() {

	// Loading case parameters from file
	arma::arma_rng::set_seed(0);

	// Loading input data from json file
	std::ifstream i("input_file.json");
	nlohmann::json input_data;
	i >> input_data;

	// Fetching input data 
	double SMA = input_data["SMA"];
	double E = input_data["E"];
	double I = input_data["I"];
	double RAAN = input_data["RAAN"];
	double PERI_OMEGA = input_data["PERI_OMEGA"];
	double M0 = input_data["M0"];
	double LATITUDE_SPIN = input_data["LATITUDE_SPIN"];
	double LONGITUDE_SPIN = input_data["LONGITUDE_SPIN"];
	double SPIN_PERIOD = input_data["SPIN_PERIOD"];
	double DENSITY = input_data["DENSITY"];
	double INSTRUMENT_FREQUENCY_SHAPE = input_data["INSTRUMENT_FREQUENCY_SHAPE"];
	double MIN_TRIANGLE_ANGLE = input_data["MIN_TRIANGLE_ANGLE"];
	double MAX_TRIANGLE_SIZE = input_data["MAX_TRIANGLE_SIZE"];
	double SURFACE_APPROX_ERROR = input_data["SURFACE_APPROX_ERROR"];
	double LOS_NOISE_SD_BASELINE = input_data["LOS_NOISE_SD_BASELINE"];
	double DISTANCE_FROM_SUN_AU = input_data["DISTANCE_FROM_SUN_AU"];
	double CR_TRUTH = input_data["CR_TRUTH"];

	bool USE_HARMONICS = input_data["USE_HARMONICS"];
	int OBSERVATION_TIMES = input_data["OBSERVATION_TIMES"]; 
	int HARMONICS_DEGREE = input_data["HARMONICS_DEGREE"];	
	int NUMBER_OF_EDGES = input_data["NUMBER_OF_EDGES"];
	int IOD_PARTICLES= input_data["IOD_PARTICLES"]; 
	int IOD_ITERATIONS  = input_data["IOD_ITERATIONS"]; 
	int IOD_RIGID_TRANSFORMS_NUMBER = input_data["IOD_RIGID_TRANSFORMS_NUMBER"]; 
	int N_ITER_BUNDLE_ADJUSTMENT = input_data["N_ITER_BUNDLE_ADJUSTMENT"];
	int N_ITER_SHAPE_FILTER = input_data["N_ITER_SHAPE_FILTER"];
	int BA_H = input_data["BA_H"];

	bool USE_BA = input_data["USE_BA"]; 
	bool USE_ICP = input_data["USE_ICP"];
	bool USE_TRUE_RIGID_TRANSFORMS = input_data["USE_TRUE_RIGID_TRANSFORMS"]; 

	arma::vec::fixed<3> MRP_0 = {input_data["MRP_0"][0],input_data["MRP_0"][1],input_data["MRP_0"][2]};
	
	std::string OUTPUT_DIR = input_data["OUTPUT_DIR"];

	double T_orbit = (OBSERVATION_TIMES - 1) * 1./INSTRUMENT_FREQUENCY_SHAPE;

	// Angular velocity in body frame
	double omega = 2 * arma::datum::pi / (SPIN_PERIOD * 3600);
	arma::vec omega_vec = {0,0,omega};
	arma::vec omega_0 = (RBK::M2(-LATITUDE_SPIN) * RBK::M3(LONGITUDE_SPIN)).t() * omega_vec;


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

	std::string path_to_shape;

#ifdef __APPLE__
	path_to_shape = "/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/"+ std::string(TARGET_SHAPE) + ".obj";
#elif __linux__
	path_to_shape = "../../../resources/shape_models/" +std::string(TARGET_SHAPE) +".obj";
#else
	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

	ShapeModelImporter::load_obj_shape_model(path_to_shape, 1, true,true_shape_model);

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
	args.set_distance_from_sun_AU(DISTANCE_FROM_SUN_AU);

	/******************************************************/
	/********* Computation of spherical harmonics *********/
	/**************** about orbited shape *****************/
	/******************************************************/
	if(USE_HARMONICS){

		vtkSmartPointer<vtkOBJReader> reader = vtkSmartPointer<vtkOBJReader>::New();
		reader -> SetFileName(path_to_shape.c_str());
		reader -> Update(); 

		vtkSmartPointer<SBGATSphericalHarmo> spherical_harmonics = vtkSmartPointer<SBGATSphericalHarmo>::New();
		spherical_harmonics -> SetInputConnection(reader -> GetOutputPort());
		spherical_harmonics -> SetDensity(DENSITY);
		spherical_harmonics -> SetScaleMeters();
		spherical_harmonics -> SetReferenceRadius(true_shape_model.get_circumscribing_radius());
		spherical_harmonics -> IsNormalized(); 
		spherical_harmonics -> SetDegree(HARMONICS_DEGREE);
		spherical_harmonics -> Update();

	// The spherical harmonics are saved to a file
		spherical_harmonics -> SaveToJson("../output/harmo_" + std::string(TARGET_SHAPE) + ".json");
		args.set_sbgat_harmonics_truth(spherical_harmonics);
	}
	/******************************************************/
	/******************************************************/
	/******************************************************/


	// Construction of the true system dynamics

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



	/******************************************************/
	/******************************************************/
	/***************( True ) Initial state ****************/
	/******************************************************/
	/******************************************************/
	/******************************************************/

	// Initial state
	arma::vec X0_augmented = arma::zeros<arma::vec>(14);

	arma::vec kep_state_vec = {SMA,E,I,RAAN,PERI_OMEGA,M0};
	OC::KepState kep_state(kep_state_vec,args.get_mu_truth());
	OC::CartState cart_state = kep_state.convert_to_cart(0);

	X0_augmented.rows(0,2) = cart_state.get_position_vector();
	X0_augmented.rows(3,5) = cart_state.get_velocity_vector();
	X0_augmented.rows(6,8) = MRP_0;
	X0_augmented.rows(9,11) = omega_0;
	X0_augmented(12) = args.get_mu_truth();
	X0_augmented(13) = CR_TRUTH;

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
	std::cout << "Propagating true state\n";
	
	StatePropagator::propagate(T_obs,X_augmented, 
		T0, 1./INSTRUMENT_FREQUENCY_SHAPE,OBSERVATION_TIMES, 
		X0_augmented,
		dynamics_system_truth,
		args,
		OUTPUT_DIR + "/","obs_point_mass");

	StatePropagator::propagate(T0, T_orbit, 10. , X0_augmented,
		dynamics_system_truth,
		args,
		OUTPUT_DIR + "/","full_orbit_point_mass");
	
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
	shape_filter_args.set_points_retained(POINTS_RETAINED);
	shape_filter_args.set_N_iter_shape_filter(N_ITER_SHAPE_FILTER);
	shape_filter_args.set_N_edges(N_EDGES);
	shape_filter_args.set_shape_degree(SHAPE_DEGREE);
	shape_filter_args.set_use_icp(USE_ICP);
	shape_filter_args.set_N_iter_bundle_adjustment(N_ITER_BUNDLE_ADJUSTMENT);
	shape_filter_args.set_use_ba(USE_BA);
	shape_filter_args.set_iod_rigid_transforms_number(IOD_RIGID_TRANSFORMS_NUMBER);
	shape_filter_args.set_iod_particles(IOD_PARTICLES);
	shape_filter_args.set_iod_iterations(IOD_ITERATIONS);
	shape_filter_args.set_use_true_rigid_transforms(USE_TRUE_RIGID_TRANSFORMS);
	shape_filter_args.set_min_triangle_angle(MIN_TRIANGLE_ANGLE);
	shape_filter_args.set_min_triangle_angle(MIN_TRIANGLE_ANGLE);
	shape_filter_args.set_max_triangle_size(MAX_TRIANGLE_SIZE);
	shape_filter_args.set_surface_approx_error(SURFACE_APPROX_ERROR);
	shape_filter_args.set_number_of_edges(NUMBER_OF_EDGES);
	shape_filter_args.set_ba_h(BA_H);
	

	std::cout << "True state at initial time: " << cart_state.get_state().t() << std::endl;
	std::cout << "\t with mu = " << cart_state.get_mu() << std::endl;

	ShapeBuilder shape_filter(&frame_graph,&lidar,&true_shape_model,&shape_filter_args);

	shape_filter.run_shape_reconstruction(times,X_augmented,OUTPUT_DIR);

	nlohmann::json output_data;
	std::string path_to_estimated_shape = "";
	std::string path_to_estimated_spherical_harmonics = "";


	arma::vec X_estimated = shape_filter.get_estimated_state();
	arma::mat covariance_estimated_state = shape_filter.get_covariance_estimated_state();

	covariance_estimated_state.save(OUTPUT_DIR + "/covariance_estimated_state.txt",arma::raw_ascii);

	std::cout << "Fetching output data...\n";
	output_data["X0_TRUE_SPACECRAFT"] = { 
		X_augmented.back()[0], 
		X_augmented.back()[1], 
		X_augmented.back()[2],
		X_augmented.back()[3], 
		X_augmented.back()[4], 
		X_augmented.back()[5]
	};

	output_data["X0_TRUE_SMALL_BODY"] = { 
		X_augmented.back()[6], 
		X_augmented.back()[7], 
		X_augmented.back()[8],
		X_augmented.back()[9], 
		X_augmented.back()[10], 
		X_augmented.back()[11]
	};

	output_data["X0_ESTIMATED_SPACECRAFT"] = { 
		X_estimated[0], 
		X_estimated[1], 
		X_estimated[2],
		X_estimated[3], 
		X_estimated[4], 
		X_estimated[5]
	};

	output_data["X0_ESTIMATED_SMALL_BODY"] = { 
		X_estimated[6], 
		X_estimated[7], 
		X_estimated[8],
		X_estimated[9], 
		X_estimated[10], 
		X_estimated[11]
	};
	output_data["CR_TRUTH"] = CR_TRUTH;

	output_data["ESTIMATED_SMALL_BODY_MU"] = X_estimated[12];
	output_data["ESTIMATED_SMALL_BODY_CR"] = 1.1;
	output_data["DISTANCE_FROM_SUN_AU"] = DISTANCE_FROM_SUN_AU;

	nlohmann::json shape_covariances_data;
	std::cout << "Exporting covariances ...\n";
	for (int e = 0; e < shape_filter.get_estimated_shape_model() -> get_NElements(); ++e){

		const arma::vec & P_X_param = shape_filter.get_estimated_shape_model() -> get_element(e).get_P_X_param();
		std::vector<double> P_X_param_vector;
		for (int i = 0; i < P_X_param.n_rows; ++i){
			P_X_param_vector.push_back(P_X_param(i));
		}
		shape_covariances_data.push_back(P_X_param_vector);
	}

	output_data["ESTIMATED_SHAPE_COVARIANCES"] = shape_covariances_data;
	output_data["ESTIMATED_SHAPE_PATH"] = OUTPUT_DIR + "/fit_shape_B_frame";
	output_data["ESTIMATED_SPHERICAL_HARMONICS"] = path_to_estimated_spherical_harmonics;

	std::ofstream o(OUTPUT_DIR + "/output_file_from_shape_reconstruction.json");
	o << output_data;

	// std::vector<std::array<double ,2> > shape_error_results;
	// std::vector<arma::vec> spurious_points;

	// // The shape error is computed here
	// for (unsigned int i = 0; i < estimated_shape_model -> get_NElements(); ++i){

	// 	Bezier * patch = static_cast<Bezier *>( estimated_shape_model -> get_elements() -> at(i).get() );
	// 	arma::vec center = patch -> evaluate(1./3,1./3);
	// 	arma::vec normal = patch -> get_normal_coordinates(1./3, 1./3);

	// 	arma::mat P = patch -> covariance_surface_point(1./3,1./3,normal);
	// 	double sd = std::sqrt(arma::dot(normal,P * normal));

	// 	Ray ray_n(center,normal);
	// 	Ray ray_mn(center,-normal);

	// 	for (unsigned int facet_index = 0; facet_index < true_shape_model.get_NElements(); ++facet_index){

	// 		Facet * facet = static_cast<Facet *>(true_shape_model.get_elements() -> at(facet_index).get());

	// 		ray_n.single_facet_ray_casting(facet,true,false);
	// 		ray_mn.single_facet_ray_casting(facet,true,false);

	// 	}

	// 	if (ray_n.get_true_range() < ray_mn.get_true_range()){
	// 		shape_error_results.push_back({sd,ray_n.get_true_range()});
	// 	}
	// 	else if (ray_n.get_true_range() > ray_mn.get_true_range()){
	// 		shape_error_results.push_back({sd,ray_mn.get_true_range()});

	// 	}
	// 	else{
	// 		std::cout << "This ray did not hit\n";
	// 	}


	// }


	// arma::mat shape_error_arma(shape_error_results.size(),2);
	// for (unsigned int j = 0; j < shape_error_results.size(); ++j){
	// 	shape_error_arma(j,0) = shape_error_results[j][0];
	// 	shape_error_arma(j,1) = shape_error_results[j][1];
	// }


	// shape_error_arma.save("../output/shape_error.txt",arma::raw_ascii);



	// args.set_estimated_mass(estimated_shape_model -> get_volume() * DENSITY);
	// args.set_estimated_inertia(estimated_shape_model -> get_inertia() );

	// std::cout << "Estimated inertia:" << std::endl;
	// std::cout << estimated_shape_model -> get_inertia() << std::endl;

	// std::cout << "True inertia:" << std::endl;
	// std::cout << true_shape_model. get_inertia() << std::endl;

	// std::cout << "\nEstimated volume: " << estimated_shape_model -> get_volume();
	// std::cout << "\nTrue volume: " << true_shape_model.get_volume();
	// std::cout << "\nVolume sd: " << estimated_shape_model -> get_volume_sd() << std::endl << std::endl;
	// // estimated_shape_model -> compute_cm_cov();

	// std::cout << "\nCOM covariance: \n" << estimated_shape_model -> get_cm_cov() << std::endl;


	// /***************************************/
	// /* END OF SHAPE RECONSTRUCTION FILTER */
	// /*************************************/


	return 0;
}












