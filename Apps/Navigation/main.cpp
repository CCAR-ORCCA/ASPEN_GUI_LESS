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
	bool USE_HARMONICS = input_data["USE_HARMONICS"];
	int OBSERVATION_TIMES = input_data["OBSERVATION_TIMES"]; 
	int NAVIGATION_TIMES = input_data["NAVIGATION_TIMES"]; 
	int HARMONICS_DEGREE = input_data["HARMONICS_DEGREE"];	
	int NUMBER_OF_EDGES = input_data["NUMBER_OF_EDGES"];
	int IOD_PARTICLES= input_data["IOD_PARTICLES"]; 
	int IOD_ITERATIONS  = input_data["IOD_ITERATIONS"]; 
	int IOD_RIGID_TRANSFORMS_NUMBER = input_data["IOD_RIGID_TRANSFORMS_NUMBER"]; 

	
	arma::vec::fixed<3> MRP_0 = {input_data["MRP_0"][0],input_data["MRP_0"][1],input_data["MRP_0"][2]};
	
	std::vector<std::vector<double>> SHAPE_COVARIANCES = input_data["ESTIMATED_SHAPE_COVARIANCES"];
	std::String ESTIMATED_SHAPE_PATH = input_data["ESTIMATED_SHAPE_PATH"];
	std::String ESTIMATED_SPHERICAL_HARMONICS = input_data["ESTIMATED_SPHERICAL_HARMONICS"];










	std::string dir = input_data["dir"];

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
arma::vec X0_augmented = arma::zeros<arma::vec>(12);

arma::vec kep_state_vec = {SMA,E,I,RAAN,PERI_OMEGA,M0};
OC::KepState kep_state(kep_state_vec,args.get_mu());
OC::CartState cart_state = kep_state.convert_to_cart(0);

X0_augmented.rows(0,2) = cart_state.get_position_vector();
X0_augmented.rows(3,5) = cart_state.get_velocity_vector();

X0_augmented.rows(6,8) = MRP_0;
X0_augmented.rows(9,11) = omega_0;

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

if(USE_HARMONICS){
	StatePropagator::propagateOrbit(T_obs,X_augmented, 
		T0, 1./INSTRUMENT_FREQUENCY_SHAPE,OBSERVATION_TIMES, 
		X0_augmented,
		Dynamics::harmonics_attitude_dxdt_inertial,args,
		dir + "/","obs_harmonics");
	StatePropagator::propagateOrbit(T0, T_orbit, 10. , X0_augmented,
		Dynamics::harmonics_attitude_dxdt_inertial,args,
		dir + "/","full_orbit_harmonics");
}
else{
	StatePropagator::propagateOrbit(T_obs,X_augmented, 
		T0, 1./INSTRUMENT_FREQUENCY_SHAPE,OBSERVATION_TIMES, 
		X0_augmented,
		Dynamics::point_mass_attitude_dxdt_inertial,args,
		dir + "/","obs_point_mass");

	StatePropagator::propagateOrbit(T0, T_orbit, 10. , X0_augmented,
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


// A-priori covariance on spacecraft state and asteroid state.
arma::vec P0_diag = {
		SIGMA_POS * SIGMA_POS,SIGMA_POS * SIGMA_POS,SIGMA_POS * SIGMA_POS,//position
		SIGMA_VEL * SIGMA_VEL,SIGMA_VEL * SIGMA_VEL,SIGMA_VEL * SIGMA_VEL,//velocity
		SIGMA_MRP * SIGMA_MRP,SIGMA_MRP * SIGMA_MRP,SIGMA_MRP * SIGMA_MRP,// mrp
		SIGMA_OMEGA * SIGMA_OMEGA,SIGMA_OMEGA * SIGMA_OMEGA,SIGMA_OMEGA * SIGMA_OMEGA // angular velocity
	};

	arma::mat P0 = arma::diagmat(P0_diag);

	arma::vec X0_true_augmented = X_augmented.back();
	arma::vec X0_estimated_augmented = X_augmented.back();

	// The initial estimated state is assembled from the output of the shape reconstruction filter
	std::cout << "Generating initial a-priori from rigid transforms ...\n";
	X0_estimated_augmented.subvec(0,2) = shape_filter_args.get_position_final();
	X0_estimated_augmented.subvec(3,5) = shape_filter_args.get_velocity_final();
	X0_estimated_augmented.subvec(6,8) = shape_filter_args.get_mrp_EN_final();
	X0_estimated_augmented.subvec(9,11) = shape_filter_args.get_omega_EN_final();


	std::cout << "True State: " << std::endl;
	std::cout << X0_true_augmented.t() << std::endl;

	std::cout << "Initial Estimated state: " << std::endl;
	std::cout << X0_estimated_augmented.t() << std::endl;

	std::cout << "Initial Error: " << std::endl;
	std::cout << (X0_true_augmented-X0_estimated_augmented).t() << std::endl;

	arma::vec nav_times(NAVIGATION_TIMES);

	// Times
	std::vector<double> nav_times_vec;
	for (unsigned int i = 0; i < NAVIGATION_TIMES; ++i){
		nav_times(i) = T_obs[T_obs.size() - 1] + double(i)/INSTRUMENT_FREQUENCY_NAV;
		nav_times_vec.push_back( nav_times(i));
	}


	NavigationFilter filter(args);
	arma::mat Q = Dynamics::create_Q(PROCESS_NOISE_SIGMA_VEL,PROCESS_NOISE_SIGMA_OMEG);
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
	int iter = filter.run(1,X0_true_augmented,X0_estimated_augmented,nav_times_vec,arma::ones<arma::mat>(1,1),Q);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << " Done running filter " << elapsed_seconds.count() << " s\n";


	filter.write_estimated_state("../output/filter/X_hat.txt");
	filter.write_true_state("../output/filter/X_true.txt");
	filter.write_T_obs(nav_times_vec,"../output/filter/nav_times.txt");
	filter.write_estimated_covariance("../output/filter/covariances.txt");
