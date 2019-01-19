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

#define TARGET_SHAPE "itokawa_64_scaled_aligned" // Target shape

int main(){


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
	double TF =  double(input_data["TF"]) * 3600;
	double T0 = 0;
	
	bool USE_HARMONICS = input_data["USE_HARMONICS"];
	int HARMONICS_DEGREE = input_data["HARMONICS_DEGREE"];	
	int NUMBER_OF_EDGES = input_data["NUMBER_OF_EDGES"];
	int IOD_PARTICLES = input_data["IOD_PARTICLES"]; 
	int IOD_ITERATIONS  = input_data["IOD_ITERATIONS"]; 
	int IOD_RIGID_TRANSFORMS_NUMBER = input_data["IOD_RIGID_TRANSFORMS_NUMBER"]; 
	int N_ITER_BUNDLE_ADJUSTMENT = input_data["N_ITER_BUNDLE_ADJUSTMENT"];
	int N_ITER_SHAPE_FILTER = input_data["N_ITER_SHAPE_FILTER"];
	int BA_H = input_data["BA_H"];

	bool USE_BA = input_data["USE_BA"]; 
	bool USE_ICP = input_data["USE_ICP"];
	bool USE_TRUE_RIGID_TRANSFORMS = input_data["USE_TRUE_RIGID_TRANSFORMS"]; 
	bool USE_BEZIER_SHAPE = input_data["USE_BEZIER_SHAPE"]; 

	arma::vec::fixed<3> MRP_0 = {input_data["MRP_0"][0],input_data["MRP_0"][1],input_data["MRP_0"][2]};

	std::string OUTPUT_DIR = input_data["OUTPUT_DIR"];

	// Kep state arguments
	// - sma : semi-major axis [L]
	// - e : eccentricity [-]
	// - i : inclination in [0,pi] [rad]
	// - Omega : right-ascension of ascending node in [0,2 pi] [rad] 
	// - omega : longitude of perigee [0,2 pi] [rad] 
	// - M0 : mean anomaly at epoch [rad]

	double au2meters = 149597870700;
	double d2r = arma::datum::pi / 180.;

	arma::vec itokaw_kep_state = {
		1.3241 * au2meters,
		0.2802,
		0 * 1.6215 * d2r,
		0 * 69.080 * d2r,
		0 * 162.81 * d2r,
		0
	};

	double mu_sun = 1.32712440018 * 10e20 ;

	OC::KepState kep_state_small_body(itokaw_kep_state,mu_sun);

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



	// Integrator extra arguments
	Args args;
	args.set_frame_graph(&frame_graph);
	args.set_true_shape_model(&true_shape_model);
	args.set_mu_truth(arma::datum::G * true_shape_model . get_volume() * DENSITY);
	args.set_mass_truth(true_shape_model . get_volume() * DENSITY);
	args.set_inertia_truth(true_shape_model.get_inertia());
	args.set_kep_state_small_body(kep_state_small_body);

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
	dynamics_system_truth.add_dynamics("r_dot",Dynamics::SRP_cannonball_truth,{"r","sigma_BN","CR_TRUTH"});
	dynamics_system_truth.add_dynamics("r_dot",Dynamics::third_body_acceleration_itokawa_sun,{"r"});


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

	StatePropagator::propagate(
		T_obs,
		X_augmented,
		T0, 
		TF, 
		10. , 
		X0_augmented,
		dynamics_system_truth,
		args,
		OUTPUT_DIR + "/","full_orbit_point_mass");

	arma::vec times(T_obs.size()); 
	for (int i = 0; i < T_obs.size(); ++i){
		times(i) = T_obs[i];
	}

	// Evaluating the force models along the trajectory
	
	arma::mat SRP_forces(3,T_obs.size());
	arma::mat third_body_forces(3,T_obs.size());
	arma::mat SH_gravity_forces(3,T_obs.size());
	arma::mat point_mass_gravity_forces(3,T_obs.size());

	for (int i = 0; i < T_obs.size(); ++i){


		arma::vec gravity_input_states = {
			X_augmented[i](0),
			X_augmented[i](1),
			X_augmented[i](2),
			X_augmented[i](6),
			X_augmented[i](7),
			X_augmented[i](8)
		};


		arma::vec point_mass_state = {
			X_augmented[i](0),
			X_augmented[i](1),
			X_augmented[i](2),
			X_augmented[i](12)
		};

		arma::vec srp_input_states = {
			X_augmented[i](0),
			X_augmented[i](1),
			X_augmented[i](2),
			X_augmented[i](6),
			X_augmented[i](7),
			X_augmented[i](8),
			X_augmented[i](13)
		};

		third_body_forces.col(i) = Dynamics::third_body_acceleration_itokawa_sun(times(i),X_augmented[i].subvec(0,2),args);
		SRP_forces.col(i) = Dynamics::SRP_cannonball_truth(times(i),srp_input_states,args);
		SH_gravity_forces.col(i) = Dynamics::spherical_harmonics_acceleration_truth(times(i),gravity_input_states,args);
		point_mass_gravity_forces.col(i) = Dynamics::point_mass_acceleration(times(i),point_mass_state,args);


	}

	third_body_forces.save(OUTPUT_DIR + "/" + "third_body_forces.txt",arma::raw_ascii);
	SRP_forces.save(OUTPUT_DIR + "/" + "SRP_forces.txt",arma::raw_ascii);
	SH_gravity_forces.save(OUTPUT_DIR + "/" + "SH_gravity_forces.txt",arma::raw_ascii);
	point_mass_gravity_forces.save(OUTPUT_DIR + "/" + "point_mass_gravity_forces.txt",arma::raw_ascii);







	return 0;
}