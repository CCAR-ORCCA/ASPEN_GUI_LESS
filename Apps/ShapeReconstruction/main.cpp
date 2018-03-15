#include <Lidar.hpp>
#include <ShapeModelTri.hpp>
#include <ShapeModelImporter.hpp>
#include <ShapeBuilder.hpp>
#include <Observations.hpp>
#include <Dynamics.hpp>
#include <Observer.hpp>
#include <System.hpp>
#include <ShapeBuilderArguments.hpp>

#include <NavigationFilter.hpp>
#include <ShapeFitterBezier.hpp>

#include <chrono>
#include <boost/progress.hpp>
#include <boost/numeric/odeint.hpp>

// Various constants that set up the visibility emulator scenario

// Lidar settings
#define ROW_RESOLUTION 128 // Goldeneye
#define COL_RESOLUTION 128 // Goldeneye
#define ROW_FOV 20 // ?
#define COL_FOV 20 // ?

// Instrument operating frequency
#define INSTRUMENT_FREQUENCY_SHAPE 0.0016
#define INSTRUMENT_FREQUENCY_NAV 0.000145


// Noise
#define FOCAL_LENGTH 1e1
#define LOS_NOISE_SD_BASELINE 5e-2 
#define LOS_NOISE_FRACTION_MES_TRUTH 0.

// Process noise (m/s^2)
#define PROCESS_NOISE_SIGMA 1e-9

// Times (s)
#define T0 0
#define TF 600000 // 10 days

// Indices
#define INDEX_INIT 400 // index where shape reconstruction takes place
#define INDEX_END 400 // end of shape fitting (must be less or equal than the number of simulation time. this is checked)

// Downsampling factor (between 0 and 1)
#define DOWNSAMPLING_FACTOR 0.1

// Ridge coef (regularization of normal equations)
#define RIDGE_COEF 0e-4

// Filter iterations
#define ITER_FILTER 4

// Number of edges in a-priori
#define N_EDGES 3000

// Shape order
#define SHAPE_DEGREE 2

// Target shape
#define TARGET_SHAPE "itokawa_64_scaled_aligned"

// Spin rate (hours)
#define SPIN_RATE 12.

// Density (kg/m^3)
#define DENSITY 1900

///////////////////////////////////////////


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

#ifdef __APPLE__
	ShapeModelImporter shape_io_truth(
		"/Users/bbercovici/GDrive/CUBoulder/Research/code/ASPEN_gui_less/resources/shape_models/"+ std::string(TARGET_SHAPE) + ".obj", 1, true);
#elif __linux__
	ShapeModelImporter shape_io_truth(
		"../../../resources/shape_models/" +std::string(TARGET_SHAPE) +".obj", 1 , true);
#else

	throw (std::runtime_error("Neither running on linux or mac os"));
#endif

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

	// Initial state
	arma::vec X0_augmented = arma::zeros<arma::vec>(12);

	double omega = 2 * arma::datum::pi / (SPIN_RATE * 3600);

	arma::vec omega_0 = {0,0,omega};
	X0_augmented.rows(9,11) = omega_0; // Omega_BN(0)

	arma::vec pos_0 = {1000,0,0};
	X0_augmented.rows(0,2) = pos_0; // r_LN(0) in body frame

	// Velocity determined from sma
	double a = arma::norm(pos_0);
	double v = sqrt(args.get_mu() * (2 / arma::norm(pos_0) - 1./ a));

	arma::vec vel_0_inertial = {0,std::cos(arma::datum::pi/6)*v,std::sin(arma::datum::pi/6)*v};
	arma::vec vel_0_body = vel_0_inertial - arma::cross(omega_0,pos_0);

	X0_augmented.rows(3,5) = vel_0_body; // r'_LN(0) in body frame

	arma::vec times = arma::regspace<arma::vec>(T0,  1./INSTRUMENT_FREQUENCY_SHAPE,  TF); 
	arma::vec times_dense = arma::regspace<arma::vec>(T0,  10,  TF); 

	std::vector<double> T_obs;
	std::vector<double> T_obs_dense;

	for (unsigned int i = 0; i < times.n_rows; ++i){
		T_obs.push_back(times(i));
	}
	for (unsigned int i = 0; i < times_dense.n_rows; ++i){
		T_obs_dense.push_back(times_dense(i));
	}

	// Containers
	std::vector<arma::vec> X_augmented;
	std::vector<arma::vec> X_augmented_dense;

	auto N_true = X0_augmented.n_rows;

	// Set active inertia here
	args.set_active_inertia(true_shape_model.get_inertia());

	System dynamics(args,N_true,Dynamics::point_mass_attitude_dxdt_body_frame );

	typedef boost::numeric::odeint::runge_kutta_cash_karp54< arma::vec > error_stepper_type;
	auto stepper = boost::numeric::odeint::make_controlled<error_stepper_type>( 1.0e-10 , 1.0e-16 );

	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_augmented, T_obs.begin(), T_obs.end(),1e-3,
		Observer::push_back_augmented_state(X_augmented));

	// The orbit is propagated with a finer timestep for visualization purposes
	boost::numeric::odeint::integrate_times(stepper, dynamics, X0_augmented, T_obs_dense.begin(), 
		T_obs_dense.end(),1e-3,
		Observer::push_back_augmented_state(X_augmented_dense));

	arma::mat X_dense(12,X_augmented_dense.size());
	for (unsigned int i = 0; i < X_augmented_dense.size(); ++i){
		X_dense.col(i) = X_augmented_dense[i];
	}
	X_dense.save("../output/trajectory.txt",arma::raw_ascii);
	


	ShapeBuilderArguments shape_filter_args;
	shape_filter_args.set_los_noise_sd_baseline(LOS_NOISE_SD_BASELINE);
	shape_filter_args.set_index_init(std::min(INDEX_INIT,int(times.size())));
	shape_filter_args.set_index_end(std::min(INDEX_END,int(times.size())));
	shape_filter_args.set_downsampling_factor(DOWNSAMPLING_FACTOR);
	shape_filter_args.set_iter_filter(ITER_FILTER);
	shape_filter_args.set_N_edges(N_EDGES);
	shape_filter_args.set_shape_degree(SHAPE_DEGREE);


	ShapeBuilder shape_filter(&frame_graph,&lidar,&true_shape_model,&shape_filter_args);


	auto start = std::chrono::system_clock::now();
	shape_filter.run_shape_reconstruction(times,X_augmented,true);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << "Point cloud acquisition and shape fitting completed in " << elapsed_seconds.count() << " s"<< std::endl;
	

	// At this stage, the bezier shape model is NOT aligned with the true shape model
	std::shared_ptr<ShapeModelBezier> estimated_shape_model = shape_filter.get_estimated_shape_model();

	// std::shared_ptr<ShapeModelBezier> estimated_shape_model = std::make_shared<ShapeModelBezier>(ShapeModelBezier(&true_shape_model,"E",&frame_graph));

	std::cout << "Estimated shape model barycenter: " << std::endl;
	estimated_shape_model -> update_mass_properties();
	std::cout << estimated_shape_model -> get_center_of_mass().t() << std::endl;
	estimated_shape_model -> shift_to_barycenter();
	estimated_shape_model -> align_with_principal_axes();
	estimated_shape_model -> update_mass_properties();
	std::cout << estimated_shape_model -> get_center_of_mass().t() << std::endl;
	std::cout << estimated_shape_model -> get_volume() << std::endl;




	estimated_shape_model -> save_both("../output/shape_model/fit_shape_aligned");
	estimated_shape_model -> construct_kd_tree_shape();
	args.set_estimated_shape_model(estimated_shape_model.get());

	/**
	END OF SHAPE RECONSTRUCTION FILTER
	*/

	/**
	BEGINNING OF NAVIGATION FILTER
	*/

	// A-priori covariance on spacecraft state and asteroid state.
	// Since the asteroid state is not estimated, it is frozen
	arma::vec P0_diag = {0.001,0.001,0.001,0.001,0.001,0.001,1e-20,1e-20,1e-20,1e-20,1e-20,1e-20};
	arma::vec P0_spacecraft_vec = {100,100,100,1e-6,1e-6,1e-6};

	P0_diag.subvec(0,5) = P0_spacecraft_vec;

	arma::mat P0 = arma::diagmat(P0_diag);


	arma::vec X0_true_augmented = X_augmented[INDEX_END];

	arma::vec X0_estimated_augmented = X_augmented[INDEX_END];


	X0_estimated_augmented.subvec(0,5) += arma::diagmat(arma::sqrt(P0_spacecraft_vec)) * arma::randn(6);


	std::cout << "True state: " << std::endl;
	std::cout << X0_true_augmented.t() << std::endl;


	std::cout << "Initial Estimated state: " << std::endl;
	std::cout << X0_estimated_augmented.t() << std::endl;

	std::cout << "Initial Error: " << std::endl;
	std::cout << (X0_true_augmented-X0_estimated_augmented).t() << std::endl;



	arma::vec nav_times = arma::regspace<arma::vec>(T0 + T_obs[INDEX_END],  1./INSTRUMENT_FREQUENCY_NAV,  TF + T_obs[INDEX_END]); 
	
	// Times
	std::vector<double> nav_times_vec;
	for (unsigned int i = 0; i < nav_times.n_rows; ++i){
		nav_times_vec.push_back( nav_times(i));
	}

	
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

	arma::mat Q = std::pow(PROCESS_NOISE_SIGMA ,2) * arma::eye<arma::mat>(3,3);

	start = std::chrono::system_clock::now();
	int iter = filter.run(1,X0_true_augmented,X0_estimated_augmented,nav_times_vec,arma::ones<arma::mat>(1,1),Q);
	end = std::chrono::system_clock::now();

	elapsed_seconds = end-start;

	std::cout << " Done running filter " << elapsed_seconds.count() << " s\n";

	filter.write_estimated_state("../output/filter/X_hat.txt");
	filter.write_true_state("../output/filter/X_true.txt");
	filter.write_T_obs(nav_times_vec,"../output/filter/nav_times.txt");
	filter.write_estimated_covariance("../output/filter/covariances.txt");




































	return 0;
}












